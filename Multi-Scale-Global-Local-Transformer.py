import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np


def load_vf_queries_from_file(file_path: str, num_vf: int, d_model: int):
    """Load VF queries initialization from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"VF queries initialization file not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.npy':
        data = np.load(file_path)
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        else:
            raise ValueError(f"Invalid .npy file format")
    elif file_ext in ['.pth', '.pt']:
        data = torch.load(file_path, map_location='cpu')
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"Invalid .pth/.pt file format")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    if data.dim() == 2:
        if data.shape[0] != num_vf or data.shape[1] != d_model:
            raise ValueError(f"Shape mismatch: expected ({num_vf}, {d_model}), got {data.shape}")
        data = data.unsqueeze(0)
    elif data.dim() == 3:
        if data.shape[1] != num_vf or data.shape[2] != d_model:
            raise ValueError(f"Shape mismatch: expected (1, {num_vf}, {d_model}), got {data.shape}")
        if data.shape[0] != 1:
            data = data[0:1]
    
    return data


def sine_2d(coords, d_model):
    """Fixed Sinusoidal 2D Positional Encoding."""
    N = coords.size(0)
    device = coords.device
    dim_per_coord = d_model // 2
    pos_encoding = torch.zeros(N, d_model, device=device)
    num_freqs = dim_per_coord // 2
    div_term = torch.exp(torch.arange(0, num_freqs, dtype=torch.float32, device=device) 
                         * -(math.log(10000.0) / (dim_per_coord - 1)))
    
    x_coords = coords[:, 0:1]
    y_coords = coords[:, 1:2]
    
    for i in range(num_freqs):
        pos_encoding[:, 2*i] = torch.sin(x_coords * div_term[i]).squeeze(-1)
        pos_encoding[:, 2*i + 1] = torch.cos(x_coords * div_term[i]).squeeze(-1)
    
    for i in range(num_freqs):
        pos_encoding[:, dim_per_coord + 2*i] = torch.sin(y_coords * div_term[i]).squeeze(-1)
        pos_encoding[:, dim_per_coord + 2*i + 1] = torch.cos(y_coords * div_term[i]).squeeze(-1)
    
    return pos_encoding


class CrossAttention(nn.Module):
    """Multi-head Cross-Attention."""
    def __init__(self, hidden_size: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = hidden_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def _shape(self, x, B, seq_len):
        return x.view(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    
    def forward(self, query_tokens, context_tokens, return_attn: bool = False):
        B, Nq, _ = query_tokens.size()
        _, Nk, _ = context_tokens.size()

        q = self.query(query_tokens)
        k = self.key(context_tokens)
        v = self.value(context_tokens)

        q = self._shape(q, B, Nq)
        k = self._shape(k, B, Nk)
        v = self._shape(v, B, Nk)

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, Nq, self.all_head_size)
        
        out = self.out(context)
        out = self.proj_dropout(out)

        if return_attn:
            return out, attn_probs
        return out


class MultiLayerCrossAttention(nn.Module):
    """Multi-layer Cross-Attention with residual connections."""
    def __init__(self, hidden_size: int = 256, num_heads: int = 8, dropout: float = 0.0, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            CrossAttention(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
    
    def forward(self, query_tokens, context_tokens, return_attn: bool = False):
        x = query_tokens
        last_attn_probs = None
        
        for i in range(self.num_layers):
            if return_attn and i == self.num_layers - 1:
                attn_out, last_attn_probs = self.layers[i](x, context_tokens, return_attn=True)
            else:
                attn_out = self.layers[i](x, context_tokens)
            x = self.layer_norms[i](x + attn_out)
            
            ffn_out = self.ffns[i](x)
            x = self.ffn_norms[i](x + ffn_out)
        
        if return_attn:
            return x, last_attn_probs
        return x


class ChannelAttention(nn.Module):
    """SE-Net style Channel Attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        out = (avg_out + max_out).view(B, C, 1, 1)
        return x * out.expand_as(x)


class VFSpatialSelfAttention(nn.Module):
    """Self-attention for VF tokens with Gaussian distance bias."""
    def __init__(self, hidden_size: int = 256, num_heads: int = 8, dropout: float = 0.0,
                 vf_coords=None, sigma: float = 0.3):
        super().__init__()
        
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.sigma = sigma
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(hidden_size)
        
        if vf_coords is not None:
            if isinstance(vf_coords, torch.Tensor):
                coords = vf_coords.clone().detach().float()
            else:
                coords = torch.tensor(vf_coords, dtype=torch.float32)
            
            num_vf = coords.size(0)
            coords_expanded_i = coords.unsqueeze(1)
            coords_expanded_j = coords.unsqueeze(0)
            dist_matrix = torch.norm(coords_expanded_i - coords_expanded_j, dim=2)
            distance_bias = -(dist_matrix ** 2) / (2 * sigma ** 2)
            self.register_buffer("distance_bias", distance_bias)
        else:
            self.register_buffer("distance_bias", None)
    
    def _shape(self, x, B, seq_len):
        return x.view(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    
    def forward(self, vf_tokens):
        B, num_vf, _ = vf_tokens.size()
        identity = vf_tokens
        
        q = self.query(vf_tokens)
        k = self.key(vf_tokens)
        v = self.value(vf_tokens)
        
        q = self._shape(q, B, num_vf)
        k = self._shape(k, B, num_vf)
        v = self._shape(v, B, num_vf)
        
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        if self.distance_bias is not None:
            attn_scores = attn_scores + self.distance_bias.unsqueeze(0).unsqueeze(0)
        
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)
        
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, num_vf, self.all_head_size)
        
        out = self.out(context)
        out = self.proj_dropout(out)
        out = self.norm(out + identity)
        
        return out


class FeatureExtractorBackbone(nn.Module):
    """Universal feature extractor backbone."""
    def __init__(self, name: str, in_ch: int = 1, pretrained: bool = True, train_last_n_layers: int = 2):
        super().__init__()
        name = name.lower()

        cfg = {
            "resnet18": {
                "builder": lambda w: models.resnet18(weights=models.ResNet18_Weights.DEFAULT if w else None),
                "nodes": {"layer2": "feat2", "layer3": "feat3", "layer4": "feat4"},
                "channels": [128, 256, 512],
                "adapt": self._adapt_resnet_stem,
            },
            "resnet34": {
                "builder": lambda w: models.resnet34(weights=models.ResNet34_Weights.DEFAULT if w else None),
                "nodes": {"layer2": "feat2", "layer3": "feat3", "layer4": "feat4"},
                "channels": [128, 256, 512],
                "adapt": self._adapt_resnet_stem,
            },
            "resnet50": {
                "builder": lambda w: models.resnet50(weights=models.ResNet50_Weights.DEFAULT if w else None),
                "nodes": {"layer2": "feat2", "layer3": "feat3", "layer4": "feat4"},
                "channels": [512, 1024, 2048],
                "adapt": self._adapt_resnet_stem,
            },
            "convnext_t": {
                "builder": lambda w: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if w else None),
                "nodes": {"features.4": "feat2", "features.6": "feat3", "features.8": "feat4"},
                "channels": [192, 384, 768],
                "adapt": self._adapt_convnext_stem,
            },
            "swin_t": {
                "builder": lambda w: models.swin_t(weights=models.Swin_T_Weights.DEFAULT if w else None),
                "nodes": {"features.1": "feat2", "features.2": "feat3", "features.3": "feat4"},
                "channels": [192, 384, 768],
                "adapt": self._adapt_swin_stem,
            },
            "efficientnet_v2_s": {
                "builder": lambda w: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if w else None),
                "nodes": {"features.4": "feat2", "features.6": "feat3", "features.7": "feat4"},
                "channels": [64, 160, 256],
                "adapt": self._adapt_effnet_stem,
            },
        }

        if name not in cfg:
            raise ValueError(f"Unsupported backbone '{name}'")

        meta = cfg[name]
        model = meta["builder"](pretrained)
        adapt_fn = meta["adapt"]
        adapt_fn(model, in_ch)

        self.extractor = create_feature_extractor(model, return_nodes=meta["nodes"])
        self.out_channels = meta["channels"]

        for p in self.parameters():
            p.requires_grad = False
        self._unfreeze_tail(model, name, train_last_n_layers)

    def _unfreeze_tail(self, model, name: str, n: int):
        name = name.lower()
        if "resnet" in name:
            stages = [model.layer4, model.layer3, model.layer2, model.layer1, model.relu, model.bn1, model.conv1]
        elif "convnext" in name:
            stages = [model.features[8], model.features[6], model.features[4], model.features[2], model.features[0]]
        elif "swin" in name:
            stages = [model.features[3], model.features[2], model.features[1], model.features[0]]
        elif "efficientnet" in name:
            stages = [model.features[7], model.features[6], model.features[5], model.features[4], 
                     model.features[3], model.features[2], model.features[1], model.features[0]]
        else:
            stages = []

        for stage in stages[:max(0, n)]:
            for p in stage.parameters():
                p.requires_grad = True

    @staticmethod
    def _adapt_resnet_stem(model, in_ch: int):
        if in_ch == 3:
            return
        old = model.conv1
        model.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            if in_ch == 1:
                model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            else:
                model.conv1.weight[:, :3].copy_(old.weight)

    @staticmethod
    def _adapt_convnext_stem(model, in_ch: int):
        stem_conv = model.features[0][0]
        if in_ch == stem_conv.in_channels:
            return
        new_conv = nn.Conv2d(in_ch, stem_conv.out_channels, kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride, padding=stem_conv.padding, bias=False)
        with torch.no_grad():
            if in_ch == 1:
                new_conv.weight.copy_(stem_conv.weight.mean(dim=1, keepdim=True))
            else:
                new_conv.weight[:, :stem_conv.in_channels].copy_(stem_conv.weight)
        model.features[0][0] = new_conv

    @staticmethod
    def _adapt_swin_stem(model, in_ch: int):
        proj = model.features[0][0].proj
        if in_ch == proj.in_channels:
            return
        new_proj = nn.Conv2d(in_ch, proj.out_channels, kernel_size=proj.kernel_size,
            stride=proj.stride, padding=proj.padding, bias=proj.bias is not None)
        with torch.no_grad():
            if in_ch == 1:
                new_proj.weight.copy_(proj.weight.mean(dim=1, keepdim=True))
            else:
                new_proj.weight[:, :proj.in_channels].copy_(proj.weight)
                if proj.bias is not None:
                    new_proj.bias.copy_(proj.bias)
        model.features[0][0].proj = new_proj

    @staticmethod
    def _adapt_effnet_stem(model, in_ch: int):
        stem_conv = model.features[0][0]
        if in_ch == stem_conv.in_channels:
            return
        new_conv = nn.Conv2d(in_ch, stem_conv.out_channels, kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride, padding=stem_conv.padding, bias=False)
        with torch.no_grad():
            if in_ch == 1:
                new_conv.weight.copy_(stem_conv.weight.mean(dim=1, keepdim=True))
            else:
                new_conv.weight[:, :stem_conv.in_channels].copy_(stem_conv.weight)
        model.features[0][0] = new_conv

    def forward(self, x):
        feats = self.extractor(x)
        return [feats["feat2"], feats["feat3"], feats["feat4"]]


class ImprovedConvBlock(nn.Module):
    """Improved convolution block with BatchNorm and GELU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 stride=1, use_residual=False, dropout=0.0):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        if self.use_residual:
            out = out + x
        
        return out


class ResNetBackbone(nn.Module):
    """Lightweight ResNet-18 backbone."""
    def __init__(self, in_ch: int = 1, pretrained: bool = True, train_last_n_layers: int = 2):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        if in_ch != 3:
            old_conv1 = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_ch, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
            with torch.no_grad():
                if in_ch == 1:
                    resnet.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
                else:
                    resnet.conv1.weight[:, :3].copy_(old_conv1.weight)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for param in self.parameters():
            param.requires_grad = False

        trainable_layers = [self.layer4, self.layer3, self.layer2, self.layer1, self.stem]
        for layer in trainable_layers[:train_last_n_layers]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        feat2 = x
        x = self.layer2(x)
        feat3 = x
        x = self.layer3(x)
        feat4 = x
        return [feat2, feat3, feat4]


class RNFLBackbone(nn.Module):
    """VGG-like backbone for RNFL feature extraction."""
    def __init__(self, in_ch: int = 1, base_channels=None, use_residual=True, dropout=0.0):
        super().__init__()
        if base_channels is None:
            base_channels = [64, 128, 256, 512]

        c1, c2, c3, c4 = base_channels
        self.maxp = nn.MaxPool2d(2)

        self.conv11 = ImprovedConvBlock(in_ch, c1, use_residual=False, dropout=dropout)
        self.conv12 = ImprovedConvBlock(c1, c1, use_residual=use_residual, dropout=dropout)
        self.conv21 = ImprovedConvBlock(c1, c2, use_residual=False, dropout=dropout)
        self.conv22 = ImprovedConvBlock(c2, c2, use_residual=use_residual, dropout=dropout)
        self.conv31 = ImprovedConvBlock(c2, c3, use_residual=False, dropout=dropout)
        self.conv32 = ImprovedConvBlock(c3, c3, use_residual=use_residual, dropout=dropout)
        self.conv41 = ImprovedConvBlock(c3, c4, use_residual=False, dropout=dropout)
        self.conv42 = ImprovedConvBlock(c4, c4, use_residual=use_residual, dropout=dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)
        feat2 = x
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp(x)
        feat3 = x
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp(x)
        feat4 = x
        return [feat2, feat3, feat4]


class ScaleInteractionModule(nn.Module):
    """Cross-scale feature interaction module."""
    def __init__(self, hidden_size: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = CrossAttention(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, scale_features):
        num_scales = len(scale_features)
        enhanced_features = []
        
        for i in range(num_scales):
            other_scales = [scale_features[j] for j in range(num_scales) if j != i]
            if len(other_scales) > 0:
                context = torch.stack(other_scales, dim=0).mean(dim=0)
                enhanced = self.cross_attn(scale_features[i], context)
                enhanced = self.norm(enhanced + scale_features[i])
                enhanced_features.append(enhanced)
            else:
                enhanced_features.append(scale_features[i])
        
        return enhanced_features


class ParallelFusionBlock(nn.Module):
    """Parallel fusion block for multi-scale features."""
    def __init__(self, d_model: int, num_scales: int, droppath: float = 0.1, use_diversity: bool = True):
        super().__init__()
        self.num_scales = num_scales
        self.use_diversity = use_diversity

        self.weight_gen = nn.Sequential(
            nn.LayerNorm(num_scales * d_model),
            nn.Linear(num_scales * d_model, num_scales)
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.drop_prob = droppath

    def drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.drop_prob == 0.0):
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.size(0),) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    def forward(self, P_in: torch.Tensor, deltas: list[torch.Tensor], return_div: bool = False):
        B, N, D = P_in.shape
        S = len(deltas)

        cat = torch.cat(deltas, dim=-1)
        logits = self.weight_gen(cat)
        w = torch.softmax(logits, dim=-1)

        stack = torch.stack(deltas, dim=2)
        fused = (w.unsqueeze(-1) * stack).sum(dim=2)

        g = self.gate(P_in)
        update = self.drop_path(g * fused)
        P_out = P_in + self.alpha * update

        if return_div and self.use_diversity and S > 1:
            norm = F.normalize(stack, dim=-1)
            sims = []
            for i in range(S):
                for j in range(i+1, S):
                    sims.append((norm[:,:,i,:] * norm[:,:,j,:]).sum(dim=-1))
            div_reg = torch.stack(sims, dim=0).mean()
            return P_out, div_reg

        return P_out


class MultiScaleRNFLToVF(nn.Module):
    """Multi-Scale Cross-Attention model: RNFL image -> VF (52-dim)."""
    def __init__(self, in_ch: int = 1, num_vf: int = 52, d_model: int = 256,
                 num_heads: int = 8, dropout: float = 0.1, sigma: float = 0.6,
                 backbone_dropout: float = 0.05, use_residual: bool = True,
                 backbone_type: str = "resnet34", resnet_pretrained: bool = True,
                 resnet_train_last_n_layers: int = 2, pos_type: str = "none",
                 vf_coords=None, num_cross_attn_layers: int = 2,
                 use_scale_interaction: bool = True, use_feature_enhancement: bool = True,
                 mlp_hidden_ratio: float = 2.0, use_vf_bottleneck: bool = False,
                 vf_latent_dim: int = 10, vf_queries_init_path: str = None,
                 use_vf_queries: bool = True, use_cross_attention: bool = True,
                 use_learnable_fusion: bool = True, use_vf_spatial_self_attn: bool = True):

        super().__init__()

        self.num_vf = num_vf
        self.d_model = d_model
        self.pos_type = pos_type
        self.use_vf_queries = use_vf_queries
        self.use_cross_attention = use_cross_attention
        self.use_learnable_fusion = use_learnable_fusion
        self.use_vf_spatial_self_attn = use_vf_spatial_self_attn

        btype = backbone_type.lower()
        if btype == "vgg":
            self.backbone = RNFLBackbone(in_ch=in_ch, use_residual=use_residual, dropout=backbone_dropout)
            out_channels = [128, 256, 512]
        elif btype == "resnet18":
            self.backbone = ResNetBackbone(in_ch=in_ch, pretrained=resnet_pretrained,
                                          train_last_n_layers=resnet_train_last_n_layers)
            out_channels = [128, 256, 512]
        elif btype in ["resnet34", "resnet50", "convnext_t", "swin_t", "efficientnet_v2_s"]:
            self.backbone = FeatureExtractorBackbone(
                name=btype, in_ch=in_ch, pretrained=resnet_pretrained,
                train_last_n_layers=resnet_train_last_n_layers)
            out_channels = self.backbone.out_channels
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        self.use_feature_enhancement = use_feature_enhancement
        if use_feature_enhancement:
            self.feature_enhancers = nn.ModuleList([
                ChannelAttention(channels=out_channels[0]),
                ChannelAttention(channels=out_channels[1]),
                ChannelAttention(channels=out_channels[2]),
            ])
        else:
            self.feature_enhancers = None
        
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(out_channels[0], d_model, kernel_size=1, bias=False),
            nn.Conv2d(out_channels[1], d_model, kernel_size=1, bias=False),
            nn.Conv2d(out_channels[2], d_model, kernel_size=1, bias=False),
        ])

        self.num_scales = len(self.scale_projs)

        if self.use_learnable_fusion:
            self.parallel_fusion = ParallelFusionBlock(d_model=d_model, num_scales=self.num_scales, 
                                                        droppath=0.1, use_diversity=False)
        else:
            self.parallel_fusion = None

        if self.use_vf_queries:
            if vf_queries_init_path is not None and os.path.exists(vf_queries_init_path):
                print(f"Loading VF queries from file: {vf_queries_init_path}")
                vf_queries_init = load_vf_queries_from_file(vf_queries_init_path, num_vf, d_model)
                self.vf_queries = nn.Parameter(vf_queries_init)
            else:
                self.vf_queries = nn.Parameter(torch.randn(1, num_vf, d_model))
        else:
            self.vf_queries = None

        if pos_type == "linear":
            self.vf_pos_embed = nn.Linear(2, d_model)
        
        if pos_type in ["linear", "sine"]:
            assert vf_coords is not None
            if not isinstance(vf_coords, torch.Tensor):
                vf_coords = torch.tensor(vf_coords, dtype=torch.float32)
            else:
                vf_coords = vf_coords.clone().detach().float()
        elif vf_coords is not None:
            if not isinstance(vf_coords, torch.Tensor):
                vf_coords = torch.tensor(vf_coords, dtype=torch.float32)
            else:
                vf_coords = vf_coords.clone().detach().float()
        
        if vf_coords is not None:
            vf_coords_normalized = vf_coords.clone()
            vf_coords_normalized[:, 0] = (vf_coords[:, 0] - 5.0) / 4.0
            vf_coords_normalized[:, 1] = (vf_coords[:, 1] - 4.5) / 3.5
            self.register_buffer("vf_coords", vf_coords_normalized)
        else:
            self.vf_coords = None

        self.num_cross_attn_layers = num_cross_attn_layers
        if num_cross_attn_layers == 1:
            self.scale_attn = nn.ModuleList([
                CrossAttention(hidden_size=d_model, num_heads=num_heads, dropout=dropout)
                for _ in range(self.num_scales)
            ])
        else:
            self.scale_attn = nn.ModuleList([
                MultiLayerCrossAttention(hidden_size=d_model, num_heads=num_heads,
                                        dropout=dropout, num_layers=num_cross_attn_layers)
                for _ in range(self.num_scales)
            ])
        
        self.scale_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_scales)])

        if self.use_vf_spatial_self_attn:
            if self.vf_coords is not None:
                self.vf_self_attn = VFSpatialSelfAttention(
                    hidden_size=d_model, num_heads=num_heads, dropout=dropout,
                    vf_coords=self.vf_coords, sigma=sigma)
            else:
                self.vf_self_attn = VFSpatialSelfAttention(
                    hidden_size=d_model, num_heads=num_heads, dropout=dropout,
                    vf_coords=None, sigma=0.4)
        else:
            self.vf_self_attn = None

        self.use_scale_interaction = use_scale_interaction
        if use_scale_interaction:
            self.scale_interaction = ScaleInteractionModule(
                hidden_size=d_model, num_heads=num_heads, dropout=dropout)
        else:
            self.scale_interaction = None
        
        self.use_vf_bottleneck = False
        self.vf_latent_dim = None
        self.vf_latent_head = None
        self.vf_decoder = None

        in_dim = d_model * (self.num_scales + 1)
        mlp_hidden = int(in_dim * mlp_hidden_ratio)
        mlp_hidden2 = max(64, mlp_hidden // 2)
        mlp_hidden3 = max(32, mlp_hidden2 // 2)
        self.out_linear = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden, mlp_hidden2),
            nn.LayerNorm(mlp_hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.35),
            nn.Linear(mlp_hidden2, mlp_hidden3),
            nn.LayerNorm(mlp_hidden3),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(mlp_hidden3, 1)
        )
        self.last_attn_entropy = None

        self.gap_head = nn.Sequential(
            nn.Conv2d(out_channels[-1], out_channels[-1] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[-1] // 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels[-1] // 2, num_vf)
        )

    def forward(self, x):
        B = x.size(0)
        feats = self.backbone(x)

        if not self.use_vf_queries:
            feat_high = feats[-1]
            vf_baseline = self.gap_head(feat_high)
            return vf_baseline

        vf_q = self.vf_queries.expand(B, -1, -1)

        if self.pos_type == "linear":
            pos = self.vf_pos_embed(self.vf_coords)
            vf_q = vf_q + pos.unsqueeze(0)
        elif self.pos_type == "sine":
            pos = sine_2d(self.vf_coords, self.d_model)
            vf_q = vf_q + pos.unsqueeze(0)
        
        scale_vf_list = []
        attn_entropy_terms = []
        
        for i, (feat, proj, attn, norm) in enumerate(zip(feats, self.scale_projs, self.scale_attn, self.scale_norms)):
            if self.use_feature_enhancement and self.feature_enhancers is not None:
                feat = self.feature_enhancers[i](feat)
            
            f = proj(feat)
            f_tokens = f.flatten(2).permute(0, 2, 1)

            if not self.use_cross_attention:
                f_global = f.mean(dim=(2, 3))
                vf_scale = vf_q + f_global.unsqueeze(1)
                vf_scale = norm(vf_scale)
                attn_probs = None
            else:
                if self.num_cross_attn_layers == 1:
                    vf_scale, attn_probs = attn(vf_q, f_tokens, return_attn=True)
                    vf_scale = norm(vf_scale + vf_q)
                else:
                    vf_scale, attn_probs = attn(vf_q, f_tokens, return_attn=True)
            
            if attn_probs is not None:
                attn_entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(-1).mean()
                attn_entropy_terms.append(attn_entropy)
            
            if self.vf_self_attn is not None:
                vf_scale = self.vf_self_attn(vf_scale)
            
            scale_vf_list.append(vf_scale)

        if self.use_scale_interaction and self.scale_interaction is not None:
            scale_vf_list = self.scale_interaction(scale_vf_list)

        if len(attn_entropy_terms) > 0:
            self.last_attn_entropy = torch.stack(attn_entropy_terms).mean()
        else:
            self.last_attn_entropy = None
        
        multi_scale_vf = torch.cat(scale_vf_list, dim=-1)
        
        if self.use_learnable_fusion and self.parallel_fusion is not None:
            vf_fused = self.parallel_fusion(vf_q, scale_vf_list)
        else:
            vf_fused = torch.stack(scale_vf_list, dim=0).mean(dim=0)
        
        vf_feat = torch.cat([multi_scale_vf, vf_fused], dim=-1)

        vf_pred = self.out_linear(vf_feat).squeeze(-1)
        return vf_pred

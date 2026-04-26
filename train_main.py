#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script: MS-SFCA model for RNFL to VF (52-dim) prediction.
"""

import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.stats as stats
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import importlib.util
from datetime import datetime
import copy

# Import MS-SFCA model
spec = importlib.util.spec_from_file_location("ms_sfca_model", "Multi-Scale-Global-Local-Transformer.py")
ms_sfca_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms_sfca_model)
MultiScaleRNFLToVF = ms_sfca_model.MultiScaleRNFLToVF

# VF coordinates for 52 test locations
VF_COORDS = np.array([
    [3.0, 1.0], [4.0, 1.0], [5.0, 1.0], [6.0, 1.0],
    [2.0, 2.0], [3.0, 2.0], [4.0, 2.0], [5.0, 2.0], [6.0, 2.0], [7.0, 2.0],
    [1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0], [5.0, 3.0], [6.0, 3.0], [7.0, 3.0], [8.0, 3.0],
    [1.0, 4.0], [2.0, 4.0], [4.0, 4.0], [5.0, 4.0], [6.0, 4.0], [7.0, 4.0], [8.0, 4.0], [9.0, 4.0],
    [1.0, 5.0], [2.0, 5.0], [4.0, 5.0], [5.0, 5.0], [6.0, 5.0], [7.0, 5.0], [8.0, 5.0], [9.0, 5.0],
    [1.0, 6.0], [2.0, 6.0], [3.0, 6.0], [4.0, 6.0], [5.0, 6.0], [6.0, 6.0], [7.0, 6.0], [8.0, 6.0],
    [2.0, 7.0], [3.0, 7.0], [4.0, 7.0], [5.0, 7.0], [6.0, 7.0], [7.0, 7.0],
    [3.0, 8.0], [4.0, 8.0], [5.0, 8.0], [6.0, 8.0]
], dtype=np.float32)


class VFLoss(nn.Module):
    """VF Loss: MSE + Smooth L1 + Cosine + z-score shape loss."""
    def __init__(self, l1=0.5, l2=0.5, l3=0.5, l4=0.5, l5=0.5, eps=1e-6):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.eps = eps
        self.last_logs = None

    def _zscore(self, x, dim=1, eps=1e-6):
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, unbiased=False, keepdim=True)
        return (x - mean) / (std + eps)

    def forward(self, pred, gt, intermediate_preds=None):
        mse_loss = F.mse_loss(pred, gt)
        smooth_l1 = F.smooth_l1_loss(pred, gt)
        cos_loss = 1 - F.cosine_similarity(pred, gt, dim=1).mean()

        zp = self._zscore(pred, dim=1, eps=self.eps)
        zg = self._zscore(gt, dim=1, eps=self.eps)
        z_mse = F.mse_loss(zp, zg)

        inter_loss = torch.tensor(0., device=pred.device)
        if intermediate_preds is not None:
            for p in intermediate_preds:
                inter_loss = inter_loss + F.smooth_l1_loss(p, gt)

        total_loss = (
            self.l1 * mse_loss +
            self.l2 * smooth_l1 +
            self.l3 * cos_loss +
            self.l5 * z_mse +
            self.l4 * inter_loss
        )

        self.last_logs = {
            'mse': mse_loss.detach(),
            'smooth_l1': smooth_l1.detach(),
            'cos': cos_loss.detach(),
            'z_mse': z_mse.detach(),
            'inter': inter_loss.detach(),
            'total': total_loss.detach(),
        }
        return total_loss


class RNFLVFDataset(Dataset):
    """Dataset for loading RNFL images and VF labels."""
    def __init__(self, root_dir, max_samples=None, augment=False, min_vf=-36.0, max_vf=9.0, 
                 is_norm=False, transform=None, use_imagenet_norm=False, resize=None, normalize=True):
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith(".npz")])
        self.augment = augment
        self.min_vf = min_vf
        self.max_vf = max_vf
        self.is_norm = is_norm
        self.use_imagenet_norm = use_imagenet_norm
        self.resize = resize
        
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {root_dir}")
        
        if max_samples is not None and max_samples > 0:
            self.files = self.files[:max_samples]
            print(f"[Dataset] Limited to {len(self.files)} samples")
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_image_transforms(
                augment=augment,
                resize=resize,
                normalize=normalize,
                use_imagenet_norm=use_imagenet_norm
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        data_path = os.path.join(self.root_dir, fname)
        data = np.load(data_path, allow_pickle=True)

        rnflt = data["rnflt"].astype(np.float32)
        tds = data["tds"].astype(np.float32)
        
        if self.is_norm:
            tds = normalize(tds, self.min_vf, self.max_vf)

        rnflt_tensor = torch.from_numpy(rnflt).float()
        rnflt_min = rnflt_tensor.min()
        rnflt_max = rnflt_tensor.max()
        if rnflt_max > rnflt_min:
            rnflt_tensor = (rnflt_tensor - rnflt_min) / (rnflt_max - rnflt_min)
        
        rnflt_tensor = rnflt_tensor.unsqueeze(0)
        
        if self.transform is not None:
            for t in self.transform.transforms if hasattr(self.transform, 'transforms') else [self.transform]:
                if isinstance(t, transforms.RandomHorizontalFlip):
                    if torch.rand(1) < 0.5:
                        rnflt_tensor = torch.flip(rnflt_tensor, dims=[2])
                elif isinstance(t, transforms.RandomVerticalFlip):
                    if torch.rand(1) < 0.5:
                        rnflt_tensor = torch.flip(rnflt_tensor, dims=[1])
                elif isinstance(t, transforms.RandomRotation):
                    degrees = t.degrees
                    if isinstance(degrees, (list, tuple)):
                        angle = torch.empty(1).uniform_(degrees[0], degrees[1]).item()
                    else:
                        angle = torch.empty(1).uniform_(-degrees, degrees).item()
                    rnflt_tensor = transforms.functional.rotate(rnflt_tensor, angle)
                elif isinstance(t, transforms.RandomAffine):
                    translate = t.translate if hasattr(t, 'translate') else (0, 0)
                    scale = t.scale if hasattr(t, 'scale') else (1.0, 1.0)
                    if translate != (0, 0) or scale != (1.0, 1.0):
                        translate_x = translate[0] * rnflt_tensor.shape[2] if isinstance(translate, tuple) else 0
                        translate_y = translate[1] * rnflt_tensor.shape[1] if isinstance(translate, tuple) else 0
                        scale_val = torch.empty(1).uniform_(scale[0], scale[1]).item() if isinstance(scale, tuple) else 1.0
                        rnflt_tensor = transforms.functional.affine(
                            rnflt_tensor, angle=0, translate=(translate_x, translate_y), 
                            scale=scale_val, shear=0
                        )
                elif isinstance(t, transforms.Resize):
                    rnflt_tensor = transforms.functional.resize(rnflt_tensor, t.size, antialias=True)
                elif isinstance(t, transforms.Normalize):
                    rnflt_tensor = transforms.functional.normalize(rnflt_tensor, t.mean, t.std)
        
        rnflt = rnflt_tensor
        tds = torch.from_numpy(tds)

        return rnflt, tds


def normalize(data, dmin, dmax):
    """Normalize data to [0, 1] range."""
    return (data - dmin) / (dmax - dmin)


def de_normalize(data, dmin, dmax):
    """Denormalize data back to original range."""
    return data * (dmax - dmin) + dmin


def get_image_transforms(augment=False, resize=None, normalize=True, 
                        mean=None, std=None, use_imagenet_norm=False):
    transform_list = []
    
    if augment:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(degrees=30))
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)))
    
    if resize is not None:
        if isinstance(resize, int):
            resize = (resize, resize)
        transform_list.append(transforms.Resize(resize))
    
    if normalize:
        if use_imagenet_norm:
            if mean is None:
                mean = [0.485]
            if std is None:
                std = [0.229]
        else:
            if mean is None:
                mean = [0.5]
            if std is None:
                std = [0.5]
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    if len(transform_list) == 0:
        return None
    
    return transforms.Compose(transform_list)


def compute_md(vf: np.ndarray):
    """Compute VF Mean Deviation."""
    return np.mean(vf, axis=1)


def _compute_metrics(all_pred: np.ndarray, all_gt: np.ndarray, min_vf=-36.0, max_vf=9.0, is_norm=False):
    """Compute metrics: MSE, RMSE, MAE, PCC, SRCC, pMAE, MD-MAE."""
    if is_norm:
        all_pred = de_normalize(all_pred, min_vf, max_vf)
        all_gt = de_normalize(all_gt, min_vf, max_vf)
    
    pred_flat = all_pred.flatten()
    target_flat = all_gt.flatten()

    diff = pred_flat - target_flat
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    pcc, _ = stats.pearsonr(pred_flat, target_flat)
    pcc = float(pcc) if not np.isnan(pcc) else 0.0

    srcc, _ = stats.spearmanr(pred_flat, target_flat)
    srcc = float(srcc) if not np.isnan(srcc) else 0.0

    pmae = float(np.mean(np.abs(all_pred - all_gt), axis=0).mean())
    md_pred = compute_md(all_pred)
    md_gt = compute_md(all_gt)
    md_mae = float(np.mean(np.abs(md_pred - md_gt)))

    return {
        "Loss": mse,
        "RMSE": rmse,
        "MAE": mae,
        "PCC": pcc,
        "SRCC": srcc,
        "pMAE": pmae,
        "MD-MAE": md_mae,
        "num_samples": int(all_gt.shape[0])
    }


def train_epoch(model, train_loader, criterion, optimizer, device, 
                attn_entropy_weight: float = 0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for rnflt, vf in tqdm(train_loader, desc="Training"):
        rnflt = rnflt.to(device)
        vf = vf.to(device)

        pred_vf = model(rnflt)
        loss = criterion(pred_vf, vf)

        if attn_entropy_weight > 0.0 and hasattr(model, "last_attn_entropy") and model.last_attn_entropy is not None:
            loss = loss + attn_entropy_weight * model.last_attn_entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, val_loader, criterion, device):
    """Validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for rnflt, vf in tqdm(val_loader, desc="Validating"):
            rnflt = rnflt.to(device)
            vf = vf.to(device)

            pred_vf = model(rnflt)
            loss = criterion(pred_vf, vf)

            total_loss += loss.item()
            num_batches += 1
            all_pred.append(pred_vf.cpu().numpy())
            all_gt.append(vf.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if len(all_pred) > 0:
        all_pred = np.concatenate(all_pred, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        metrics = _compute_metrics(all_pred, all_gt, min_vf=-36.0, max_vf=9.0, is_norm=False)
    else:
        metrics = {
            "Loss": 0.0, "RMSE": 0.0, "MAE": 0.0, "PCC": 0.0, "SRCC": 0.0, "pMAE": 0.0, "MD-MAE": 0.0, "num_samples": 0
        }

    return avg_loss, metrics


def evaluate_dataset(model, data_loader, criterion, device, split='test'):
    """Evaluate dataset."""
    model.eval()
    all_pred = []
    all_gt = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for b, (rnflt, vf) in enumerate(tqdm(data_loader, desc=f"Evaluating {split}")):
            rnflt = rnflt.to(device)
            vf = vf.to(device)

            pred_vf = model(rnflt)
            loss = criterion(pred_vf, vf)
            total_loss += loss.item()
            num_batches += 1

            pred_np = pred_vf.cpu().numpy()
            gt_np = vf.cpu().numpy()
            all_pred.append(pred_np)
            all_gt.append(gt_np)

    if len(all_pred) > 0:
        all_pred = np.concatenate(all_pred, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        metrics = _compute_metrics(all_pred, all_gt, min_vf=-36.0, max_vf=9.0, is_norm=False)
    else:
        metrics = {
            "Loss": 0.0, "RMSE": 0.0, "MAE": 0.0, "PCC": 0.0, "SRCC": 0.0, "pMAE": 0.0, "MD-MAE": 0.0, "num_samples": 0
        }
    
    return metrics


def get_attn_entropy_weight(epoch: int, target_weight: float, warmup_epochs: int = 10, rampup_epochs: int = 25):
    """Compute attention entropy regularization weight with warm-up."""
    if target_weight <= 0.0:
        return 0.0
    
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < rampup_epochs:
        return target_weight * 0.5
    else:
        return target_weight


def train_single_model(pos_type, args, train_loader, val_loader, device, base_train_id):
    """Train a single model."""
    pos_suffix = {'none': 'NoPos', 'linear': 'Linear2D', 'sine': 'Sine2D'}[pos_type]
    
    train_id = f"{base_train_id}_{pos_suffix}"
    run_dir = os.path.join(args.results_root, f'train_{train_id}')
    log_dir = os.path.join(args.logs_root, f'train_{train_id}')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    config = vars(args).copy()
    config['train_id'] = train_id
    config['pos_type'] = pos_type
    config['run_dir'] = run_dir
    config['log_dir'] = log_dir
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training model with pos_type='{pos_type}' ({pos_suffix})")
    print(f"Train ID: {train_id}")
    print(f"{'='*80}")
    
    # Create model
    model = MultiScaleRNFLToVF(
        in_ch=1,
        num_vf=52,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dropout=args.dropout,
        sigma=args.sigma,
        backbone_dropout=args.dropout * 0.3,
        use_residual=True,
        backbone_type=args.backbone_type,
        resnet_pretrained=not args.no_resnet_pretrained,
        resnet_train_last_n_layers=args.resnet_train_last_n_layers,
        num_cross_attn_layers=args.num_cross_attn_layers,
        mlp_hidden_ratio=args.mlp_hidden_ratio,
        use_vf_bottleneck=args.use_vf_bottleneck,
        vf_latent_dim=args.vf_latent_dim,
        vf_queries_init_path=args.vf_queries_init_path,
        pos_type=pos_type,
        vf_coords=VF_COORDS,
    ).to(device)
    
    print(f"Model created: pos_type={pos_type}, d_model={args.d_model}, num_heads={args.num_heads}, "
          f"backbone_type={args.backbone_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = VFLoss(
        l1=1.0,
        l2=0.3,
        l3=0.05,
        l4=0.3,
        l5=0.1,
        eps=1e-6
    )
    print("Using VFLoss: l1=1.0, l2=0.3, l3=0.05, l4=0.3, l5=0.1")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-7
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_records = {'train': [], 'valid': []}
    epoch_logs = []
    best_info = {'epoch': None, 'valid_loss': None, 'train_loss': None, 'lr': None, 'pos_type': pos_type}
    best_model = None
    early_stop_counter = 0
    
    print(f'\nTraining started: {train_id}')
    if args.early_stop_patience > 0:
        print(f'Early stopping enabled with patience={args.early_stop_patience}')
    
    for epoch in range(1, args.epochs + 1):
        current_attn_entropy_weight = get_attn_entropy_weight(
            epoch=epoch,
            target_weight=args.attn_entropy_weight,
            warmup_epochs=10,
            rampup_epochs=25
        )
        
        if args.disp_gap > 0 and (epoch % args.disp_gap == 0 or epoch == 1):
            print(f"\nEpoch {epoch}/{args.epochs}, lr: {optimizer.param_groups[0]['lr']:.6e}")
            if args.attn_entropy_weight > 0.0:
                print(f"Attn Entropy Weight: {current_attn_entropy_weight:.4f}")
        
        epoch_log = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'pos_type': pos_type}
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            attn_entropy_weight=current_attn_entropy_weight
        )
        train_losses.append(train_loss)
        train_records['train'].append(train_loss)
        epoch_log['train_loss'] = train_loss
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        train_records['valid'].append(val_loss)
        epoch_log['valid_loss'] = val_loss
        epoch_log['valid_rmse'] = val_metrics['RMSE']
        epoch_log['valid_mae'] = val_metrics['MAE']
        epoch_log['valid_pcc'] = val_metrics['PCC']
        epoch_log['valid_srcc'] = val_metrics['SRCC']
        epoch_log['valid_pmae'] = val_metrics['pMAE']
        epoch_log['valid_md_mae'] = val_metrics['MD-MAE']
        
        scheduler.step(val_loss)
        
        if args.disp_gap > 0 and (epoch % args.disp_gap == 0 or epoch == 1):
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}, RMSE: {val_metrics['RMSE']:.4f}, "
                  f"MAE: {val_metrics['MAE']:.4f}, PCC: {val_metrics['PCC']:.4f}, "
                  f"SRCC: {val_metrics['SRCC']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_info = {
                'epoch': epoch,
                'valid_loss': val_loss,
                'train_loss': train_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'valid_rmse': val_metrics['RMSE'],
                'valid_mae': val_metrics['MAE'],
                'valid_pcc': val_metrics['PCC'],
                'valid_srcc': val_metrics['SRCC'],
                'valid_pmae': val_metrics['pMAE'],
                'valid_md_mae': val_metrics['MD-MAE'],
                'pos_type': pos_type
            }
            checkpoint_path = os.path.join(run_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'pos_type': pos_type,
            }, checkpoint_path)

            if args.disp_gap > 0:
                print(f"Saved best model (val_loss: {val_loss:.6f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        epoch_logs.append(epoch_log)
        
        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save training records
    losses_path = os.path.join(run_dir, 'losses.pkl')
    joblib.dump(train_records, losses_path)
    
    log_json = os.path.join(log_dir, 'train_log.json')
    with open(log_json, 'w') as f:
        json.dump(epoch_logs, f, indent=2)
    
    log_csv = os.path.join(log_dir, 'train_log.csv')
    csv_headers = ['epoch', 'lr', 'pos_type', 'train_loss', 'valid_loss', 'valid_rmse', 'valid_mae', 
                   'valid_pcc', 'valid_srcc', 'valid_pmae', 'valid_md_mae']
    with open(log_csv, 'w') as f:
        f.write(','.join(csv_headers) + '\n')
        for row in epoch_logs:
            line = [str(row.get(h, '')) for h in csv_headers]
            f.write(','.join(line) + '\n')
    
    best_info_path = os.path.join(log_dir, 'best_info.json')
    with open(best_info_path, 'w') as f:
        json.dump(best_info, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Training completed for pos_type='{pos_type}'!")
    print(f"Best epoch: {best_info['epoch']}, Best val loss: {best_info['valid_loss']:.6f}")
    print("="*80)
    
    if best_model is None:
        best_model = model
    
    print("\nEvaluating with best model...")
    
    val_metrics = evaluate_dataset(best_model, val_loader, criterion, device, split='val')
    
    metrics = {
        "pos_type": pos_type,
        "best_epoch": best_info['epoch'],
        "best_lr": best_info['lr'],
        "Val": val_metrics,
        "Train": {"Loss": float(train_losses[-1]) if train_losses else None}
    }
    
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    best_metrics_path = os.path.join(log_dir, 'best_metrics.json')
    best_metrics = {
        "pos_type": pos_type,
        "best_epoch": best_info['epoch'],
        "Val": val_metrics
    }
    with open(best_metrics_path, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Final Results (pos_type='{pos_type}'):")
    print(f"Val  - RMSE: {val_metrics['RMSE']:.4f}, MAE: {val_metrics['MAE']:.4f}, "
          f"PCC: {val_metrics['PCC']:.4f}, SRCC: {val_metrics['SRCC']:.4f}")
    print("="*80)
    
    return best_info


def main():
    parser = argparse.ArgumentParser(description='Train MS-SFCA model for RNFL to VF prediction')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='your_data_path',
                        help='Data root directory path')
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='Max training samples (0=all, -1=half, >0=specific)')
    parser.add_argument('--max_val_samples', type=int, default=0,
                        help='Max validation samples')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    
    # Model architecture parameters
    parser.add_argument('--d_model', type=int, default=384, help='Model hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma for VF self-attention')
    parser.add_argument('--backbone_type', type=str, default='resnet34',
                        choices=['vgg', 'resnet18', 'resnet34', 'resnet50', 'convnext_t', 'swin_t', 'efficientnet_v2_s'],
                        help='Backbone type')
    parser.add_argument('--no_resnet_pretrained', action='store_true', help='Disable pretrained weights')
    parser.add_argument('--resnet_train_last_n_layers', type=int, default=2,
                        help='Number of layers to train from the end')
    
    # Model enhancement parameters
    parser.add_argument('--num_cross_attn_layers', type=int, default=2,
                        help='Number of Cross-Attention layers per scale')
    parser.add_argument('--mlp_hidden_ratio', type=float, default=3.0,
                        help='MLP hidden layer ratio')
    
    # Low-rank VF parameters
    parser.add_argument('--use_vf_bottleneck', action='store_true', default=False,
                        help='Use VF bottleneck')
    parser.add_argument('--vf_latent_dim', type=int, default=10, choices=[8, 10, 12],
                        help='VF latent dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='Early stopping patience (0=disabled)')
    parser.add_argument('--attn_entropy_weight', type=float, default=0.01,
                        help='Attention entropy regularization weight')
    
    # Data preprocessing parameters
    parser.add_argument('--use_imagenet_norm', action='store_true',
                        help='Use ImageNet normalization')
    parser.add_argument('--resize', type=int, default=224, help='Image resize size')
    parser.add_argument('--no_normalize', action='store_true', help='Disable image normalization')
    
    # Positional encoding parameter
    parser.add_argument('--pos_type', type=str, default='linear',
                        help='Positional encoding type')
    
    # VF Queries initialization parameter
    parser.add_argument('--vf_queries_init_path', type=str, default=None,
                        help='VF queries initialization file path')
    
    # Output parameters
    parser.add_argument('--results_root', type=str, default='./results',
                        help='Results root directory')
    parser.add_argument('--logs_root', type=str, default='./logs',
                        help='Logs root directory')
    parser.add_argument('--train_id', type=str, default=None,
                        help='Custom training ID')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    parser.add_argument('--disp_gap', type=int, default=2,
                        help='Display interval (epochs)')
    
    args = parser.parse_args()

    base_train_id = args.train_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Dataset
    temp_train_dir = os.path.join(args.data_root, 'train')
    temp_val_dir = os.path.join(args.data_root, 'val')
    
    train_files = sorted([f for f in os.listdir(temp_train_dir) if f.endswith(".npz")])
    val_files = sorted([f for f in os.listdir(temp_val_dir) if f.endswith(".npz")])
    total_train = len(train_files)
    total_val = len(val_files)
    
    if args.max_train_samples == -1:
        max_train = max(1, total_train // 2)
        print(f"[Data] Using half of training data: {max_train}/{total_train}")
    elif args.max_train_samples == 0:
        max_train = None
        print(f"[Data] Using all training data: {total_train}")
    else:
        max_train = args.max_train_samples
    
    if args.max_val_samples == -1:
        max_val = max(1, total_val // 2)
        print(f"[Data] Using half of validation data: {max_val}/{total_val}")
    elif args.max_val_samples == 0:
        max_val = None
        print(f"[Data] Using all validation data: {total_val}")
    else:
        max_val = args.max_val_samples
    
    min_vf, max_vf = -36.0, 9.0
    
    normalize_images = not args.no_normalize
    train_dataset = RNFLVFDataset(
        temp_train_dir, 
        max_samples=max_train, 
        augment=args.augment, 
        min_vf=min_vf, 
        max_vf=max_vf, 
        is_norm=False,
        transform=None,
        use_imagenet_norm=args.use_imagenet_norm,
        resize=args.resize,
        normalize=normalize_images
    )
    val_dataset = RNFLVFDataset(
        temp_val_dir, 
        max_samples=max_val, 
        augment=False,
        min_vf=min_vf, 
        max_vf=max_vf, 
        is_norm=False,
        transform=None,
        use_imagenet_norm=args.use_imagenet_norm,
        resize=args.resize,
        normalize=normalize_images
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    print("\n" + "="*60)
    print("Data Preprocessing Configuration:")
    print("="*60)
    print(f"  Augmentation: {'Enabled' if args.augment else 'Disabled'}")
    print(f"  ImageNet Normalization: {'Enabled' if args.use_imagenet_norm else 'Disabled'}")
    print(f"  Resize: {args.resize}")
    print(f"  Normalize: {'Disabled' if args.no_normalize else 'Enabled'}")
    print("="*60 + "\n")

    pos_types = [args.pos_type]
    
    all_results = {}
    all_metrics = {}
    
    for pos_type in pos_types:
        try:
            best_info = train_single_model(
                pos_type=pos_type,
                args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                base_train_id=base_train_id
            )
            all_results[pos_type] = best_info
            
            pos_suffix = {'none': 'NoPos', 'linear': 'Linear2D', 'sine': 'Sine2D'}[pos_type]
            train_id = f"{base_train_id}_{pos_suffix}"
            metrics_path = os.path.join(args.results_root, f'train_{train_id}', 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    all_metrics[pos_type] = json.load(f)
        except Exception as e:
            print(f"\nError training {pos_type} model: {e}")
            import traceback
            traceback.print_exc()
            all_results[pos_type] = None
            all_metrics[pos_type] = None
    
    print("\n" + "="*80)
    print("Final Results:")
    print("="*80)
    for pos_type in pos_types:
        if all_results[pos_type] is not None:
            info = all_results[pos_type]
            print(f"pos_type='{pos_type}': Epoch={info.get('epoch', 'N/A')}, "
                  f"Val Loss={info.get('valid_loss', 0):.6f}, "
                  f"RMSE={info.get('valid_rmse', 0):.4f}, "
                  f"PCC={info.get('valid_pcc', 0):.4f}")
        else:
            print(f"pos_type='{pos_type}': FAILED")
    print("="*80)


if __name__ == '__main__':
    main()

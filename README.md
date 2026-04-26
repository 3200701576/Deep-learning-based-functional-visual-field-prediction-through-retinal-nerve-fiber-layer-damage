# MS-SFCA: Multi-Scale Cross-Attention for RNFL to VF Prediction

A PyTorch implementation of a multi-scale cross-attention model that predicts 52-dimensional Visual Field (VF) measurements from Retinal Nerve Fiber Layer (RNFL) images.

## Project Structure

```
QMSF/
├── Multi-Scale-Global-Local-Transformer.py  # Core model architecture
├── train_main.py                             # Training script
├── vgg.py                                    # VGG-style convolution block
└── resnet.py                                 # ResNet components (unused by default)
```

## Model Architecture

The MS-SFCA model employs a multi-scale cross-attention mechanism:

1. **Backbone**: Extracts multi-scale features from RNFL images (supports ResNet18/34/50, ConvNeXt-Tiny, Swin-T, EfficientNet-V2-S, VGG)
2. **Scale Interaction Module**: Enables cross-scale feature interaction
3. **Multi-Layer Cross-Attention**: Connects VF query tokens with RNFL features
4. **VF Spatial Self-Attention**: Models spatial relationships between VF locations with Gaussian distance bias
5. **Parallel Fusion Block**: Adaptively combines multi-scale features

## Requirements

```bash
pip install torch torchvision numpy scipy tqdm joblib
```

## Quick Start

### Training

```bash
python train_main.py --data_root /path/to/your/data --epochs 100 --batch_size 4
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | required | Path to data directory |
| `--d_model` | 384 | Model hidden dimension |
| `--num_heads` | 8 | Number of attention heads |
| `--backbone_type` | resnet34 | Backbone architecture |
| `--num_cross_attn_layers` | 2 | Cross-attention layers per scale |
| `--batch_size` | 4 | Batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--pos_type` | linear | Positional encoding type |


### Data Format

The training data should be organized as:
```
data_root/
├── train/
│   ├── sample1.npz  # Contains 'rnflt' and 'tds' arrays
│   ├── sample2.npz
│   └── ...
└── val/
    ├── sample1.npz
    └── ...
```

Each `.npz` file must contain:
- `rnflt`: RNFL image (H x W)
- `tds`: VF label (52 dimensions)

## Evaluation Metrics

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- PCC (Pearson Correlation Coefficient)
- SRCC (Spearman Rank Correlation Coefficient)
- pMAE (Point-wise MAE)
- MD-MAE (Mean Deviation MAE)


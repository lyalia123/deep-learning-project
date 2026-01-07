# Section 4 — Modern Architectures & Transfer Learning (Student B)

This folder contains:
- **ResNet vs plain CNN** comparison (CIFAR-10)
- **Gradient flow** proxy analysis (gradient norms in early layers)
- **Transfer learning** (scratch vs pretrained ResNet18)
- **Data augmentation** study (≥5 augmentations)

## Install
```bash
pip install -r ../requirements.txt
```

## 1) ResNet vs plain CNN
```bash
python resnet_vs_plain.py --epochs 10 --batch-size 128
```

## 2) Transfer learning (scratch vs pretrained)
```bash
python transfer_learning.py --epochs 5 --batch-size 64 --resize 224
```

## 3) Augmentation study
```bash
python augmentation_study.py --epochs 5 --batch-size 64 --resize 224
```

All outputs are saved under `section4_transfer/runs/...`.

# Section 3 — Convolutional Neural Networks (Student B)

This folder contains a **NumPy-from-scratch** CNN for MNIST + required analyses:
- CNN training (Conv → Pool → FC) with backprop
- Filter + feature map visualization
- Receptive field study
- Pooling comparison (Max vs Avg)

## Quick start

Create env (example):
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r ../requirements.txt
```

Train NumPy CNN on MNIST:
```bash
python train_mnist_numpy.py --epochs 5 --batch-size 64 --lr 0.01
```

Visualize filters + feature maps (after training):
```bash
python visualize_filters.py --ckpt runs/mnist_numpy/best.pkl
```

Pooling comparison:
```bash
python pooling_comparison.py --epochs 3
```

Receptive field study:
```bash
python receptive_field.py
```

Outputs are saved under `section3_cnn/runs/...`.

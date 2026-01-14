# Deep Learning Project — From MLP Baselines to CNNs & Transfer Learning

This repository contains a 4-part deep learning project:
1) **MLP from scratch (NumPy)**  
2) **Optimization algorithms + gradient checking (NumPy)**  
3) **CNN from scratch (NumPy) + required analyses (MNIST)**  
4) **Modern architectures & transfer learning (PyTorch) (CIFAR-10)**

> Note: Section 4 uses **CIFAR-10** (approved by the instructor as a substitute dataset).

---

## Repository structure
├── section1_mlp/ # MLP from scratch (NumPy)
├── section2_optimization/ # Optimizers + gradient checking (NumPy)
├── section3_cnn/ # CNN from scratch (NumPy) + analyses (MNIST)
├── section4_transfer/ # ResNet vs PlainCNN + transfer learning + augmentation (PyTorch)
├── shared_utils/ # Reproducibility + plotting + dataloaders
├── tests/ # Unit tests for all sections
├── requirements.txt
└── README.md


## Install base dependencies
pip install -r requirements.txt

## Tests
pytest -q


## Section 1 — MLP (NumPy, from scratch)

What’s inside:
* A NumPy MLP with forward/backprop and SGD step

* Classification on MNIST (via OpenML) with activation comparison (ReLU/Sigmoid/Tanh)

Extra demos: regression / universal approximation

Run (from inside the folder because imports are local):
```bash
cd section1_mlp
python train.py
python test.py
python mlp_regression.py
python universal_approximation.py
```


Outputs: plots (shown interactively).

## Section 2 — Optimization + Gradient Checking (NumPy)

What’s inside:

* Modular MLP (Linear/Activation layers)

* Optimizers: SGD (momentum/nesterov), RMSprop, Adam

* Gradient checking demo + optimizer comparison on toy datasets

Run (from inside the folder because imports are local):
```bash
cd section2_optimization
python train.py
python gradient_checking.py
```

Outputs: console logs + plots.

## Section 3 — CNN (NumPy from scratch) + required analyses (MNIST)

This section contains a NumPy-from-scratch CNN for MNIST + required analyses:

* CNN training (Conv → Pool → FC) with backprop

* Filter + feature map visualization

* Receptive field study (theoretical)

* Pooling comparison (Max vs Avg)

* Quick start (run from repo root)

Train NumPy CNN on MNIST:
```bash
python section3_cnn/train_mnist_numpy.py --epochs 5 --batch-size 64 --lr 0.01
```

Visualize filters + feature maps (after training):
```bash
python section3_cnn/visualize_filters.py --ckpt section3_cnn/runs/mnist_numpy/best.pkl
```

Pooling comparison:
```bash
python section3_cnn/pooling_comparison.py --epochs 3
```

Receptive field study:
```bash
python section3_cnn/receptive_field.py
```

Outputs are saved under:

section3_cnn/runs/mnist_numpy/ (loss/acc curves + best.pkl)

section3_cnn/runs/vis/ (filters + feature maps)

section3_cnn/runs/pooling_compare/ (max vs avg comparison plots)

## Section 4 — Modern architectures & Transfer Learning (PyTorch)

This section contains:

* ResNet vs plain CNN comparison (CIFAR-10)

* Gradient flow proxy analysis (gradient norm in early layers)

* Transfer learning (scratch vs pretrained ResNet18)

* Data augmentation study (≥5 augmentations)


1) ResNet vs PlainCNN (CIFAR-10, 32×32)
python section4_transfer/resnet_vs_plain.py --epochs 10 --batch-size 128

2) Transfer learning (scratch vs pretrained ResNet18)

Uses resized CIFAR-10 (default 224) to match ImageNet-pretrained ResNet18.

python section4_transfer/transfer_learning.py --epochs 5 --batch-size 64 --resize 224

3) Augmentation study (≥5 augmentations)
python section4_transfer/augmentation_study.py --epochs 5 --batch-size 64 --resize 224


Outputs are saved under:

section4_transfer/runs/resnet_vs_plain/

section4_transfer/runs/transfer_learning/

section4_transfer/runs/augmentation/




## Report
https://www.overleaf.com/read/cxmccphpzvph#dc60dc
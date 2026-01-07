from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torchvision import transforms, models

from shared_utils.seed import set_seed
from shared_utils.torch_data import cifar10_loaders
from shared_utils.plotting import ensure_dir, save_curve
from section4_transfer.models import PlainCNN
from section4_transfer.train_utils import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--runs-dir", type=str, default="section4_transfer/runs/resnet_vs_plain")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 normalization (common values)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    loaders = cifar10_loaders(batch_size=args.batch_size, train_tfm=train_tfm, test_tfm=test_tfm)

    # Plain CNN
    plain = PlainCNN(num_classes=10)
    plain_logs = train(plain, loaders.train, loaders.test, device, args.epochs, args.lr, args.runs_dir, label="plain_cnn")

    # ResNet18 (skip connections)
    resnet = models.resnet18(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    resnet_logs = train(resnet, loaders.train, loaders.test, device, args.epochs, args.lr, args.runs_dir, label="resnet18")

    # combined curves
    out = ensure_dir(args.runs_dir)
    xs = list(range(1, args.epochs + 1))
    save_curve(Path(out) / "compare_test_acc.png", xs, {
        "plain_cnn_test": [l.test_acc for l in plain_logs],
        "resnet18_test": [l.test_acc for l in resnet_logs],
    }, ylabel="test accuracy", title="ResNet vs Plain CNN (CIFAR-10)")

    save_curve(Path(out) / "compare_grad_norm.png", xs, {
        "plain_first_conv_grad_norm": [l.grad_norm_first_layer for l in plain_logs],
        "resnet_first_conv_grad_norm": [l.grad_norm_first_layer for l in resnet_logs],
    }, ylabel="L2 norm", title="Gradient flow proxy (first conv)")

    print(f"Saved results to: {args.runs_dir}")


if __name__ == "__main__":
    main()

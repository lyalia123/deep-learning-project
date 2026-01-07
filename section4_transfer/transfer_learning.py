from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models

from shared_utils.seed import set_seed
from shared_utils.torch_data import cifar10_loaders
from shared_utils.plotting import ensure_dir, save_curve
from section4_transfer.train_utils import train


def make_loaders(batch_size: int, resize: int):
    mean = (0.485, 0.456, 0.406)  # ImageNet
    std = (0.229, 0.224, 0.225)

    train_tfm = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(resize, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return cifar10_loaders(batch_size=batch_size, train_tfm=train_tfm, test_tfm=test_tfm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--runs-dir", type=str, default="section4_transfer/runs/transfer_learning")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(args.batch_size, args.resize)

    # (A) ResNet18 from scratch
    scratch = models.resnet18(weights=None)
    scratch.fc = nn.Linear(scratch.fc.in_features, 10)
    scratch_logs = train(scratch, loaders.train, loaders.test, device, args.epochs, args.lr, args.runs_dir, label="scratch_resnet18")

    # (B) Pretrained ResNet18 + fine-tuning
    pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    pretrained.fc = nn.Linear(pretrained.fc.in_features, 10)

    # Option 1: fine-tune all layers (simple for assignment)
    finetune_logs = train(pretrained, loaders.train, loaders.test, device, args.epochs, args.lr, args.runs_dir, label="pretrained_finetune_resnet18")

    # Combined plot
    out = ensure_dir(args.runs_dir)
    xs = list(range(1, args.epochs + 1))
    save_curve(Path(out) / "compare_test_acc.png", xs, {
        "scratch": [l.test_acc for l in scratch_logs],
        "pretrained_finetune": [l.test_acc for l in finetune_logs],
    }, ylabel="test accuracy", title="Transfer learning: scratch vs pretrained (CIFAR-10 resized)")

    print(f"Saved results to: {args.runs_dir}")


if __name__ == "__main__":
    main()

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


def make_loaders(batch_size: int, resize: int, augmented: bool):
    mean = (0.485, 0.456, 0.406)  # ImageNet
    std = (0.229, 0.224, 0.225)

    if augmented:
        # >=5 augmentations:
        # 1) RandomResizedCrop, 2) HFlip, 3) ColorJitter, 4) RandomRotation, 5) RandomGrayscale, + optional RandomErasing
        train_tfm = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomResizedCrop(resize, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        train_tfm = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
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
    ap.add_argument("--runs-dir", type=str, default="section4_transfer/runs/augmentation")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders_noaug = make_loaders(args.batch_size, args.resize, augmented=False)
    loaders_aug = make_loaders(args.batch_size, args.resize, augmented=True)

    # Use pretrained backbone to emphasize generalization differences
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base.fc = nn.Linear(base.fc.in_features, 10)

    # train no-aug
    noaug_logs = train(base, loaders_noaug.train, loaders_noaug.test, device, args.epochs, args.lr, args.runs_dir, label="pretrained_no_aug")

    # re-init a new model for fair comparison
    base2 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base2.fc = nn.Linear(base2.fc.in_features, 10)
    aug_logs = train(base2, loaders_aug.train, loaders_aug.test, device, args.epochs, args.lr, args.runs_dir, label="pretrained_with_aug")

    out = ensure_dir(args.runs_dir)
    xs = list(range(1, args.epochs + 1))
    save_curve(Path(out) / "compare_test_acc.png", xs, {
        "no_aug": [l.test_acc for l in noaug_logs],
        "with_aug": [l.test_acc for l in aug_logs],
    }, ylabel="test accuracy", title="Data augmentation impact (pretrained ResNet18)")

    print(f"Saved results to: {args.runs_dir}")


if __name__ == "__main__":
    main()

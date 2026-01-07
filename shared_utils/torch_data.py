from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

@dataclass
class DataLoaders:
    train: DataLoader
    test: DataLoader

def mnist_loaders(batch_size: int = 64, root: str = "./data", num_workers: int = 2) -> DataLoaders:
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return DataLoaders(train=train, test=test)

def cifar10_loaders(batch_size: int = 128, root: str = "./data", num_workers: int = 2, train_tfm=None, test_tfm=None) -> DataLoaders:
    if train_tfm is None:
        train_tfm = transforms.Compose([transforms.ToTensor()])
    if test_tfm is None:
        test_tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tfm)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tfm)
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return DataLoaders(train=train, test=test)

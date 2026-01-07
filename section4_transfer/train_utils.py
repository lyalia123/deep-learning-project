from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared_utils.plotting import ensure_dir, save_curve


@dataclass
class EpochLog:
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    grad_norm_first_layer: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    return total_loss / total, correct / total


def grad_norm(param: torch.Tensor) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().norm().item())


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: str,
    label: str,
    weight_decay: float = 0.0,
):
    out = ensure_dir(out_dir)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()

            # gradient flow proxy: grad norm on the *first* conv layer weights
            first_weight = None
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    first_weight = m.weight
                    break
            gn = grad_norm(first_weight) if first_weight is not None else 0.0

            opt.step()

            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device)

        logs.append(EpochLog(train_loss, train_acc, test_loss, test_acc, gn))
        print(f"[{label}] epoch {epoch}: train_acc={train_acc*100:.2f}% test_acc={test_acc*100:.2f}% grad_norm={gn:.4f}")

        xs = list(range(1, len(logs) + 1))
        save_curve(Path(out) / f"{label}_acc.png", xs, {f"{label}_train": [l.train_acc for l in logs], f"{label}_test": [l.test_acc for l in logs]}, ylabel="accuracy")
        save_curve(Path(out) / f"{label}_loss.png", xs, {f"{label}_train": [l.train_loss for l in logs], f"{label}_test": [l.test_loss for l in logs]}, ylabel="loss")
        save_curve(Path(out) / f"{label}_grad_norm.png", xs, {f"{label}_first_conv_grad_norm": [l.grad_norm_first_layer for l in logs]}, ylabel="L2 norm")

    # save weights
    torch.save(model.state_dict(), Path(out) / f"{label}_model.pt")
    return logs

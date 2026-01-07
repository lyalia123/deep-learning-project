from __future__ import annotations
import os
from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_curve(out_path: str | os.PathLike, xs, ys_dict: dict[str, list[float]], xlabel="epoch", ylabel="value", title=None):
    plt.figure()
    for k, ys in ys_dict.items():
        plt.plot(xs, ys, label=k)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from section3_cnn.cnn_numpy import Sequential
from shared_utils.plotting import ensure_dir


def load_one_mnist(root: str = "./data", idx: int = 0):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    x, y = ds[idx]
    x = x.numpy().astype(np.float32)  # (1,28,28)
    return x[None, ...], int(y)  # add batch dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pkl")
    ap.add_argument("--out-dir", type=str, default="section3_cnn/runs/vis")
    ap.add_argument("--sample-idx", type=int, default=0)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)

    with open(args.ckpt, "rb") as f:
        obj = pickle.load(f)
    model: Sequential = obj["model"]

    # ---- visualize first conv filters
    # layer 0 is Conv2D in our default architecture
    conv0 = model.layers[0]
    W = conv0.W  # (out_ch, in_ch, kh, kw)
    out_ch = W.shape[0]

    cols = min(8, out_ch)
    rows = int(np.ceil(out_ch / cols))

    plt.figure(figsize=(cols * 1.5, rows * 1.5))
    for i in range(out_ch):
        plt.subplot(rows, cols, i + 1)
        filt = W[i, 0]
        plt.imshow(filt, cmap="gray")
        plt.axis("off")
        plt.title(f"f{i}")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "filters_conv1.png", dpi=200)
    plt.close()

    # ---- feature maps on one sample
    x, y = load_one_mnist(idx=args.sample_idx)
    feats = []
    cur = x
    for layer in model.layers:
        cur = layer.forward(cur, train=False)
        # store after conv/relu/pool stages
        if layer.__class__.__name__ in ["Conv2D", "ReLU", "MaxPool2D", "AvgPool2D"]:
            feats.append((layer.__class__.__name__, cur))

    # pick the first Conv output feature maps
    conv_out = None
    for name, t in feats:
        if name == "Conv2D":
            conv_out = t
            break

    if conv_out is not None:
        fm = conv_out[0]  # (C,H,W)
        c = min(16, fm.shape[0])
        cols = 4
        rows = int(np.ceil(c / cols))
        plt.figure(figsize=(cols * 2.2, rows * 2.2))
        for i in range(c):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(fm[i], cmap="gray")
            plt.axis("off")
            plt.title(f"ch{i}")
        plt.suptitle(f"Feature maps after first Conv (label={y})")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "feature_maps_conv1.png", dpi=200)
        plt.close()

    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()

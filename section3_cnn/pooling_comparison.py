from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from shared_utils.plotting import ensure_dir, save_curve

# We just call the training script twice (max vs avg) by importing its helpers
from section3_cnn.train_mnist_numpy import load_mnist_numpy, iterate_minibatches, evaluate
from section3_cnn.cnn_numpy import (
    Conv2D, ReLU, MaxPool2D, AvgPool2D, Flatten, Linear,
    Sequential, SoftmaxCrossEntropy, sgd_step
)
from shared_utils.seed import set_seed


def train(pool_type: str, epochs: int, batch_size: int, lr: float, run_dir: str):
    (X_train, y_train), (X_test, y_test) = load_mnist_numpy()
    pool = MaxPool2D() if pool_type == "max" else AvgPool2D()

    model = Sequential([
        Conv2D(1, 8, kernel_size=3, stride=1, pad=1, weight_scale=0.02),
        ReLU(),
        pool,
        Conv2D(8, 16, kernel_size=3, stride=1, pad=1, weight_scale=0.02),
        ReLU(),
        MaxPool2D(),
        Flatten(),
        Linear(16 * 7 * 7, 128, weight_scale=0.02),
        ReLU(),
        Linear(128, 10, weight_scale=0.02),
    ])
    loss_fn = SoftmaxCrossEntropy()

    train_losses, test_accs = [], []
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True):
            logits = model.forward(xb, train=True)
            loss = loss_fn.forward(logits, yb)
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            sgd_step(model.params_and_grads(), lr=lr)
            epoch_losses.append(loss)

        train_losses.append(float(np.mean(epoch_losses)))
        test_accs.append(evaluate(model, X_test, y_test))

    out = ensure_dir(run_dir)
    save_curve(Path(out) / f"loss_{pool_type}.png", list(range(1, epochs+1)), {f"{pool_type}_loss": train_losses}, ylabel="loss")
    save_curve(Path(out) / f"acc_{pool_type}.png", list(range(1, epochs+1)), {f"{pool_type}_acc": test_accs}, ylabel="accuracy")
    return train_losses, test_accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--runs-dir", type=str, default="section3_cnn/runs/pooling_compare")
    args = ap.parse_args()

    set_seed(args.seed)
    max_loss, max_acc = train("max", args.epochs, args.batch_size, args.lr, args.runs_dir)
    avg_loss, avg_acc = train("avg", args.epochs, args.batch_size, args.lr, args.runs_dir)

    # combined plot
    save_curve(Path(args.runs_dir) / "acc_compare.png",
               list(range(1, args.epochs+1)),
               {"max_pool": max_acc, "avg_pool": avg_acc},
               ylabel="test accuracy", title="Pooling comparison (MNIST NumPy CNN)")
    print(f"Saved results to: {args.runs_dir}")


if __name__ == "__main__":
    main()

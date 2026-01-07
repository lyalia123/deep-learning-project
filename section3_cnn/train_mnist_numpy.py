from __future__ import annotations
import argparse, os, pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

from torchvision import datasets, transforms

from section3_cnn.cnn_numpy import (
    Conv2D, ReLU, MaxPool2D, AvgPool2D, Flatten, Linear,
    Sequential, SoftmaxCrossEntropy, accuracy_from_logits, sgd_step
)
from shared_utils.seed import set_seed
from shared_utils.plotting import save_curve, ensure_dir


def load_mnist_numpy(root: str = "./data"):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    X_train = train_ds.data.numpy().astype(np.float32) / 255.0  # (N,28,28)
    y_train = train_ds.targets.numpy().astype(np.int64)
    X_test  = test_ds.data.numpy().astype(np.float32) / 255.0
    y_test  = test_ds.targets.numpy().astype(np.int64)

    # add channel dim
    X_train = X_train[:, None, :, :]
    X_test  = X_test[:, None, :, :]
    return (X_train, y_train), (X_test, y_test)


def iterate_minibatches(X, y, batch_size: int, shuffle: bool = True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def evaluate(model: Sequential, X, y, batch_size: int = 256):
    accs = []
    for xb, yb in iterate_minibatches(X, y, batch_size=batch_size, shuffle=False):
        logits = model.forward(xb, train=False)
        accs.append(accuracy_from_logits(logits, yb))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--pool", choices=["max", "avg"], default="max")
    ap.add_argument("--runs-dir", type=str, default="section3_cnn/runs/mnist_numpy")
    args = ap.parse_args()

    set_seed(args.seed)

    (X_train, y_train), (X_test, y_test) = load_mnist_numpy()

    pool = MaxPool2D() if args.pool == "max" else AvgPool2D()

    # Simple CNN:
    # (1,28,28) -> conv(8) -> relu -> pool -> conv(16) -> relu -> pool -> flatten -> fc -> fc
    model = Sequential([
        Conv2D(1, 8, kernel_size=3, stride=1, pad=1, weight_scale=0.02),
        ReLU(),
        pool,
        Conv2D(8, 16, kernel_size=3, stride=1, pad=1, weight_scale=0.02),
        ReLU(),
        MaxPool2D(),  # keep second pooling max (common baseline)
        Flatten(),
        Linear(16 * 7 * 7, 128, weight_scale=0.02),
        ReLU(),
        Linear(128, 10, weight_scale=0.02),
    ])
    loss_fn = SoftmaxCrossEntropy()

    run_dir = ensure_dir(args.runs_dir)
    ckpt_path = run_dir / "best.pkl"

    train_losses, test_accs = [], []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # training
        epoch_losses = []
        for xb, yb in tqdm(iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True),
                           total=len(X_train)//args.batch_size, desc=f"epoch {epoch}"):
            logits = model.forward(xb, train=True)
            loss = loss_fn.forward(logits, yb)
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            sgd_step(model.params_and_grads(), lr=args.lr, weight_decay=args.weight_decay)
            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses))
        test_acc = evaluate(model, X_test, y_test)

        train_losses.append(train_loss)
        test_accs.append(test_acc)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} test_acc={test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            with open(ckpt_path, "wb") as f:
                pickle.dump({"model": model, "epoch": epoch, "test_acc": test_acc}, f)

        save_curve(run_dir / "curves_loss.png", list(range(1, len(train_losses)+1)), {"train_loss": train_losses},
                   ylabel="loss", title="NumPy CNN training loss")
        save_curve(run_dir / "curves_acc.png", list(range(1, len(test_accs)+1)), {"test_acc": test_accs},
                   ylabel="accuracy", title="NumPy CNN test accuracy")

    print(f"Best test accuracy: {best_acc*100:.2f}%")
    print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()

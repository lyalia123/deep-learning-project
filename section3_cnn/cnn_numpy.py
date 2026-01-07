"""
NumPy CNN from scratch (forward + backward), built for MNIST (N,1,28,28).

Key design choices:
- im2col/col2im implementation for Conv and Pool layers for speed.
- Softmax + Cross Entropy combined (stable).
- SGD optimizer.

This is intentionally educational-but-usable: readable code, not maximally optimized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np


def _pad2d(x: np.ndarray, pad: int) -> np.ndarray:
    if pad == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")


def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> Tuple[np.ndarray, Tuple]:
    """
    Convert input (N,C,H,W) into columns: (N*out_h*out_w, C*kh*kw)
    Returns cols and a cache for col2im.
    """
    N, C, H, W = x.shape
    x_p = _pad2d(x, pad)
    Hp, Wp = x_p.shape[2], x_p.shape[3]

    out_h = (Hp - kh) // stride + 1
    out_w = (Wp - kw) // stride + 1

    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(kw), kh)
    j0 = np.tile(j0, C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)

    cols = x_p[:, k, i, j]  # (N, C*kh*kw, out_h*out_w)
    cols = cols.transpose(0, 2, 1).reshape(N * out_h * out_w, -1)
    cache = (x.shape, kh, kw, stride, pad, out_h, out_w, k, i, j)
    return cols, cache


def col2im(cols: np.ndarray, cache: Tuple) -> np.ndarray:
    """
    Inverse of im2col.
    """
    (x_shape, kh, kw, stride, pad, out_h, out_w, k, i, j) = cache
    N, C, H, W = x_shape
    Hp, Wp = H + 2 * pad, W + 2 * pad
    x_p = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)

    cols_reshaped = cols.reshape(N, out_h * out_w, C * kh * kw).transpose(0, 2, 1)
    np.add.at(x_p, (slice(None), k, i, j), cols_reshaped)

    if pad == 0:
        return x_p
    return x_p[:, :, pad:-pad, pad:-pad]


class Layer:
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params_and_grads(self):
        return []


class ReLU(Layer):
    def __init__(self):
        self.mask: Optional[np.ndarray] = None

    def forward(self, x, train=True):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Flatten(Layer):
    def __init__(self):
        self.orig_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x, train=True):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)


class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int, weight_scale: float = 0.01):
        self.W = weight_scale * np.random.randn(in_dim, out_dim).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.x: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, x, train=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx

    def params_and_grads(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class Conv2D(Layer):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, pad: int = 1, weight_scale: float = 0.01):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kh = self.kw = kernel_size
        self.stride = stride
        self.pad = pad

        self.W = weight_scale * np.random.randn(out_ch, in_ch, self.kh, self.kw).astype(np.float32)
        self.b = np.zeros((out_ch,), dtype=np.float32)

        self.cache = None
        self.x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x, train=True):
        cols, cache = im2col(x, self.kh, self.kw, self.stride, self.pad)  # (N*out_h*out_w, C*kh*kw)
        W_col = self.W.reshape(self.out_ch, -1).T  # (C*kh*kw, out_ch)
        out = cols @ W_col + self.b  # (N*out_h*out_w, out_ch)

        # reshape to (N, out_ch, out_h, out_w)
        (x_shape, _kh, _kw, _stride, _pad, out_h, out_w, *_rest) = cache
        out = out.reshape(x_shape[0], out_h, out_w, self.out_ch).transpose(0, 3, 1, 2)

        self.cache = (cols, cache)
        self.x_shape = x.shape
        return out

    def backward(self, dout):
        cols, cache = self.cache
        N = self.x_shape[0]
        out_h = (self.x_shape[2] + 2 * self.pad - self.kh) // self.stride + 1
        out_w = (self.x_shape[3] + 2 * self.pad - self.kw) // self.stride + 1

        dout_col = dout.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, self.out_ch)

        self.db = dout_col.sum(axis=0)
        self.dW = (cols.T @ dout_col).T.reshape(self.W.shape)

        W_col = self.W.reshape(self.out_ch, -1)
        dcols = dout_col @ W_col  # (N*out_h*out_w, C*kh*kw)
        dx = col2im(dcols, cache)
        return dx

    def params_and_grads(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class MaxPool2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kh = self.kw = kernel_size
        self.stride = stride
        self.cache = None
        self.argmax = None
        self.x_shape = None

    def forward(self, x, train=True):
        N, C, H, W = x.shape
        cols, cache = im2col(x, self.kh, self.kw, self.stride, pad=0)  # (N*out_h*out_w, C*kh*kw)
        cols = cols.reshape(-1, C, self.kh * self.kw)  # (N*out_h*out_w, C, k*k)
        argmax = np.argmax(cols, axis=2)
        out = np.max(cols, axis=2)  # (N*out_h*out_w, C)

        out_h = (H - self.kh) // self.stride + 1
        out_w = (W - self.kw) // self.stride + 1
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.cache = (cache, C)
        self.argmax = argmax
        self.x_shape = x.shape
        return out

    def backward(self, dout):
        cache, C = self.cache
        N, Cx, H, W = self.x_shape
        out_h = (H - self.kh) // self.stride + 1
        out_w = (W - self.kw) // self.stride + 1

        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*out_h*out_w, C)
        dcols = np.zeros((dout_flat.shape[0], C, self.kh * self.kw), dtype=dout.dtype)
        idx = self.argmax.reshape(-1, C)
        np.put_along_axis(dcols, idx[..., None], dout_flat[..., None], axis=2)
        dcols = dcols.reshape(dout_flat.shape[0], -1)  # (N*out_h*out_w, C*kh*kw)

        dx = col2im(dcols, cache)
        return dx

    def params_and_grads(self):
        return []


class AvgPool2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kh = self.kw = kernel_size
        self.stride = stride
        self.cache = None
        self.x_shape = None

    def forward(self, x, train=True):
        N, C, H, W = x.shape
        cols, cache = im2col(x, self.kh, self.kw, self.stride, pad=0)
        cols = cols.reshape(-1, C, self.kh * self.kw)
        out = cols.mean(axis=2)  # (N*out_h*out_w, C)

        out_h = (H - self.kh) // self.stride + 1
        out_w = (W - self.kw) // self.stride + 1
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.cache = (cache, C)
        self.x_shape = x.shape
        return out

    def backward(self, dout):
        cache, C = self.cache
        N, Cx, H, W = self.x_shape
        out_h = (H - self.kh) // self.stride + 1
        out_w = (W - self.kw) // self.stride + 1

        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        dcols = np.repeat(dout_flat[..., None] / (self.kh * self.kw), self.kh * self.kw, axis=2)
        dcols = dcols.reshape(dout_flat.shape[0], -1)
        dx = col2im(dcols, cache)
        return dx


class SoftmaxCrossEntropy:
    """
    Combines:
      softmax(logits) + cross-entropy loss
    Returns: loss (float)
    """
    def __init__(self):
        self.probs = None
        self.y = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self.probs = probs
        self.y = y
        N = logits.shape[0]
        loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
        return float(loss)

    def backward(self) -> np.ndarray:
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y] -= 1.0
        grad /= N
        return grad


class Sequential:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params_and_grads(self):
        for li, layer in enumerate(self.layers):
            for name, p, g in layer.params_and_grads():
                yield f"{li}.{layer.__class__.__name__}.{name}", p, g


@dataclass
class TrainState:
    epoch: int
    step: int
    best_acc: float


def accuracy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == y).mean())


def sgd_step(params_and_grads, lr: float, weight_decay: float = 0.0):
    for _, p, g in params_and_grads:
        if g is None:
            continue
        if weight_decay != 0.0:
            g = g + weight_decay * p
        p -= lr * g

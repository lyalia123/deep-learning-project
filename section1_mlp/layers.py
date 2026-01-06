import numpy as np

class Linear:
    def __init__(self, input_dim: int, output_dim: int):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        m = self.X.shape[0]
        self.dW = self.X.T @ dZ / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T
        return dX

    def step(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db

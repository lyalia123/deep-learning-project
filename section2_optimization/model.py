from layers import Linear
from activations import relu, relu_derivative
import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        # добавляем сразу, чтобы optimizer мог их взять
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        m = self.X.shape[0]
        self.dW = self.X.T @ dZ / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T
        return dX

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, X):
        self.Z1 = self.fc1.forward(X)
        self.A1 = relu(self.Z1)
        self.Z2 = self.fc2.forward(self.A1)
        return self.Z2

    def backward(self, dLoss):
        dZ2 = dLoss
        dA1 = self.fc2.backward(dZ2)
        dZ1 = dA1 * relu_derivative(self.Z1)
        self.fc1.backward(dZ1)

    def parameters(self):
        # возвращает список для оптимизаторов
        return [{'W': self.fc1.W, 'b': self.fc1.b, 'dW': self.fc1.dW, 'db': self.fc1.db},
                {'W': self.fc2.W, 'b': self.fc2.b, 'dW': self.fc2.dW, 'db': self.fc2.db}]

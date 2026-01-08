import numpy as np
from layers import ReLU, Sigmoid, Tanh, softmax

class MLP:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation="relu"):
        # ===== инициализация весов =====
        if activation == "relu":
            scale = lambda n_in: np.sqrt(2 / n_in)
        else:  # sigmoid, tanh
            scale = lambda n_in: np.sqrt(1 / n_in)

        self.W1 = np.random.randn(input_dim, hidden_dim1) * scale(input_dim)
        self.b1 = np.zeros((1, hidden_dim1))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * scale(hidden_dim1)
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2, output_dim) * scale(hidden_dim2)
        self.b3 = np.zeros((1, output_dim))

        activations = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
        self.act_fn = activations[activation]

        self.cache = {}

    def forward(self, X):
        self.cache["X"] = X
        self.cache["Z1"] = Z1 = X.dot(self.W1) + self.b1
        self.cache["A1"] = A1 = self.act_fn.forward(Z1)
        self.cache["Z2"] = Z2 = A1.dot(self.W2) + self.b2
        self.cache["A2"] = A2 = self.act_fn.forward(Z2)
        self.cache["Z3"] = Z3 = A2.dot(self.W3) + self.b3
        self.cache["A3"] = A3 = softmax(Z3)
        return A3

    def backward(self, dLoss):
        X, Z1, A1, Z2, A2 = self.cache["X"], self.cache["Z1"], self.cache["A1"], self.cache["Z2"], self.cache["A2"]
        m = dLoss.shape[0]

        # Output layer
        dZ3 = dLoss
        dW3 = A2.T.dot(dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        # Hidden layer 2
        dA2 = dZ3.dot(self.W3.T)
        dZ2 = self.act_fn.backward(dA2, Z2)
        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer 1
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = self.act_fn.backward(dA1, Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Normalize
        self.grads = {
            "W1": dW1/m, "b1": db1/m,
            "W2": dW2/m, "b2": db2/m,
            "W3": dW3/m, "b3": db3/m
        }

    def step(self, lr):
        for param, grad in zip(
            [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3],
            [self.grads["W1"],self.grads["b1"],self.grads["W2"],self.grads["b2"],self.grads["W3"],self.grads["b3"]]
        ):
            param -= lr * grad

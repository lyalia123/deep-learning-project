import numpy as np

class ReLU:
    @staticmethod
    def forward(Z):
        return np.maximum(0, Z)

    @staticmethod
    def backward(dA, Z):
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ

class Sigmoid:
    @staticmethod
    def forward(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def backward(dA, Z):
        A = Sigmoid.forward(Z)
        return dA * A * (1 - A)

class Tanh:
    @staticmethod
    def forward(Z):
        return np.tanh(Z)

    @staticmethod
    def backward(dA, Z):
        A = np.tanh(Z)
        return dA * (1 - A**2)

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

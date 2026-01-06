import numpy as np

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def cross_entropy_derivative(y_pred, y_true):
    return (y_pred - y_true) / y_true.shape[0]

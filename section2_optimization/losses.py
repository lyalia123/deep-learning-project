import numpy as np

def softmax(x):
    """Numerically stable softmax"""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(pred, y, epsilon=1e-12):
    """
    Compute cross-entropy loss
    pred: softmax probabilities (m × c)
    y: one-hot encoded labels (m × c)
    """
    m = y.shape[0]
    # Clip probabilities to avoid log(0)
    pred = np.clip(pred, epsilon, 1. - epsilon)
    loss = -np.sum(y * np.log(pred)) / m
    return loss

def cross_entropy_backward(pred, y):
    """
    Gradient of cross-entropy loss w.r.t. softmax input
    Simplified derivative for softmax + cross-entropy combination
    """
    m = y.shape[0]
    return (pred - y) / m

def mse(pred, y):
    """Mean Squared Error for regression"""
    return np.mean((pred - y) ** 2)

def mse_backward(pred, y):
    """Gradient of MSE"""
    return 2 * (pred - y) / y.shape[0]
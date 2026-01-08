import numpy as np

def cross_entropy(probs, Y):
    m = Y.shape[0]
    probs = np.clip(probs, 1e-12, 1-1e-12)
    return -np.sum(Y * np.log(probs)) / m

def cross_entropy_derivative(probs, Y):
    # для softmax + one-hot
    return (probs - Y) / Y.shape[0]

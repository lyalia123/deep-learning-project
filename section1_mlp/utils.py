import numpy as np

def accuracy(y_pred, y_true):
    return (y_pred.argmax(axis=1) == y_true.argmax(axis=1)).mean()

def create_batches(X, Y, batch_size):
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size]

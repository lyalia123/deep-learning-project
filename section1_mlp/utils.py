import numpy as np

def accuracy(probs, Y):
    preds = np.argmax(probs, axis=1)
    labels = np.argmax(Y, axis=1)
    return np.mean(preds == labels)

def create_batches(X, Y, batch_size=64):
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    for start in range(0, m, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]

import numpy as np

def one_hot_encode(y, num_classes=None):
    """
    Convert integer labels to one-hot encoding
    """
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    
    # Handle both 1D and 2D y arrays
    if y.shape[1] == 1:
        one_hot[np.arange(m), y.flatten()] = 1
    else:
        # If already one-hot, return as is
        if y.shape[1] == num_classes:
            return y
        else:
            raise ValueError(f"y shape {y.shape} doesn't match num_classes {num_classes}")
    
    return one_hot

def accuracy(pred, y):
    """
    Calculate accuracy from predictions and labels
    pred: softmax probabilities or logits (m × c)
    y: one-hot encoded labels (m × c)
    """
    if len(y.shape) > 1 and y.shape[1] > 1:
        # y is one-hot encoded
        y_labels = np.argmax(y, axis=1)
    else:
        # y is already integer labels
        y_labels = y.flatten()
    
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        # pred is probabilities or logits
        pred_labels = np.argmax(pred, axis=1)
    else:
        # pred is already predictions
        pred_labels = pred.flatten()
    
    return np.mean(pred_labels == y_labels)

def create_minibatches(X, y, batch_size=32, shuffle=True):
    """
    Create mini-batches from dataset
    """
    m = X.shape[0]
    
    if shuffle:
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
    
    for i in range(0, m, batch_size):
        end = min(i + batch_size, m)
        yield X[i:end], y[i:end]

def create_batches(X, Y, batch_size=64):
    """
    Alternative name for compatibility
    """
    return create_minibatches(X, Y, batch_size, shuffle=True)
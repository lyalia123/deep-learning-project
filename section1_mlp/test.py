import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from model import MLP
from losses import cross_entropy, cross_entropy_derivative
from mlp_regression import MLPReg  # импортируем отдельный модуль

np.random.seed(42)

# ------------------ Utility functions ------------------
def small_batch_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y)
    return X[:256], Y[:256]

# ------------------ Test 1: Forward pass ------------------
def test_forward_softmax():
    X, Y = small_batch_mnist()
    model = MLP(784, 16, 16, 10, activation="relu")
    probs = model.forward(X)
    assert probs.shape == (X.shape[0], 10), f"Unexpected shape: {probs.shape}"
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), "Softmax rows do not sum to 1"

# ------------------ Test 2: Backward pass ------------------
def test_backward_shapes():
    X, Y = small_batch_mnist()
    model = MLP(784, 16, 16, 10, activation="relu")
    probs = model.forward(X)
    dLoss = cross_entropy_derivative(probs, Y)
    model.backward(dLoss)
    for key in ["W1","b1","W2","b2","W3","b3"]:
        assert key in model.grads
        assert model.grads[key].shape == getattr(model, key).shape, f"Grad shape mismatch for {key}"

# ------------------ Test 3: Training step decreases loss ------------------
def test_training_step():
    X, Y = small_batch_mnist()
    model = MLP(784, 16, 16, 10, activation="relu")
    probs = model.forward(X)
    initial_loss = cross_entropy(probs, Y)
    dLoss = cross_entropy_derivative(probs, Y)
    model.backward(dLoss)
    model.step(lr=0.1)
    probs2 = model.forward(X)
    new_loss = cross_entropy(probs2, Y)
    assert new_loss <= initial_loss, "Loss did not decrease after one training step"

# ------------------ Test 4: Universal Approximation ------------------
def test_mlp_regression():
    X = np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1,1)
    Y = np.sin(X)
    X_norm = (X - X.min()) / (X.max()-X.min())
    Y_norm = (Y - Y.min()) / (Y.max()-Y.min())
    mlp = MLPReg(1, 8, 8)
    epochs = 50
    lr = 0.01
    losses = []
    for _ in range(epochs):
        preds = mlp.forward(X_norm)
        loss = np.mean((preds - Y_norm)**2)
        losses.append(loss)
        mlp.backward(X_norm, Y_norm, preds)
        mlp.step(lr)
    assert losses[-1] < losses[0], "Regression MLP did not learn (loss did not decrease)"

# ------------------ Run tests manually ------------------
if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from model import MLP
from losses import cross_entropy, cross_entropy_derivative

np.random.seed(42)

# ---------- Utility functions ----------
def small_batch_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y)
    # берём маленький батч для тестов
    return X[:256], Y[:256]

def assert_close(a, b, tol=1e-6):
    assert np.all(np.abs(a - b) < tol), f"Arrays not close! Max diff: {np.max(np.abs(a-b))}"

# ---------- Test 1: Forward pass softmax ----------
def test_forward_softmax():
    X, Y = small_batch_mnist()
    model = MLP(784, 16, 16, 10, activation="relu")
    probs = model.forward(X)
    # Проверяем форму
    assert probs.shape == (X.shape[0], 10), f"Unexpected output shape: {probs.shape}"
    # Сумма вероятностей = 1
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), f"Softmax rows do not sum to 1"

# ---------- Test 2: Backward pass ----------
def test_backward_shapes():
    X, Y = small_batch_mnist()
    model = MLP(784, 16, 16, 10, activation="relu")
    probs = model.forward(X)
    dLoss = cross_entropy_derivative(probs, Y)
    model.backward(dLoss)
    # Проверяем размеры градиентов
    for key in ["W1", "W2", "W3", "b1", "b2", "b3"]:
        assert key in model.grads, f"{key} missing in grads"
        assert model.grads[key].shape == getattr(model, key).shape, f"Grad shape mismatch for {key}"

# ---------- Test 3: Training decreases loss ----------
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

# ---------- Test 4: Universal Approximation (MLPReg) ----------
def test_mlp_regression():
    from __main__ import MLPReg  # если тест в том же проекте
    X = np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1,1)
    Y = np.sin(X)
    # нормализация
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
    # Проверяем, что loss уменьшился
    assert losses[-1] < losses[0], "Regression MLP did not learn (loss did not decrease)"

# ---------- Run tests ----------
if __name__ == "__main__":
    test_forward_softmax()
    print("Forward pass test passed!")
    test_backward_shapes()
    print("Backward pass test passed!")
    test_training_step()
    print("Training step test passed!")
    test_mlp_regression()
    print("MLP regression (universal approximation) test passed!")

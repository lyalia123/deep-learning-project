import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from model import MLP
from activations import softmax
from losses import cross_entropy, cross_entropy_derivative
from utils import accuracy, create_batches

# reproducibility
np.random.seed(42)

# =========================
# Load MNIST
# =========================
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int).to_numpy().reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# make sure numpy arrays
X_train = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
Y_train = Y_train.to_numpy() if hasattr(Y_train, "to_numpy") else Y_train
X_test = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
Y_test = Y_test.to_numpy() if hasattr(Y_test, "to_numpy") else Y_test

# =========================
# Training config
# =========================
lr = 0.1
epochs = 20
batch_size = 64

activations = ["relu", "sigmoid", "tanh"]
results = {}

# =========================
# Train models
# =========================
for act in activations:
    print(f"\n===== Training with {act.upper()} =====")

    model = MLP(
        input_dim=784,
        hidden_dim1=256,
        hidden_dim2=128,
        output_dim=10,
        activation=act
    )

    acc_history = []

    for epoch in range(epochs):
        for X_batch, Y_batch in create_batches(X_train, Y_train, batch_size):
            logits = model.forward(X_batch)
            probs = softmax(logits)

            loss = cross_entropy(probs, Y_batch)
            dLoss = cross_entropy_derivative(probs, Y_batch)

            model.backward(dLoss)
            model.step(lr)

        # evaluation
        test_logits = model.forward(X_test)
        test_probs = softmax(test_logits)
        acc = accuracy(test_probs, Y_test)
        acc_history.append(acc)

        print(
            f"Activation={act} | "
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Test Acc: {acc:.4f}"
        )

    results[act] = acc_history

# =========================
# Final summary
# =========================
print("\n===== FINAL ACCURACY =====")
for act in results:
    print(f"{act.upper()}: {results[act][-1]:.4f}")

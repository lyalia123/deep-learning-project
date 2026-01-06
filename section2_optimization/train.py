import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from model import TwoLayerMLP
from losses import cross_entropy, cross_entropy_derivative
from utils import accuracy, create_batches
from optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

np.random.seed(42)

# =========================
# Load MNIST
# =========================
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data.astype(np.float32)/255
y = mnist.target.astype(int).to_numpy().reshape(-1,1)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# make sure numpy arrays
X_train = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
Y_train = Y_train.to_numpy() if hasattr(Y_train, "to_numpy") else Y_train
X_test = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
Y_test = Y_test.to_numpy() if hasattr(Y_test, "to_numpy") else Y_test

# =========================
# Config
# =========================
epochs = 5
batch_size = 64
lr = 0.01

optimizers = ["SGD", "Adam", "RMSprop"]
results = {}

# =========================
# Training loop
# =========================
for opt_name in optimizers:
    print(f"\n=== Training with {opt_name} ===")
    model = TwoLayerMLP(784, 128, 10)
    
    if opt_name == "SGD":
        opt = SGD(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        opt = RMSprop(model.parameters(), lr=lr)
    else:
        opt = Adam(model.parameters(), lr=lr)

    acc_history = []
    loss_history = []

    for epoch in range(epochs):
        for X_batch, Y_batch in create_batches(X_train, Y_train, batch_size):
            logits = model.forward(X_batch)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            loss = cross_entropy(probs, Y_batch)
            dLoss = cross_entropy_derivative(probs, Y_batch)

            model.backward(dLoss)
            opt.step()

        # evaluation
        test_logits = model.forward(X_test)
        probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        acc = accuracy(probs, Y_test)

        acc_history.append(acc)
        loss_history.append(loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    results[opt_name] = {'acc': acc_history, 'loss': loss_history}

# =========================
# Plot results
# =========================
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
for opt_name in results:
    plt.plot(results[opt_name]['loss'], label=opt_name)
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1,2,2)
for opt_name in results:
    plt.plot(results[opt_name]['acc'], label=opt_name)
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

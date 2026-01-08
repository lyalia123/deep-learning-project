import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model import MLP
from losses import cross_entropy, cross_entropy_derivative
from utils import accuracy, create_batches

np.random.seed(42)

# -------- MNIST --------
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32)/255.0
y = mnist.target.astype(int).reshape(-1,1)
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# -------- Config --------
lr = 0.1
epochs = 20
batch_size = 128
activations = ["relu","sigmoid","tanh"]

results = {}
trained_models = {}

# -------- Training --------
for act in activations:
    print(f"\n===== Training with {act.upper()} =====")
    model = MLP(input_dim=784, hidden_dim1=256, hidden_dim2=128, output_dim=10, activation=act)
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        for X_batch, Y_batch in create_batches(X_train, Y_train, batch_size):
            probs = model.forward(X_batch)
            loss = cross_entropy(probs, Y_batch)
            dLoss = cross_entropy_derivative(probs, Y_batch)
            model.backward(dLoss)
            model.step(lr)

        test_probs = model.forward(X_test)
        acc = accuracy(test_probs, Y_test)

        loss_history.append(loss)
        acc_history.append(acc)

        print(f"Activation={act} | Epoch {epoch+1}/{epochs} | Loss={loss:.4f} | Test Acc={acc:.4f}")

    results[act] = acc_history
    trained_models[act] = model

    # Loss plot per activation
    plt.plot(loss_history, label=act.upper())

plt.title("Loss Curves per Activation")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.show()

# Accuracy comparison
plt.figure()
for act in results:
    plt.plot(results[act], label=act.upper())
plt.title("Accuracy per Epoch – Activation Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Activation distributions
# Activation distributions
X_sample = X_test[:1000]
plt.figure(figsize=(15,8))
plot_idx = 1

for act in activations:
    model = trained_models[act]
    _ = model.forward(X_sample)
    
    A1 = model.cache["A1"].flatten()
    A2 = model.cache["A2"].flatten()
    
    plt.subplot(3,2,plot_idx)
    plt.hist(A1, bins=50)
    plt.title(f"{act.upper()} - Hidden Layer 1")
    plot_idx += 1
    
    plt.subplot(3,2,plot_idx)
    plt.hist(A2, bins=50)
    plt.title(f"{act.upper()} - Hidden Layer 2")
    plot_idx += 1

plt.tight_layout()
plt.show()


# Confusion matrices
for act in activations:
    model = trained_models[act]
    preds = model.forward(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix – {act.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

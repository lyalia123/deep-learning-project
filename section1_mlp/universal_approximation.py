import numpy as np
import matplotlib.pyplot as plt

from model import MLP
from losses import mse, mse_derivative

np.random.seed(42)

# =========================
# Dataset: sin function
# =========================
x = np.linspace(-1, 1, 200).reshape(-1, 1)
y = np.sin(3 * x)

# =========================
# Training config
# =========================
epochs = 3000
lr = 0.05

hidden_sizes = [5, 20, 50]

plt.figure(figsize=(8, 5))

for h in hidden_sizes:
    print(f"\nTraining with {h} hidden neurons")

    model = MLP(
        input_dim=1,
        hidden_dim1=h,
        hidden_dim2=0,   # ❗ второй слой отключаем
        output_dim=1,
        activation="tanh"
    )

    for epoch in range(epochs):
        y_pred = model.forward(x)
        loss = mse(y_pred, y)
        dLoss = mse_derivative(y_pred, y)

        model.backward(dLoss)
        model.step(lr)

    # prediction
    y_hat = model.forward(x)

    plt.plot(x, y_hat, label=f"{h} neurons")

# true function
plt.plot(x, y, "k--", label="True sin(3x)")
plt.legend()
plt.title("Universal Approximation with MLP")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

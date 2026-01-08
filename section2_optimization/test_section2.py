import numpy as np
from model import MLP
from optimizers import SGD, RMSprop, Adam, GradientDescent
from gradient_checking import gradient_check
from utils import create_minibatches, one_hot_encode
from losses import cross_entropy, cross_entropy_backward

np.random.seed(42)

# ------------------ Test 1: Gradient checking ------------------
def test_gradient_check():
    X = np.random.randn(5, 3)
    y = np.eye(2)[np.random.randint(0, 2, size=5)]
    model = MLP(input_dim=3, hidden_dim=4, output_dim=2, activation="relu")
    
    # Передаем функцию потерь
    passed, max_error = gradient_check(model, X, y, loss_fn=cross_entropy)
    assert passed, f"Gradient check failed! Max error: {max_error}"


# ------------------ Test 2: Mini-batch creation ------------------
def test_create_minibatches():
    X = np.arange(20).reshape(10,2)
    y = np.arange(10).reshape(10,1)
    batches = list(create_minibatches(X, y, batch_size=3))
    total_samples = sum(batch[0].shape[0] for batch in batches)
    assert total_samples == X.shape[0], "Total samples mismatch in minibatches"
    assert batches[-1][0].shape[0] <= 3, "Last batch size too large"

# ------------------ Test 3: Optimizers update parameters ------------------
def test_optimizers_update():
    X = np.random.randn(10, 2)
    y = np.eye(2)[np.random.randint(0, 2, size=10)]
    model = MLP(input_dim=2, hidden_dim=4, output_dim=2, activation="relu")  # исправлено
    old_params = {k: v.copy() for k, v in model.parameters().items()}

    optimizers = [
        SGD(lr=0.1, momentum=0.9),
        RMSprop(lr=0.01),
        Adam(lr=0.01),
        GradientDescent(lr=0.1)
    ]

    for opt in optimizers:
        pred = model.predict_proba(X)
        loss = cross_entropy(pred, y)
        grad_loss = cross_entropy_backward(pred, y)
        model.backward(grad_loss)
        opt.step(model)
        new_params = model.parameters()
        changed = any(not np.allclose(old_params[k], new_params[k]) for k in old_params)
        assert changed, f"{opt.__class__.__name__} did not update any parameters"

# ------------------ Test 4: Training step reduces loss ------------------
def test_training_step_loss_decrease():
    X = np.random.randn(50, 2)
    y = np.eye(2)[np.random.randint(0, 2, size=50)]
    model = MLP(input_dim=2, hidden_dim=4, output_dim=2, activation="relu")  # исправлено
    optimizer = SGD(lr=0.1, momentum=0.9)

    pred1 = model.predict_proba(X)
    loss1 = cross_entropy(pred1, y)

    grad_loss = cross_entropy_backward(pred1, y)
    model.backward(grad_loss)
    optimizer.step(model)

    pred2 = model.predict_proba(X)
    loss2 = cross_entropy(pred2, y)
    assert loss2 <= loss1, "Loss did not decrease after one training step"

# ------------------ Run tests manually ------------------
if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])

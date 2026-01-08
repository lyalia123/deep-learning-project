import os
import sys
import importlib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEC1 = os.path.join(ROOT, "section1_mlp")

def import_sec1_model_and_loss():
    # Убрать пути секций если уже добавлялись
    for p in [os.path.join(ROOT, "section2_optimization"), SEC1]:
        if p in sys.path:
            sys.path.remove(p)

    # Добавить section1_mlp первым
    sys.path.insert(0, SEC1)

    # Очистить плоские модули, чтобы не тянуло "layers" из другой секции
    for name in ["layers", "model", "losses", "utils", "activations", "train", "gradient_checking", "optimizers"]:
        sys.modules.pop(name, None)

    model_mod = importlib.import_module("model")
    loss_mod = importlib.import_module("losses")
    return model_mod, loss_mod

def one_hot(y, num_classes):
    y = y.astype(int)
    Y = np.zeros((y.shape[0], num_classes), dtype=float)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y

def test_mlp_forward_shape_and_loss():
    model_mod, loss_mod = import_sec1_model_and_loss()
    MLP = model_mod.MLP
    cross_entropy = loss_mod.cross_entropy

    np.random.seed(0)
    X = np.random.randn(8, 20)
    y = np.random.randint(0, 3, size=(8,))

    model = MLP(input_dim=20, hidden_dim1=16, hidden_dim2=16, output_dim=3, activation="relu")

    probs = model.forward(X)  # forward возвращает probs
    assert probs.shape == (8, 3)

    Y = one_hot(y, 3)
    loss = cross_entropy(probs, Y)
    assert np.isfinite(loss)

def test_mlp_backward_runs():
    model_mod, _ = import_sec1_model_and_loss()
    MLP = model_mod.MLP

    np.random.seed(1)
    X = np.random.randn(8, 10)
    y = np.random.randint(0, 4, size=(8,))

    model = MLP(input_dim=10, hidden_dim1=12, hidden_dim2=12, output_dim=4, activation="tanh")

    probs = model.forward(X)
    Y = one_hot(y, 4)

    dprobs = (probs - Y) / X.shape[0]
    model.backward(dprobs)  # backward(dprobs) по вашей сигнатуре

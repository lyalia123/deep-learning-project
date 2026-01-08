import os
import sys
import numpy as np
import importlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEC2 = os.path.join(ROOT, "section2_optimization")

def import_sec2_modules():
    # 1) убрать возможные секции из sys.path, чтобы не путались layers/model
    for p in [os.path.join(ROOT, "section1_mlp"), SEC2]:
        if p in sys.path:
            sys.path.remove(p)

    # 2) добавить section2_optimization первым в sys.path
    sys.path.insert(0, SEC2)

    # 3) вычистить плоские модули (важно!)
    for name in ["layers", "model", "losses", "utils", "optimizers", "gradient_checking", "train"]:
        sys.modules.pop(name, None)

    # 4) импортировать плоско, как ожидает код секции
    model = importlib.import_module("model")
    optim = importlib.import_module("optimizers")
    return model, optim

def test_section2_mlp_forward_shape():
    model_mod, _ = import_sec2_modules()
    MLP = model_mod.MLP

    X = np.random.randn(8, 10)
    mlp = MLP(input_dim=10, hidden_dim=8, output_dim=3, activation="relu")
    logits = mlp.forward(X)
    assert logits.shape == (8, 3)

def test_section2_optimizers_exist():
    _, optim_mod = import_sec2_modules()
    assert hasattr(optim_mod, "SGD")
    assert hasattr(optim_mod, "Adam")

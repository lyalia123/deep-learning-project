import torch
from section4_transfer.models import PlainCNN

def test_plaincnn_forward():
    model = PlainCNN(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

def test_training_step():
    model = PlainCNN(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0

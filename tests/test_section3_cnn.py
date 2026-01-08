import numpy as np
from section3_cnn.cnn_numpy import Conv2D, MaxPool2D

def test_conv_forward_shape():
    x = np.random.randn(2, 1, 28, 28)
    conv = Conv2D(1, 4, kernel_size=3, pad=1)
    out = conv.forward(x)
    assert out.shape == (2, 4, 28, 28)

def test_pool_forward_shape():
    x = np.random.randn(2, 4, 28, 28)
    pool = MaxPool2D(kernel_size=2, stride=2)
    out = pool.forward(x)
    assert out.shape == (2, 4, 14, 14)

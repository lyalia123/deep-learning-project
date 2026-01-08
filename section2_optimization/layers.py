import numpy as np

class Linear:
    def __init__(self, in_features, out_features, initialization="he"):
        self.in_features = in_features
        self.out_features = out_features
        
        if initialization == "he":
            # He initialization for ReLU
            scale = np.sqrt(2.0 / in_features)
        elif initialization == "xavier":
            # Xavier initialization for sigmoid/tanh
            scale = np.sqrt(1.0 / in_features)
        else:
            scale = 0.01
            
        self.W = scale * np.random.randn(in_features, out_features)
        self.b = np.zeros((1, out_features))
        
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        """Forward pass: Z = XW + b"""
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, grad_out):
        """Backward pass: computes dW, db, and returns dX"""
        # dW = Xᵀ @ dL/dZ
        self.dW = self.x.T @ grad_out
        
        # db = sum(dL/dZ, axis=0)
        self.db = np.sum(grad_out, axis=0, keepdims=True)
        
        # dX = dL/dZ @ Wᵀ
        dx = grad_out @ self.W.T
        return dx
    
    def zero_grad(self):
        """Reset gradients"""
        self.dW = None
        self.db = None

class ReLU:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        """Forward pass: A = max(0, Z)"""
        self.mask = (x > 0).astype(float)
        return x * self.mask
    
    def backward(self, grad_out):
        """Backward pass: dZ = dA ⊙ (Z > 0)"""
        return grad_out * self.mask

class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        """Forward pass: A = 1 / (1 + exp(-Z))"""
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, grad_out):
        """Backward pass: dZ = dA * A * (1 - A)"""
        return grad_out * self.out * (1 - self.out)

class Tanh:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        """Forward pass: A = tanh(Z)"""
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, grad_out):
        """Backward pass: dZ = dA * (1 - A²)"""
        return grad_out * (1 - self.out ** 2)
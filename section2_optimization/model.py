import numpy as np
from layers import Linear, ReLU, Sigmoid, Tanh
from losses import softmax

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu", initialization="he"):
        """
        2-layer MLP for classification
        """
        self.layers = []
        
        # Layer 1: Linear + Activation
        self.fc1 = Linear(input_dim, hidden_dim, initialization)
        
        if activation == "relu":
            self.act1 = ReLU()
        elif activation == "sigmoid":
            self.act1 = Sigmoid()
        elif activation == "tanh":
            self.act1 = Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Layer 2: Linear (output layer)
        self.fc2 = Linear(hidden_dim, output_dim, initialization)
        
        # Store layers in list for easy iteration
        self.layers = [self.fc1, self.act1, self.fc2]
        
    def forward(self, x):
        """Forward pass through the network"""
        z1 = self.fc1.forward(x)
        a1 = self.act1.forward(z1)
        z2 = self.fc2.forward(a1)
        
        # Store intermediate values for backward pass
        self.cache = {
            'x': x,
            'z1': z1,
            'a1': a1,
            'z2': z2
        }
        
        return z2  # Return pre-softmax for loss computation
    
    def predict_proba(self, x):
        """Get softmax probabilities"""
        z2 = self.forward(x)
        return softmax(z2)
    
    def predict(self, x):
        """Get class predictions"""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)
    
    def backward(self, grad_loss):
        """Backward pass through the network"""
        # Backward through output layer (Linear)
        grad_a1 = self.fc2.backward(grad_loss)
        
        # Backward through activation
        grad_z1 = self.act1.backward(grad_a1)
        
        # Backward through first layer (Linear)
        _ = self.fc1.backward(grad_z1)
    
    def parameters(self):
        """Get all trainable parameters"""
        return {
            'W1': self.fc1.W, 'b1': self.fc1.b,
            'W2': self.fc2.W, 'b2': self.fc2.b
        }
    
    def gradients(self):
        """Get all gradients"""
        return {
            'W1': self.fc1.dW, 'b1': self.fc1.db,
            'W2': self.fc2.dW, 'b2': self.fc2.db
        }
    
    def zero_grad(self):
        """Reset all gradients"""
        self.fc1.zero_grad()
        self.fc2.zero_grad()
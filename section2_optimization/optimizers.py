import numpy as np

class SGD:
    """Stochastic Gradient Descent with momentum"""
    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = {}
        
    def step(self, model):
        """Update parameters using SGD"""
        params = model.parameters()
        grads = model.gradients()
        
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # Momentum update
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
            
            if self.nesterov:
                # Nesterov accelerated gradient
                params[key] += self.momentum * self.velocity[key] - self.lr * grads[key]
            else:
                # Standard momentum
                params[key] += self.velocity[key]

class RMSprop:
    """RMSprop optimizer"""
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}
        
    def step(self, model):
        """Update parameters using RMSprop"""
        params = model.parameters()
        grads = model.gradients()
        
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # Update cache: E[g²] = βE[g²] + (1-β)g²
            self.cache[key] = self.beta * self.cache[key] + (1 - self.beta) * grads[key]**2
            
            # Parameter update: θ = θ - η * g / sqrt(E[g²] + ε)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)

class Adam:
    """Adam optimizer"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate
        self.t = 0   # Time step
        
    def step(self, model):
        """Update parameters using Adam"""
        params = model.parameters()
        grads = model.gradients()
        
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # Parameter update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class GradientDescent:
    """Vanilla Gradient Descent (for comparison)"""
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def step(self, model):
        """Update parameters using simple gradient descent"""
        params = model.parameters()
        grads = model.gradients()
        
        for key in params:
            params[key] -= self.lr * grads[key]
import numpy as np

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p['W'] -= self.lr * p['dW']
            p['b'] -= self.lr * p['db']

class RMSprop:
    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = [{'dW': np.zeros_like(p['W']), 'db': np.zeros_like(p['b'])} for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            self.s[i]['dW'] = self.beta*self.s[i]['dW'] + (1-self.beta)*(p['dW']**2)
            self.s[i]['db'] = self.beta*self.s[i]['db'] + (1-self.beta)*(p['db']**2)
            p['W'] -= self.lr * p['dW'] / (np.sqrt(self.s[i]['dW']) + self.eps)
            p['b'] -= self.lr * p['db'] / (np.sqrt(self.s[i]['db']) + self.eps)

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [{'dW': np.zeros_like(p['W']), 'db': np.zeros_like(p['b'])} for p in params]
        self.v = [{'dW': np.zeros_like(p['W']), 'db': np.zeros_like(p['b'])} for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i]['dW'] = self.beta1*self.m[i]['dW'] + (1-self.beta1)*p['dW']
            self.m[i]['db'] = self.beta1*self.m[i]['db'] + (1-self.beta1)*p['db']
            self.v[i]['dW'] = self.beta2*self.v[i]['dW'] + (1-self.beta2)*(p['dW']**2)
            self.v[i]['db'] = self.beta2*self.v[i]['db'] + (1-self.beta2)*(p['db']**2)
            
            m_hat_W = self.m[i]['dW'] / (1 - self.beta1**self.t)
            m_hat_b = self.m[i]['db'] / (1 - self.beta1**self.t)
            v_hat_W = self.v[i]['dW'] / (1 - self.beta2**self.t)
            v_hat_b = self.v[i]['db'] / (1 - self.beta2**self.t)
            
            p['W'] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            p['b'] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

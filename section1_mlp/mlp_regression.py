import numpy as np

class MLPReg:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        self.W1 = np.random.randn(input_dim, hidden_dim1) * 0.1
        self.b1 = np.zeros((1, hidden_dim1))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.1
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2,1) * 0.1
        self.b3 = np.zeros((1,1))

    def forward(self,X):
        self.Z1 = X.dot(self.W1)+self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.A1.dot(self.W2)+self.b2
        self.A2 = np.tanh(self.Z2)
        self.Z3 = self.A2.dot(self.W3)+self.b3
        self.A3 = self.Z3
        return self.A3

    def backward(self,X,Y,preds):
        m = X.shape[0]
        dZ3 = (preds-Y)/m
        dW3 = self.A2.T.dot(dZ3)
        db3 = np.sum(dZ3,axis=0,keepdims=True)
        dA2 = dZ3.dot(self.W3.T)
        dZ2 = dA2 * (1 - self.A2**2)
        dW2 = self.A1.T.dot(dZ2)
        db2 = np.sum(dZ2,axis=0,keepdims=True)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * (1 - self.A1**2)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1,axis=0,keepdims=True)
        self.grads = {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}

    def step(self,lr):
        self.W1 -= lr*self.grads["W1"]
        self.b1 -= lr*self.grads["b1"]
        self.W2 -= lr*self.grads["W2"]
        self.b2 -= lr*self.grads["b2"]
        self.W3 -= lr*self.grads["W3"]
        self.b3 -= lr*self.grads["b3"]

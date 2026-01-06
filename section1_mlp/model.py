from layers import Linear
from activations import (
    relu, relu_derivative,
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative
)

class MLP:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation="relu"):
        # activation
        if activation == "relu":
            self.act = relu
            self.act_der = relu_derivative
        elif activation == "sigmoid":
            self.act = sigmoid
            self.act_der = sigmoid_derivative
        elif activation == "tanh":
            self.act = tanh
            self.act_der = tanh_derivative
        else:
            raise ValueError("Unknown activation")

        self.fc1 = Linear(input_dim, hidden_dim1)

        if hidden_dim2 > 0:
            # two hidden layers
            self.fc2 = Linear(hidden_dim1, hidden_dim2)
            self.fc3 = Linear(hidden_dim2, output_dim)
            self.use_second_layer = True
        else:
            # one hidden layer
            self.fc2 = Linear(hidden_dim1, output_dim)
            self.use_second_layer = False

    def forward(self, X):
        self.Z1 = self.fc1.forward(X)
        self.A1 = self.act(self.Z1)

        if self.use_second_layer:
            self.Z2 = self.fc2.forward(self.A1)
            self.A2 = self.act(self.Z2)
            self.Z3 = self.fc3.forward(self.A2)
            return self.Z3
        else:
            self.Z2 = self.fc2.forward(self.A1)
            return self.Z2

    def backward(self, dLoss):
        if self.use_second_layer:
            dZ3 = dLoss
            dA2 = self.fc3.backward(dZ3)
            dZ2 = dA2 * self.act_der(self.Z2)

            dA1 = self.fc2.backward(dZ2)
            dZ1 = dA1 * self.act_der(self.Z1)

            self.fc1.backward(dZ1)
        else:
            dZ2 = dLoss
            dA1 = self.fc2.backward(dZ2)
            dZ1 = dA1 * self.act_der(self.Z1)
            self.fc1.backward(dZ1)

    def step(self, lr):
        self.fc1.step(lr)
        self.fc2.step(lr)
        if self.use_second_layer:
            self.fc3.step(lr)

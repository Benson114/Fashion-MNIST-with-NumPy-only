import numpy as np


class Layer:
    def __init__(self):
        self.input_cache = None


class Activation(Layer):
    def __init__(self, activation_type):
        super().__init__()
        self.type = activation_type
        self.leakyrelu_alpha = 0.01 if activation_type == "leakyrelu" else None  # LeakyReLU parameter

    def forward(self, z):
        self.input_cache = z
        if self.type == "relu":
            return np.maximum(0, z)
        elif self.type == "leakyrelu":
            return np.where(z > 0, z, z * self.leakyrelu_alpha)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.type == "softmax":
            exps = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, dA, z):
        if self.type == "relu":
            return dA * (z > 0)
        elif self.type == "leakyrelu":
            dz = np.ones_like(z)
            dz[z < 0] = self.leakyrelu_alpha
            return dA * dz
        elif self.type == "sigmoid" or self.type == "softmax":
            sig = self.forward(z)
            return dA * sig * (1 - sig)


class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = np.random.normal(0, pow(input_dim, -0.5), (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.input_cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, dZ, x):
        dW = np.dot(x.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dx = np.dot(dZ, self.W.T)
        return dW, db, dx

    def zero_grad(self):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

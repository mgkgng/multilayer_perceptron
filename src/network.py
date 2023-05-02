import numpy as np

class Network:
    def __init__(self, sizes):
        if len(sizes) < 4:
            raise ValueError('At least two hidden layers are required.')
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.loss = []

    def activate(self, z, output=False):
        if output:
            return np.exp(z) / np.sum(np.exp(z)) # softmax
        return 1.0 / (1.0 + np.exp(-z)) # sigmoid
    
    def feedforward(self, a):
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = self.activate(np.dot(w, a) + b)
        return self.activate(np.dot(self.weights[-1], a) + self.biases[-1], output=True)
import numpy as np

class Network:
    def __init__(self, sizes):
        if len(sizes) < 4:
            raise ValueError('At least two hidden layers are required.')
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.zs = []
        self.activations = []
        self.loss = []
        self.val_loss = []

    def activation(self, z, output=False):
        if output is True:
            return np.exp(z) / np.sum(np.exp(z)) # softmax
        return 1.0 / (1.0 + np.exp(-z)) # sigmoid

    def activation_prime(self, z, output=False):
        return self.activation(z) * (1 - self.activation(z))
    
    def feedforward(self, a):
        self.zs = []
        self.activations = [a]

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a) + b
            self.zs.append(z)
            a = self.activation(z)
            self.activations.append(a)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        self.zs.append(z)
        a = self.activation(z, output=True)
        self.activations.append(a)
        return a

    def cross_entropy(self, y_hat, y):
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)

    def backprop(self, X, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        pred = self.feedforward(X)

        err = pred - y
        loss = self.cross_entropy(pred, y)
        self.loss.append(loss)

        delta = err * self.activation_prime(self.zs[-1], output=True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.activations[-2].transpose())

        for l in range(2, self.num_layers - 1):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.activation_prime(self.zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, self.activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def train(self, X_train, y_train, X_val, y_val, epochs, lr):
        for epoch in range(epochs):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            for x, y in zip(X_train, y_train):
                delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(-1, 1), y.reshape(-1, 1))
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
            self.weights = [w - (lr / len(X_train)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (lr / len(X_train)) * nb for b, nb in zip(self.biases, nabla_b)]

            # Validation loss & accuracy
            correct = 0
            for x, y in zip(X_val, y_val):
                pred = self.feedforward(x.reshape(-1, 1))
                loss = self.cross_entropy(pred, y.reshape(-1, 1))
                self.val_loss.append(loss)
                if np.argmax(pred) == np.argmax(y):
                    correct += 1

            print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {np.mean(self.val_loss):.4f} - Val Accuracy: {correct / len(X_val) * 100:.2f}%')
            self.loss = []
            self.val_loss = []


import numpy as np
from mini_batch import MiniBatchGenerator

class Network:
    def __init__(self, sizes, patience=15, min_delta=0.0001):
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
        self.patience = patience
        self.min_delta = min_delta


    def activation(self, z, output=False):
        if output is True:
            return np.exp(z) / np.sum(np.exp(z)) # softmax
        return 1.0 / (1.0 + np.exp(-z)) # sigmoid

    def activation_prime(self, z, output=False):
        return self.activation(z) * (1 - self.activation(z))
    
    def feedforward(self, a, w_lookahead=None, b_lookahead=None):
        # For Nesterov Accelerated Gradient
        weights = self.weights if w_lookahead is None else w_lookahead
        biases = self.biases if b_lookahead is None else b_lookahead     

        self.zs = []
        self.activations = [a]

        for b, w in zip(biases[:-1], weights[:-1]):
            z = np.dot(w, a) + b
            self.zs.append(z)
            a = self.activation(z)
            self.activations.append(a)
        z = np.dot(weights[-1], a) + biases[-1]
        self.zs.append(z)
        a = self.activation(z, output=True)
        self.activations.append(a)
        return a

    def cross_entropy(self, y_hat, y):
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)

    def backprop(self, X, y, w_lookahead=None, b_lookahead=None):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        pred = self.feedforward(X, w_lookahead, b_lookahead)

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

    def validate(self, X_val, y_val):
        correct = 0
        for x, y in zip(X_val, y_val):
            pred = self.feedforward(x.reshape(-1, 1))
            loss = self.cross_entropy(pred, y.reshape(-1, 1))
            self.val_loss.append(loss)
            if np.argmax(pred) == np.argmax(y):
                correct += 1
        return np.mean(self.val_loss), correct / len(X_val)

    def train(self, X_train, y_train, X_val, y_val, epochs, lr):
        for epoch in range(epochs):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            for x, y in zip(X_train, y_train):
                delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(-1, 1), y.reshape(-1, 1))
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
            self.weights = [w - lr * (nw / len(X_train)) for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - lr * (nb / len(X_train)) for b, nb in zip(self.biases, nabla_b)]

            val_loss, val_acc = self.validate(X_val, y_val)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss = []
            self.val_loss = []

    def check_early_stopping(self, val_loss, epoch_nb):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.stagnated_epochs_nb = 0
        else:
            self.stagnated_epochs_nb += 1
        
        if self.stagnated_epochs_nb >= self.patience:
            print(f'Early stopping after {epoch_nb + 1} epochs')
            return False
        return True


    # Stochastic Gradient Descent
    def SGD(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size):
        mini_batches = MiniBatchGenerator(X_train, y_train, batch_size)
        for epoch in range(epochs):
            for X_batch, y_batch in mini_batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                for x, y in zip(X_batch, y_batch):
                    delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(-1, 1), y.reshape(-1, 1))                        
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                
                self.weights = [w - lr * (nw / len(X_batch)) for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - lr * (nb / len(X_batch)) for b, nb in zip(self.biases, nabla_b)]

            val_loss, val_acc = self.validate(X_val, y_val)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss = []
            self.val_loss = []

    # Nesterov Accelerated Gradient
    def NAG(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size, mu=0.9, early_stopping=False):
        mini_batches = MiniBatchGenerator(X_train, y_train, batch_size)
        vel_w = [np.zeros(w.shape) for w in self.weights]
        vel_b = [np.zeros(b.shape) for b in self.biases]
        if early_stopping:
            self.best_val_loss = np.inf
            self.stagnated_epochs_nb = 0

        for epoch in range(epochs):
            for X_batch, y_batch in mini_batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                # Lookahead weights and biases
                w_lookahead = [w - mu * vw for w, vw in zip(self.weights, vel_w)]
                b_lookahead = [b - mu * vb for b, vb in zip(self.biases, vel_b)]

                for x, y in zip(X_batch, y_batch):
                    delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(-1, 1), y.reshape(-1, 1), w_lookahead, b_lookahead)                        
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                
                # Update velocity
                vel_b = [mu * vb + lr * (nb / len(X_batch)) for vb, nb in zip(vel_b, nabla_b)]
                vel_w = [mu * vw + lr * (nw / len(X_batch)) for vw, nw in zip(vel_w, nabla_w)]

                # Update weights and biases (NAG)
                self.weights = [w - vw for w, vw in zip(self.weights, vel_w)]
                self.biases = [b - vb for b, vb in zip(self.biases, vel_b)]
                
            val_loss, val_acc = self.validate(X_val, y_val)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss = []
            self.val_loss = []

            if early_stopping:
                res = self.check_early_stopping(val_loss, epoch)
                if res is False:
                    break
            

    # Root Mean Square Propagation
    # beta: decay rate to update the moving average of the squared gradient
    def RMSProp(self, X_train, y_train, X_val, y_val, batch_size, epochs, lr, beta=0.9, epsilon=1e-8):
        pass

    # Adaptive Moment Estimation
    def Adam(self, ):
        pass
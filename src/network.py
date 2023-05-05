import numpy as np
import matplotlib.pyplot as plt
from mini_batch import MiniBatchGenerator

class Network:
    def __init__(self, sizes, **kwargs):
        if len(sizes) < 4:
            raise ValueError('At least two hidden layers are required.')
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = kwargs.get('biases', [np.random.randn(y, 1) for y in sizes[1:]])
        self.weights = kwargs.get('weights', [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])])
        self.zs = []
        self.activations = []
        self.loss = []
        self.val_loss = []
        self.loss_progress = []
        self.patience = kwargs.get('patience', 15)
        self.min_delta = kwargs.get('min_delta', 0.001)
        self.best_weights = None
        self.best_biases = None
        self.compare = kwargs.get('compare', False)

    def activation(self, z, output=False, epsilon=1e-8):
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

    def cross_entropy(self, y_hat, y, epsilon=1e-8):
        return -np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon)) / len(y)

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

    def check_early_stopping(self, val_loss, epoch_nb):
        # TODO? restore best weights and biases
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.stagnated_epochs_nb = 0
            return 0
        
        self.stagnated_epochs_nb += 1
        if self.stagnated_epochs_nb >= self.patience:
            print(f'Early stopping after {epoch_nb + 1} epochs')
            return 2
        return 1

    def train_default(self, X_train, y_train, X_val, y_val, epochs, lr):
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
            if self.compare == False:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss = []
            self.val_loss = []
            self.loss_progress.append(val_loss)
        print('Training finished')
        if self.compare == False:
            self.plot_progress()

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
            if self.compare == False:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss = []
            self.val_loss = []
            self.loss_progress.append(val_loss)
        print('Training finished')
        if self.compare == False:
            self.plot_progress()

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
            if self.compare == False:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss = []
            self.val_loss = []
            self.loss_progress.append(val_loss)

            if early_stopping:
                res = self.check_early_stopping(val_loss, epoch)
                if res == 0:
                    self.best_weights = self.weights
                    self.best_biases = self.biases
                elif res == 2:
                    # Restore best weights and biases
                    self.weights = self.best_weights
                    self.biases = self.best_biases
                    break
        print('Training finished')
        if self.compare == False:
            self.plot_progress()

    # Root Mean Square Propagation
    # beta: decay rate to update the moving average of the squared gradient
    def RMSProp(self, X_train, y_train, X_val, y_val, batch_size, epochs, lr, beta=0.9, epsilon=1e-8, early_stopping=False):
        mini_batches = MiniBatchGenerator(X_train, y_train, batch_size)
        squared_w = [np.zeros(w.shape) for w in self.weights]
        squared_b = [np.zeros(b.shape) for b in self.biases]
        if early_stopping:
            self.best_val_loss = np.inf
            self.stagnated_epochs_nb = 0

        for epoch in range(epochs):
            for X_batch, y_batch in mini_batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                for x, y in zip(X_batch, y_batch):
                    delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(-1, 1), y.reshape(-1, 1))                        
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                
                dw = [nw / len(X_batch) for nw in nabla_w]
                db = [nb / len(X_batch) for nb in nabla_b]

                squared_w = [beta * sw + (1 - beta) * np.square(dw) for sw, dw in zip(squared_w, dw)]
                squared_b = [beta * sb + (1 - beta) * np.square(db) for sb, db in zip(squared_b, db)]

                self.weights = [w - (lr * dw) / (np.sqrt(sw) + epsilon) for w, dw, sw in zip(self.weights, dw, squared_w)]
                self.biases = [b - (lr * db) / (np.sqrt(sb) + epsilon) for b, db, sb in zip(self.biases, db, squared_b)]

            val_loss, val_acc = self.validate(X_val, y_val)
            if self.compare == False:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss = []
            self.val_loss = []
            self.loss_progress.append(val_loss)

            if early_stopping:
                res = self.check_early_stopping(val_loss, epoch)
                if res == 0:
                    self.best_weights = self.weights
                    self.best_biases = self.biases
                elif res == 2:
                    # Restore best weights and biases
                    self.weights = self.best_weights
                    self.biases = self.best_biases
                    break

        print('Training finished')
        if self.compare == False:
            self.plot_progress()

    
    # Adaptive Moment Estimation
    def Adam(self, X_train, y_train, X_val, y_val, batch_size, epochs, lr, mu=0.9, beta=0.999, epsilon=1e-8, early_stopping=False):
        mini_batches = MiniBatchGenerator(X_train, y_train, batch_size)
        m_w = [np.zeros(w.shape) for w in self.weights]
        m_b = [np.zeros(b.shape) for b in self.biases]
        v_w = [np.zeros(w.shape) for w in self.weights]
        v_b = [np.zeros(b.shape) for b in self.biases]

        if early_stopping:
            self.best_val_loss = np.inf
            self.stagnated_epochs_nb = 0

        for epoch in range(epochs):
            for X_batch, y_batch in mini_batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                for x, y in zip(X_batch, y_batch):
                    delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(-1, 1), y.reshape(-1, 1))                        
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                
                # Loss
                dw = [nw / len(X_batch) for nw in nabla_w]
                db = [nb / len(X_batch) for nb in nabla_b]

                # Update moving average of the gradient
                m_w = [mu * mw + (1 - mu) * dw for mw, dw in zip(m_w, dw)]
                m_b = [mu * mb + (1 - mu) * db for mb, db in zip(m_b, db)]

                # Update moving average of the squared gradient
                v_w = [beta * vw + (1 - beta) * np.square(dw) for vw, dw in zip(v_w, dw)]
                v_b = [beta * vb + (1 - beta) * np.square(db) for vb, db in zip(v_b, db)]

                # Compute bias-corrected first moment estimate
                m_w_corr = [mw / (1 - mu ** (epoch + 1)) for mw in m_w]
                m_b_corr = [mb / (1 - mu ** (epoch + 1)) for mb in m_b]

                # Compute bias-corrected second raw moment estimate
                v_w_corr = [vw / (1 - beta ** (epoch + 1)) for vw in v_w]
                v_b_corr = [vb / (1 - beta ** (epoch + 1)) for vb in v_b]

                # Update parameters
                self.weights = [w - (lr * mw_corr) / (np.sqrt(vw_corr) + epsilon) for w, mw_corr, vw_corr in zip(self.weights, m_w_corr, v_w_corr)]
                self.biases = [b - (lr * mb_corr) / (np.sqrt(vb_corr) + epsilon) for b, mb_corr, vb_corr in zip(self.biases, m_b_corr, v_b_corr)]

            val_loss, val_acc = self.validate(X_val, y_val)
            if self.compare == False:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(self.loss):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss_progress.append(val_loss)
            self.loss = []
            self.val_loss = []

            if early_stopping:
                res = self.check_early_stopping(val_loss, epoch)
                if res == 0:
                    self.best_weights = self.weights
                    self.best_biases = self.biases
                elif res == 2:
                    # TODO better -> in order to restore the best weights, should keep the best one and compare the loss each time
                    # Restore best weights and biases
                    self.weights = self.best_weights
                    self.biases = self.best_biases
                    break
        print('Training finished')
        if self.compare == False:
            self.plot_progress()

    def Adam_KFold(self, kf_gen, batch_size, epochs, lr, mu=0.9, beta=0.999, epsilon=1e-8, early_stopping=False):
        m_w = [np.zeros(w.shape) for w in self.weights]
        m_b = [np.zeros(b.shape) for b in self.biases]
        v_w = [np.zeros(w.shape) for w in self.weights]
        v_b = [np.zeros(b.shape) for b in self.biases]

        if early_stopping:
            self.best_val_loss = np.inf
            self.stagnated_epochs_nb = 0

        for epoch in range(epochs):
            for X_train, y_train, X_val, y_val in kf_gen:
                k_fold_loss = []
                k_fold_val_loss = []
                k_fold_val_acc = []
                mini_batches = MiniBatchGenerator(X_train, y_train, batch_size)

                for X_batch, y_batch in mini_batches:
                    nabla_b = [np.zeros(b.shape) for b in self.biases]
                    nabla_w = [np.zeros(w.shape) for w in self.weights]

                    for x, y in zip(X_batch, y_batch):
                        delta_nabla_b, delta_nabla_w = self.backprop(x.reshape(-1, 1), y.reshape(-1, 1))                        
                        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    
                    # Loss
                    dw = [nw / len(X_batch) for nw in nabla_w]
                    db = [nb / len(X_batch) for nb in nabla_b]

                    # Update moving average of the gradient
                    m_w = [mu * mw + (1 - mu) * dw for mw, dw in zip(m_w, dw)]
                    m_b = [mu * mb + (1 - mu) * db for mb, db in zip(m_b, db)]

                    # Update moving average of the squared gradient
                    v_w = [beta * vw + (1 - beta) * np.square(dw) for vw, dw in zip(v_w, dw)]
                    v_b = [beta * vb + (1 - beta) * np.square(db) for vb, db in zip(v_b, db)]

                    # Compute bias-corrected first moment estimate
                    m_w_corr = [mw / (1 - mu ** (epoch + 1)) for mw in m_w]
                    m_b_corr = [mb / (1 - mu ** (epoch + 1)) for mb in m_b]

                    # Compute bias-corrected second raw moment estimate
                    v_w_corr = [vw / (1 - beta ** (epoch + 1)) for vw in v_w]
                    v_b_corr = [vb / (1 - beta ** (epoch + 1)) for vb in v_b]

                    # Update parameters
                    self.weights = [w - (lr * mw_corr) / (np.sqrt(vw_corr) + epsilon) for w, mw_corr, vw_corr in zip(self.weights, m_w_corr, v_w_corr)]
                    self.biases = [b - (lr * mb_corr) / (np.sqrt(vb_corr) + epsilon) for b, mb_corr, vb_corr in zip(self.biases, m_b_corr, v_b_corr)]

                    k_val_loss, k_val_acc = self.validate(X_val, y_val)
                    k_fold_loss.append(np.mean(self.loss))
                    k_fold_val_loss.append(k_val_loss)
                    k_fold_val_acc.append(k_val_acc)
                    
                    self.loss = []
                    self.val_loss = []

            loss = np.mean(k_fold_loss)
            val_loss = np.mean(k_fold_val_loss)
            val_acc = np.mean(k_fold_val_acc)
            if self.compare == False:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc * 100:.2f}%')
            self.loss_progress.append(val_loss)

            if early_stopping:
                res = self.check_early_stopping(val_loss, epoch)
                if res == 0:
                    self.best_weights = self.weights
                    self.best_biases = self.biases
                elif res == 2:
                    # TODO better -> in order to restore the best weights, should keep the best one and compare the loss each time
                    # Restore best weights and biases
                    self.weights = self.best_weights
                    self.biases = self.best_biases
                    break
        print('Training finished')
        if self.compare == False:
            self.plot_progress()

    def plot_progress(self):
        if self.compare == True:
            return
        plt.plot(self.loss_progress)
        plt.title('Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.show()

    def predict(self, X):
        y_pred = np.argmax(self.feedforward(X.T), axis=0) 
        print('Prediction finished')
        return y_pred 

    def save(self, path):
        for i, w in enumerate(self.weights):
            np.save(path + 'weights' + str(i), w)
        for i, b in enumerate(self.biases):
            np.save(path + 'biases' + str(i), b)
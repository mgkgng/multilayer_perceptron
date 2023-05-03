import numpy as np

class MiniBatchGenerator:
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(X.shape[0])

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self._generate()

    def _generate(self):
        for start_idx in range(0, self.X.shape[0] - self.batch_size + 1, self.batch_size):
            batch = self.indices[start_idx:start_idx + self.batch_size]
            yield self.X[batch], self.y[batch]

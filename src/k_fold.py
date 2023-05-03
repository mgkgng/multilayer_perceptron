import numpy as np
from sklearn.model_selection import KFold

class KFoldGenerator:
    def __init__(self, X, y, n_splits, shuffle=True):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle)
        
    def __iter__(self):
        for train_idx, val_idx in self.kf.split(self.X, self.y):
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]
            yield X_train, y_train, X_val, y_val

# StandardScaler
import numpy as np


class StandardScaler:

    def __init__(self):
        """object instantiation"""
        self.mean = 0
        self.sd = 0

    def fit(self, X):
        """fit scaler by learning mean and standard deviation per feature """
        self.mean = X.mean(axis=0)
        self.sd = np.std(X, axis=0)

    def transform(self, X):
        """transform X by learned mean and standard deviation, and return it """
        new = (X - self.mean) / self.sd
        return new

    def fit_transform(self, X):
        """fit scaler by learning mean and standard deviation per feature, and then transform X """
        self.fit(X)
        return self.transform(X)
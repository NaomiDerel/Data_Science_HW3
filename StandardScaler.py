import numpy as np


def our_std_axis0(X):
    l = np.shape(X)[0]
    return np.std(X, axis=0) * l / (l - 1)


def our_std_axis1(X):
    l = np.shape(X)[1]
    return np.std(X, axis=1) * l / (l - 1)


class StandardScaler:

    def __init__(self):
        """object instantiation"""
        self.mean = 0
        self.sd = 0

    def fit(self, X):
        """fit scaler by learning mean and standard deviation per feature """
        self.mean = X.mean(axis=0)
        self.sd = our_std_axis0(X)

    def transform(self, X):
        """transform X by learned mean and standard deviation, and return it """
        transformed = (X - self.mean) / self.sd
        return transformed

    def fit_transform(self, X):
        """fit scaler by learning mean and standard deviation per feature, and then transform X """
        self.fit(X)
        return self.transform(X)

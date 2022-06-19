import numpy as np


class StandardScaler:

    def __init__(self):
        """ object instantiation """
        self.mean = 0
        self.sd = 0

    def fit(self, X):
        """ fit scaler by learning mean and standard deviation per feature """
        self.mean = X.mean(axis=0)
        self.sd = X.std(axis=0, ddof=1)

    def transform(self, X):
        """ transform X by learned mean and standard deviation, and return it """
        transformed = (X - self.mean) / self.sd
        return transformed

    def fit_transform(self, X):
        """ fit scaler by learning mean and standard deviation per feature, and then transform X """
        self.fit(X)
        return self.transform(X)

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)


def load_data(path):
    """
    reads and returns the pandas DataFrame
    :param path: path to data set
    :return: data as pandas
    """
    df = pd.read_csv(path)
    return df


def adjust_labels(y):
    """
    adjust labels of season from {0,1,2,3} to {0,1}
    :param y: labels for season
    :return: new adjusted labels
    """
    new = []
    for i in y:
        if i == 0 or i == 1:
            new.append(0)
        elif i == 2 or i == 3:
            new.append(1)
    return new


class StandardScaler:
    def __init__(self):
        """
        object instantiation
        """
        self.mean = 0
        self.sd = 0

    def fit(self, X):
        """
        fit scaler by learning mean and standard deviation per feature
        :param X: data
        """
        self.mean = X.mean(axis=0)
        self.sd = X.std(axis=0, ddof=1)

    def transform(self, X):
        """
        transform X by learned mean and standard deviation, and return it
        :param X: data
        :return: transformed data by mean and sd
        """
        transformed = (X - self.mean) / self.sd
        return transformed

    def fit_transform(self, X):
        """
        fit scaler by learning mean and standard deviation per feature, and then transform X
        :param X: data
        :return: transformed data
        """
        self.fit(X)
        return self.transform(X)


import numpy as np
from scipy import stats
from abc import abstractmethod
from StandardScaler import StandardScaler


class KNN:
    def __init__(self, k):
        """ object instantiation, save k and define a scaler object """
        self.k = k
        self.scaler = StandardScaler()
        self.X_trained = 0
        self.y_trained = 0

    def fit(self, X_train, y_train):
        """ fit scaler and save X_train and y_train """
        self.X_trained = self.scaler.fit_transform(X_train)
        self.y_trained = y_train

    @abstractmethod
    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        pass

    def neighbours_indices(self, x):
        """ for a given point x, find indices of k closest points in the training set """
        new_arr = self.X_trained
        temp = []
        for point in new_arr:
            temp.append(self.dist(x, point))

        indexes = np.argsort(np.array(temp))
        return indexes[0:self.k]

    @staticmethod
    def dist(x1, x2):
        """ returns Euclidean distance between x1 and x2 """
        dist = np.square(np.subtract(x1, x2))
        dist_sum = np.sum(dist)
        distance = dist_sum ** 0.5
        return distance


class ClassificationKNN(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels.
        :param X_test:
        :return:
        """
        X_test_fitted = self.scaler.transform(X_test)
        prd_labels = []
        for point in X_test_fitted:
            nearest_points = self.neighbours_indices(point)
            nearest_values = [self.y_trained[i] for i in nearest_points]
            most_common = stats.mode(nearest_values, axis=None)[0][0]
            prd_labels.append(most_common)
        return prd_labels


class RegressionKNN(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation"""
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        X_test_fitted = self.scaler.transform(X_test)
        prd_labels = []
        for point in X_test_fitted:
            nearest_points = np.array(self.neighbours_indices(point))
            nearest_values = [self.y_trained[i] for i in nearest_points]
            prd_labels.append(np.mean(nearest_values))
        return prd_labels

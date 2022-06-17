# knn.py

import numpy as np
from scipy import stats
from abc import abstractmethod
from StandardScaler import StandardScaler


# from data import StandardScaler

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
        # self.y_trained = self.scaler.fit_transform(y_train)
        self.y_trained = y_train

    @abstractmethod
    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        pass

    def neighbours_indices(self, x):
        """ for a given point x, find indices of k closest points in the training set """
        new_arr = self.X_trained
        nearest_indices = []
        temp = []
        for point in new_arr:
            temp.append(self.dist(x, point))
        temp = np.array(temp)
        indexes = np.argsort(temp)
        return indexes[0:self.k]

    @staticmethod
    def dist(x1, x2):
        """ returns Euclidean distance between x1 and x2 """
        dist = np.square(np.subtract(x1, x2))  # subtracts the vectors x,y and squares each place in the result
        dist_sum = np.sum(dist)  # numpy sum on all the places in the vector
        distance = dist_sum ** 0.5  # square root of the sum
        return distance


class ClassificationKNN(KNN):

    def __init__(self, k):
        """ object instantiation, parent class instantiation"""
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
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
            nearest_points = self.neighbours_indices(point)
            nearest_points = np.array(nearest_points)
            nearest_values = [self.y_trained[i] for i in nearest_points]

            mean_value = np.mean(nearest_values)
            prd_labels.append(round(mean_value))
        return prd_labels

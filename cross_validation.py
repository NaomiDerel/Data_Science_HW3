# cross_validation.py

import numpy as np
from StandardScaler import our_std_axis0 , our_std_axis1


def cross_validation_score(model, X, y, folds, metric):
    """ run cross validation on X and y with specific model by given folds. Evaluate by given metric. """

    results = []
    for train_indices, validation_indices in folds.split(X):

        model.fit(X[:][train_indices], y[:][train_indices])

        # print(model.X_trained)
        # print(model.X_trained.shape)
        # print(X[validation_indices])

        # scaler = StandardScaler()
        # X_validation_trained = scaler.fit_transform(X[:][validation_indices])
        # y_validation_trained = scaler.fit_transform(y[validation_indices])

        y_pred = model.predict(X[:][validation_indices])

        results.append(metric(y[validation_indices], y_pred))

    return results


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    """ run cross validation on X and y for every model induced by values from k_list by given folds.
        Evaluate each model by given metric. """

    np_array = np.array([cross_validation_score(model(k), X, y, folds, metric) for k in k_list])
    m = np.mean(np_array, axis=1)
    s = our_std_axis1(np_array)
    w = np.std(np_array, axis=1)
    return m, s

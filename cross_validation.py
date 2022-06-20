import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """
    run cross validation on X and y with specific model by given folds. Evaluate by given metric
    :param model: KNN model - regression or classification
    :param X: data sel
    :param y: labels for data set
    :param folds: divided data into train and test sets
    :param metric: f1_score or rmse to calculate quality of predictions
    :return: results for predictions
    """
    results = []
    for train_indices, validation_indices in folds.split(X):
        model.fit(X[:][train_indices], y[:][train_indices])
        y_pred = model.predict(X[:][validation_indices])
        results.append(metric(y[validation_indices], y_pred))

    return results


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    """
    run cross validation on X and y for every model induced by values from k_list by given folds.
    Evaluate each model by given metric
    :param model: KNN model
    :param k_list: k values for KNN model
    :param X: data set
    :param y: labels for data
    :param folds: divided data into train and test sets
    :param metric: f1_score or rmse to calculate quality of predictions
    :return:
    """
    np_array = np.array([cross_validation_score(model(k), X, y, folds, metric) for k in k_list])
    return np.mean(np_array, axis=1), np.std(np_array, axis=1, ddof=1)

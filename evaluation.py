import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """
    returns f1_score of binary classification task with true labels y_true and predicted labels y_pred
    :param y_true: true labels for test set
    :param y_pred: predicted labels for train set
    :return: f1_score_value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score_value = (2 * recall * precision) / (recall + precision)

    return f1_score_value


def rmse(y_true, y_pred):
    """
    returns RMSE of regression task with true labels y_true and predicted labels y_pred
    :param y_true: true labels for test set
    :param y_pred: predicted labels for train set
    :return: rmse value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = y_true.shape[0]
    temp_np = np.square(np.subtract(y_true, y_pred))
    sum = np.sum(temp_np)
    RMSE = ((1/n) * sum) ** 0.5

    return RMSE


def visualize_results(k_list, scores, metric_name, title, path):
    """
    plot a results graph of cross validation scores
    :param k_list: k values to preform KNN by
    :param scores: mean scores for each k
    :param metric_name: f1_score or rmse
    :param title: title for graph
    :param path: path to save graph to
    """
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(metric_name)
    plt.plot(k_list, scores[0])
    plt.savefig(path, format="png")
    plt.show()


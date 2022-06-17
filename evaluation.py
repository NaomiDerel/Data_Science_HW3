# evaluation.py

import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """ returns f1_score of binary classification task with true labels y_true and predicted labels y_pred """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    print("tp: " + str(TP) + ", fn: " + str(FN) + ", fp: " + str(FP) + ", tn: " + str(TN))

    recall = TP / (TP + FN)
    print(recall)
    precision = TP / (TP + FP)
    print(precision)

    if recall == 0 and precision == 0:
        f1_score_value = 0
    else:
        f1_score_value = (2 * recall * precision) / (recall + precision)

    print(f1_score_value)
    return f1_score_value


def rmse(y_true, y_pred):
    """ returns RMSE of regression task with true labels y_true and predicted labels y_pred """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = y_true.shape[0]
    temp_np = np.square(np.subtract(y_true, y_pred))
    sum = np.sum(temp_np)

    RMSE = ((1 / n) * sum) ** 0.5

    return RMSE


def visualize_results(k_list, scores, metric_name, title, path):
    """ plot a results graph of cross validation scores """

    plt.scatter(k_list, scores)  # scatter plot of the first and second columns - the 2 features

    # giving names to the axis and the whole graph
    plt.xlabel("values of k")
    plt.ylabel(metric_name)
    plt.title(title)

    plt.show()  ## isn't needed, just for tests

    plt.savefig(path, format="png")  # saving the files in pdf format with the given path



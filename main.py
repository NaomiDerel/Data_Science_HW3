import sys

import numpy as np

from data import load_data, get_folds, adjust_labels, add_noise
from cross_validation import model_selection_cross_validation, cross_validation_score
from knn import ClassificationKNN, RegressionKNN
from evaluation import f1_score, rmse

# from sklearn.metrics import f1_score


def main():
    path = "london_sample_500.csv"
    df = load_data(path)
    folds = get_folds()
    k_list = [3, 5, 11, 25, 51, 75, 101]

    print("Part1 - Classification")
    y_true = np.array(adjust_labels(df["season"]))
    df_X = df[["t1", "t2", "wind_speed", "hum"]]
    np_data = add_noise(df_X.to_numpy())

    classification_results_np = model_selection_cross_validation(ClassificationKNN, k_list, np_data, y_true, folds, f1_score)

    i = 0
    for k in k_list:
        print("k=" + str(k) +
              ", mean score: " + "{:.4f}".format(classification_results_np[0][i]) +
              ", std of scores: " + "{:.4f}".format(classification_results_np[1][i]))
        i += 1

    print()

    print("Part2 - Regression")
    y_true = np.array(df["hum"])
    df_X = df[["t1", "t2", "wind_speed"]]
    np_data = add_noise(df_X.to_numpy())

    regression_results_np = model_selection_cross_validation(RegressionKNN, k_list, np_data, y_true, folds, rmse)

    i = 0
    for k in k_list:
        print("k=" + str(k) +
              ", mean score: " + "{:.4f}".format(regression_results_np[0][i]) +
              ", std of scores: " + "{:.4f}".format(regression_results_np[1][i]))
        i += 1


if __name__ == '__main__':
    main()

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

    print("Part 1 - Classification")

    y_true = np.array(adjust_labels(df["season"]))
    df_X = df[["t1", "t2", "wind_speed", "hum"]]
    np_data = add_noise(df_X.to_numpy())

    # print(data_with_noise)
    # print(cross_validation_score(ClassificationKNN(3), data_with_noise, y_true, folds, f1_score))

    results_np = model_selection_cross_validation(ClassificationKNN, k_list, np_data, y_true, folds, f1_score)

    i = 0
    for k in k_list:
        print("k=" + str(k) + ", mean score: " + str(results_np[0][i]) + ", std of scores: " + str(results_np[0][i]))
        i += 1


if __name__ == '__main__':
    main()

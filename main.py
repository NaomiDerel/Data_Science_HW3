import sys
import numpy as np

from data import load_data, get_folds, adjust_labels, add_noise
from cross_validation import model_selection_cross_validation
from knn import ClassificationKNN, RegressionKNN
from evaluation import f1_score, rmse, visualize_results


def main(argv):
    """
    calculates and prints Classification and Regression according to requirements.
    :param argv: path to file.
    """
    # path = "london_sample_500.csv"
    path = argv[1]
    df = load_data(path)
    folds = get_folds()
    k_list = [3, 5, 11, 25, 51, 75, 101]

    results = print_results(df, "Part1 - Classification", "season", True, ["t1", "t2", "wind_speed", "hum"],
                            ClassificationKNN, k_list, folds, f1_score)
    visualize_results(k_list, results, "F1_Score", "Classification", "plot1.png")

    print()

    results = print_results(df, "Part2 - Regression", "hum", False, ["t1", "t2", "wind_speed"],
                            RegressionKNN, k_list, folds, rmse)
    visualize_results(k_list, results, "RMSE", "Regression", "plot2.png")


def print_results(df, title, label, adjustable, features, KNNType, k_list, folds, metric):
    """
    prints mean and standard-deviation results for each k in k_list according to requirements.
    :param df: full data frame
    :param title: for each part
    :param label: parameter to predict
    :param adjustable: true if we should adjust the label, false otherwise
    :param features: features in data to predict by
    :param KNNType: classification or regression KNN model
    :param k_list: k values to preform KNN by
    :param folds: divided data into train and test data
    :param metric: f1_score or rmse to calculate quality of predictions
    :return: mean and std scores for each k
    """
    print(title)

    if adjustable:
        y_true = np.array(adjust_labels(df[label]))
    else:
        y_true = np.array(df[label])
    df_X = df[features]
    np_data = add_noise(df_X.to_numpy())

    results = model_selection_cross_validation(KNNType, k_list, np_data, y_true, folds, metric)

    i = 0
    for k in k_list:
        print("k=" + str(k) +
              ", mean score: " + "{:.4f}".format(results[0][i]) +
              ", std of scores: " + "{:.4f}".format(results[1][i]))
        i += 1

    return results


if __name__ == '__main__':
    main(sys.argv)

import os

import numpy as np
import pandas as pd


def save(X_train, X_test, y_train, y_test, file_path, file_name):
    """Save dataset to csv file

    :param X_train: numpy array of train features
    :param X_test: numpy array of test features
    :param y_train: numpy array vector of train labels matching the order of X_train by rows
    :param y_test: numpy array vector of test labels matching the order of X_test by rows
    :param file_path: directory
    :param file_name: file name with extension
    :return: Pandas dataframe merging train and test data with the respective labels in a column
    """

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    df = pd.DataFrame(X, columns=["A", "B"])
    df["label"] = y

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    df.to_csv(file_path + file_name, sep="\t")

    return df


def counts(df):
    """Print value counts for each column of a dataframe

    :param df: Pandas dataframe
    :return: void
    """
    for c in df.columns:
        print("---- %s ---" % c)
        print(df[c].value_counts())

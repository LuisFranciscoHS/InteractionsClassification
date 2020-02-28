import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.Python import config
from src.Python.datasets import reactome, intact
from src.Python.lib.dictionaries import flatten_dictionary


def set_column(X, ref_ppis, label):
    """Add column to X with values {0, 1} when the interaction appears in the reference ppis

    :param X: pandas dataframe with the first two columns the pair of interacting proteins
    :param ref_ppis: dictionary one --> set of protein interactions
    :param label: name of the new column
    :return: same pandas dataframe with the added new column
    """

    result = []
    for index, row in X.iterrows():
        if row[0] in ref_ppis.keys() and row[1] in ref_ppis[row[0]]:
            result.append(1)
        else:
            result.append(0)

    X[label] = result
    return X


def load_data(test_size=0.5, num_ppis=100):
    """PPI Dataset simmilar to the study from Ontario Institute for Cancer Research
    (github.com/LuisFranciscoHS/InteractionsClassification/wiki/Toronto-Dataset)

    Args:
        test_size: percentage of interactions to test the model
        num_ppis: total number of interactions to be returned. Includes positive and negative interactions

    Returns: 2 tuples or (matrix X, vector y). Given N the total number of ppis:
        * X_train (N*(1-test_size) examples, 7 features), y_train (N*(1-test_size) labels):
        * X_test (N*test_size examples, 7 features), y_test (N*test_size labels):

    """
    num_positive_ppis = num_negative_ppis = num_ppis / 2
    if num_ppis % 2 == 1:
        num_positive_ppis += 1

    reactome_ppis = reactome.get_ppis(num_positive_ppis)  # Dictionary object
    reactome_ppis_flat = pd.DataFrame(flatten_dictionary(reactome_ppis))  # Pandas DataFrame

    random_ppis = reactome.get_random_ppis(num_negative_ppis, reactome_ppis_flat)  # Pandas DataFrame

    ppis = pd.concat([reactome_ppis_flat, random_ppis])

    set_column(ppis, intact.get_ppis("9606"), "Human PPI")
    set_column(ppis, intact.get_ppis("4932"), "Yeast PPI")

    ppis.drop(ppis.columns[[0, 1]], axis=1, inplace=True)
    y = np.concatenate((np.ones((int(num_positive_ppis),)), np.zeros((int(num_negative_ppis),))), axis=0)
    # X = ppis.to_numpy()
    return train_test_split(ppis, y, test_size=test_size, random_state=0)


def save():
    pass


if __name__ == "__main__":
    config.set_root_wd()
    load_data()

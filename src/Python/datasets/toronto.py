import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.Python import config
from src.Python.datasets import reactome, intact
from src.Python.lib import dictionaries
from src.Python.lib.dictionaries import flatten_dictionary


def set_column(i, X, ppis, ref_ppis):
    """Set column i values to 0 or 1 with a dictionary of interactions interactions that are annotated in the dictionary

    Args:
        i: column index
        X: numpy array to set the resulting feature values
        ppis: two-column dataframe of ppis corresponding to each features row in X
        ref_ppis: dictionary one_to_set with known ppis of certain type
    Returns:
        Modified numpy array X
    """

    X[:, i] = 0
    for index, row in ppis.iterrows():
        if row[0] in ref_ppis:
            if row[1] in ref_ppis[row[0]]:
                X[index, i] = 1

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

    # Initialize dataset with the correct size
    X = np.ones((num_ppis, 7))
    y = np.ones((num_ppis,))

    num_positive_ppis = num_negative_ppis = num_ppis/2
    if num_ppis%2 == 1:
        num_positive_ppis += 1

    reactome_ppis = reactome.get_ppis(num_positive_ppis)    # Dictionary object
    reactome_ppis_flat = pd.DataFrame(flatten_dictionary(reactome_ppis)) # Pandas DataFrame

    random_ppis = reactome.get_random_ppis(num_negative_ppis, reactome_ppis_flat)   # Pandas DataFrame

    ppis = pd.concat([reactome_ppis_flat, random_ppis])

    human_ppis = intact.get_ppis()

    set_column(0, X, ppis, ppis)


    # Check if each PPI is annotated for Human in IntAct

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return ((X_train, y_train), (X_test, y_test))

def save():
    pass

if __name__ == "__main__":
    config.set_root_wd()
    load_data()
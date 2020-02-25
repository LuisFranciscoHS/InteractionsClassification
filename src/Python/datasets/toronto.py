import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.Python import config
from src.Python.datasets import reactome
from src.Python.lib.dictionaries import flatten_dictionary


def load_data(test_size=0.5):
    """PPI Dataset simmilar to the study from Ontario Institute for Cancer Research
    (github.com/LuisFranciscoHS/InteractionsClassification/wiki/Toronto-Dataset)

    Args:
        test_size: percentage of interactions to test the model

    Returns: 2 tuples or (matrix X, vector y). Given N the total number of ppis:
        * X_train (N*(1-test_size) examples, 7 features), y_train (N*(1-test_size) labels):
        * X_test (N*test_size examples, 7 features), y_test (N*test_size labels):

    """

    # Gather interactions from Reactome
    ppis = reactome.get_ppis(500)
    df = pd.DataFrame(flatten_dictionary(ppis))

    # For each of them get table saying
    X = np.ones((50000, 7))
    y = np.ones((50000,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return ((X_train, y_train), (X_test, y_test))

def save():
    pass

if __name__ == "__main__":
    config.set_root_wd()
    load_data()
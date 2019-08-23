import numpy as np
import sklearn

from dataset import reactome

def merge_n_columns():
    pass #TODO Implement merge_n_columns function

def load_dataset(config):

    # Get Reactome dataset
    dataset_reactome = reactome.load_dataset(config)

    # Get String dataset TODO

    # Get negative dataset TODO
    

    # Bind datasets together

    return sklearn.utils.Bunch(
        features=dataset_reactome['features'],
        target=dataset_reactome['target'],
        interactions=dataset_reactome['interactions'])

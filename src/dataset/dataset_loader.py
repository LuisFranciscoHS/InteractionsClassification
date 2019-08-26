import numpy as np
import sklearn

from config import read_config
from dataset import reactome


def merge_features(*features):
    """Creates one np.array of sample vectors. Each vector has k elements for k features.
    Each feature should be a simple list."""
    if len(features) == 0:
        raise ValueError('Missing feature arrays to merge')

    result = np.array([[sample] for sample in features[0]])
    for feature in features[1:]:
        result = np.concatenate((result, np.array([feature]).T), axis=1)
    print(result)
    return result


def merge_datasets(*datasets):
    """Concatenates the target vector, the features array and the interactions list"""
    if len(datasets) == 0:
        raise ValueError('Missing datasets to merge')

    for dataset in datasets[1:]:
        datasets[0]['features'] = np.concatenate((datasets[0]['features'], dataset['features']), axis=0)
        datasets[0]['targets'] = np.concatenate((datasets[0]['targets'], dataset['targets']), axis=0)
        datasets[0]['interactions'] = datasets[0]['interactions'] +  dataset['interactions']
    return datasets[0]


def load_dataset(path_config):

    config = read_config(path_config)

    # Get Reactome dataset
    dataset_reactome = reactome.load_dataset(config)

    # Get String dataset TODO

    # Get negative dataset TODO
    dataset_negative = {}

    # Bind datasets together
    merge_datasets(dataset_reactome, dataset_negative)

    return sklearn.utils.Bunch(
        features=dataset_reactome['features'],
        target=dataset_reactome['target'],
        interactions=dataset_reactome['interactions'])

import numpy as np
import sklearn

from config import read_config
from dataset import reactome, string_database
from dataset.string_database import get_or_create_features


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


def load_dataset(config):
    """Create Bunch with the features, target and interactions from STRING with Reactome interactions as the positive
    dataset and the rest of interactions as negative dataset."""
    print('Creating dataset from STRING...')
    features = string_database.get_or_create_features(config)
    interactions = features.index
    targets = string_database.create_targets(config, interactions)
    features_names = features.columns
    target_names = ('non-functional', 'functional')

    return sklearn.utils.Bunch(features=features, targets=targets, interactions=interactions,
                               feature_names=features_names, target_names=target_names)
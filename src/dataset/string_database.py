#%%
import gzip
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
import sklearn

import dictionaries
from config import read_config
from dataset.download import download_if_not_exists

#%%
def gather_files(config):
    """Downloads and extracts the String protein network for Human and the mapping from Entrez to Uniprot."""
    download_if_not_exists(config['PATH_STRING'], config['STRING_ID_MAP'] + '.gz',
                           config['URL_STRING_ID_MAP'], 'String id mapping')
    with gzip.open(config['PATH_STRING'] + config['STRING_ID_MAP'] + '.gz') as gz_file:
        with open(config['PATH_STRING'] + config['STRING_ID_MAP'], 'wb') as file:
            file.writelines(gz_file.readlines())

    download_if_not_exists(config['PATH_STRING'], config['STRING_PROTEIN_NETWORK'] + '.gz',
                           config['URL_STRING_PROTEIN_NETWORK'], 'String protein network')
    with gzip.open(config['PATH_STRING'] + config['STRING_PROTEIN_NETWORK'] + '.gz') as gz_file:
        with open(config['PATH_STRING'] + config['STRING_PROTEIN_NETWORK'], 'wb') as file:
            file.writelines(gz_file.readlines())


#%%
def create_ensembl_uniprot_mapping(config):
    """Creates a one to one dictionary"""
    print("Reading Entrez -- UniProt mapping...")
    temp_mapping = dictionaries.read_dictionary_one_to_set(config['PATH_STRING'], config['STRING_ID_MAP'],
                                                           order_pairs=False, col_indices=(2, 1), ignore_header=False)
    return {k: {p.split('|')[0] for p in v} for k, v in temp_mapping.items()}  # Extract the Uniprot accessions


#%%
def create_interactions(config):
    """Creates a list of tuples from the String interactions with UniProt accessions"""
    mapping = create_ensembl_uniprot_mapping(config)
    ensemble_ppis = dictionaries.read_dictionary_one_to_set(config['PATH_STRING'], config['STRING_PROTEIN_NETWORK'],
                                                            order_pairs=False, col_indices=(0, 1), ignore_header=True)
    ppis = dictionaries.create_ppis_dictionary(ensemble_ppis, mapping)
    return dictionaries.flatten_dictionary(ppis)


#%%
def create_features(config, mapping=None):
    """Creates pandas dataframe with a vector of features for each interaction in string.
    The features are: """

    features = pd.read_csv(config['PATH_STRING'] + config['STRING_PROTEIN_NETWORK'], sep='\t')
    # Replace identifiers
    if mapping is None:
        mapping = create_ensembl_uniprot_mapping(config)
    features['item_id_a'] = features['item_id_a'].apply(lambda x: next(iter(mapping.get(x, 'A00000'))))
    features['item_id_b'] = features['item_id_b'].apply(lambda x: next(iter(mapping.get(x, 'A00000'))))

    return features


#%%
def create_targets(config):
    # reactome_ppis = reactome.get_interactions(config['PATH_SWISSPROT'], config['FILE_SWISSPROT_PROTEINS'],
    #                                           config['URL_SWISSPROT_PROTEINS'],
    #                                           config['PATH_REACTOME'], config['REACTOME_INTERNAL_EDGES'],
    #                                           config['REACTOME_PPIS'],
    #                                           config['PATH_TOOLS'], config['FILE_PATHWAYMATCHER'],
    #                                           config['URL_PATHWAYMATCHER'])
    # targets = np.ones(len(interactions), dtype=int)
    # feature_in_biogrid = dictionaries.in_dictionary(interactions, biogrid_ppis)
    return np.array([])


def load_dataset(config):
    """Create dictionary with the features, target and interactions from Reactome as the positive dataset."""

    gather_files(config)

    features = create_features(config)
    interactions = create_interactions(features)
    targets = create_targets(config)

    return sklearn.utils.Bunch(features=features,
                               targets=targets,
                               interactions=interactions,
                               feature_names=['non-functional', 'functional'])


if __name__ == '__main__':
    print(os.getcwd())
    dataset = load_dataset(read_config('../../'))
    print("Loaded String dataset")
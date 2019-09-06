#%%
import gzip
import os
import re

import pandas as pd

import dictionaries
from config import read_config
from dataset import reactome, dataset_loader
from dataset.download import download_if_not_exists


#%%
def gather_files(config):
    """Downloads and extracts the String protein network for Human and the mapping from Entrez to Uniprot."""
    print("Downloading STRING files...")
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
def fill_dataframe(original_csv, new_dataframe):
    print("Filling feature values...")
    for sample in original_csv.itertuples():
        mode, score, pair = sample[3], sample[7], (sample[1], sample[2])
        # print(f"Mode: {mode}, Score: {score}, Pair: {pair}\n")
        new_dataframe.at[pair, 'score_' + mode + '_mode'] = score


#%%
def check_no_missing_values(frame):
    print("Checking not missing values...")
    for col in frame.columns:
        if pd.isnull(frame[col]).any():
            return False
    return True


#%%
def get_or_create_features(config, mapping=None):
    """Creates pandas dataframe with a vector of features for each interaction in string.
    The features are: """
    print("Creating features...")
    features = {}
    if not os.path.exists(config['PATH_STRING'] + config['STRING_FEATURES']):
        gather_files(config)

        print("Reading data from STRING file...")
        original_features = pd.read_csv(config['PATH_STRING'] + config['STRING_PROTEIN_NETWORK'], sep='\t')
        # Replace identifiers
        if mapping is None:
            mapping = create_ensembl_uniprot_mapping(config)
        original_features['item_id_a'] = original_features['item_id_a'].apply(lambda x: next(iter(mapping.get(x, 'A00000'))))
        original_features['item_id_b'] = original_features['item_id_b'].apply(lambda x: next(iter(mapping.get(x, 'A00000'))))

        # Create desired columns
        indexes = list({(row[1], row[2]) for row in original_features.itertuples()})
        columns = ['score_' + mode + '_mode' for mode in original_features['mode'].unique()]
        features = pd.DataFrame(columns=columns, index=indexes)
        for column in features.columns:
            features[column] = 0.0

        fill_dataframe(original_features, features)

        if check_no_missing_values(features):
            print("Features READY")

        features.to_csv(config['PATH_STRING'] + config['STRING_FEATURES'], header=True)
    else:
        features = pd.read_csv(config['PATH_STRING'] + config['STRING_FEATURES'], index_col=0)
        features.index = [tuple(re.sub("['() ]", "", i).split(',')) for i in features.index]
        print("Features READY")
    return features


#%%
def create_targets(config, features):
    """Set if each interaction pair is functional interaction or not.
    In other words, check if the pair interacts in Reactome.
    1 = yes = True, 0 = no = False"""
    print("Creating targets...")
    reactome_ppis = reactome.get_interactions(config['PATH_SWISSPROT'], config['FILE_SWISSPROT_PROTEINS'],
                                              config['URL_SWISSPROT_PROTEINS'],
                                              config['PATH_REACTOME'], config['REACTOME_INTERNAL_EDGES'],
                                              config['REACTOME_PPIS'],
                                              config['PATH_TOOLS'], config['FILE_PATHWAYMATCHER'],
                                              config['URL_PATHWAYMATCHER'])
    result = pd.Series(dictionaries.in_dictionary(features.index, reactome_ppis))
    result.index = features.index
    print("Added index to targets.")
    return result


#%%
if __name__ == '__main__':
    print(os.getcwd())
    dataset = dataset_loader.load_dataset(read_config('../../'))
    print("Loaded STRING dataset")
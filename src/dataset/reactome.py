# %% Load modules and configuration
import os
import subprocess

import numpy as np
import requests
import sklearn

from config_loader import read_config

import dictionaries
from dataset import biogrid
from dataset.download import download_if_not_exists


def create_pathwaymatcher_files(path_swissprot, file_swissprot_proteins, url_swissprot,
                                path_reactome, file_reactome_internal_edges,
                                path_pathwaymatcher, file_pathwaymatcher, url_pathwaymatcher):
    """Create protein interaction network of all Reactome by mapping all proteins in SwissProt with PathwayMatcher."""
    if not os.path.exists(path_reactome + file_reactome_internal_edges):

        # Download SwissProt protein list
        download_if_not_exists(path_swissprot, file_swissprot_proteins, url_swissprot, 'SwissProt protein list')

        # Download PathwayMatcher executable
        download_if_not_exists(path_pathwaymatcher, file_pathwaymatcher, url_pathwaymatcher, 'PathwayMatcher')

        command = f"java -jar {path_pathwaymatcher}{file_pathwaymatcher} match-uniprot -i {path_swissprot}{file_swissprot_proteins} -o {path_reactome} -gu";
        os.system(command)

        # Remove extra PathwayMatcher result files
        extra_files = "analysis.tsv", "proteinExternalEdges.tsv", "proteinVertices.tsv", "search.tsv"
        [os.remove(f"{path_reactome}{file}") for file in extra_files if os.path.exists(f"{path_reactome}{file}")]

        print(f"PathwayMatcher files READY")


def get_interactions(path_swissprot, file_swissprot_proteins, url_swissprot,
                     path_reactome, file_reactome_internal_edges, file_reactome_ppis,
                     path_pathwaymatcher, file_pathwaymatcher, url_pathwaymatcher):
    """Get dictionary of enumerated interactions: pair of accessions --> index"""
    ppis = {}
    if not os.path.exists(path_reactome + file_reactome_ppis):
        create_pathwaymatcher_files(path_swissprot, file_swissprot_proteins, url_swissprot,
                                    path_reactome, file_reactome_internal_edges,
                                    path_pathwaymatcher, file_pathwaymatcher, url_pathwaymatcher)

        print("Reading Reactome interactions...")
        ppis = dictionaries.read_dictionary_one_to_set(path_reactome, file_reactome_internal_edges,
                                                       order_pairs=True, col_indices=(0, 1), ignore_header=True)
        dictionaries.write_dictionary_one_to_set(ppis, path_reactome, file_reactome_ppis)
    else:
        print("Reading Reactome unique interactions...")
        ppis = dictionaries.read_dictionary_one_to_set(path_reactome, file_reactome_ppis)

    print("Reactome interactions READY")
    return ppis


def load_dataset(config):
    """Create dictionary with the features, target and interactions from Reactome as the positive dataset."""

    reactome_ppis = get_interactions(config['PATH_SWISSPROT'], config['FILE_SWISSPROT_PROTEINS'], config['URL_SWISSPROT_PROTEINS'],
                                     config['PATH_REACTOME'], config['REACTOME_INTERNAL_EDGES'], config['REACTOME_PPIS'],
                                     config['PATH_TOOLS'], config['FILE_PATHWAYMATCHER'], config['URL_PATHWAYMATCHER'])

    biogrid_ppis = biogrid.get_interactions(config['PATH_BIOGRID'], config['URL_BIOGRID_ALL'], config['BIOGRID_ALL'],
                                            config['BIOGRID_GGIS'], config['BIOGRID_PPIS'], config['BIOGRID_ENTREZ_TO_UNIPROT'],
                                            batch_size=config['ID_MAPPING_BATCH_SIZE'])

    # Check which Reactome interactions appear in Biogrid for human
    interactions = dictionaries.flatten_dictionary(reactome_ppis)
    targets = np.ones(len(interactions), dtype=int)
    feature_in_biogrid = dictionaries.in_dictionary(interactions, biogrid_ppis)
    print("Reactome interactions reported in Biogrid: ", feature_in_biogrid.count(1))

    from dataset.dataset_loader import merge_features
    features = merge_features(feature_in_biogrid)
    feature_names = ['in_biogrid']
    print("Shape of data: ", features.shape)
    print("Shape of target: ", targets.shape)

    return sklearn.utils.Bunch(features=features, targets=targets, interactions=interactions, feature_names=feature_names)


if __name__ == '__main__':
    dataset = load_dataset(read_config('../../'))
    print("Loaded Reactome dataset")

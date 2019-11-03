import gzip
import os
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd
import sklearn

from src.Python.lib import conversions, dictionaries
from src.Python.lib.dictionaries import read_dictionary_one_to_set, create_ppis_dictionary, write_dictionary_one_to_set, \
    read_set_from_columns
from src.Python.lib.download import download_if_not_exists


def read_swissprot_proteins():
    pass


def get_create_targets():
    """Create pandas dataframe with a row for each pair of eligible proteins. Each sample row has these colums:
        - If they are interacting in Reactome: Positive
        - Else if: If not in String or "low" score in String: Negative
        - Else: we don't use them.

        INPUT:
    - File with elegible proteins (Swissprot)
    - File with proteins interacting in Reactome
    - File with proteins interacting in String along with its interactors

        OUTPUT:
    - File with three columns:
            * PROTEIN1 (String, single word)
            * PROTEIN2 (String, single word)
            * IS_FUNCTIONAL (Integer, only 0: non-functional, or 1: functional)

    This method is to be executed once, to build the dataset labels.
        """
    proteins_swissprot = read_swissprot_proteins()
    # interactions_reactome = read_reactome_interactions()
    # interactions_string = read_string_interactions()
    #
    return pd.DataFrame(np.arange(15).reshape(5, 3))


def get_or_create_features():
    """
    Create a table of protein features, reorganize UniProt features into acceptable format for the learner.

    INPUT:
    - File with proteins and attributes directly downloaded from UniProt.
    The file should be downloaded as uncompressed, tab-separated format.
    Columns: Entry, Length, Mass, Subcellular location [CC]

    OUTPUT:
    - File with one protein with its features, one protein per line. Attributes with the format fixed for the learner.
        * Entry (String), protein accession: stays the same
        * Length (integer), number of amino acids: remove commas from string
        * Mass (integer): number of daltons: remove commas
        * SL-<Location Name> (boolean): there is one column for each possible location of the mature protein
    """
    pass


def create_dataset_features_table():
    """
    Create a matrix with the features for each interaction. Gets the

    INPUT:
    - Protein features
    - List of interactions
    """
    pass


def load():
    # If dataset features and label files are not there, call create methods
    targets = get_create_targets()
    features = get_or_create_features()
    features_names = features.columns
    return sklearn.utils.Bunch(features=features, targets=targets, feature_names=features_names)


# def get_train_and_test_X_y(path, discard_percentage=0.7):
#     """Get the feature matrix 'X_train', 'X_test' and the label vector 'y_train', 'y_test' for the
#     interactions dataset."""
#     print("Reading config from: ", os.getcwd())
#     dataset = load_dataset(read_config(path))
#     scaler = StandardScaler()
#     scaler.fit(dataset['features'])
#     dataset['features'] = scaler.transform(dataset['features'])
#
#     X_sample, X_discard, y_sample, y_discard = train_test_split(dataset['features'],
#                                                                 dataset['targets'],
#                                                                 random_state=33, test_size=discard_percentage)
#     return train_test_split(X_sample, y_sample, random_state=33, test_size=0.2)

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
        datasets[0]['interactions'] = datasets[0]['interactions'] + dataset['interactions']
    return datasets[0]


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

    reactome_ppis = get_interactions(config['PATH_SWISSPROT'], config['FILE_SWISSPROT_PROTEINS'],
                                     config['URL_SWISSPROT_PROTEINS'],
                                     config['PATH_REACTOME'], config['REACTOME_INTERNAL_EDGES'],
                                     config['REACTOME_PPIS'],
                                     config['PATH_TOOLS'], config['FILE_PATHWAYMATCHER'], config['URL_PATHWAYMATCHER'])

    biogrid_ppis = get_interactions(config['PATH_BIOGRID'], config['URL_BIOGRID_ALL'], config['BIOGRID_ALL'],
                                            config['BIOGRID_GGIS'], config['BIOGRID_PPIS'],
                                            config['BIOGRID_ENTREZ_TO_UNIPROT'],
                                            batch_size=config['ID_MAPPING_BATCH_SIZE'])

    # Check which Reactome interactions appear in Biogrid for human
    interactions = dictionaries.flatten_dictionary(reactome_ppis)
    targets = np.ones(len(interactions), dtype=int)
    feature_in_biogrid = dictionaries.in_dictionary(interactions, biogrid_ppis)
    print("Reactome interactions reported in Biogrid: ", feature_in_biogrid.count(1))

    features = merge_features(feature_in_biogrid)
    feature_names = ['in_biogrid']
    print("Shape of data: ", features.shape)
    print("Shape of target: ", targets.shape)

    return sklearn.utils.Bunch(features=features, targets=targets, interactions=interactions,
                               feature_names=feature_names)


def create_ggi_file(url, path_biogrid, filename_all, filename_ggis):
    """Download Biogrid human gene interactions"""
    if not os.path.exists(path_biogrid + filename_ggis):
        # Download the zip with all the species, then extract and delete the others
        download_if_not_exists(path_biogrid, filename_all, url, 'Biogrid gene interactions')

        # Decompress the file
        with ZipFile(path_biogrid + filename_all, 'r') as zip:
            print("Extracting human gene interactions file...")
            zip.extract(filename_ggis, path_biogrid)
        os.remove(f"{path_biogrid}{filename_all}")

    print("File Biogrid ggi READY")


def create_gene_to_protein_mapping(path_biogrid, filename_ggis, url, filename_all,
                                   filename_entrez_to_uniprot, batch_size):
    # Check if required file exists
    if not os.path.exists(path_biogrid + filename_ggis):
        create_ggi_file(url, path_biogrid, filename_all, filename_ggis)

    if not os.path.exists(path_biogrid + filename_entrez_to_uniprot):

        unique_genes = read_set_from_columns(path_biogrid, filename_ggis, col_indices=(1, 2), ignore_header=True)
        # Convert gene to protein ids by batches
        with open(path_biogrid + filename_entrez_to_uniprot, "w") as file_gene_to_protein:
            start, end = 0, batch_size
            total = len(unique_genes)
            while True:
                end = min(end, total)
                print(f"  Converting genes {start} to {end}")
                mapping = conversions.map_ids(list(unique_genes)[start:end])
                for key, values in mapping.items():
                    for value in values:
                        file_gene_to_protein.write(f"{key}\t{value}\n")

                if end == total:
                    break
                start += batch_size
                end += batch_size
    print("Gene --> Protein mapping READY")


def get_interactions(path_biogrid, url, filename_all, filename_ggis, filename_ppis,
                     filename_entrez_to_uniprot, batch_size=1000):
    """Returns dictionary of Biogrid protein interactions enumerated"""
    ppis = {}
    if not os.path.exists(path_biogrid + filename_ppis):
        print("Creating biogrid protein interaction file...")

        create_ggi_file(url, path_biogrid, filename_all, filename_ggis)
        ggis = read_dictionary_one_to_set(path_biogrid, filename_ggis, order_pairs=True, col_indices=(1, 2))

        create_gene_to_protein_mapping(path_biogrid, filename_ggis, url, filename_all,
                                       filename_entrez_to_uniprot, batch_size)
        entrez_to_uniprot = read_dictionary_one_to_set(path_biogrid, filename_entrez_to_uniprot)

        # Create dictionary with converted unique protein pairs
        ppis = create_ppis_dictionary(ggis, entrez_to_uniprot)
        write_dictionary_one_to_set(ppis, path_biogrid, filename_ppis)
    else:
        ppis = read_dictionary_one_to_set(path_biogrid, filename_ppis, order_pairs=True)
    print("Biogrid protein interactions READY")
    return ppis


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


# %%
def create_ensembl_uniprot_mapping(config):
    """Creates a one to one dictionary"""
    print("Reading Entrez -- UniProt mapping...")
    temp_mapping = dictionaries.read_dictionary_one_to_set(config['PATH_STRING'], config['STRING_ID_MAP'],
                                                           order_pairs=False, col_indices=(2, 1), ignore_header=False)
    return {k: {p.split('|')[0] for p in v} for k, v in temp_mapping.items()}  # Extract the Uniprot accessions


# %%
def fill_dataframe(original_csv, new_dataframe):
    print("Filling feature values...")
    for sample in original_csv.itertuples():
        mode, score, pair = sample[3], sample[7], (sample[1], sample[2])
        # print(f"Mode: {mode}, Score: {score}, Pair: {pair}\n")
        new_dataframe.at[pair, 'score_' + mode + '_mode'] = score


# %%
def check_no_missing_values(frame):
    print("Checking not missing values...")
    for col in frame.columns:
        if pd.isnull(frame[col]).any():
            return False
    return True


# %%
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
        original_features['item_id_a'] = original_features['item_id_a'].apply(
            lambda x: next(iter(mapping.get(x, 'A00000'))))
        original_features['item_id_b'] = original_features['item_id_b'].apply(
            lambda x: next(iter(mapping.get(x, 'A00000'))))

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


# %%
def create_targets(config, features):
    """Set if each interaction pair is functional interaction or not.
    In other words, check if the pair interacts in Reactome.
    1 = yes = True, 0 = no = False"""
    print("Creating targets...")
    reactome_ppis = get_interactions(config['PATH_SWISSPROT'], config['FILE_SWISSPROT_PROTEINS'],
                                              config['URL_SWISSPROT_PROTEINS'],
                                              config['PATH_REACTOME'], config['REACTOME_INTERNAL_EDGES'],
                                              config['REACTOME_PPIS'],
                                              config['PATH_TOOLS'], config['FILE_PATHWAYMATCHER'],
                                              config['URL_PATHWAYMATCHER'])
    result = pd.Series(dictionaries.in_dictionary(features.index, reactome_ppis))
    result.index = features.index
    print("Added index to targets.")
    return result

# Functions to download and read the PPIs from Reactome

import os
import random

import pandas as pd

from src.Python import config
from src.Python.generic import dictionaries
from src.Python.generic.download import download_if_not_exists


def read_swissprot_proteins(path_swissprot, file_swissprot_proteins, url_swissprot):
    if not os.path.exists(path_swissprot + file_swissprot_proteins):
        download_if_not_exists(path_swissprot, file_swissprot_proteins, url_swissprot, 'SwissProt protein list')


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

    print(path_reactome + file_reactome_internal_edges)

    print(f"PathwayMatcher files READY")


def get_ppis(examples=10,
             path_swissprot=config.PATH_SWISSPROT,
             file_swissprot_proteins=config.FILE_SWISSPROT_PROTEINS,
             url_swissprot=config.URL_SWISSPROT_PROTEINS,
             path_reactome=config.PATH_REACTOME,
             file_reactome_internal_edges=config.REACTOME_INTERACTIONS,
             file_reactome_ppis=config.REACTOME_PPIS,
             path_pathwaymatcher=config.PATH_TOOLS,
             file_pathwaymatcher=config.FILE_PATHWAYMATCHER,
             url_pathwaymatcher=config.URL_PATHWAYMATCHER):
    """Returns dictionary of lexicographical interactions: accessions --> accessions set"""
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

    ppi_subset = {}
    example = 0

    if examples > 8000:
        for key, values in ppis.items():
            for value in values:
                ppi_subset.setdefault(key.strip(), set()).add(value.strip())
                example += 1
                if example >= examples:
                    break
            if example >= examples:
                break
    else:
        random.seed(77)
        keys = random.sample(list(ppis.keys()), int(examples))
        for key in keys:
            ppi_subset.setdefault(key.strip(), set()).add(random.sample(ppis[key], 1)[0].strip())

    print("Reactome interactions READY")
    return ppi_subset


def get_random_ppis(n, ppis):
    """ Create n protein pairs using the same proteins from the parameter interactions

    :param n: number of protein pairs
    :param ppis: Two-column Pandas DataFrame, one protein-protein interaction per row.
    These contain the available proteins to make the new pairs.
    :return: two column numpy array with n rows
    """

    # Create set of proteins
    proteins = set()
    for I in range(len(ppis)):
        proteins.add(ppis.iloc[I, 0])
        proteins.add(ppis.iloc[I, 1])

    random_ppis = set()

    while len(random_ppis) < n:
        pair = random.sample(proteins, 2)
        if pair[0] > pair[1]:
            pair[0], pair[1] = pair[1], pair[0]
        if (pair[0], pair[1]) not in random_ppis and ((ppis[0] != pair[0]) & (ppis[1] != pair[1])).any():
            random_ppis.add((pair[0], pair[1]))

    return pd.DataFrame(random_ppis)
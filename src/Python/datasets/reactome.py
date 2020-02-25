# Functions to download and read the PPIs from Reactome

import os

from src.Python import config
from src.Python.lib import dictionaries
from src.Python.lib.download import download_if_not_exists


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
    for key, values in ppis.items():
        for value in values:
            if example < len(ppis.keys()) and example < examples:
                ppi_subset.setdefault(key.strip(), set()).add(value.strip())
                example += 1
            else:
                break

    print("Reactome interactions READY")
    return ppi_subset

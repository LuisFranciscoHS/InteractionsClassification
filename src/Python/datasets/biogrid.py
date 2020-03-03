import os

from Python import config
from Python.generic.conversions import create_mapping
from Python.generic.dictionaries import read_dictionary_one_to_set, create_ppis_dictionary, \
    write_dictionary_one_to_set, read_set_from_columns, convert_dict_to_set
from Python.generic.download import download_file_from_zip


def get_ppis(path_biogrid=config.PATH_BIOGRID,
             url=config.URL_BIOGRID_ALL,
             filename_all=config.BIOGRID_ALL,
             filename_ggis=config.BIOGRID_HUMAN_GGIS,
             filename_ppis=config.BIOGRID_HUMAN_PPIS,
             filename_entrez_to_uniprot=config.ENTREZ_TO_UNIPROT,
             batch_size=1000):
    """
    Get of Biogrid protein protein interactions
    It must convert the gene names to protein accessions.

    :param path_biogrid:
    :param url: To download the file
    :param filename_all: original zipped interactions file
    :param filename_ggis: original gene interactions file from Biogrid
    :param filename_ppis: resulting ppi file after converting genes to proteins
    :param filename_entrez_to_uniprot:
    :param batch_size: for queries to the id mapping online service
    :return: dictionary (one --> set)
    """
    ppis = {}
    if not os.path.exists(path_biogrid + filename_ppis):
        print("Creating biogrid protein interaction file...")

        download_file_from_zip(url, path_biogrid, filename_all, filename_ggis)
        ggis = read_dictionary_one_to_set(path_biogrid, filename_ggis, order_pairs=True, col_indices=(1, 2))
        unique_genes = convert_dict_to_set(ggis)

        create_mapping(path_biogrid, unique_genes, filename_entrez_to_uniprot)
        entrez_to_uniprot = read_dictionary_one_to_set(path_biogrid, filename_entrez_to_uniprot)

        ppis = create_ppis_dictionary(ggis, entrez_to_uniprot)
        write_dictionary_one_to_set(ppis, path_biogrid, filename_ppis)
    else:
        ppis = read_dictionary_one_to_set(path_biogrid, filename_ppis, order_pairs=True)
    print(filename_ppis, " READY")

    return ppis

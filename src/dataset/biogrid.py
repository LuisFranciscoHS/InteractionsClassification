import os

import conversions
from config_loader import read_config
from dataset.download import download_if_not_exists
from dictionaries import read_dictionary_one_to_set, write_dictionary_one_to_set, read_set_from_columns, \
    create_ppis_dictionary
from zipfile import ZipFile


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


if __name__ == '__main__':
    config = read_config("../../config.txt")
    ppis = get_interactions(config['PATH_BIOGRID'], config['URL_BIOGRID_ALL'], config['BIOGRID_ALL'],
                            config['BIOGRID_GGIS'], config['BIOGRID_PPIS'], config['BIOGRID_ENTREZ_TO_UNIPROT'])
    print("Loaded Reactome dataset")
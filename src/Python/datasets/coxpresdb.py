import errno
import os

from Python import config
from Python.generic.conversions import create_mapping
from Python.generic.dictionaries import convert_dict_to_set, read_dictionary_one_to_set, invert


def get_ppis(reactome_ppis, threshold=5000.0):
    """ Get human co-expressed pairs of proteins. They are not necessarily ppi, but to keep same naming structure.

    :param reactome_ppis: dictionary one --> set
    :param threshold: Maximum correlation score to be considered co-expressed
    :return: dictionary one accession --> accession set of gene ids
    """

    if not os.path.exists(config.PATH_COXPRESDB + config.COXPRESDB_HUMAN):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config.PATH_COXPRESDB + config.COXPRESDB_HUMAN)

    protein_set_to_convert = convert_dict_to_set(reactome_ppis)
    create_mapping(config.PATH_COXPRESDB, protein_set_to_convert, config.UNIPROT_TO_ENTREZ,
                   from_database_name="ACC", to_database_name="P_ENTREZGENEID")
    uniprot_to_entrez = read_dictionary_one_to_set(config.PATH_COXPRESDB, config.UNIPROT_TO_ENTREZ)
    entrez_to_uniprot = invert(uniprot_to_entrez)

    ppis_dict = {}
    for protein in reactome_ppis.keys():
        if protein in uniprot_to_entrez:
            for gene in uniprot_to_entrez[protein]:
                if not os.path.exists(config.PATH_COXPRESDB + config.COXPRESDB_HUMAN + os.path.sep + gene):
                    # print(f"Not found file {config.COXPRESDB_HUMAN + os.path.sep + gene}")
                    continue
                with open(config.PATH_COXPRESDB + config.COXPRESDB_HUMAN + os.path.sep + gene) as file:
                    file.readline()
                    for line in file:
                        fields = line.split('\t')
                        if 2 > len(fields):
                            raise ValueError(f"File does not have the expected 2 columns.")
                        gene, mr = fields[0], fields[1]
                        if float(mr) <= threshold:
                            if gene in entrez_to_uniprot:
                                for acc in entrez_to_uniprot[gene.strip()]:
                                    ppis_dict.setdefault(protein, set()).add(acc)
                        else:
                            break

    print("Coexpressed interactions READY")
    return ppis_dict

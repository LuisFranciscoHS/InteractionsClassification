import os
import zipfile

from Python.generic.download import download_if_not_exists
from src.Python import config


def get_ppis(taxid, n=2000000):
    """
    Reads intact ppis from the file

    :return: dictionary one accession --> accession set
    """

    if not os.path.exists(config.PATH_INTACT + config.FILE_INTACT_PPIS):
        download_if_not_exists(config.PATH_INTACT, config.FILE_INTACT_PPIS, config.URL_INTACT,
                           'Intact interactions')

        with zipfile.ZipFile(config.PATH_INTACT + config.FILE_INTACT_PPIS, 'r') as zip_ref:
            zip_ref.extractall(dir)

    ppis_dict = {}
    with open(config.PATH_INTACT + config.FILE_INTACT_PPIS, encoding='utf-8', errors='ignore') as f:
        line = f.readline()
        cont = 0
        for line in f:
            line = f.readline()
            fields = line.split("\t")
            if "uniprotkb:" in fields[0] and "uniprotkb:" in fields[1] \
                    and taxid in fields[9] and taxid in fields[10]:
                p1 = fields[0].replace("uniprotkb:", "")
                p2 = fields[1].replace("uniprotkb:", "")

                if p1 > p2:
                    p1, p2 = p2, p1

                if p1 not in ppis_dict:
                    ppis_dict.setdefault(p1, set()).add(p2)
                    cont += 1
                elif p2 not in p1:
                    ppis_dict[p1].add(p2)
                    cont += 1
                if cont > n:
                    break

    print("Intact interactions for ", taxid, " READY")
    return ppis_dict

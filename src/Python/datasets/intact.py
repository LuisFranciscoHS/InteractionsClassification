from src.Python import config


def get_ppis(n = 2000000):
    """
    Reads intact ppis from the file

    :return: dictionary one accession --> accession set
    """
    ppis = {}
    with open(config.FILE_INTACT_PPIS,encoding='utf-8', errors='ignore') as f:
        line = f.readline()
        cont = 0
        for line in f:
            line = f.readline()
            fields = line.split("\t")
            if "uniprotkb:" in fields[0] and "uniprotkb:" in fields[1]:
                p1 = fields[0].replace("uniprotkb:", "")
                p2 = fields[1].replace("uniprotkb:", "")
                if p1 < p2:
                    ppis.setdefault(p1, set()).add(p2)
                else:
                    ppis.setdefault(p2, set()).add(p1)
                cont += 1
                if cont > n: break

    return ppis

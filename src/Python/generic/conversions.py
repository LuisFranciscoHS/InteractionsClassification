# %%
import os

from src.Python.generic.dictionaries import convert_tab_to_dict


def map_ids(ids, from_database_name='P_ENTREZGENEID', to_database_name='ACC'):
    """Map id list using UniProt identifier mapping service (https://www.uniprot.org/help/api_idmapping)\n
    Returns dictionary with mapping."""
    import urllib.parse
    import urllib.request

    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': from_database_name,
        'to': to_database_name,
        'format': 'tab',
        'query': ' '.join(ids),
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()
        # print(str(response.decode('utf-8')))
        return convert_tab_to_dict(str(response.decode('utf-8')))


def create_mapping(output_dir, gene_set, file_entrez_to_uniprot,
                   from_database_name='P_ENTREZGENEID',
                   to_database_name='ACC',
                   batch_size=1000):
    """Create file with mapping from one set of identifiers to another.
    Default value is from Entrez Genes Id to UniProt Protein accessions.

    :param output_dir:
    :param gene_set:
    :param file_entrez_to_uniprot: Result file
    :param from_database_name:
    :param to_database_name:
    :param batch_size: Number of genes to convert at a time with the UniProt mapping service
    :return: void
    """

    if not os.path.exists(output_dir + file_entrez_to_uniprot):

        # Convert gene to protein ids by batches
        with open(output_dir + file_entrez_to_uniprot, "w") as file_gene_to_protein:
            start, end = 0, batch_size
            total = len(gene_set)
            while True:
                end = min(end, total)
                print(f"  Converting genes {start} to {end}")
                mapping = map_ids(list(gene_set)[start:end],
                                  from_database_name=from_database_name,
                                  to_database_name=to_database_name)
                for key, values in mapping.items():
                    for value in values:
                        file_gene_to_protein.write(f"{key}\t{value}\n")

                if end == total:
                    break
                start += batch_size
                end += batch_size
    print(f"{from_database_name} --> {to_database_name} mapping READY")

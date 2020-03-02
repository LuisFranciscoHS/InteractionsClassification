# %%
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

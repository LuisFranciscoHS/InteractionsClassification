#%%
def convert_tab_to_dict(response):
    """Convert tab response to dictionary.\n
    Response is a string made of rows with two tab separated columns."""
    from collections import defaultdict

    if not type(response) == type("hola"):
        print("The argument is not a string.")
        return {}
    result = defaultdict(list)
    for entry in response.splitlines()[1:]:
        from_id, to_id = tuple(entry.split())
        result[from_id].append(to_id)
    return result

#%%
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
        #print(str(response.decode('utf-8')))
        return convert_tab_to_dict(str(response.decode('utf-8')))
#%%
# Read configuration variables

path_config = "config.txt"
variables = {}
file_config = open(path_config, "r")
for line in file_config:
    (key, value) = line.split()
    variables[key] = value
print(variables)

#%%
# Download the protein list from SwissProt
import os
path_swissprot_proteins = "data/UniProt/swissprot_human_proteins.tab"

if not os.path.exists(path_swissprot_proteins):
    print("Downloading protein list from SwissProt...")
    os.makedirs(os.path.dirname(path_swissprot_proteins), exist_ok=True)
    import requests
    request_result = requests.get("https://www.uniprot.org/uniprot/?query=reviewed:yes+AND+organism:9606&columns=id&format=tab")

    with open(path_swissprot_proteins, "w") as file_swissprot_proteins:
        file_swissprot_proteins.write(request_result.text)

print("SwissProt list READY")

#%%

# Run PathwayMatcher to get interactions in Reactome
path_pathwaymatcher = "tools/PathwayMatcher.jar"

if not os.path.exists(path_pathwaymatcher):
        print("Downloading PathwayMatcher...")
        url_pathwaymatcher = "https://github.com/PathwayAnalysisPlatform/PathwayMatcher/releases/download/1.9.1/PathwayMatcher.jar"

        r = requests.get(url_pathwaymatcher)
        os.makedirs(os.path.dirname(path_pathwaymatcher), exist_ok=True)
        open(path_pathwaymatcher, 'wb').write(r.content)

print("PathwayMatcher READY")

import subprocess
subprocess.run("java -jar tools/PathwayMatcher.jar match-uniprot -i data/UniProt/swissprot_human_proteins.tab -o data/Reactome/ -gu",
    capture_output=False)

print("Reactome interactions READY")

#%%

# Collect features of each Reactome interaction

# Biogrid: if there is a reported humap PPI

## Download Biogrid human gene interactions

import os
path_biogrid_gene_interactions = "data/Biogrid/human_gene_interactions.tab"

if not os.path.exists(path_biogrid_gene_interactions):
    print("Downloading Biogrid human gene interactions...")
    os.makedirs(os.path.dirname(path_biogrid_gene_interactions), exist_ok=True)
    import requests
    request_result = requests.get("https://www.uniprot.org/uniprot/?query=reviewed:yes+AND+organism:9606&columns=id&format=tab")

    with open(path_swissprot_proteins, "w") as file_swissprot_proteins:
        file_swissprot_proteins.write(request_result.text)

print("Biogrid gene interactions READY")

#%%

## Convert Biogrid interactions to protein interactions
import urllib.parse
import urllib.request

url = 'https://www.uniprot.org/uploadlists/'

params = {
    'from': 'P_ENTREZGENEID',
    'to': 'ACC',
    'format': 'tab',
    'query': '2318\n84665\n88\n90\n339\n6416',
    'reviewed': 'yes'
}

data = urllib.parse.urlencode(params)
data = data.encode('utf-8')
req = urllib.request.Request(url, data)
with urllib.request.urlopen(req) as f:
   response = f.read()
print(response.decode('utf-8'))
print("Biogrid protein interactions READY")

#%%

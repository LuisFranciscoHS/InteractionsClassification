#%%
import os
import requests
import subprocess
import config
import sys

#sys.path.append("src/")

# Read configuration variables
config = config.read_config()

#%%
# Download the protein list from SwissProt
if not os.path.exists(config['PATH_SWISSPROT_PROTEINS']):
    print("Downloading protein list from SwissProt...")
    os.makedirs(os.path.dirname(config['PATH_SWISSPROT_PROTEINS']), exist_ok=True)
    request_result = requests.get(config['URL_SWISSPROT_PROTEINS'])

    with open(config['PATH_SWISSPROT_PROTEINS'], "w") as file_swissprot_proteins:
        file_swissprot_proteins.write(request_result.text)

print("SwissProt list READY")

#%%
# Collect features of each Reactome interaction
# Run PathwayMatcher to get interactions in Reactome
if not os.path.exists(config['PATH_REACTOME'] + config['REACTOME_PPI']):

    if not os.path.exists(config['PATH_PATHWAYMATCHER']):
        print("Downloading PathwayMatcher...")
        request_result = requests.get(config['URL_PATHWAYMATCHER'])
        os.makedirs(os.path.dirname(config['PATH_PATHWAYMATCHER']), exist_ok=True)
        open(config['PATH_PATHWAYMATCHER'], 'wb').write(request_result.content)

    print("PathwayMatcher READY")

    subprocess.run(f"java -jar {config['PATH_PATHWAYMATCHER']} match-uniprot -i {config['PATH_SWISSPROT_PROTEINS']} -o {config['PATH_REACTOME']} -gu",
            capture_output=False)

# Remove extra PathwayMatcher result files
extra_files = "analysis.tsv", "proteinExternalEdges.tsv", "proteinVertices.tsv", "search.tsv"
for extra_file in extra_files:
    if os.path.exists(f"{config['PATH_REACTOME']}{extra_file}"):
        os.remove(f"{config['PATH_REACTOME']}{extra_file}")

print("Reactome interactions READY")

#%%
import biogrid
import conversions

# Biogrid: if there is a reported humap PPI
if not os.path.exists(config['PATH_BIOGRID'] + config['BIOGRID_PPI']):
    print("Creating biogrid protein interaction file...")
    if not os.path.exists(config['PATH_BIOGRID'] + config['BIOGRID_ENTREEZ_TO_UNIPROT']):
        biogrid.create_gene_to_protein_mapping()

    # Traverse gene interactions and convert them to protein interactions
    #     

print("Biogrid protein interactions READY")

#%%

## Check which Reactome interactions appear in Biogrid for human

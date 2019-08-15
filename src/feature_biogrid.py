#%% Load modules and configuration
import os
import requests
import subprocess
import sys
from collections import defaultdict
import numpy as np

if not "src/" in sys.path:
    sys.path.append("src/")

import config
import biogrid
import conversions
import dictionaries

config = config.read_config()

#%% Set correct working directory
# import os
# try:
# 	os.chdir(os.path.join(os.getcwd(), '..\\..\Project\InteractionsClassification'))
# 	print(os.getcwd())
# except:
# 	pass

#%% Download the protein list from SwissProt
if not os.path.exists(config['PATH_SWISSPROT_PROTEINS']):
    print("Downloading protein list from SwissProt...")
    os.makedirs(os.path.dirname(config['PATH_SWISSPROT_PROTEINS']), exist_ok=True)
    request_result = requests.get(config['URL_SWISSPROT_PROTEINS'])

    with open(config['PATH_SWISSPROT_PROTEINS'], "w") as file_swissprot_proteins:
        file_swissprot_proteins.write(request_result.text)

print("SwissProt list READY")

#%% Collect features of each Reactome interaction
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

# Read unique interaction pairs (proteins in lexicographyc order)
print("Reading unique interactions...")
reactome_ppis = {}
for interaction in open(config['PATH_REACTOME'] + config['REACTOME_PPI']):
    from_id, to_id, *_ = interaction.split()
    if from_id > to_id:
        from_id, to_id = to_id, from_id
    reactome_ppis.setdefault(from_id, set()).add(to_id)

# Convert each set to tuple
for key in reactome_ppis.keys():
    reactome_ppis[key] = tuple(reactome_ppis[key])

print("Reactome interactions READY")

#%% Biogrid: if there is a reported humap PPI
if not os.path.exists(config['PATH_BIOGRID'] + config['BIOGRID_PPI']):
    print("Creating biogrid protein interaction file...")
    if not os.path.exists(config['PATH_BIOGRID'] + config['BIOGRID_ENTREZ_TO_UNIPROT']):
        biogrid.create_gene_to_protein_mapping(
            config['PATH_BIOGRID'],
            config['BIOGRID_GI'], 
            config['URL_BIOGRID_ALL'],
            config['BIOGRID_ALL'],
            config['BIOGRID_ENTREZ_TO_UNIPROT'],
            config['ID_MAPPING_BATCH_SIZE']
        )
    biogrid_gi = biogrid.read_gene_interactions(config['PATH_BIOGRID'], config['BIOGRID_GI'])
    entrez_to_uniprot = biogrid.read_entrez_to_uniprot_mapping(config['PATH_BIOGRID'], config['BIOGRID_ENTREZ_TO_UNIPROT'])
    
    # Create dictionary with converted unique protein pairs
    biogrid_ppi = biogrid.create_ppi_dictionary(biogrid_gi, entrez_to_uniprot)

    file_biogrid_ppi = open(config['PATH_BIOGRID'] + config['BIOGRID_PPI'], 'w')
    for protein, interactors in biogrid_ppi.items():
        for interactor in interactors:
            file_biogrid_ppi.write(f"{protein}\t{interactor}\n")
    file_biogrid_ppi.close()
    
print("Biogrid protein interactions READY")

#%% Check which Reactome interactions appear in Biogrid for human

biogrid_ppi = dictionaries.read_dictionary(config['PATH_BIOGRID'], config['BIOGRID_PPI'])

interactions = []
feature_biogrid_ppi = []
for protein, interactors  in reactome_ppis.items():
    for interactor in interactors:
        interactions.append((protein, interactor))
        feature_biogrid_ppi.append(1 if protein in biogrid_ppi and interactor in biogrid_ppi[protein] else 0)

targets = np.ones(len(interactions))
features = np.array(feature_biogrid_ppi)

print("Reactome interactions reported in Biogrid: ", feature_biogrid_ppi.count(1))
print("Shape of data: ", features.shape)
print("Shape of target: ", targets.shape)


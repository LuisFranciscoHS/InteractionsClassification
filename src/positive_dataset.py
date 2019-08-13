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



#%%

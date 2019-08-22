import os
import requests

#%%
def download_if_not_exists(file_path, file_name, file_url, label):
    if not os.path.exists(file_path + file_name):
        print(f"Downloading {label.upper()}...")
        if len(file_path) != 0:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        request_result = requests.get(file_url)
        open(file_path + file_name, 'wb').write(request_result.content)
        print(f"{label.upper()} READY")

#%% Download swissprot
download_if_not_exists('', "protein_list.txt", "https://www.uniprot.org/uniprot/?query=*&fil=organism%3A%22Homo+sapiens+%28Human%29+%5B9606%5D%22+AND+reviewed%3Ayes")
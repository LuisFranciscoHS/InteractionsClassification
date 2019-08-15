import os
import requests

def download_if_not_exists(file_path, file_name, file_url, label):
    if not os.path.exists(file_path + file_name):
        print(f"Downloading {label.upper()}...")
        request_result = requests.get(file_url)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        open(file_path + file_name, 'w').write(request_result.content)
import os
import requests
import shutil
import urllib.request as request
from contextlib import closing


def download_if_not_exists(file_path, file_name, file_url, label):
    if not os.path.exists(file_path + file_name):
        print(f"Downloading {label.upper()}...")
        if len(file_path) != 0:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if file_url.startswith("ftp"):
            with closing(request.urlopen(file_url)) as r:
                with open(file_path + file_name, 'wb') as f:
                    shutil.copyfileobj(r, f)
        else:
            request_result = requests.get(file_url)
            open(file_path + file_name, 'wb').write(request_result.content)

        print(f"{label.upper()} READY")

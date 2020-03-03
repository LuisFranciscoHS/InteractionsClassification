import os
from zipfile import ZipFile

import requests
import shutil
import urllib.request as request
from contextlib import closing


def download_if_not_exists(output_dir, url, label):
    """Download file from url

    :param output_dir: output directory
    :param url: full url with path and file name
    :param label: Alias of the file for the messages
    :return: void
    """
    file_name = os.path.basename(url)
    if not os.path.exists(output_dir + file_name):
        print(f"Downloading {label.upper()}...")
        if len(output_dir) != 0:
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)

        if url.startswith("ftp"):
            with closing(request.urlopen(url)) as r:
                with open(output_dir + file_name, 'wb') as f:
                    shutil.copyfileobj(r, f)
        else:
            request_result = requests.get(url)
            open(output_dir + file_name, 'wb').write(request_result.content)

        print(f"{label.upper()} READY")


def download_file_from_zip(url, output_dir, file):
    """Download and extract file from zip package

    :param url: With path and file name
    :param output_dir:
    :param file: File to be extracted from inside the zip package
    :return: void
    """
    if not os.path.exists(output_dir + file):
        # Download the zip with all the species, then extract and delete the others
        zip_file = os.path.basename(url)
        download_if_not_exists(output_dir, url, file)

        # Decompress the file
        with ZipFile(output_dir + zip_file, 'r') as zip:
            print("Extracting ", zip_file, "...")
            zip.extract(file, output_dir)
        os.remove(f"{output_dir}{zip_file}")
    print("File ", file, " READY")

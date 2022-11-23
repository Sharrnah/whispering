import zipfile
import os
from best_download import download_file


def download_extract(urls, extract_dir, checksum):
    local_dl_file = os.path.join(extract_dir, os.path.basename(urls[0]))

    success = download_file(urls, local_file=local_dl_file, expected_checksum=checksum, max_retries=3)
    if success:
        with zipfile.ZipFile(local_dl_file, "r") as f:
            f.extractall(extract_dir)
        # remove the zip file after extraction
        os.remove(local_dl_file)

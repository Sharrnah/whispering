import json
import threading
import time
import zipfile
import os
from best_download import download_file
from contextlib import closing
import urllib.request
from urllib.parse import urlparse
import hashlib

import settings
import websocket


# import logging
# logging.basicConfig(filename="download.log", level=logging.INFO)


def download_extract(urls, extract_dir, checksum):
    success = False
    local_dl_file = os.path.join(extract_dir, os.path.basename(urls[0]))

    use_ui_downloader = settings.GetOption("ui_download")
    if use_ui_downloader and websocket.UI_CONNECTED["value"] and websocket.UI_CONNECTED["websocket"] is not None:
        # send websocket message to UI
        websocket.AnswerMessage(websocket.UI_CONNECTED["websocket"], json.dumps({"type": "download", "data": {"urls": urls, "extract_dir": local_dl_file, "checksum": checksum}}))
        while True:
            if os.path.isfile(local_dl_file + ".finished"):
                if sha256_checksum(local_dl_file + ".finished") == checksum:
                    success = True
                break
            else:
                # if the finished file doesn't exist, wait for a second before checking again
                time.sleep(1)

        # remove the zip file after extraction
        if success:
            os.remove(local_dl_file + ".finished")

    else:
        success = download_file(urls, local_file=local_dl_file, expected_checksum=checksum, max_retries=3)
        if success:
            with zipfile.ZipFile(local_dl_file, "r") as f:
                f.extractall(extract_dir)
            # remove the zip file after extraction
            os.remove(local_dl_file)

    return success


def sha256_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file_simple(url, target_path, expected_sha256=None):
    progress_lock = threading.Lock()
    file_name = os.path.basename(urlparse(url).path)

    def show_progress(count, total_size):
        with progress_lock:
            percentage = int(count * 100 / total_size)
            print(f'\rDownloading {file_name}: {percentage}%', end='')

    if os.path.isdir(target_path):
        target_path = os.path.join(target_path, file_name)

    with closing(urllib.request.urlopen(url)) as remote_file:
        headers = remote_file.info()
        total_size = int(headers.get('Content-Length', -1))

        with open(target_path, 'wb') as local_file:
            block_size = 8192
            downloaded_size = 0
            for block in iter(lambda: remote_file.read(block_size), b''):
                local_file.write(block)
                downloaded_size += len(block)
                show_progress(downloaded_size, total_size)
            print()

    if expected_sha256:
        actual_sha256 = sha256_checksum(target_path)
        if actual_sha256.lower() != expected_sha256.lower():
            os.remove(target_path)
            raise ValueError(f"Downloaded file has incorrect SHA256 hash. Expected {expected_sha256}, but got {actual_sha256}.")
        else:
            print("SHA256 hash verified.")


def download_thread(url, extract_dir, checksum):
    dl_thread = threading.Thread(target=download_file_simple, args=(url, extract_dir, checksum))
    dl_thread.start()
    dl_thread.join()

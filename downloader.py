import json
import threading
import time
import zipfile
import tarfile
import os
from best_download import download_file
import requests
import hashlib

import shutil

import settings
import websocket


# import logging
# logging.basicConfig(filename="download.log", level=logging.INFO)


def extract_tar_gz(file_path, output_dir):
    with tarfile.open(file_path, "r:gz") as tar_file:
        tar_file.extractall(path=output_dir)
    # remove the zip file after extraction
    os.remove(file_path)


def extract_zip(file_path, output_dir):
    with zipfile.ZipFile(file_path, "r") as zip_file:
        zip_file.extractall(path=output_dir)
    # remove the zip file after extraction
    os.remove(file_path)


def move_files(source_dir, target_dir):
    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        # Check if it's a file
        if os.path.isfile(source_path):
            shutil.move(source_path, target_path)


def download_extract(urls, extract_dir, checksum, title="", extract_format="", alt_fallback=False,
                     fallback_extract_func=None, fallback_extract_func_args=None, force_non_ui_dl=False):
    success = False
    local_dl_file = os.path.join(extract_dir, os.path.basename(urls[0]))

    use_ui_downloader = settings.GetOption("ui_download")
    if not force_non_ui_dl and use_ui_downloader and websocket.UI_CONNECTED["value"] and websocket.UI_CONNECTED["websocket"] is not None:
        # send websocket message to UI
        websocket.AnswerMessage(websocket.UI_CONNECTED["websocket"], json.dumps({"type": "download",
                                                                                 "data": {"urls": urls,
                                                                                          "extract_dir": local_dl_file,
                                                                                          "checksum": checksum,
                                                                                          "title": title,
                                                                                          "extract_format": extract_format}}))
        while True:
            if os.path.isfile(local_dl_file + ".finished"):
                if sha256_checksum(local_dl_file + ".finished") == checksum:
                    success = True
                break
            else:
                # if the finished file doesn't exist, wait for a second before checking again
                time.sleep(1)

        # remove the zip file after extraction, or just rename if not a compressed file
        if success:
            if extract_format != "none":
                os.remove(local_dl_file + ".finished")
            else:
                os.rename(local_dl_file + ".finished", local_dl_file)

    else:
        if not alt_fallback and fallback_extract_func is None:
            success = download_file(urls, local_file=local_dl_file, expected_checksum=checksum, max_retries=3)
            if success and extract_format != "none":
                with zipfile.ZipFile(local_dl_file, "r") as f:
                    f.extractall(extract_dir)
                # remove the zip file after extraction
                os.remove(local_dl_file)
        else:
            try:
                download_file_normal(urls[0], extract_dir, checksum)
            except Exception as e:
                download_file_simple(urls[0], extract_dir, checksum)
            if fallback_extract_func is not None:
                fallback_extract_func(*fallback_extract_func_args)

    return success


def sha256_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file_normal(url, target_path, expected_sha256=None, num_retries=3):
    download_file(url, local_directory=target_path, expected_checksum=expected_sha256, max_retries=num_retries)


def download_file_simple(url, target_path, expected_sha256=None, num_retries=3, timeout=60):
    file_name = os.path.basename(url)
    if os.path.isdir(target_path):
        target_path = os.path.join(target_path, file_name)
    headers = {'User-Agent': 'Mozilla/5.0'}
    while num_retries > 0:
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()  # Raise an exception if the GET request returned an unsuccessful status code
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            downloaded_size_in_bytes = 0
            with open(target_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=None):
                    downloaded_size_in_bytes += len(chunk)
                    percentage = (downloaded_size_in_bytes / total_size_in_bytes) * 100
                    print(f'\rDownloading {file_name}: {percentage:.2f}%', end='')
                    file.write(chunk)
            print()  # Ensure the output goes to the next line after the download completes
            if total_size_in_bytes != 0 and downloaded_size_in_bytes != total_size_in_bytes:
                raise Exception("ERROR, something went wrong while downloading file")
            if expected_sha256:
                actual_sha256 = hashlib.sha256(open(target_path, 'rb').read()).hexdigest()
                if actual_sha256 != expected_sha256.lower():
                    os.remove(target_path)
                    raise ValueError(
                        f"Downloaded file has incorrect SHA256 hash. Expected {expected_sha256}, but got {actual_sha256}.")
            print("File downloaded successfully.")
            return
        except (requests.HTTPError, requests.ConnectionError) as e:
            num_retries -= 1
            print(f"Download failed due to network error: {e}")
            if num_retries > 0:
                print(f"Retrying... (Attempts left: {num_retries})")
            else:
                print("Aborting download.")
                raise e
        except Exception as e:
            print(f"Download failed due to unexpected error: {e}")
            raise e


def download_thread(url, extract_dir, checksum, num_retries=3, timeout=60):
    dl_thread = threading.Thread(target=download_file_simple, args=(url, extract_dir, checksum, num_retries, timeout))
    dl_thread.start()
    dl_thread.join()

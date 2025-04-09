import json
import threading
import time
import zipfile
import tarfile
import os
from pathlib import Path

#from best_download import download_file
from robust_downloader import download
import requests
import hashlib

import shutil

import settings
import websocket

running_downloads = []  # Global list tracking ongoing downloads

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
    file_name = os.path.basename(urls[0])
    local_dl_file = os.path.join(extract_dir, file_name)

    # Check if the download is already running
    if local_dl_file in running_downloads:
        print(f"Download for {file_name} is already in progress. Skipping duplicate start.")
        return False

    # Add to running downloads list
    running_downloads.append(local_dl_file)
    try:
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
                if os.path.isfile(local_dl_file + ".finished") and os.path.isfile(local_dl_file):
                    if sha256_checksum(local_dl_file) == checksum or checksum == "":
                        success = True
                    break
                else:
                    # if the finished file doesn't exist, wait for a second before checking again
                    time.sleep(1)

            # remove the zip file after extraction, or just rename if not a compressed file
            if success:
                time.sleep(1)
                os.remove(local_dl_file + ".finished")
                if extract_format != "none":
                    os.remove(local_dl_file)

        else:
            if not alt_fallback and fallback_extract_func is None:
                try:
                    import random
                    selected_url = random.choice(urls)
                    download(selected_url, filename=file_name, folder=extract_dir, sha256=checksum, retry_max=5)
                    if extract_format != "none":
                        with zipfile.ZipFile(str(local_dl_file), "r") as f:
                            f.extractall(extract_dir)
                        # remove the zip file after extraction
                        os.remove(local_dl_file)
                    success = True
                except Exception as e:
                    print(e)
                    success = False
            else:
                import random
                selected_url = random.choice(urls)
                try:
                    download_file_normal(selected_url, extract_dir, checksum)
                except Exception as first_exception:
                    if len(urls) > 1:
                        download_successful = False
                        last_exception = None
                        # Remove the selected URL from the list
                        remaining_urls = [url for url in urls if url != selected_url]
                        for url in remaining_urls:
                            try:
                                download_file_simple(url, extract_dir, checksum)
                                download_successful = True
                                success = True
                                break  # Exit the loop if download is successful
                            except Exception as e:
                                last_exception = e
                                continue  # Try the next URL if this one fails
                        if not download_successful:
                            success = False
                            print("All download attempts failed.")
                            if last_exception:
                                print(f"Last encountered exception: {last_exception}")
                    else:
                        success = False
                        print(first_exception)
                if fallback_extract_func is not None:
                    fallback_extract_func(*fallback_extract_func_args)
        return success
    finally:
        if local_dl_file in running_downloads:
            running_downloads.remove(local_dl_file)

def sha256_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file_normal(url, target_path, expected_sha256=None, num_retries=3):
    file_name = os.path.basename(url)
    download(url, filename=file_name, folder=target_path, sha256=expected_sha256, retry_max=num_retries)


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


# =====================================================
# Functions to check filehashes from a list of hashes.
# =====================================================
def save_hashes(model_path, file_checksums):
    hash_checked_path = model_path / "hash_checked"
    with open(hash_checked_path, 'w') as f:
        json.dump(file_checksums, f)


def load_hashes(model_path):
    hash_checked_path = model_path / "hash_checked"
    if not hash_checked_path.is_file():
        return None
    with open(hash_checked_path, 'r') as f:
        return json.load(f)


def check_file_hashes(path, hash_list) -> bool:
    """
    Go over the list of hashes in hash_list and check if the file exists and if the hash matches.
    hash_list example:
    {
        "generation_config.json": "1149807b43a0dd788e052bfcb47c012b0b182946b66c63b3ecdf9aad2d9b5f66",
        "config.json": "b5b4368433a25df0943929beaf6833db03b767b150990ee078fe62c5a7b31434",
        # ...
    }
    Returns True if all hashes match, False otherwise.
    """
    for file_name, expected_hash in hash_list.items():
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            return False
        actual_hash = sha256_checksum(file_path)
        if actual_hash.lower() != expected_hash.lower():
            return False
    return True


def download_model(download_settings, state=None):
    """
    Download the model from the given URL and extract it to the specified directory.
    Args:
        download_settings: {
            "model_path": [string] Path to the directory where the model will be downloaded (model cache path under '.cache' for example Path(Path.cwd() / ".cache" / "phi4") ).
            "model_link_dict": Dictionary containing model links and checksums. has to be in the format:
                MODEL_LINKS = {
                    "GOT_OCR_2.0": {
                        "urls": [
                            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/GOT_OCR_2.0/GOT-OCR-2.0.zip",
                            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/GOT_OCR_2.0/GOT-OCR-2.0.zip",
                            "https://s3.libs.space:9000/ai-models/GOT_OCR_2.0/GOT-OCR-2.0.zip",
                        ],
                        "checksum": "d98db661dd7d76943807b316685d9561b4cf85403fee1ba749fb691e038a587b",
                        "file_checksums": {
                            "config.json": "cbe8aacd6cd84a2d58eafcd0045c6ac40e02e3a448f24b8cee51cc81d8bdccf2",
                            "generation_config.json": "31915c5a692f43c5765a20cfc5f9403bcd250f5721a0d931bb703169c08993b4",
                            "model.safetensors": "6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4",
                            "preprocessor_config.json": "ef9a0dc0935cac11f4230ca30d00a52bedfa52b6633e409e9fbd2ea56373aa7e",
                            "special_tokens_map.json": "7c2368a3889fdfb37c24cabeb031b53f47934f357b54e56e8e389909a338ea47",
                            "tokenizer.json": "36b382a3c48c9a143c30139dac6c8230ddfb0b46a3dc43082af6052abe99d9de",
                            "tokenizer_config.json": "8b0542937d32a67da8ea2d1288b870e325be383a962c65d201864299560a2b8e"
                        },
                        "path": "", # Path to the subdirectory where the model will be downloaded.
                    },
                }
            "model_name": Name of the model to download.
            "title": Title for the download process.
            "alt_fallback": Boolean indicating whether to use an alternative fallback method.
            "force_non_ui_dl": Boolean indicating whether to force non-UI download.
            "extract_format": Format of the file to be extracted (e.g., "zip", "tar.gz" or "none").
        }
        state: dictionary {"is_downloading": False} in class to check if the model is already downloading.
    """

    model_path = download_settings["model_path"]
    model_link_dict = download_settings["model_link_dict"]
    model_name = download_settings["model_name"]
    title = download_settings["title"]
    alt_fallback = download_settings["alt_fallback"]
    force_non_ui_dl = download_settings["force_non_ui_dl"]
    extract_format = download_settings["extract_format"].lower()

    fallback_extract_func = None
    match extract_format:
        case "zip":
            fallback_extract_func = extract_zip
        case "tar.gz":
            fallback_extract_func = extract_tar_gz
        case "none", "":
            fallback_extract_func = None

    if state is None:
        state = {"is_downloading": False}

    model_directory = Path(model_path)
    if "path" in model_link_dict[model_name] and model_link_dict[model_name]["path"] != "":
        model_directory = Path(model_path / model_link_dict[model_name]["path"])
    os.makedirs(str(model_directory.resolve()), exist_ok=True)

    hash_checked_file = model_directory / "hash_checked"

    # if one of the files does not exist, break the loop and download the files
    needs_download = False
    for file, expected_checksum in model_link_dict[model_name]["file_checksums"].items():
        if not Path(model_directory / file).exists():
            needs_download = True
            break

    if not needs_download:
        if not hash_checked_file.is_file():
            needs_download = not check_file_hashes(
                str(model_directory.resolve()),
                model_link_dict[model_name]["file_checksums"]
            )
            if not needs_download:
                save_hashes(model_directory, model_link_dict[model_name]["file_checksums"])
        else:
            expected_hashes = model_link_dict[model_name]["file_checksums"]
            loaded_hashes = load_hashes(model_directory)
            if not loaded_hashes:
                if check_file_hashes(model_directory, expected_hashes):
                    needs_download = False
                else:
                    needs_download = True

    if needs_download and not state["is_downloading"]:
        print(f"download started... {title}")
        state["is_downloading"] = True
        filename = os.path.basename(model_link_dict[model_name]["urls"][0])
        download_success = download_extract(model_link_dict[model_name]["urls"],
                                                       str(model_directory.resolve()),
                                                       model_link_dict[model_name]["checksum"],
                                                       alt_fallback=alt_fallback,
                                                       force_non_ui_dl=force_non_ui_dl,
                                                       fallback_extract_func=fallback_extract_func,
                                                       fallback_extract_func_args=(
                                                           str(Path(model_directory / filename)),
                                                           str(Path(model_directory).resolve()),
                                                       ),
                                                       title=f"{title}", extract_format=extract_format)
        if not download_success:
            print(f"Download failed: {title}")
        else:
            save_hashes(model_directory, model_link_dict[model_name]["file_checksums"])
    state["is_downloading"] = False

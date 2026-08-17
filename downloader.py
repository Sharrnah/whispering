import json
import hashlib
import os
import random
import shutil
import threading
import time
import tarfile
import zipfile
from pathlib import Path

from robust_downloader import download
import requests

import settings
import websocket

running_downloads = set()  # Global set tracking ongoing downloads.
running_downloads_lock = threading.Lock()

HASH_MARKER_FILENAME = "hash_checked"
DOWNLOAD_FINISHED_SUFFIX = ".finished"
DOWNLOAD_FAILED_SUFFIX = ".failed"
HASH_CHUNK_SIZE = 1024 * 1024
# A UI process that exits before writing a receipt must not leave Python blocked forever.
UI_DOWNLOAD_WAIT_TIMEOUT_SECONDS = 24 * 60 * 60

# import logging
# logging.basicConfig(filename="download.log", level=logging.INFO)


def _safe_archive_target(output_dir, member_name):
    output_root = Path(output_dir).resolve()
    member_path = (output_root / member_name).resolve()
    try:
        member_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes the extraction directory: {member_name}") from exc


def _ui_extracts_archive(file_name, extract_format):
    normalized_format = extract_format.lower()
    if normalized_format == "none":
        return False
    if normalized_format:
        return True
    normalized_name = file_name.lower()
    return normalized_name.endswith(".zip") or normalized_name.endswith(".tar.gz")


def extract_tar_gz(file_path, output_dir):
    with tarfile.open(file_path, "r:gz") as tar_file:
        for member in tar_file.getmembers():
            _safe_archive_target(output_dir, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsupported TAR member type: {member.name}")
        tar_file.extractall(path=output_dir)
    os.remove(file_path)


def extract_zip(file_path, output_dir):
    with zipfile.ZipFile(file_path, "r") as zip_file:
        for member in zip_file.infolist():
            _safe_archive_target(output_dir, member.filename)
        zip_file.extractall(path=output_dir)
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
    if not urls:
        print("Download failed: no URLs were provided.")
        return False

    os.makedirs(extract_dir, exist_ok=True)
    file_name = os.path.basename(urls[0])
    local_dl_file = os.path.join(extract_dir, file_name)
    finished_file = local_dl_file + DOWNLOAD_FINISHED_SUFFIX
    failed_file = local_dl_file + DOWNLOAD_FAILED_SUFFIX

    with running_downloads_lock:
        if local_dl_file in running_downloads:
            print(f"Download for {file_name} is already in progress. Skipping duplicate start.")
            return False
        running_downloads.add(local_dl_file)
    try:
        use_ui_downloader = settings.GetOption("ui_download")
        if not force_non_ui_dl and use_ui_downloader and websocket.UI_CONNECTED["value"] and websocket.UI_CONNECTED["websocket"] is not None:
            # A receipt belongs to one request only. Remove receipts left by a prior run
            # before asking the UI to start a new download.
            for receipt_file in (finished_file, failed_file):
                try:
                    os.remove(receipt_file)
                except FileNotFoundError:
                    pass

            websocket.AnswerMessage(websocket.UI_CONNECTED["websocket"], json.dumps({"type": "download",
                                                                                     "data": {"urls": urls,
                                                                                              "extract_dir": local_dl_file,
                                                                                              "checksum": checksum,
                                                                                              "title": title,
                                                                                              "extract_format": extract_format}}))
            wait_started = time.monotonic()
            while True:
                if os.path.isfile(failed_file):
                    try:
                        error_message = Path(failed_file).read_text(encoding="utf-8").strip()
                    except OSError:
                        error_message = ""
                    try:
                        os.remove(failed_file)
                    except FileNotFoundError:
                        pass
                    print(f"UI download failed for {file_name}: {error_message or 'unknown error'}")
                    return False

                if os.path.isfile(finished_file):
                    if not os.path.isfile(local_dl_file):
                        try:
                            os.remove(finished_file)
                        except FileNotFoundError:
                            pass
                        print(f"UI download produced a success receipt without {file_name}.")
                        return False
                    # The Go downloader writes .finished only after it has checked the
                    # archive SHA-256 and extracted it successfully. Trust that receipt
                    # instead of hashing a multi-GB archive a second time in Python.
                    try:
                        os.remove(finished_file)
                    except FileNotFoundError:
                        pass
                    if _ui_extracts_archive(file_name, extract_format):
                        try:
                            os.remove(local_dl_file)
                        except FileNotFoundError:
                            pass
                    return True

                if not websocket.UI_CONNECTED["value"]:
                    print(f"UI disconnected while downloading {file_name}.")
                    return False
                if time.monotonic() - wait_started >= UI_DOWNLOAD_WAIT_TIMEOUT_SECONDS:
                    print(f"UI download timed out waiting for a receipt: {file_name}")
                    return False
                time.sleep(1)

        candidate_urls = list(urls)
        random.shuffle(candidate_urls)
        last_exception = None
        for attempt_index, selected_url in enumerate(candidate_urls):
            try:
                # Keep the robust/resumable downloader for the first mirror. The
                # simple downloader is a compatibility fallback for servers that do
                # not implement byte ranges the way robust-downloader expects.
                if attempt_index == 0:
                    archive_verified = download_file_normal(
                        selected_url,
                        extract_dir,
                        checksum or None,
                        num_retries=5,
                        filename=file_name,
                    )
                else:
                    archive_verified = download_file_simple(
                        selected_url,
                        extract_dir,
                        checksum or None,
                        filename=file_name,
                    )

                if not os.path.isfile(local_dl_file):
                    raise FileNotFoundError(f"Downloader did not create {local_dl_file}")
                if checksum and not archive_verified:
                    actual_checksum = sha256_checksum(local_dl_file)
                    if actual_checksum.lower() != checksum.lower():
                        try:
                            os.remove(local_dl_file)
                        except FileNotFoundError:
                            pass
                        if attempt_index == 0:
                            # The pinned robust-downloader revision can return early
                            # for a corrupt pre-existing file whose size happens to
                            # match the server. Delete it and make one clean attempt.
                            archive_verified = download_file_normal(
                                selected_url,
                                extract_dir,
                                checksum,
                                num_retries=5,
                                filename=file_name,
                            )
                            if not archive_verified or not os.path.isfile(local_dl_file):
                                raise ValueError(
                                    f"Downloaded file has incorrect SHA256 hash. "
                                    f"Expected {checksum}, but got {actual_checksum}."
                                )
                        else:
                            raise ValueError(
                                f"Downloaded file has incorrect SHA256 hash. "
                                f"Expected {checksum}, but got {actual_checksum}."
                            )

                if fallback_extract_func is not None:
                    extract_args = fallback_extract_func_args
                    if extract_args is None:
                        extract_args = (local_dl_file, extract_dir)
                    fallback_extract_func(*extract_args)
                elif extract_format != "none" and not (alt_fallback and extract_format == ""):
                    if extract_format.lower() == "tar.gz":
                        extract_tar_gz(local_dl_file, extract_dir)
                    else:
                        # Historically an empty format meant ZIP on the standard
                        # path; the alt_fallback path above still downloads as-is.
                        extract_zip(local_dl_file, extract_dir)
                return True
            except Exception as exc:
                last_exception = exc
                print(f"Download attempt failed for {selected_url}: {exc}")

        print("All download attempts failed.")
        if last_exception is not None:
            print(f"Last encountered exception: {last_exception}")
        return False
    finally:
        with running_downloads_lock:
            running_downloads.discard(local_dl_file)


def sha256_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_SIZE), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file_normal(url, target_path, expected_sha256=None, num_retries=3, filename=None):
    file_name = filename or os.path.basename(url)
    target_file = Path(target_path) / file_name
    stat_before_download = None
    if target_file.is_file():
        current_stat = target_file.stat()
        stat_before_download = (current_stat.st_size, current_stat.st_mtime_ns)
    download(url, filename=file_name, folder=target_path, sha256=expected_sha256, retry_max=num_retries)
    if not expected_sha256:
        return False
    if stat_before_download is None:
        return True

    current_stat = target_file.stat()
    stat_after_download = (current_stat.st_size, current_stat.st_mtime_ns)
    # The pinned robust-downloader verifies a resumed file after writing to it,
    # but can return early for an unchanged same-size corrupt file. Ask the caller
    # for an independent hash only when no write occurred.
    return stat_after_download != stat_before_download


def download_file_simple(url, target_path, expected_sha256=None, num_retries=3, timeout=60, filename=None):
    file_name = filename or os.path.basename(url)
    if os.path.isdir(target_path):
        target_path = os.path.join(target_path, file_name)
    headers = {'User-Agent': 'Mozilla/5.0'}
    while num_retries > 0:
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                downloaded_size_in_bytes = 0
                sha256_hash = hashlib.sha256()
                with open(target_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=HASH_CHUNK_SIZE):
                        if not chunk:
                            continue
                        downloaded_size_in_bytes += len(chunk)
                        sha256_hash.update(chunk)
                        if total_size_in_bytes:
                            percentage = (downloaded_size_in_bytes / total_size_in_bytes) * 100
                            print(f'\rDownloading {file_name}: {percentage:.2f}%', end='')
                        else:
                            print(f'\rDownloading {file_name}: {downloaded_size_in_bytes} bytes', end='')
                        file.write(chunk)
            print()  # Ensure the output goes to the next line after the download completes
            if total_size_in_bytes != 0 and downloaded_size_in_bytes != total_size_in_bytes:
                raise Exception("ERROR, something went wrong while downloading file")
            if expected_sha256:
                actual_sha256 = sha256_hash.hexdigest()
                if actual_sha256 != expected_sha256.lower():
                    os.remove(target_path)
                    raise ValueError(
                        f"Downloaded file has incorrect SHA256 hash. Expected {expected_sha256}, but got {actual_sha256}.")
            print("File downloaded successfully.")
            return bool(expected_sha256)
        except requests.RequestException as e:
            num_retries -= 1
            print(f"Download failed due to network error: {e}")
            if num_retries > 0:
                print(f"Retrying... (Attempts left: {num_retries})")
            else:
                print("Aborting download.")
                raise
        except Exception as e:
            print(f"Download failed due to unexpected error: {e}")
            raise


def download_thread(url, extract_dir, checksum, num_retries=3, timeout=60):
    dl_thread = threading.Thread(target=download_file_simple, args=(url, extract_dir, checksum, num_retries, timeout))
    dl_thread.start()
    dl_thread.join()


# =====================================================
# Functions to check filehashes from a list of hashes.
# =====================================================
def save_hashes(model_path, file_checksums):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    hash_checked_path = model_path / HASH_MARKER_FILENAME
    temporary_path = model_path / (
        f".{HASH_MARKER_FILENAME}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(temporary_path, 'w', encoding='utf-8') as file:
            json.dump(file_checksums, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, hash_checked_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def load_hashes(model_path):
    hash_checked_path = Path(model_path) / HASH_MARKER_FILENAME
    if not hash_checked_path.is_file():
        return None
    try:
        with open(hash_checked_path, 'r', encoding='utf-8') as file:
            loaded_hashes = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return loaded_hashes if isinstance(loaded_hashes, dict) else None


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


def model_needs_download(model_path, expected_hashes) -> bool:
    """Check a managed model while avoiding repeated hashes on normal startup.

    ``hash_checked`` is a receipt for one exact manifest. If every expected file
    exists and the receipt equals that manifest, no model file is read. A missing,
    invalid, or stale receipt causes one full verification and is replaced
    atomically only when all files match.
    """
    model_path = Path(model_path)
    if any(not (model_path / file_name).is_file() for file_name in expected_hashes):
        return True

    if load_hashes(model_path) == expected_hashes:
        return False

    if check_file_hashes(str(model_path.resolve()), expected_hashes):
        save_hashes(model_path, expected_hashes)
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

    model_path = Path(download_settings["model_path"])
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

    model_directory = model_path
    if "path" in model_link_dict[model_name] and model_link_dict[model_name]["path"] != "":
        model_directory = Path(model_path / model_link_dict[model_name]["path"])
    os.makedirs(str(model_directory.resolve()), exist_ok=True)

    model_entry = model_link_dict[model_name]
    expected_hashes = model_entry["file_checksums"]
    if not model_needs_download(model_directory, expected_hashes):
        return True
    if state["is_downloading"]:
        return False

    print(f"download started... {title}")
    state["is_downloading"] = True
    hash_checked_file = model_directory / HASH_MARKER_FILENAME
    try:
        # Never allow an old receipt to authenticate files written by a new or
        # interrupted extraction.
        try:
            hash_checked_file.unlink()
        except FileNotFoundError:
            pass

        filename = os.path.basename(model_entry["urls"][0])
        download_success = download_extract(
            model_entry["urls"],
            str(model_directory.resolve()),
            model_entry["checksum"],
            alt_fallback=alt_fallback,
            force_non_ui_dl=force_non_ui_dl,
            fallback_extract_func=fallback_extract_func,
            fallback_extract_func_args=(
                str(Path(model_directory / filename)),
                str(model_directory.resolve()),
            ),
            title=title,
            extract_format=extract_format,
        )
        if not download_success:
            print(f"Download failed: {title}")
            return False

        # This is the only full model-file verification on a successful install.
        # Later startups use the exact-manifest receipt above and do not rehash.
        if not check_file_hashes(str(model_directory.resolve()), expected_hashes):
            print(f"Downloaded model files failed verification: {title}")
            return False

        save_hashes(model_directory, expected_hashes)
        return True
    except Exception as exc:
        print(f"Download failed: {title}: {exc}")
        return False
    finally:
        state["is_downloading"] = False

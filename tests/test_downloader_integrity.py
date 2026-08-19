import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import downloader


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


class DownloaderIntegrityTests(unittest.TestCase):
    def test_exact_manifest_receipt_skips_large_file_hashing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / "model.bin").write_bytes(b"model")
            expected_hashes = {"model.bin": _sha256(b"model")}
            downloader.save_hashes(model_path, expected_hashes)

            with mock.patch.object(
                downloader,
                "check_file_hashes",
                side_effect=AssertionError("cache hit must not rehash model files"),
            ):
                self.assertFalse(
                    downloader.model_needs_download(model_path, expected_hashes)
                )

    def test_stale_manifest_is_rechecked_once_and_replaced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / "model.bin").write_bytes(b"model")
            expected_hashes = {"model.bin": _sha256(b"model")}
            downloader.save_hashes(model_path, {"old.bin": "0" * 64})

            with mock.patch.object(
                downloader,
                "check_file_hashes",
                wraps=downloader.check_file_hashes,
            ) as check_hashes:
                self.assertFalse(
                    downloader.model_needs_download(model_path, expected_hashes)
                )

            check_hashes.assert_called_once()
            self.assertEqual(downloader.load_hashes(model_path), expected_hashes)

    def test_invalid_manifest_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / downloader.HASH_MARKER_FILENAME).write_bytes(b"\xffnot json")
            self.assertIsNone(downloader.load_hashes(model_path))

    def test_download_model_verifies_files_before_writing_receipt(self):
        model_bytes = b"verified model"
        expected_hashes = {"model.bin": _sha256(model_bytes)}
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir)
            model_path = cache_path / "test-model"
            state = {"is_downloading": False}
            download_settings = {
                "model_path": cache_path,
                "model_link_dict": {
                    "test-model": {
                        "urls": ["https://example.invalid/test-model.zip"],
                        "checksum": "1" * 64,
                        "file_checksums": expected_hashes,
                        "path": "test-model",
                    }
                },
                "model_name": "test-model",
                "title": "test model",
                "alt_fallback": False,
                "force_non_ui_dl": True,
                "extract_format": "zip",
            }

            def create_extracted_file(*args, **kwargs):
                del args, kwargs
                (model_path / "model.bin").write_bytes(model_bytes)
                return True

            with mock.patch.object(
                downloader, "download_extract", side_effect=create_extracted_file
            ) as download_extract:
                self.assertTrue(
                    downloader.download_model(download_settings, state)
                )

            self.assertEqual(downloader.load_hashes(model_path), expected_hashes)
            self.assertFalse(state["is_downloading"])
            download_extract.assert_called_once()

            with mock.patch.object(
                downloader,
                "check_file_hashes",
                side_effect=AssertionError("trusted receipt must avoid rehashing"),
            ), mock.patch.object(downloader, "download_extract") as second_download:
                self.assertTrue(
                    downloader.download_model(download_settings, state)
                )
            second_download.assert_not_called()

    def test_failed_post_download_verification_does_not_write_receipt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir)
            model_path = cache_path / "test-model"
            state = {"is_downloading": False}
            download_settings = {
                "model_path": cache_path,
                "model_link_dict": {
                    "test-model": {
                        "urls": ["https://example.invalid/test-model.zip"],
                        "checksum": "1" * 64,
                        "file_checksums": {"model.bin": _sha256(b"expected")},
                        "path": "test-model",
                    }
                },
                "model_name": "test-model",
                "title": "test model",
                "alt_fallback": False,
                "force_non_ui_dl": True,
                "extract_format": "zip",
            }

            def create_wrong_file(*args, **kwargs):
                del args, kwargs
                (model_path / "model.bin").write_bytes(b"wrong")
                return True

            with mock.patch.object(
                downloader, "download_extract", side_effect=create_wrong_file
            ):
                self.assertFalse(
                    downloader.download_model(download_settings, state)
                )

            self.assertFalse((model_path / downloader.HASH_MARKER_FILENAME).exists())
            self.assertFalse(state["is_downloading"])

    def test_non_ui_zip_fallback_reports_success(self):
        archive_bytes = b"archive"
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = Path(temp_dir)
            archive_path = extract_dir / "model.zip"
            extracted_path = extract_dir / "model.bin"

            def create_archive(*args, **kwargs):
                del args, kwargs
                archive_path.write_bytes(archive_bytes)

            def extract_archive(file_path, output_dir):
                self.assertEqual(Path(file_path), archive_path)
                Path(output_dir, "model.bin").write_bytes(b"model")
                Path(file_path).unlink()

            with mock.patch.object(downloader.settings, "GetOption", return_value=False), \
                    mock.patch.object(
                        downloader, "download_file_normal", side_effect=create_archive
                    ):
                success = downloader.download_extract(
                    ["https://example.invalid/model.zip"],
                    str(extract_dir),
                    _sha256(archive_bytes),
                    extract_format="zip",
                    fallback_extract_func=extract_archive,
                    fallback_extract_func_args=(str(archive_path), str(extract_dir)),
                    force_non_ui_dl=True,
                )

            self.assertTrue(success)
            self.assertTrue(extracted_path.is_file())

    def test_wrong_same_size_existing_archive_is_retried_cleanly(self):
        expected_archive = b"correct"
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = Path(temp_dir)
            archive_path = extract_dir / "model.zip"
            archive_path.write_bytes(b"corrupt")
            attempts = 0

            def robust_download(*args, **kwargs):
                nonlocal attempts
                del args, kwargs
                attempts += 1
                if attempts == 1:
                    return False
                archive_path.write_bytes(expected_archive)
                return True

            with mock.patch.object(downloader.settings, "GetOption", return_value=False), \
                    mock.patch.object(
                        downloader, "download_file_normal", side_effect=robust_download
                    ):
                success = downloader.download_extract(
                    ["https://example.invalid/model.zip"],
                    str(extract_dir),
                    _sha256(expected_archive),
                    extract_format="none",
                    force_non_ui_dl=True,
                )

            self.assertTrue(success)
            self.assertEqual(attempts, 2)
            self.assertEqual(archive_path.read_bytes(), expected_archive)

    def test_simple_downloader_streams_without_content_length(self):
        payload = b"streamed response"

        class FakeResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                self.chunk_size = chunk_size
                return iter((payload[:5], payload[5:]))

        response = FakeResponse()
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                downloader.requests, "get", return_value=response
            ):
                verified = downloader.download_file_simple(
                    "https://example.invalid/model.bin",
                    temp_dir,
                    _sha256(payload),
                )

            self.assertTrue(verified)
            self.assertEqual(response.chunk_size, downloader.HASH_CHUNK_SIZE)
            self.assertEqual(Path(temp_dir, "model.bin").read_bytes(), payload)

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = Path(temp_dir) / "model"
            extract_dir.mkdir()
            archive_path = Path(temp_dir) / "model.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.bin", b"escape")

            with self.assertRaisesRegex(ValueError, "escapes"):
                downloader.extract_zip(archive_path, extract_dir)

            self.assertFalse((Path(temp_dir) / "outside.bin").exists())
            self.assertTrue(archive_path.exists())

    def test_ui_failure_receipt_returns_without_waiting_forever(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = Path(temp_dir)
            archive_path = extract_dir / "model.zip"

            def report_failure(*args, **kwargs):
                del args, kwargs
                Path(str(archive_path) + downloader.DOWNLOAD_FAILED_SUFFIX).write_text(
                    "checksum mismatch", encoding="utf-8"
                )

            connected = {"value": True, "websocket": object()}
            with mock.patch.object(downloader.settings, "GetOption", return_value=True), \
                    mock.patch.dict(downloader.websocket.UI_CONNECTED, connected, clear=True), \
                    mock.patch.object(
                        downloader.websocket, "AnswerMessage", side_effect=report_failure
                    ):
                success = downloader.download_extract(
                    ["https://example.invalid/model.zip"],
                    str(extract_dir),
                    "1" * 64,
                    extract_format="zip",
                )

            self.assertFalse(success)
            self.assertFalse(
                Path(str(archive_path) + downloader.DOWNLOAD_FAILED_SUFFIX).exists()
            )

    def test_ui_success_receipt_avoids_duplicate_archive_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = Path(temp_dir)
            archive_path = extract_dir / "model.zip"

            def report_success(*args, **kwargs):
                del args, kwargs
                archive_path.write_bytes(b"already verified by Go")
                Path(str(archive_path) + downloader.DOWNLOAD_FINISHED_SUFFIX).touch()

            connected = {"value": True, "websocket": object()}
            with mock.patch.object(downloader.settings, "GetOption", return_value=True), \
                    mock.patch.dict(downloader.websocket.UI_CONNECTED, connected, clear=True), \
                    mock.patch.object(
                        downloader.websocket, "AnswerMessage", side_effect=report_success
                    ), mock.patch.object(
                        downloader,
                        "sha256_checksum",
                        side_effect=AssertionError("Go receipt must avoid a second archive hash"),
                    ):
                success = downloader.download_extract(
                    ["https://example.invalid/model.zip"],
                    str(extract_dir),
                    "1" * 64,
                    extract_format="zip",
                )

            self.assertTrue(success)
            self.assertFalse(archive_path.exists())


if __name__ == "__main__":
    unittest.main()

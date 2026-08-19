import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from settings import SettingsManager


class SettingsPersistenceTests(unittest.TestCase):
    def test_stale_backend_save_preserves_external_profile_changes(self):
        """A stopping backend must not undo a profile just saved by the UI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.yaml"
            profile_path.write_text(
                yaml.safe_dump(
                    {
                        "process_id": 111,
                        "energy": 300,
                        "txt_translator": "NLLB200_CT2",
                        "txt_translator_size": "small",
                    }
                ),
                encoding="utf-8",
            )

            stale_backend = SettingsManager()
            stale_backend.load_yaml(profile_path)
            with mock.patch.object(stale_backend, "debounced_save_yaml"):
                stale_backend.set_option("process_id", 222)
                stale_backend.set_option("energy", 420)
                stale_backend.set_option("txt_translator", "")

            # The profile editor is a separate process. It saves the new model
            # while the old backend still holds the previous profile in memory.
            profile_path.write_text(
                yaml.safe_dump(
                    {
                        "process_id": 111,
                        "energy": 300,
                        "txt_translator": "milmmt",
                        "txt_translator_size": "MiLMMT-46-1B-v1.0",
                    }
                ),
                encoding="utf-8",
            )

            stale_backend.flush_pending_save()

            saved = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["process_id"], 222)
            self.assertEqual(saved["energy"], 420)
            self.assertEqual(saved["txt_translator"], "milmmt")
            self.assertEqual(
                saved["txt_translator_size"], "MiLMMT-46-1B-v1.0"
            )
            self.assertEqual(stale_backend.get_option("txt_translator"), "milmmt")

    def test_backend_change_after_external_save_remains_the_newest_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.yaml"
            profile_path.write_text(
                yaml.safe_dump({"txt_translator": "NLLB200_CT2"}),
                encoding="utf-8",
            )

            backend = SettingsManager()
            backend.load_yaml(profile_path)

            profile_path.write_text(
                yaml.safe_dump({"txt_translator": "milmmt"}),
                encoding="utf-8",
            )
            with mock.patch.object(backend, "debounced_save_yaml"):
                backend.set_option("txt_translator", "hunyuan_mt")
            backend.save_yaml(profile_path)

            saved = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["txt_translator"], "hunyuan_mt")

    def test_shutdown_flushes_pending_profile_write(self):
        """Shutdown must not lose settings still inside the debounce window."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.yaml"
            profile_path.write_text(
                yaml.safe_dump({"energy": 300, "process_id": 123}),
                encoding="utf-8",
            )

            backend = SettingsManager()
            backend.load_yaml(profile_path)
            with mock.patch.object(backend, "debounced_save_yaml"):
                backend.set_option("energy", 420)
                backend.set_option("process_id", 0)

            backend.flush_pending_save()

            saved = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["energy"], 420)
            self.assertEqual(saved["process_id"], 0)
            self.assertEqual(backend._dirty_settings, {})


if __name__ == "__main__":
    unittest.main()

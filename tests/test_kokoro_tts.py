import sys
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch

from Models.TTS.kokoro import german as kokoro_german
from Models.TTS.kokoro import pipeline as kokoro_pipeline
from Models.TTS.kokoro import russian as kokoro_russian
from Models.TTS.kokoro.model import _load_module_state


with mock.patch.dict(
    sys.modules,
    {
        "Plugins": types.SimpleNamespace(plugin_custom_event_call=mock.Mock(return_value=None)),
        "downloader": types.SimpleNamespace(download_model=mock.Mock(return_value=True)),
    },
):
    from Models.TTS import kokoro_tts


class KokoroTTSAdapterTests(unittest.TestCase):
    def test_runtime_loads_legacy_and_parametrized_weight_norm_strictly(self):
        from torch import nn
        from torch.nn.utils.parametrizations import weight_norm

        source = weight_norm(nn.Conv1d(3, 4, 3))
        source_state = source.state_dict()
        checkpoint_states = (
            {f"module.{key}": value.clone() for key, value in source_state.items()},
            {
                "module.bias": source_state["bias"].clone(),
                "module.weight_g": source_state[
                    "parametrizations.weight.original0"
                ].clone(),
                "module.weight_v": source_state[
                    "parametrizations.weight.original1"
                ].clone(),
            },
        )

        for checkpoint_state in checkpoint_states:
            with self.subTest(keys=tuple(checkpoint_state)):
                target = weight_norm(nn.Conv1d(3, 4, 3))
                _load_module_state(target, checkpoint_state, "decoder")
                for key, expected in source_state.items():
                    torch.testing.assert_close(target.state_dict()[key], expected)

        nested_source = nn.Sequential(weight_norm(nn.Conv1d(3, 4, 3)))
        nested_state = nested_source.state_dict()
        nested_legacy_state = {}
        for key, value in nested_state.items():
            legacy_key = key.replace(
                "parametrizations.weight.original0",
                "weight_g",
            ).replace(
                "parametrizations.weight.original1",
                "weight_v",
            )
            nested_legacy_state[f"module.{legacy_key}"] = value.clone()
        nested_target = nn.Sequential(weight_norm(nn.Conv1d(3, 4, 3)))
        _load_module_state(nested_target, nested_legacy_state, "decoder")
        for key, expected in nested_state.items():
            torch.testing.assert_close(nested_target.state_dict()[key], expected)

        incomplete_state = dict(checkpoint_states[0])
        incomplete_state.pop("module.bias")
        with self.assertRaisesRegex(RuntimeError, "decoder.*incompatible"):
            _load_module_state(
                weight_norm(nn.Conv1d(3, 4, 3)),
                incomplete_state,
                "decoder",
            )

        model_source = (
            Path(kokoro_tts.__file__).parent / "kokoro" / "model.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("strict=False", model_source)

    def test_runtime_accepts_only_checkpoint_omitted_identity_norm_parameters(self):
        from torch import nn

        class PredictorFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.F0 = nn.ModuleList([nn.Module()])
                self.F0[0].norm1 = nn.Module()
                self.F0[0].norm1.norm = nn.InstanceNorm1d(2, affine=True)
                self.anchor = nn.Linear(2, 2)

        source = PredictorFixture()
        checkpoint_state = {
            f"module.{key}": value.clone()
            for key, value in source.state_dict().items()
            if key not in {
                "F0.0.norm1.norm.weight",
                "F0.0.norm1.norm.bias",
            }
        }

        target = PredictorFixture()
        _load_module_state(target, checkpoint_state, "predictor")
        torch.testing.assert_close(target.anchor.weight, source.anchor.weight)
        torch.testing.assert_close(
            target.F0[0].norm1.norm.weight,
            torch.ones_like(target.F0[0].norm1.norm.weight),
        )
        torch.testing.assert_close(
            target.F0[0].norm1.norm.bias,
            torch.zeros_like(target.F0[0].norm1.norm.bias),
        )

        checkpoint_state.pop("module.anchor.bias")
        with self.assertRaisesRegex(RuntimeError, "predictor.*incompatible"):
            _load_module_state(PredictorFixture(), checkpoint_state, "predictor")

    def test_thorsten_checkpoint_has_complete_hosted_manifest(self):
        entry = kokoro_tts.TTS_MODEL_LINKS[kokoro_tts.THORSTEN_MODEL]

        self.assertEqual(
            set(entry["file_checksums"]),
            {
                "LICENSE",
                "README.md",
                "config.json",
                "model.pth",
                "voices/thorsten.pt",
            },
        )
        self.assertEqual(len(entry["checksum"]), 64)
        self.assertNotEqual(entry["checksum"], "0" * 64)
        self.assertEqual(
            entry["source_revision"],
            "734e593d320a3d876bede7020f773dfd481a0cc7",
        )
        self.assertEqual(len(entry["urls"]), 3)
        self.assertTrue(all("huggingface" not in url.lower() for url in entry["urls"]))
        self.assertTrue(all(len(value) == 64 for value in entry["file_checksums"].values()))
        self.assertEqual(
            kokoro_tts.model_list["German"],
            [kokoro_tts.THORSTEN_MODEL],
        )

    def test_russian_checkpoint_has_complete_hosted_manifest(self):
        entry = kokoro_tts.TTS_MODEL_LINKS[kokoro_tts.RUSSIAN_MODEL]

        self.assertEqual(
            set(entry["file_checksums"]),
            {
                "README.md",
                "RUACCENT_LICENSE",
                "RUACCENT_README.md",
                "config.json",
                "kokoro-ru-v2-base.pth",
                "russian-frontend.zip",
                "voices/masha.pt",
                "voices/sveta.pt",
            },
        )
        self.assertEqual(len(entry["checksum"]), 64)
        self.assertNotEqual(entry["checksum"], "0" * 64)
        self.assertEqual(
            entry["source_revision"],
            "27d078fe1c0cab919613a64e906919214385f21d",
        )
        self.assertEqual(
            entry["frontend_revision"],
            "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855",
        )
        self.assertEqual(
            entry["ruaccent_source_revision"],
            "3ac0ad5f6508f6c1a4a604220042232c70a7baf9",
        )
        self.assertEqual(len(entry["urls"]), 3)
        self.assertTrue(all("huggingface" not in url.lower() for url in entry["urls"]))
        self.assertTrue(all(len(value) == 64 for value in entry["file_checksums"].values()))
        self.assertEqual(
            kokoro_tts.model_list["Russian"],
            [kokoro_tts.RUSSIAN_MODEL],
        )

    def test_russian_pipeline_requires_managed_frontend_assets_and_rejects_oov(self):
        language_assets = {
            "model_directory": Path("model"),
            "frontend_archive": "russian-frontend.zip",
            "frontend_checksum": "a" * 64,
        }
        fake_g2p = mock.Mock(return_value=("phonemes", set()))
        with mock.patch.object(
            kokoro_russian,
            "RussianG2P",
            return_value=fake_g2p,
        ) as g2p_class:
            pipeline = kokoro_pipeline.KPipeline(
                "ru",
                model=False,
                language_assets=language_assets,
            )

        self.assertEqual(pipeline.lang_code, "r")
        g2p_class.assert_called_once_with(**language_assets)
        self.assertEqual(pipeline._phonemize("Привет"), "phonemes")
        fake_g2p.return_value = ("phonemes", {"unsupported"})
        with self.assertRaisesRegex(ValueError, "unsupported symbols"):
            pipeline._phonemize("Привет")

        with self.assertRaisesRegex(ValueError, "managed frontend assets"):
            kokoro_pipeline.KPipeline("ru", model=False)

    def test_russian_frontend_extracts_verified_archive_and_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory)
            archive_path = model_directory / "frontend.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("required.txt", "ready")

            with mock.patch.object(
                kokoro_russian,
                "_FRONTEND_REQUIRED_FILES",
                ("required.txt",),
            ):
                extracted = kokoro_russian._extract_frontend(
                    model_directory,
                    archive_path.name,
                    "b" * 64,
                )
                self.assertEqual((extracted / "required.txt").read_text(), "ready")
                self.assertEqual(
                    kokoro_russian._extract_frontend(
                        model_directory,
                        archive_path.name,
                        "b" * 64,
                    ),
                    extracted,
                )

            traversal_archive = model_directory / "traversal.zip"
            with zipfile.ZipFile(traversal_archive, "w") as archive:
                archive.writestr("../escape.txt", "bad")
                archive.writestr("required.txt", "ready")
            with mock.patch.object(
                kokoro_russian,
                "_FRONTEND_REQUIRED_FILES",
                ("required.txt",),
            ), self.assertRaisesRegex(ValueError, "escapes extraction"):
                kokoro_russian._extract_frontend(
                    model_directory,
                    traversal_archive.name,
                    "c" * 64,
                )
            self.assertFalse((model_directory / "escape.txt").exists())

    def test_german_g2p_accepts_misaki_return_shapes_and_maps_short_ue(self):
        for g2p_result in ("hʏːtə", ("hʏːtə", None)):
            with self.subTest(return_type=type(g2p_result).__name__):
                g2p = mock.Mock(return_value=g2p_result)
                with mock.patch.object(
                    kokoro_pipeline.espeak,
                    "EspeakG2P",
                    return_value=g2p,
                ):
                    pipeline = kokoro_pipeline.KPipeline("de", model=False)

                self.assertEqual(pipeline.lang_code, "d")
                self.assertEqual(pipeline._phonemize("Hüte"), "hyːtə")

    def test_german_frontend_matches_checkpoint_text_normalization(self):
        self.assertEqual(
            kokoro_german.normalize_text_de("Dr. Müller zahlt 3,14 € um 08:30 Uhr."),
            "Doktor Müller zahlt drei Euro und vierzehn Cent um acht Uhr dreißig.",
        )

        g2p = object.__new__(kokoro_german.GermanG2P)
        g2p.espeak = mock.Mock(side_effect=AssertionError("override should bypass eSpeak"))
        phonemes, tokens = g2p("API JSON")
        self.assertEqual(phonemes, "eɪpiːˈaɪ dʒˈeɪsən")
        self.assertIsNone(tokens)

    def test_german_byte_family_uses_english_diphthong_respelling(self):
        byte_source = (
            "Byte, Bytes, Kilobyte, Megabyte, Gigabytes, Mebibyte, "
            "Gibibytes und Bytecode."
        )
        byte_respelled = (
            "Bait, Baits, Kilobait, Megabait, Gigabaits, Mebibait, "
            "Gibibaits und Baitcode."
        )
        self.assertEqual(
            kokoro_german.respell_for_espeak_de(
                byte_source + " Type und Hypnose."
            ),
            byte_respelled + " Type und Hypnose.",
        )
        self.assertEqual(kokoro_german.override_for("Hype"), "hˈaɪp")
        self.assertEqual(kokoro_german.override_for("Hypes"), "hˈaɪps")

        g2p = object.__new__(kokoro_german.GermanG2P)
        g2p.espeak = mock.Mock(return_value=("checkpoint phonemes", None))
        self.assertEqual(g2p(byte_source), ("checkpoint phonemes", None))
        g2p.espeak.assert_called_once_with(byte_respelled)

    def test_specialized_models_force_their_language_and_stock_rejects_them(self):
        self.assertEqual(
            kokoro_tts.KokoroTTS._effective_language(
                kokoro_tts.THORSTEN_MODEL,
                "a",
            ),
            "d",
        )
        self.assertEqual(
            kokoro_tts.KokoroTTS._effective_language(
                kokoro_tts.RUSSIAN_MODEL,
                "a",
            ),
            "r",
        )
        with mock.patch("builtins.print"):
            self.assertEqual(
                kokoro_tts.KokoroTTS._effective_language(
                    kokoro_tts.DEFAULT_MODEL,
                    "de",
                ),
                "a",
            )
            self.assertEqual(
                kokoro_tts.KokoroTTS._effective_language(
                    kokoro_tts.DEFAULT_MODEL,
                    "ru",
                ),
                "a",
            )

    def test_load_model_uses_thorsten_checkpoint_and_german_pipeline(self):
        adapter = object.__new__(kokoro_tts.KokoroTTS)
        adapter.model = None
        adapter.pipeline = None
        adapter.last_language = ""
        adapter.loaded_model_name = ""
        adapter.loaded_device = ""
        adapter.compute_device = "cpu"
        adapter.compute_device_str = "cpu"
        adapter.voice_list = []
        adapter._voice_cache = {}
        adapter._model_lock = threading.RLock()
        adapter.download_state = {"is_downloading": False}
        adapter.download_model = mock.Mock(return_value=True)
        adapter.update_voices = mock.Mock()

        fake_model = mock.Mock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model
        fake_pipeline = mock.Mock()

        def get_option(name):
            return {
                "tts_model": ["German", kokoro_tts.THORSTEN_MODEL],
                "tts_ai_device": "cpu",
            }.get(name)

        with mock.patch.object(
            kokoro_tts.settings,
            "GetOption",
            side_effect=get_option,
        ), mock.patch.object(
            kokoro_tts,
            "KModel",
            return_value=fake_model,
        ) as model_class, mock.patch.object(
            kokoro_tts,
            "KPipeline",
            return_value=fake_pipeline,
        ) as pipeline_class:
            adapter.load_model("a")

        adapter.download_model.assert_called_once_with(kokoro_tts.THORSTEN_MODEL)
        config_path, model_path = model_class.call_args.args
        self.assertEqual(Path(config_path).name, "config.json")
        self.assertEqual(Path(model_path).name, "model.pth")
        self.assertEqual(Path(model_path).parent.name, kokoro_tts.THORSTEN_MODEL)
        pipeline_class.assert_called_once_with(
            lang_code="d",
            model=fake_model,
            device="cpu",
        )
        self.assertEqual(adapter.loaded_model_name, kokoro_tts.THORSTEN_MODEL)
        self.assertEqual(adapter.last_language, "d")
        adapter.update_voices.assert_called_once_with(kokoro_tts.THORSTEN_MODEL)

    def test_load_model_uses_russian_checkpoint_and_managed_frontend(self):
        adapter = object.__new__(kokoro_tts.KokoroTTS)
        adapter.model = None
        adapter.pipeline = None
        adapter.last_language = ""
        adapter.loaded_model_name = ""
        adapter.loaded_device = ""
        adapter.compute_device = "cpu"
        adapter.compute_device_str = "cpu"
        adapter.voice_list = []
        adapter._voice_cache = {}
        adapter._model_lock = threading.RLock()
        adapter.download_state = {"is_downloading": False}
        adapter.download_model = mock.Mock(return_value=True)
        adapter.update_voices = mock.Mock()

        fake_model = mock.Mock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model
        fake_pipeline = mock.Mock()

        def get_option(name):
            return {
                "tts_model": ["Russian", kokoro_tts.RUSSIAN_MODEL],
                "tts_ai_device": "cpu",
            }.get(name)

        with mock.patch.object(
            kokoro_tts.settings,
            "GetOption",
            side_effect=get_option,
        ), mock.patch.object(
            kokoro_tts,
            "KModel",
            return_value=fake_model,
        ) as model_class, mock.patch.object(
            kokoro_tts,
            "KPipeline",
            return_value=fake_pipeline,
        ) as pipeline_class:
            adapter.load_model("a")

        adapter.download_model.assert_called_once_with(kokoro_tts.RUSSIAN_MODEL)
        config_path, model_path = model_class.call_args.args
        self.assertEqual(Path(config_path).name, "config.json")
        self.assertEqual(Path(model_path).name, "kokoro-ru-v2-base.pth")
        self.assertEqual(Path(model_path).parent.name, kokoro_tts.RUSSIAN_MODEL)
        pipeline_kwargs = pipeline_class.call_args.kwargs
        self.assertEqual(pipeline_kwargs["lang_code"], "r")
        self.assertIs(pipeline_kwargs["model"], fake_model)
        self.assertEqual(pipeline_kwargs["device"], "cpu")
        self.assertEqual(
            Path(pipeline_kwargs["language_assets"]["model_directory"]).name,
            kokoro_tts.RUSSIAN_MODEL,
        )
        self.assertEqual(
            pipeline_kwargs["language_assets"]["frontend_archive"],
            "russian-frontend.zip",
        )
        self.assertEqual(adapter.loaded_model_name, kokoro_tts.RUSSIAN_MODEL)
        self.assertEqual(adapter.last_language, "r")
        adapter.update_voices.assert_called_once_with(kokoro_tts.RUSSIAN_MODEL)

    def test_incompatible_profile_voice_falls_back_to_thorsten(self):
        adapter = object.__new__(kokoro_tts.KokoroTTS)
        adapter.loaded_model_name = kokoro_tts.THORSTEN_MODEL
        voice_filename = str(Path("voices") / "thorsten.pt")
        adapter.voice_list = [
            {
                "name": "thorsten",
                "voice_filename": voice_filename,
            }
        ]
        adapter._voice_cache = {}
        voice_tensor = mock.sentinel.voice_tensor

        with mock.patch.object(
            kokoro_tts.torch,
            "load",
            return_value=voice_tensor,
        ) as torch_load:
            selected_name, selected_tensor = adapter._get_voice_tensor("af_heart")

        self.assertEqual(selected_name, "thorsten")
        self.assertIs(selected_tensor, voice_tensor)
        torch_load.assert_called_once_with(
            voice_filename,
            weights_only=True,
            map_location="cpu",
        )

    def test_thorsten_exposes_only_its_checkpoint_voice(self):
        adapter = object.__new__(kokoro_tts.KokoroTTS)
        adapter.voice_list = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory)
            voice_path = (
                cache_root
                / kokoro_tts.THORSTEN_MODEL
                / "voices"
                / "thorsten.pt"
            )
            voice_path.parent.mkdir(parents=True)
            voice_path.touch()
            with mock.patch.object(kokoro_tts, "cache_path", cache_root):
                adapter.update_voices(kokoro_tts.THORSTEN_MODEL)

        self.assertEqual(len(adapter.voice_list), 1)
        self.assertEqual(adapter.voice_list[0]["name"], "thorsten")

    def test_russian_exposes_only_base_checkpoint_voices(self):
        adapter = object.__new__(kokoro_tts.KokoroTTS)
        adapter.voice_list = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory)
            voices_directory = cache_root / kokoro_tts.RUSSIAN_MODEL / "voices"
            voices_directory.mkdir(parents=True)
            for voice_name in ("sveta", "masha", "dima"):
                (voices_directory / f"{voice_name}.pt").touch()
            with mock.patch.object(kokoro_tts, "cache_path", cache_root):
                adapter.update_voices(kokoro_tts.RUSSIAN_MODEL)

        self.assertEqual(
            [voice["name"] for voice in adapter.voice_list],
            ["sveta", "masha"],
        )

    def test_native_frontends_preserve_punctuation_for_normalization(self):
        for model_name in kokoro_tts.NATIVE_FRONTEND_MODELS:
            with self.subTest(model=model_name):
                adapter = object.__new__(kokoro_tts.KokoroTTS)
                adapter.loaded_model_name = model_name
                adapter._model_lock = threading.RLock()
                adapter.pipeline = mock.Mock(return_value=iter(()))
                adapter._get_voice_tensor = mock.Mock(
                    return_value=("voice", mock.sentinel.voice_tensor)
                )

                def get_option(name):
                    return {
                        "tts_volume": 1.0,
                        "tts_prosody_rate": "",
                        "tts_voice": "voice",
                    }[name]

                with mock.patch.object(
                    kokoro_tts.settings,
                    "GetOption",
                    side_effect=get_option,
                ):
                    list(adapter.stream_tts_segments("Text, with punctuation."))

                adapter.pipeline.assert_called_once_with(
                    "Text, with punctuation.",
                    voice=mock.sentinel.voice_tensor,
                    speed=1,
                    split_pattern=None,
                )


if __name__ == "__main__":
    unittest.main()

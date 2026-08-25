import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from scipy.io.wavfile import read as read_wav
from Models.TTS import audio8_tts


class Audio8ManifestTests(unittest.TestCase):
    def test_model_uses_one_shared_codec_archive(self):
        self.assertEqual(set(audio8_tts.MODEL_LANGUAGES), {audio8_tts.DEFAULT_MODEL})
        self.assertIn("de", audio8_tts.MODEL_LANGUAGES[audio8_tts.DEFAULT_MODEL])
        entry = audio8_tts.TTS_MODEL_LINKS[audio8_tts.DEFAULT_MODEL]
        self.assertEqual(entry["base_model"], audio8_tts.CODEC_MODEL)
        self.assertNotIn("codec.pth", entry["file_checksums"])
        self.assertEqual(entry["checksum"], "0" * 64)
        self.assertTrue(all("huggingface.co" not in url for url in entry["urls"]))
        codec = audio8_tts.TTS_MODEL_LINKS[audio8_tts.CODEC_MODEL]
        self.assertEqual(
            codec["file_checksums"]["codec.pth"],
            "c310505aa11fe2f6cc63b8d3130dc7e77e73227774f5c62575769b1f47a8d048",
        )
        self.assertTrue(
            all(
                len(checksum) == 64
                for checksum in audio8_tts.VOICE_MODEL_LINKS["voices"][
                    "file_checksums"
                ].values()
            )
        )

    def test_placeholder_archive_fails_before_network_request(self):
        adapter = object.__new__(audio8_tts.Audio8TTS)
        adapter.download_state = {"is_downloading": False}
        download = mock.Mock()
        fake_downloader = types.SimpleNamespace(
            model_needs_download=mock.Mock(return_value=True),
            download_model=download,
        )
        with mock.patch.dict(sys.modules, {"downloader": fake_downloader}):
            with self.assertRaisesRegex(RuntimeError, "not currently available"):
                adapter.download_model(audio8_tts.CODEC_MODEL)
        download.assert_not_called()

    def test_windows_flash_extension_is_explicitly_packaged(self):
        project_root = Path(__file__).resolve().parents[1]
        spec = (project_root / "audioWhisper.spec").read_text(encoding="utf-8")
        requirements = (project_root / "requirements.nvidia.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("'flash_attn_2_cuda'", spec)
        self.assertIn("flash-attn @", requirements)


class Audio8CompatibilityTests(unittest.TestCase):
    def test_attention_auto_uses_existing_flash_kernel_only_when_compatible(self):
        flash_function = mock.Mock()
        with mock.patch.object(
            audio8_tts,
            "_get_flash_attention_function",
            return_value=flash_function,
        ):
            backend, selected = audio8_tts._resolve_attention_backend(
                "auto",
                torch.device("cuda"),
                torch.bfloat16,
            )
        self.assertEqual(backend, "flash_attention_2")
        self.assertIs(selected, flash_function)

        backend, selected = audio8_tts._resolve_attention_backend(
            "auto",
            torch.device("cpu"),
            torch.float32,
        )
        self.assertEqual(backend, "sdpa")
        self.assertIsNone(selected)

    def test_explicit_flash_request_falls_back_when_kernel_cannot_load(self):
        with mock.patch.object(
            audio8_tts,
            "_get_flash_attention_function",
            side_effect=OSError("incompatible wheel"),
        ):
            backend, selected = audio8_tts._resolve_attention_backend(
                "flash_attention_2",
                torch.device("cuda"),
                torch.bfloat16,
            )
        self.assertEqual(backend, "sdpa")
        self.assertIsNone(selected)

    def test_flash_patch_tracks_only_the_valid_cache_prefix(self):
        attention = types.SimpleNamespace(forward=mock.Mock())
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(max_seq_len=16),
            layers=[types.SimpleNamespace(attention=attention)],
            _setup_generation_caches=mock.Mock(),
            _slow_step=mock.Mock(return_value=("logits", "hidden")),
        )
        apply_rope = mock.Mock()
        audio8_tts._enable_audio8_flash_attention(
            model,
            types.SimpleNamespace(_apply_rope=apply_rope),
            mock.Mock(),
        )

        model._setup_generation_caches(1, 16, torch.bfloat16)
        result = model._slow_step(torch.zeros((1, 11, 5)), "position", "ids", "mask")

        self.assertEqual(model._audio8_valid_cache_length, 5)
        self.assertEqual(attention._audio8_valid_cache_length, 5)
        self.assertEqual(result, ("logits", "hidden"))
        self.assertIs(attention._audio8_apply_rope, apply_rope)

    def test_flash_patch_is_idempotent(self):
        class FakeAttention:
            training = False

            def forward(self, *args):
                del args
                return "compatible output"

        attention = FakeAttention()
        original_function = FakeAttention.forward
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(max_seq_len=16),
            layers=[types.SimpleNamespace(attention=attention)],
            _setup_generation_caches=mock.Mock(),
            _slow_step=mock.Mock(),
        )
        modeling = types.SimpleNamespace(_apply_rope=mock.Mock())

        audio8_tts._enable_audio8_flash_attention(model, modeling, mock.Mock())
        first_hook = attention.forward
        replacement_flash_function = mock.Mock()
        audio8_tts._enable_audio8_flash_attention(
            model,
            modeling,
            replacement_flash_function,
        )

        self.assertIs(attention.forward, first_hook)
        self.assertIs(
            attention._audio8_original_forward.__func__,
            original_function,
        )
        self.assertIs(attention._audio8_flash_function, replacement_flash_function)
        result = attention.forward(torch.zeros((1, 1, 1)), None, None)
        self.assertEqual(result, "compatible output")

        # An instance corrupted by the old double-install behavior also repairs
        # its fallback from the untouched runtime class instead of recursing.
        attention._audio8_original_forward = attention.forward
        repaired = attention.forward(torch.zeros((1, 1, 1)), None, None)
        self.assertEqual(repaired, "compatible output")
        self.assertIs(attention._audio8_original_forward.__func__, original_function)

    def test_verified_runtime_rejects_incomplete_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "runtime is incomplete"):
                audio8_tts._load_verified_runtime(directory)

    def test_transformers5_meta_load_rotary_buffers_are_restored(self):
        config = types.SimpleNamespace(
            max_seq_len=8,
            head_dim=4,
            num_codebooks=3,
            fast_head_dim=2,
            rope_base=10000,
        )
        model = torch.nn.Module()
        model.config = config
        model.register_buffer("freqs_cis", torch.zeros((8, 2, 2), dtype=torch.bfloat16))
        model.register_buffer("fast_freqs_cis", torch.zeros((3, 1, 2), dtype=torch.bfloat16))

        def precompute(length, head_dim, base):
            self.assertEqual(base, 10000)
            return torch.ones((length, head_dim // 2, 2), dtype=torch.bfloat16)

        modeling = types.SimpleNamespace(_precompute_rope=precompute)
        audio8_tts._restore_rotary_buffers(model, modeling)

        self.assertTrue(torch.equal(model.freqs_cis, torch.ones_like(model.freqs_cis)))
        self.assertTrue(
            torch.equal(model.fast_freqs_cis, torch.ones_like(model.fast_freqs_cis))
        )


class Audio8AdapterTests(unittest.TestCase):
    def make_adapter(self):
        adapter = object.__new__(audio8_tts.Audio8TTS)
        adapter.sample_rate = audio8_tts.SAMPLE_RATE
        adapter.special_settings = dict(audio8_tts.Audio8TTS.special_settings_defaults)
        adapter.compute_device_str = "cpu"
        adapter.compute_device = torch.device("cpu")
        adapter.loaded_configuration = None
        adapter.reference_code_cache = {}
        adapter.stop_event = mock.Mock(is_set=mock.Mock(return_value=False))
        adapter.last_generation = {"audio": None, "sample_rate": None, "text": ""}
        return adapter

    def test_model_selection_accepts_default_and_rejects_stale_values(self):
        adapter = self.make_adapter()
        with mock.patch.object(
            audio8_tts.settings,
            "GetOption",
            return_value=["0.6B", audio8_tts.DEFAULT_MODEL],
        ):
            self.assertEqual(adapter._get_model_name(), audio8_tts.DEFAULT_MODEL)
        with mock.patch.object(
            audio8_tts.settings,
            "GetOption",
            return_value=["Other", "stale-model"],
        ):
            self.assertEqual(adapter._get_model_name(), audio8_tts.DEFAULT_MODEL)

    def test_attention_defaults_to_auto(self):
        adapter = self.make_adapter()
        self.assertEqual(adapter.special_settings["attention"], "auto")

    def test_model_loading_uses_the_generation_lock(self):
        adapter = self.make_adapter()
        adapter.generation_lock = mock.MagicMock()
        adapter._load_model_locked = mock.Mock()

        adapter.load_model()

        adapter.generation_lock.__enter__.assert_called_once_with()
        adapter.generation_lock.__exit__.assert_called_once()
        adapter._load_model_locked.assert_called_once_with()

    def test_clone_auto_uses_sidecar_and_falls_back_without_one(self):
        adapter = self.make_adapter()
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "speaker.wav"
            audio_path.touch()
            sidecar = audio_path.with_suffix(".txt")
            sidecar.write_text("Exact words in the sample.", encoding="utf-8")
            with mock.patch.object(adapter, "_resolve_reference", return_value=audio_path):
                path, transcript = adapter._clone_reference()
            self.assertEqual(path, audio_path)
            self.assertEqual(transcript, "Exact words in the sample.")

            sidecar.unlink()
            with mock.patch.object(adapter, "_resolve_reference", return_value=audio_path):
                path, transcript = adapter._clone_reference()
            self.assertIsNone(path)
            self.assertEqual(transcript, "")

    def test_required_clone_mode_reports_missing_transcript(self):
        adapter = self.make_adapter()
        adapter.special_settings["clone_mode"] = "required"
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "speaker.wav"
            audio_path.touch()
            with mock.patch.object(adapter, "_resolve_reference", return_value=audio_path):
                with self.assertRaisesRegex(ValueError, "exact reference transcript"):
                    adapter._clone_reference()

    def test_long_text_is_bounded_to_upstream_recommendation(self):
        adapter = self.make_adapter()
        adapter.special_settings["streaming_segment_characters"] = 1000
        segments = adapter._segments(("This is a sentence. " * 40).strip())
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(segment) <= 150 for segment in segments))

    def test_pcm_and_wav_export_use_normalized_float_and_pcm16(self):
        adapter = self.make_adapter()
        wave = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
        pcm = np.frombuffer(adapter.return_pcm_audio(wave), dtype="<f4")
        np.testing.assert_array_equal(pcm, np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32))
        rate, wav = read_wav(io.BytesIO(adapter.return_wav_file_binary(wave)))
        self.assertEqual(rate, audio8_tts.SAMPLE_RATE)
        self.assertEqual(wav.dtype, np.dtype("int16"))
        np.testing.assert_array_equal(wav, np.array([-32767, -16384, 16384, 32767], dtype=np.int16))

    def test_segment_streaming_queues_the_same_pcm_that_it_returns(self):
        adapter = self.make_adapter()
        streamer = mock.Mock()
        adapter.audio_streamer = streamer
        with (
            mock.patch.object(adapter, "load"),
            mock.patch.object(adapter, "_clone_reference", return_value=(None, "")),
            mock.patch.object(adapter, "_generation_generator", return_value=None),
            mock.patch.object(adapter, "_segments", return_value=["one", "two"]),
            mock.patch.object(
                adapter,
                "_generate_segment",
                side_effect=[torch.tensor([[0.1, 0.2]]), torch.tensor([[0.3]])],
            ),
            mock.patch.object(adapter, "_silence", return_value=torch.tensor([[0.0]])),
            mock.patch.object(adapter, "init_audio_stream_playback"),
        ):
            wave, sample_rate = adapter._synthesize("one two", streamed=True)

        queued = b"".join(call.args[0] for call in streamer.add_audio_chunk.call_args_list)
        self.assertEqual(queued, adapter.return_pcm_audio(wave))
        self.assertEqual(sample_rate, audio8_tts.SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()

import inspect
import io
import sys
import tempfile
import types
import unittest
import wave
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TTS_RUNTIME_PARENT = PROJECT_ROOT / "Models" / "TTS"
if str(TTS_RUNTIME_PARENT) not in sys.path:
    sys.path.insert(0, str(TTS_RUNTIME_PARENT))

from Models.TTS import index_tts
from Models.TTS.text_segmentation import chunk_text, parse_voice_tagged_text


class _FakeAudioStreamer:
    def __init__(self):
        self.chunks = []

    def add_audio_chunk(self, chunk):
        self.chunks.append(chunk)


class _FakeSegmentModel:
    def __init__(self):
        self.arguments = []
        self.generated = []

    def infer(self, **kwargs):
        self.arguments.append(kwargs)
        scale = 3276.7 * len(self.arguments)
        audio = torch.tensor([[scale, -scale]], dtype=torch.float32)
        self.generated.append(audio)
        return index_tts.SAMPLE_RATE, audio


class IndexTTSAdapterTests(unittest.TestCase):
    def test_hosted_archive_manifest_is_offline_and_complete(self):
        entry = index_tts.TTS_MODEL_LINKS[index_tts.DEFAULT_MODEL]
        self.assertEqual(entry["checksum"], "0" * 64)
        self.assertEqual(entry["path"], "IndexTTS-2.5")
        self.assertEqual(len(entry["urls"]), 3)
        self.assertTrue(all("huggingface" not in url.lower() for url in entry["urls"]))

        required = {
            "LICENSE",
            "README.md",
            "codec.pth",
            "config.yaml",
            "feat1.pt",
            "feat2.pt",
            "gpt.pth",
            "multilingual_zh_ja_yue_char_del.tiktoken",
            "s2mel.pth",
            "wav2vec2bert_stats.pt",
            r"hf_cache\w2v-bert-2.0\config.json",
            r"hf_cache\w2v-bert-2.0\model.safetensors",
            r"hf_cache\w2v-bert-2.0\preprocessor_config.json",
            r"hf_cache\semantic_codec_model.safetensors",
            r"hf_cache\campplus_cn_common.bin",
            r"hf_cache\bigvgan\config.json",
            r"hf_cache\bigvgan\bigvgan_generator.pt",
        }
        self.assertEqual(set(entry["file_checksums"]), required)
        self.assertTrue(all(len(value) == 64 for value in entry["file_checksums"].values()))

    def test_vendored_runtime_contains_no_remote_model_downloader(self):
        forbidden = ("hf_hub_download", "snapshot_download", "modelscope.pipelines")
        runtime_files = [PROJECT_ROOT / "Models" / "TTS" / "index_tts.py"]
        runtime_files.extend((TTS_RUNTIME_PARENT / "indextts").rglob("*.py"))
        combined_source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
        for symbol in forbidden:
            self.assertNotIn(symbol, combined_source)

    def test_placeholder_archive_hash_prevents_network_request(self):
        adapter = object.__new__(index_tts.IndexTTS)
        adapter.download_state = {"is_downloading": False}
        download = mock.Mock()
        fake_downloader = types.SimpleNamespace(
            model_needs_download=mock.Mock(return_value=True),
            download_model=download,
        )
        with mock.patch.dict(sys.modules, {"downloader": fake_downloader}):
            with self.assertRaisesRegex(RuntimeError, "archive is not ready"):
                adapter.download_model()
        download.assert_not_called()

    def test_wave_tensor_normalizes_upstream_int16_scale(self):
        wave = index_tts.IndexTTS._wave_tensor(torch.tensor([[-32767.0, 0.0, 32767.0]]))
        self.assertEqual(tuple(wave.shape), (1, 3))
        torch.testing.assert_close(wave, torch.tensor([[-1.0, 0.0, 1.0]]))

    def test_wav_export_is_standard_pcm16(self):
        adapter = object.__new__(index_tts.IndexTTS)
        adapter.sample_rate = index_tts.SAMPLE_RATE
        audio = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]], dtype=torch.float32)

        wav_bytes = adapter.return_wav_file_binary(audio, sample_rate=16000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnframes(), 5)
            samples = np.frombuffer(wav_file.readframes(5), dtype="<i2")
        np.testing.assert_array_equal(
            samples,
            np.array([-32767, -16384, 0, 16384, 32767], dtype=np.int16),
        )

    def test_chatterbox_chunking_uses_shorter_streaming_target(self):
        text = (
            "IndexTTS streaming should begin early. The second sentence should be generated afterward. "
            "The third sentence provides enough text to see whether punctuation causes a split. "
            "A fourth sentence makes this representative of a longer paragraph sent through TTS."
        )

        self.assertEqual(len(chunk_text(text, goal_length=200)), 1)
        chunks = chunk_text(text, goal_length=120)
        self.assertEqual([len(chunk) for chunk in chunks], [89, 82, 83])

    def test_voice_tags_only_switch_at_line_starts(self):
        sections = parse_voice_tagged_text(
            "Narration [not_a_tag].\n[alice] Hello there.\nStill Alice.\n[bob]\nAnd now Bob."
        )
        self.assertEqual(
            sections,
            [
                ("main", "Narration [not_a_tag]."),
                ("alice", "Hello there.\nStill Alice."),
                ("bob", "And now Bob."),
            ],
        )

    def test_segment_streaming_chunks_text_switches_voices_and_preserves_playback(self):
        adapter = object.__new__(index_tts.IndexTTS)
        adapter.generation_lock = __import__("threading").RLock()
        adapter.stop_event = __import__("threading").Event()
        adapter.sample_rate = index_tts.SAMPLE_RATE
        adapter.last_generation = {"audio": None, "sample_rate": None}
        adapter.audio_streamer = _FakeAudioStreamer()
        adapter.model = _FakeSegmentModel()
        adapter.special_settings = {
            **index_tts.IndexTTS.special_settings_defaults,
            "streaming_segment_goal_length": 120,
            "pause_between_segments_ms": 2,
            "pause_between_voice_change_ms": 3,
        }
        adapter.load = lambda: None
        adapter._seed_generation = mock.Mock()
        adapter._resolve_voice = lambda _: "main.wav"
        adapter.update_voices = lambda: [
            {"name": "alice", "audio_filename": "alice.wav"},
            {"name": "bob", "audio_filename": "bob.wav"},
        ]
        adapter._inference_kwargs = lambda _: {"lang": "en"}
        adapter.init_audio_stream_playback = lambda: None
        adapter._finish_audio = lambda audio, **_: index_tts.IndexTTS._wave_tensor(audio)

        wave, sample_rate = adapter.tts_streaming(
            "[alice] Hello from Alice.\n[bob] Hello from Bob."
        )

        expected = torch.cat(
            [
                index_tts.IndexTTS._wave_tensor(adapter.model.generated[0]),
                torch.zeros((1, int(index_tts.SAMPLE_RATE * 0.003))),
                index_tts.IndexTTS._wave_tensor(adapter.model.generated[1]),
            ],
            dim=-1,
        )
        self.assertEqual(sample_rate, index_tts.SAMPLE_RATE)
        torch.testing.assert_close(wave, expected)
        sent = b"".join(adapter.audio_streamer.chunks)
        self.assertEqual(sent, adapter.return_pcm_audio(expected))
        self.assertEqual(
            [call["spk_audio_prompt"] for call in adapter.model.arguments],
            ["alice.wav", "bob.wav"],
        )
        self.assertTrue(all(not call["stream_return"] for call in adapter.model.arguments))
        adapter._seed_generation.assert_called_once_with()

    def test_non_streaming_voice_tags_are_kept_in_wav_generation(self):
        adapter = object.__new__(index_tts.IndexTTS)
        adapter.generation_lock = __import__("threading").RLock()
        adapter.stop_event = __import__("threading").Event()
        adapter.sample_rate = index_tts.SAMPLE_RATE
        adapter.last_generation = {"audio": None, "sample_rate": None}
        adapter.model = _FakeSegmentModel()
        adapter.special_settings = {
            **index_tts.IndexTTS.special_settings_defaults,
            "pause_between_voice_change_ms": 4,
        }
        adapter.load = lambda: None
        adapter._seed_generation = mock.Mock()
        adapter._resolve_voice = lambda _: "main.wav"
        adapter.update_voices = lambda: [
            {"name": "alice", "audio_filename": "alice.wav"},
            {"name": "bob", "audio_filename": "bob.wav"},
        ]
        adapter._inference_kwargs = lambda _: {"lang": "en"}
        adapter._finish_audio = lambda audio, **_: index_tts.IndexTTS._wave_tensor(audio)

        wave, sample_rate = adapter.tts("[alice] First.\n[bob] Second.")

        self.assertEqual(sample_rate, index_tts.SAMPLE_RATE)
        self.assertEqual(
            [call["spk_audio_prompt"] for call in adapter.model.arguments],
            ["alice.wav", "bob.wav"],
        )
        expected_length = 2 + int(index_tts.SAMPLE_RATE * 0.004) + 2
        self.assertEqual(wave.shape[-1], expected_length)
        adapter._seed_generation.assert_called_once_with()


class IndexTTSTransformersCompatibilityTests(unittest.TestCase):
    def test_reference_conditioning_lru_reuses_preprocessing_and_tracks_file_changes(self):
        from indextts import infer_v2_5

        runtime = object.__new__(infer_v2_5.IndexTTS2)
        runtime.device = "cpu"
        runtime.speaker_conditioning_cache = OrderedDict()
        runtime.emotion_conditioning_cache = OrderedDict()
        runtime._load_and_cut_audio = mock.Mock(
            return_value=(torch.zeros((1, 320), dtype=torch.float32), 16000)
        )
        runtime._resample_audio = mock.Mock(side_effect=lambda audio, *_: audio)
        runtime.extract_features = mock.Mock(
            return_value={
                "input_features": torch.zeros((1, 4, 3)),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            }
        )
        runtime.get_emb = mock.Mock(return_value=torch.ones((1, 4, 3)))
        runtime.mel_fn = mock.Mock(return_value=torch.ones((1, 80, 5)))
        runtime.campplus_model = mock.Mock(return_value=torch.ones((1, 192)))
        length_regulator = mock.Mock(return_value=(torch.ones((1, 5, 4)),))
        runtime.s2mel = types.SimpleNamespace(
            models={"length_regulator": length_regulator}
        )

        with mock.patch.object(
            infer_v2_5.torchaudio.compliance.kaldi,
            "fbank",
            return_value=torch.ones((5, 80)),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                first_path = Path(temp_dir) / "first.wav"
                second_path = Path(temp_dir) / "second.wav"
                first_path.write_bytes(b"first")
                second_path.write_bytes(b"second")

                first = runtime._prepare_speaker_conditioning(first_path)
                runtime.emotion_conditioning_cache.clear()
                default_emotion = runtime._prepare_emotion_conditioning(
                    first_path, speaker_conditioning=first
                )
                second = runtime._prepare_speaker_conditioning(second_path)
                first_again = runtime._prepare_speaker_conditioning(first_path)

                self.assertIs(first, first_again)
                self.assertIs(default_emotion, first["spk_cond_emb"])
                self.assertIsNot(first, second)
                self.assertEqual(runtime._load_and_cut_audio.call_count, 2)
                self.assertEqual(runtime.get_emb.call_count, 2)
                self.assertEqual(length_regulator.call_count, 2)

                first_path.write_bytes(b"first changed")
                changed = runtime._prepare_speaker_conditioning(first_path)

        self.assertIsNot(first, changed)
        self.assertEqual(runtime._load_and_cut_audio.call_count, 3)
        self.assertEqual(runtime.get_emb.call_count, 3)

    def test_reference_conditioning_cache_is_bounded_lru(self):
        from indextts.infer_v2_5 import IndexTTS2

        runtime = object.__new__(IndexTTS2)
        runtime.REFERENCE_CACHE_MAX_ENTRIES = 2
        cache = OrderedDict()
        runtime._lru_put(cache, "a", 1)
        runtime._lru_put(cache, "b", 2)
        self.assertEqual(runtime._lru_get(cache, "a"), 1)
        runtime._lru_put(cache, "c", 3)

        self.assertEqual(list(cache), ["a", "c"])
        self.assertNotIn("b", cache)

    def test_pinned_wetext_normalizer_loads_both_languages(self):
        from indextts.utils.front import TextNormalizer

        normalizer = TextNormalizer()
        normalizer.load()
        self.assertIn("two", normalizer.normalize("I have 2 apples."))
        chinese = normalizer.normalize("\u6211\u67092\u4e2a\u82f9\u679c\u3002")
        self.assertNotIn("2", chinese)
        self.assertIn("\u82f9\u679c", chinese)

    def test_bigvgan_loader_accepts_current_hub_mixin_arguments(self):
        from indextts.s2mel.modules.bigvgan.bigvgan import BigVGAN

        parameters = inspect.signature(BigVGAN._from_pretrained).parameters
        self.assertIsNone(parameters["proxies"].default)
        self.assertFalse(parameters["resume_download"].default)
        self.assertFalse(parameters["local_files_only"].default)
        with self.assertRaisesRegex(FileNotFoundError, "verified local"):
            BigVGAN._from_pretrained(model_id="not-a-local-model-directory")

    def test_current_transformers_generation_supports_greedy_and_beam_cache(self):
        from torch import nn
        from transformers import DynamicCache, GPT2Config, GPT2Model

        from indextts.gpt.model_v2 import GPT2InferenceModel, LearnedPositionEmbeddings

        config = GPT2Config(
            vocab_size=16,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_cache=True,
        )
        model = GPT2InferenceModel(
            config,
            GPT2Model(config),
            LearnedPositionEmbeddings(16, 8),
            nn.Embedding(16, 8),
            nn.LayerNorm(8),
            nn.Linear(8, 16),
            kv_cache=True,
        ).eval()
        model.store_mel_emb(torch.randn(1, 2, 8))
        prompt = torch.tensor([[0, 0, 1]])
        mask = torch.ones_like(prompt)

        empty_layer_cache = DynamicCache(config=config)
        self.assertTrue(empty_layer_cache)
        self.assertEqual(empty_layer_cache.get_seq_length(), 0)
        prepared = model.prepare_inputs_for_generation(
            prompt,
            past_key_values=empty_layer_cache,
            attention_mask=mask,
            use_cache=True,
        )
        self.assertEqual(prepared["input_ids"].shape, prompt.shape)

        greedy = model.generate(prompt, attention_mask=mask, max_new_tokens=2, num_beams=1)
        beams = model.generate(prompt, attention_mask=mask, max_new_tokens=2, num_beams=3)

        self.assertGreater(greedy.shape[-1], prompt.shape[-1])
        self.assertGreater(beams.shape[-1], prompt.shape[-1])


if __name__ == "__main__":
    unittest.main()

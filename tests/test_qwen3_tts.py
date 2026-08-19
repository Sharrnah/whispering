import io
import sys
import tempfile
import threading
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from Models.TTS import qwen3_tts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeAudioStreamer:
    def __init__(self):
        self.chunks = []

    def add_audio_chunk(self, chunk):
        self.chunks.append(chunk)


class _FakeSpeechTokenizer:
    def __init__(self, samples_per_frame=4):
        self.samples_per_frame = samples_per_frame
        self.decode_inputs = []

    def get_decode_upsample_rate(self):
        return self.samples_per_frame

    def decode(self, encoded):
        codes = encoded[0]["audio_codes"].detach().cpu()
        self.decode_inputs.append(codes.clone())
        # Represent each codec frame by four identical samples.  This makes it
        # easy to prove that context is decoded but never emitted a second time.
        audio = np.repeat(
            codes[:, 0].numpy().astype(np.float32) / 10.0,
            self.samples_per_frame,
        )
        return [audio], qwen3_tts.SAMPLE_RATE


class _FakeBatchSpeechTokenizer(_FakeSpeechTokenizer):
    def decode(self, encoded):
        codes_per_lane = [item["audio_codes"].detach().cpu() for item in encoded]
        self.decode_inputs.append([codes.clone() for codes in codes_per_lane])
        wavs = [
            np.repeat(
                codes[:, 0].numpy().astype(np.float32) / 10.0,
                self.samples_per_frame,
            )
            for codes in codes_per_lane
        ]
        return wavs, qwen3_tts.SAMPLE_RATE


class Qwen3TTSAdapterTests(unittest.TestCase):
    def test_all_feature_checkpoints_have_hosted_manifests(self):
        expected_models = {
            "Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
        self.assertEqual(qwen3_tts.MODEL_LIST_FLAT, expected_models)
        for model_name in expected_models | {qwen3_tts.TOKENIZER_MODEL}:
            entry = qwen3_tts.TTS_MODEL_LINKS[model_name]
            self.assertEqual(entry["checksum"], "0" * 64)
            self.assertEqual(len(entry["urls"]), 3)
            self.assertTrue(all("huggingface" not in url.lower() for url in entry["urls"]))
            self.assertTrue(entry["source_revision"])
            self.assertTrue(all(len(value) == 64 for value in entry["file_checksums"].values()))

        tokenizer_files = set(qwen3_tts.TTS_MODEL_LINKS[qwen3_tts.TOKENIZER_MODEL]["file_checksums"])
        self.assertEqual(
            tokenizer_files,
            {"config.json", "configuration.json", "model.safetensors", "preprocessor_config.json"},
        )
        for model_name in expected_models:
            self.assertNotIn("speech_tokenizer/model.safetensors", qwen3_tts.TTS_MODEL_LINKS[model_name]["file_checksums"])

    def test_vendored_runtime_cannot_download_models_from_hugging_face(self):
        forbidden = ("snapshot_download", "hf_hub_download", "download_weights_from_hf_specific")
        runtime_files = [PROJECT_ROOT / "Models" / "TTS" / "qwen3_tts.py"]
        runtime_files.extend((PROJECT_ROOT / "Models" / "TTS" / "qwen3_tts_runtime").rglob("*.py"))
        combined_source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
        for symbol in forbidden:
            self.assertNotIn(symbol, combined_source)

    def test_transformers_5_load_restores_rotary_frequency_buffers(self):
        from Models.TTS.qwen3_tts_runtime.core.models import modeling_qwen3_tts

        config = types.SimpleNamespace(
            rope_scaling=None,
            max_position_embeddings=32768,
            rope_theta=1_000_000.0,
            head_dim=128,
            hidden_size=1024,
            num_attention_heads=16,
        )
        rotary = modeling_qwen3_tts.Qwen3TTSTalkerRotaryEmbedding(config)
        rotary.inv_freq.zero_()
        rotary.original_inv_freq.zero_()

        modeling_qwen3_tts._restore_rotary_embedding_buffers(torch.nn.Sequential(rotary))

        expected, _ = modeling_qwen3_tts._compute_default_rope_parameters(config)
        torch.testing.assert_close(rotary.inv_freq, expected)
        torch.testing.assert_close(rotary.original_inv_freq, expected)
        self.assertEqual(float(rotary.inv_freq[0]), 1.0)

    def test_placeholder_archive_hash_prevents_network_request(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.download_state = {"is_downloading": False}
        download = mock.Mock()
        fake_downloader = types.SimpleNamespace(
            model_needs_download=mock.Mock(return_value=True),
            download_model=download,
        )
        with mock.patch.dict(sys.modules, {"downloader": fake_downloader}):
            with self.assertRaisesRegex(RuntimeError, "archive .* is not ready"):
                adapter.download_model(qwen3_tts.DEFAULT_MODEL)
        download.assert_not_called()

    def test_generation_defaults_match_official_examples(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.special_settings = dict(qwen3_tts.Qwen3TTS.special_settings_defaults)
        adapter.compute_device = torch.device("cpu")
        adapter.stop_event = threading.Event()

        generation = adapter._generation_kwargs()

        self.assertTrue(generation["do_sample"])
        self.assertEqual(generation["temperature"], 0.9)
        self.assertEqual(generation["top_k"], 50)
        self.assertEqual(generation["top_p"], 1.0)
        self.assertEqual(generation["repetition_penalty"], 1.05)
        self.assertTrue(generation["subtalker_dosample"])
        self.assertEqual(generation["subtalker_temperature"], 0.9)
        self.assertEqual(generation["subtalker_top_k"], 50)
        self.assertEqual(generation["subtalker_top_p"], 1.0)
        self.assertEqual(generation["max_new_tokens"], 2048)
        self.assertTrue(generation["use_cache"])

    def test_unstable_half_precision_uses_a_safe_effective_precision(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.compute_device = torch.device("cuda")
        cases = (
            ("auto", True, "bfloat16"),
            ("auto", False, "float32"),
            ("float16", True, "bfloat16"),
            ("float16", False, "float32"),
            ("bfloat16", False, "float32"),
            ("float32", True, "float32"),
        )
        for requested, bf16_supported, expected in cases:
            with self.subTest(
                requested=requested,
                bf16_supported=bf16_supported,
            ):
                adapter.special_settings = {"precision": requested}
                with mock.patch.object(
                    torch.cuda,
                    "is_bf16_supported",
                    return_value=bf16_supported,
                ), mock.patch("builtins.print"):
                    self.assertEqual(adapter._effective_precision(), expected)

    def test_model_selection_accepts_profile_and_ui_display_forms(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        values = [
            ["Voice design", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            "Qwen3-TTS-12Hz-0.6B-CustomVoice (Built-in voices and control)",
        ]
        with mock.patch.object(qwen3_tts.settings, "GetOption", side_effect=lambda _: values.pop(0)):
            with mock.patch.object(qwen3_tts.settings, "SetOption") as set_option:
                self.assertEqual(adapter._get_model_name(), "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
                self.assertEqual(adapter._get_model_name(), "Qwen3-TTS-12Hz-0.6B-CustomVoice")
        set_option.assert_called_once_with(
            "tts_model",
            ["Built-in voices and control", "Qwen3-TTS-12Hz-0.6B-CustomVoice"],
        )

    def test_clone_prompt_uses_sidecar_transcript_and_is_cached(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.special_settings = {"clone_mode": "auto", "reference_text": ""}
        adapter.voice_prompt_cache = {}
        adapter.model = mock.Mock()
        adapter.model.create_voice_clone_prompt.return_value = [object()]
        with tempfile.TemporaryDirectory() as temp_dir:
            voice = Path(temp_dir) / "voice.wav"
            voice.write_bytes(b"not decoded in this unit test")
            voice.with_suffix(".txt").write_text("Exact reference words.", encoding="utf-8")
            adapter._resolve_voice = lambda _: voice

            first = adapter._clone_prompt(None)
            second = adapter._clone_prompt(None)

        self.assertIs(first, second)
        adapter.model.create_voice_clone_prompt.assert_called_once_with(
            ref_audio=str(voice),
            ref_text="Exact reference words.",
            x_vector_only_mode=False,
        )

    def test_clone_prompt_cache_is_metadata_aware_and_keeps_recent_voices(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.special_settings = {"clone_mode": "x_vector", "reference_text": ""}
        adapter.voice_prompt_cache = {}
        adapter.VOICE_PROMPT_CACHE_MAX_ENTRIES = 2
        adapter.model = mock.Mock()
        adapter.model.create_voice_clone_prompt.side_effect = lambda **kwargs: [
            kwargs["ref_audio"]
        ]
        adapter._resolve_voice = lambda path: Path(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.wav"
            second = Path(temp_dir) / "second.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            adapter._clone_prompt(first)
            adapter._clone_prompt(second)
            adapter._clone_prompt(first)
            self.assertEqual(adapter.model.create_voice_clone_prompt.call_count, 2)

            first.write_bytes(b"first changed")
            adapter._clone_prompt(first)
            self.assertEqual(adapter.model.create_voice_clone_prompt.call_count, 3)
            self.assertEqual(len(adapter.voice_prompt_cache), 2)

    def test_x_vector_prompt_skips_unused_reference_codec_encoding(self):
        from Models.TTS.qwen3_tts_runtime.inference.qwen3_tts_model import Qwen3TTSModel

        runtime = object.__new__(Qwen3TTSModel)
        speech_tokenizer = mock.Mock()
        runtime.model = types.SimpleNamespace(
            tts_model_type="base",
            tokenizer_type="qwen3_tts_tokenizer_12hz",
            tts_model_size="0.6B",
            speech_tokenizer=speech_tokenizer,
            speaker_encoder_sample_rate=24000,
            extract_speaker_embedding=mock.Mock(return_value=torch.ones(16)),
        )
        runtime._normalize_audio_inputs = mock.Mock(
            return_value=[(np.zeros(240, dtype=np.float32), 24000)]
        )

        prompt = runtime.create_voice_clone_prompt(
            ref_audio="voice.wav",
            ref_text=None,
            x_vector_only_mode=True,
        )

        speech_tokenizer.encode.assert_not_called()
        self.assertIsNone(prompt[0].ref_code)
        self.assertTrue(prompt[0].x_vector_only_mode)

    def test_mixed_clone_prompt_encodes_only_icl_references(self):
        from Models.TTS.qwen3_tts_runtime.inference.qwen3_tts_model import Qwen3TTSModel

        runtime = object.__new__(Qwen3TTSModel)
        speech_tokenizer = mock.Mock()
        reference_code = torch.ones((3, 16), dtype=torch.long)
        speech_tokenizer.encode.return_value = types.SimpleNamespace(
            audio_codes=[reference_code]
        )
        runtime.model = types.SimpleNamespace(
            tts_model_type="base",
            tokenizer_type="qwen3_tts_tokenizer_12hz",
            tts_model_size="0.6B",
            speech_tokenizer=speech_tokenizer,
            speaker_encoder_sample_rate=24000,
            extract_speaker_embedding=mock.Mock(return_value=torch.ones(16)),
        )
        x_vector_audio = np.zeros(120, dtype=np.float32)
        icl_audio = np.ones(120, dtype=np.float32)
        runtime._normalize_audio_inputs = mock.Mock(
            return_value=[(x_vector_audio, 24000), (icl_audio, 24000)]
        )

        prompt = runtime.create_voice_clone_prompt(
            ref_audio=["x-vector.wav", "icl.wav"],
            ref_text=[None, "Exact words."],
            x_vector_only_mode=[True, False],
        )

        encoded_audio, = speech_tokenizer.encode.call_args.args
        self.assertEqual(len(encoded_audio), 1)
        self.assertIs(encoded_audio[0], icl_audio)
        self.assertEqual(speech_tokenizer.encode.call_args.kwargs, {"sr": 24000})
        self.assertIsNone(prompt[0].ref_code)
        self.assertIs(prompt[1].ref_code, reference_code)

    def test_segment_streaming_sends_and_returns_identical_audio(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.generation_lock = threading.RLock()
        adapter.stop_event = threading.Event()
        adapter.sample_rate = qwen3_tts.SAMPLE_RATE
        adapter.last_generation = {"audio": None, "sample_rate": None, "text": ""}
        adapter.audio_streamer = _FakeAudioStreamer()
        adapter.load = lambda: None
        adapter.init_audio_stream_playback = lambda: None
        adapter._streaming_segments = lambda _: ["one", "two"]
        generated = iter((torch.tensor([[0.1, -0.1]]), torch.tensor([[0.2, -0.2]])))
        adapter._generate_one = lambda *_: next(generated)
        adapter._finish_audio = lambda audio, **_: audio
        adapter._silence = lambda: torch.tensor([[0.0]])

        result, sample_rate = adapter.tts_streaming_segments("one two")

        expected = torch.tensor([[0.1, -0.1, 0.0, 0.2, -0.2]])
        self.assertEqual(sample_rate, qwen3_tts.SAMPLE_RATE)
        torch.testing.assert_close(result, expected)
        self.assertEqual(b"".join(adapter.audio_streamer.chunks), adapter.return_pcm_audio(expected))

    def test_codec_streamer_emits_only_new_audio_with_reference_context(self):
        tokenizer = _FakeSpeechTokenizer()
        emitted = []
        streamer = qwen3_tts._QwenCodecAudioStreamer(
            tokenizer,
            lambda audio, sample_rate: emitted.append((audio.copy(), sample_rate)),
            eos_token_id=99,
            frames_per_chunk=2,
            context_frames=2,
        )
        streamer.set_prefix_codes(torch.tensor([[7] * 16, [8] * 16]))

        streamer.put(torch.tensor([[1] * 16]))
        self.assertEqual(emitted, [])
        streamer.put(torch.tensor([[2] * 16]))
        streamer.put(torch.tensor([[3] * 16]))
        streamer.put(torch.tensor([[99] * 16]))
        streamer.end()

        self.assertEqual(len(emitted), 2)
        np.testing.assert_allclose(emitted[0][0], np.repeat([0.1, 0.2], 4))
        np.testing.assert_allclose(emitted[1][0], np.repeat([0.3], 4))
        self.assertTrue(all(sample_rate == qwen3_tts.SAMPLE_RATE for _, sample_rate in emitted))
        np.testing.assert_array_equal(tokenizer.decode_inputs[0][:, 0], [7, 8, 1, 2])
        np.testing.assert_array_equal(tokenizer.decode_inputs[1][:, 0], [1, 2, 3])

    def test_batch_codec_streamer_routes_lanes_and_finishes_them_independently(self):
        tokenizer = _FakeBatchSpeechTokenizer()
        emitted = {0: [], 1: [], 2: []}
        ended = []
        streamer = qwen3_tts._QwenBatchCodecAudioStreamer(
            tokenizer,
            lambda lane, audio, _sample_rate: emitted[lane].append(audio.copy()),
            ended.append,
            eos_token_id=99,
            pad_token_id=98,
            batch_size=3,
            frames_per_chunk=2,
            context_frames=1,
        )
        streamer.set_prefix_codes(
            [
                torch.tensor([[7] * 16]),
                torch.tensor([[8] * 16]),
                torch.tensor([[9] * 16]),
            ]
        )

        for rows in (
            (1, 3, 5),
            (2, 4, 6),
            (99, 7, 8),
            (98, 9, 99),
        ):
            streamer.put(torch.tensor([[value] * 16 for value in rows]))
        streamer.end()

        np.testing.assert_allclose(np.concatenate(emitted[0]), np.repeat([0.1, 0.2], 4))
        np.testing.assert_allclose(
            np.concatenate(emitted[1]),
            np.repeat([0.3, 0.4, 0.7, 0.9], 4),
        )
        np.testing.assert_allclose(
            np.concatenate(emitted[2]),
            np.repeat([0.5, 0.6, 0.8], 4),
        )
        self.assertEqual(sorted(ended), [0, 1, 2])
        self.assertEqual(len(ended), 3)
        self.assertEqual(len(tokenizer.decode_inputs[0]), 3)
        np.testing.assert_array_equal(tokenizer.decode_inputs[0][0][:, 0], [7, 1, 2])
        np.testing.assert_array_equal(tokenizer.decode_inputs[0][1][:, 0], [8, 3, 4])
        np.testing.assert_array_equal(tokenizer.decode_inputs[0][2][:, 0], [9, 5, 6])

    def test_codec_streaming_queues_and_returns_the_same_audio(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.generation_lock = threading.RLock()
        adapter.stop_event = threading.Event()
        adapter.sample_rate = qwen3_tts.SAMPLE_RATE
        adapter.last_generation = {"audio": None, "sample_rate": None, "text": ""}
        adapter.audio_streamer = _FakeAudioStreamer()
        adapter.special_settings = {
            **qwen3_tts.Qwen3TTS.special_settings_defaults,
            "streaming_codec_frames": 2,
            "streaming_decoder_context_frames": 2,
        }
        tokenizer = _FakeSpeechTokenizer()
        adapter.model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                speech_tokenizer=tokenizer,
                config=types.SimpleNamespace(
                    talker_config=types.SimpleNamespace(codec_eos_token_id=99)
                ),
            )
        )
        adapter.load = lambda: None
        adapter.init_audio_stream_playback = lambda: None

        queued_during_generation = []

        def generate(_text, _ref_audio, codec_streamer=None):
            for value in (1, 2, 3):
                codec_streamer.put(torch.tensor([[value] * 16]))
                queued_during_generation.append(len(adapter.audio_streamer.chunks))
            codec_streamer.end()
            return torch.tensor([[9.0]])

        adapter._generate_one = generate
        with mock.patch.object(qwen3_tts.settings, "GetOption", return_value=1.0):
            result, sample_rate = adapter.tts_streaming_codec("hello")

        expected = torch.from_numpy(
            np.repeat([[0.1, 0.2, 0.3]], 4, axis=1).astype(np.float32)
        )
        self.assertEqual(sample_rate, qwen3_tts.SAMPLE_RATE)
        torch.testing.assert_close(result, expected)
        self.assertEqual(queued_during_generation, [0, 0, 0])
        self.assertEqual(len(adapter.audio_streamer.chunks), 1)
        self.assertEqual(b"".join(adapter.audio_streamer.chunks), adapter.return_pcm_audio(expected))

    def test_fixed_codec_buffer_mode_preserves_low_latency_queueing(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.generation_lock = threading.RLock()
        adapter.stop_event = threading.Event()
        adapter.sample_rate = qwen3_tts.SAMPLE_RATE
        adapter.last_generation = {"audio": None, "sample_rate": None, "text": ""}
        adapter.audio_streamer = _FakeAudioStreamer()
        adapter.special_settings = {
            **qwen3_tts.Qwen3TTS.special_settings_defaults,
            "streaming_buffer_mode": "fixed",
            "streaming_codec_frames": 2,
        }
        tokenizer = _FakeSpeechTokenizer()
        adapter.model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                speech_tokenizer=tokenizer,
                config=types.SimpleNamespace(
                    talker_config=types.SimpleNamespace(codec_eos_token_id=99)
                ),
            )
        )
        adapter.load = lambda: None
        adapter.init_audio_stream_playback = lambda: None
        queued_during_generation = []

        def generate(_text, _ref_audio, codec_streamer=None):
            codec_streamer.put(torch.tensor([[1] * 16]))
            codec_streamer.put(torch.tensor([[2] * 16]))
            queued_during_generation.append(len(adapter.audio_streamer.chunks))
            codec_streamer.end()
            return torch.zeros((1, 0))

        adapter._generate_one = generate
        with mock.patch.object(qwen3_tts.settings, "GetOption", return_value=0.0):
            adapter.tts_streaming_codec("hello")

        self.assertEqual(queued_during_generation, [1])

    def test_lookahead_groups_avoid_a_single_tail_when_possible(self):
        self.assertEqual(
            qwen3_tts.Qwen3TTS._lookahead_groups(["a", "b", "c", "d"], 3),
            [["a", "b"], ["c", "d"]],
        )
        self.assertEqual(
            qwen3_tts.Qwen3TTS._lookahead_groups(
                ["a", "b", "c", "d", "e", "f", "g"],
                3,
            ),
            [["a", "b", "c"], ["d", "e"], ["f", "g"]],
        )

    def test_parallel_lookahead_falls_back_for_an_indivisible_utterance(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.sample_rate = qwen3_tts.SAMPLE_RATE
        adapter._lookahead_segments = lambda _text: ["one short phrase"]
        expected = (torch.tensor([[0.25, -0.25]]), qwen3_tts.SAMPLE_RATE)
        adapter.tts_streaming_codec = mock.Mock(return_value=expected)

        result = adapter.tts_streaming_lookahead("one short phrase", "voice.wav")

        self.assertIs(result, expected)
        adapter.tts_streaming_codec.assert_called_once_with(
            "one short phrase",
            "voice.wav",
        )

    def test_parallel_lookahead_returns_and_queues_lanes_in_text_order(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.generation_lock = threading.RLock()
        adapter.stop_event = threading.Event()
        adapter.sample_rate = qwen3_tts.SAMPLE_RATE
        adapter.last_generation = {"audio": None, "sample_rate": None, "text": ""}
        adapter.audio_streamer = _FakeAudioStreamer()
        adapter.special_settings = {
            **qwen3_tts.Qwen3TTS.special_settings_defaults,
            "streaming_mode": "lookahead",
            "streaming_lookahead_batch_size": 3,
            "streaming_lookahead_pause_ms": 0,
        }
        adapter.streaming_rtf_history = {}
        tokenizer = _FakeBatchSpeechTokenizer()
        adapter.model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                speech_tokenizer=tokenizer,
                config=types.SimpleNamespace(
                    talker_config=types.SimpleNamespace(
                        codec_eos_token_id=99,
                        codec_pad_id=98,
                    )
                ),
            )
        )
        adapter.load = lambda: None
        adapter.init_audio_stream_playback = lambda: None
        adapter._lookahead_segments = lambda _text: ["one", "two", "three", "four"]
        values = {"one": 1, "two": 3, "three": 5, "four": 7}
        generated_groups = []

        def generate_batch(group, _ref_audio, codec_streamer=None):
            generated_groups.append(list(group))
            for offset in (0, 1):
                codec_streamer.put(
                    torch.tensor(
                        [[values[item] + offset] * 16 for item in group]
                    )
                )
            codec_streamer.end()
            return []

        adapter._generate_batch = generate_batch

        def get_option(key):
            return {
                "tts_volume": 1.0,
                "tts_streamed_min_play_time": 0.0,
                "tts_prosody_rate": "medium",
            }.get(key)

        with mock.patch.object(qwen3_tts.settings, "GetOption", side_effect=get_option):
            result, sample_rate = adapter.tts_streaming_lookahead("ignored")

        expected = torch.from_numpy(
            np.repeat([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 4, axis=1)
            .astype(np.float32)
        )
        self.assertEqual(sample_rate, qwen3_tts.SAMPLE_RATE)
        self.assertEqual(generated_groups, [["one", "two"], ["three", "four"]])
        torch.testing.assert_close(result, expected)
        self.assertEqual(
            b"".join(adapter.audio_streamer.chunks),
            adapter.return_pcm_audio(expected),
        )

    def test_adaptive_buffer_target_accounts_for_slower_than_realtime_generation(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.special_settings = {
            **qwen3_tts.Qwen3TTS.special_settings_defaults,
            "language": "en",
            "streaming_buffer_safety_ms": 500,
        }

        with mock.patch.object(qwen3_tts.settings, "GetOption", return_value=0.0):
            target, duration = adapter._adaptive_buffer_target(
                "one two three four five six seven eight.",
                expected_rtf=2.0,
            )

        self.assertGreater(duration, 3.5)
        self.assertAlmostEqual(target, min(duration, duration * 0.5 + 0.5))

    def test_codec_streaming_preserves_explicit_text_conditioning_mode(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.special_settings = {"model_text_mode": "full_text"}
        adapter.model = mock.Mock()
        adapter.model.generate_custom_voice.return_value = ([np.zeros(8, dtype=np.float32)], 24000)
        adapter._get_model_name = lambda: "Qwen3-TTS-12Hz-0.6B-CustomVoice"
        adapter._language = lambda: "English"
        adapter._generation_kwargs = lambda: {}
        adapter._speaker = lambda: "Ryan"
        adapter._instruction = lambda _mode: ""
        codec_streamer = object()

        adapter._generate_one("hello", codec_streamer=codec_streamer)

        call = adapter.model.generate_custom_voice.call_args
        self.assertTrue(call.kwargs["non_streaming_mode"])
        self.assertIs(call.kwargs["codec_streamer"], codec_streamer)
        self.assertFalse(call.kwargs["decode_audio"])

    def test_batch_generation_forwards_lists_to_controlled_voice_modes(self):
        for model_name, method_name in (
            ("Qwen3-TTS-12Hz-0.6B-CustomVoice", "generate_custom_voice"),
            ("Qwen3-TTS-12Hz-1.7B-VoiceDesign", "generate_voice_design"),
        ):
            with self.subTest(model_name=model_name):
                adapter = object.__new__(qwen3_tts.Qwen3TTS)
                adapter.special_settings = {"model_text_mode": "auto"}
                adapter.sample_rate = qwen3_tts.SAMPLE_RATE
                adapter.model = mock.Mock()
                getattr(adapter.model, method_name).return_value = (
                    [],
                    qwen3_tts.SAMPLE_RATE,
                )
                adapter._get_model_name = lambda value=model_name: value
                adapter._language = lambda: "English"
                adapter._generation_kwargs = lambda: {}
                adapter._speaker = lambda: "Ryan"
                adapter._instruction = lambda _mode: "Speak warmly."
                codec_streamer = object()

                result = adapter._generate_batch(
                    ["one", "two", "three"],
                    codec_streamer=codec_streamer,
                )

                self.assertEqual(result, [])
                call = getattr(adapter.model, method_name).call_args
                self.assertEqual(call.kwargs["text"], ["one", "two", "three"])
                self.assertEqual(
                    call.kwargs["language"],
                    ["English", "English", "English"],
                )
                self.assertEqual(
                    call.kwargs["instruct"],
                    ["Speak warmly."] * 3,
                )
                self.assertIs(call.kwargs["codec_streamer"], codec_streamer)
                self.assertFalse(call.kwargs["decode_audio"])
                if method_name == "generate_custom_voice":
                    self.assertEqual(call.kwargs["speaker"], ["Ryan"] * 3)

    def test_wav_export_is_pcm16(self):
        adapter = object.__new__(qwen3_tts.Qwen3TTS)
        adapter.sample_rate = qwen3_tts.SAMPLE_RATE
        audio = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
        wav_bytes = adapter.return_wav_file_binary(audio, 24000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 24000)
            samples = np.frombuffer(wav_file.readframes(3), dtype="<i2")
        np.testing.assert_array_equal(samples, np.array([-32767, 0, 32767], dtype=np.int16))


if __name__ == "__main__":
    unittest.main()

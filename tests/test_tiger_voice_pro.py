import ast
import inspect
import io
import queue
import threading
import time
import traceback
import types
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.io.wavfile import write as write_wav

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _simple_resample(samples, recorded_sample_rate, target_sample_rate, **_kwargs):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if recorded_sample_rate == target_sample_rate:
        return samples
    target_length = int(round(samples.size * target_sample_rate / recorded_sample_rate))
    old_positions = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    new_positions = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


def _plugin_harness_class():
    source = (PROJECT_ROOT / "Plugins" / "tiger-voice-pro_plugin.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    plugin_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TigerVoiceProPlugin"
    )
    selected_names = {
        "_normalize_realtime_chunk",
        "_log_realtime_backlog",
        "_enqueue_realtime_input",
        "_ensure_output_sola_state",
        "_apply_sola_crossfade",
        "_audio_to_mono_float32",
        "_restore_mono_shape",
        "_float_audio_as_dtype",
        "_float_audio_to_wav_bytes",
        "realtime_sts",
        "on_plugin_tts_after_audio_call",
    }
    methods = [
        node
        for node in plugin_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in selected_names
    ]
    harness = ast.ClassDef(
        name="TigerVoiceProHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {
        "io": io,
        "np": np,
        "queue": queue,
        "soundfile": soundfile,
        "threading": threading,
        "torch": torch,
        "write_wav": write_wav,
        "audio_tools": types.SimpleNamespace(resample_audio=_simple_resample),
    }
    exec(compile(module, "tiger-voice-pro-methods", "exec"), namespace)
    return namespace["TigerVoiceProHarness"]


def _compile_module_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


PLUGIN_CLASS = _plugin_harness_class()


def _plugin_harness():
    plugin = PLUGIN_CLASS.__new__(PLUGIN_CLASS)
    plugin.verbose = False
    plugin._rt_input_drop_count = 0
    plugin._rt_last_backlog_log_time = 0.0
    return plugin


def test_realtime_chunk_uses_explicit_channel_count():
    mono = np.arange(1024, dtype=np.int16)
    normalized_mono = PLUGIN_CLASS._normalize_realtime_chunk(
        mono.tobytes(), input_channels=1
    )
    assert normalized_mono.shape == (1024,)
    np.testing.assert_allclose(normalized_mono, mono.astype(np.float32) / 32768.0)

    stereo = np.tile(np.array([1200, -1200], dtype=np.int16), 512)
    normalized_stereo = PLUGIN_CLASS._normalize_realtime_chunk(
        stereo.tobytes(), input_channels=2
    )
    assert normalized_stereo.shape == (512,)
    np.testing.assert_allclose(normalized_stereo, 0.0)


def test_realtime_wrapper_serializes_callback_threads():
    plugin = _plugin_harness()
    plugin._rt_ingest_lock = threading.RLock()
    state_lock = threading.Lock()
    state = {"active": 0, "maximum": 0, "calls": 0}

    def fake_realtime_sts(_audio, _sample_rate, input_channels=1):
        assert input_channels == 1
        with state_lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.01)
        with state_lock:
            state["active"] -= 1
            state["calls"] += 1

    plugin._realtime_sts_serialized = fake_realtime_sts
    threads = [
        threading.Thread(target=plugin.realtime_sts, args=(b"pcm", 48000, 1))
        for _ in range(8)
    ]
    for worker in threads:
        worker.start()
    for worker in threads:
        worker.join()

    assert state["calls"] == 8
    assert state["maximum"] == 1


def test_recorder_forwards_actual_channel_count_to_realtime_plugin():
    calls = []

    class FakePlugin:
        @staticmethod
        def is_enabled(_default):
            return True

        @staticmethod
        def realtime_sts(audio, sample_rate, input_channels=1):
            calls.append((audio, sample_rate, input_channels))

    namespace = _compile_module_functions(
        PROJECT_ROOT / "audio_processing_recording.py",
        {"call_plugin_realtime_sts"},
        {"inspect": inspect, "traceback": traceback},
    )
    namespace["call_plugin_realtime_sts"](
        [FakePlugin()], b"pcm", 48000, input_channels=2
    )
    assert calls == [(b"pcm", 48000, 2)]


def test_recorder_keeps_legacy_realtime_plugin_signature_compatible():
    calls = []

    class LegacyPlugin:
        @staticmethod
        def is_enabled(_default):
            return True

        @staticmethod
        def realtime_sts(audio, sample_rate):
            calls.append((audio, sample_rate))

    namespace = _compile_module_functions(
        PROJECT_ROOT / "audio_processing_recording.py",
        {"call_plugin_realtime_sts"},
        {"inspect": inspect, "traceback": traceback},
    )
    plugin = LegacyPlugin()
    namespace["call_plugin_realtime_sts"](
        [plugin], b"pcm", 48000, input_channels=2
    )

    assert calls == [(b"pcm", 48000)]
    assert plugin._realtime_sts_accepts_input_channels is False


def test_successful_recorder_open_reports_requested_channels():
    class FakePyAudio:
        @staticmethod
        def open(**_kwargs):
            return object()

    class FakeProcessor:
        input_channel_num = None

        @staticmethod
        def callback(*_args):
            return None

    fake_pyaudio_module = types.SimpleNamespace(
        paInt16=8,
        PyAudio=lambda: FakePyAudio(),
    )
    namespace = _compile_module_functions(
        PROJECT_ROOT / "audio_tools.py",
        {"calculate_chunk_size", "start_recording_audio_stream"},
        {"pyaudio": fake_pyaudio_module},
    )

    processor = FakeProcessor()
    _stream, _needs_resample, _sample_rate, channels = (
        namespace["start_recording_audio_stream"](
            sample_rate=16000,
            channels=1,
            chunk=512,
            py_audio=FakePyAudio(),
            audio_processor=processor,
        )
    )
    assert channels == 1
    assert processor.input_channel_num == 1


def test_stereo_recorder_fallback_reports_two_channels():
    open_calls = []

    class FakePyAudio:
        @staticmethod
        def open(**kwargs):
            open_calls.append((kwargs["channels"], kwargs["rate"]))
            if len(open_calls) == 1:
                raise OSError("requested format is unavailable")
            return object()

        @staticmethod
        def get_device_info_by_index(_device_index):
            return {"defaultSampleRate": 48000, "maxInputChannels": 2}

    class FakeProcessor:
        input_channel_num = None
        needs_sample_rate_conversion = None
        recorded_sample_rate = None

        @staticmethod
        def callback(*_args):
            return None

    fake_pyaudio_module = types.SimpleNamespace(
        paInt16=8,
        PyAudio=lambda: FakePyAudio(),
    )
    namespace = _compile_module_functions(
        PROJECT_ROOT / "audio_tools.py",
        {"calculate_chunk_size", "start_recording_audio_stream"},
        {"pyaudio": fake_pyaudio_module},
    )

    processor = FakeProcessor()
    _stream, needs_resample, recorded_rate, channels = (
        namespace["start_recording_audio_stream"](
            device_index=7,
            sample_rate=16000,
            channels=1,
            chunk=512,
            py_audio=FakePyAudio(),
            audio_processor=processor,
        )
    )

    assert open_calls == [(1, 16000), (2, 48000)]
    assert needs_resample is True
    assert recorded_rate == 48000
    assert channels == 2
    assert processor.input_channel_num == 2


def test_full_input_queue_replaces_only_the_oldest_window():
    plugin = _plugin_harness()
    plugin.audio_input_queue = queue.Queue(maxsize=2)
    plugin.audio_input_queue.put_nowait("oldest")
    plugin.audio_input_queue.put_nowait("middle")

    assert plugin._enqueue_realtime_input("newest") is True
    assert plugin.audio_input_queue.get_nowait() == "middle"
    assert plugin.audio_input_queue.get_nowait() == "newest"
    assert plugin._rt_input_drop_count == 1


def test_sola_uses_model_output_sample_rate():
    plugin = _plugin_harness()
    plugin.rt_crossfade_time = 0.04
    plugin.rt_block_time = 0.5
    plugin.rt_sola_sample_rate = None
    plugin.rt_sola_buffer = None

    processed = np.linspace(-0.2, 0.2, int(1.05 * 24000), dtype=np.float32)
    output = plugin._apply_sola_crossfade(processed, sample_rate=24000)

    assert plugin.rt_output_crossfade_samples == 960
    assert plugin.rt_output_sola_search_samples == 240
    assert plugin.rt_sola_buffer.shape == (960,)
    assert output.shape == (12000,)


def test_tts_after_hook_decodes_wav_container_without_header_samples():
    plugin = _plugin_harness()
    plugin.model = object()
    plugin.is_enabled = lambda _default=False: True
    plugin.get_plugin_setting = lambda name, *_args: {
        "tts_enable": True,
        "voice": "target.wav",
    }[name]

    converted = np.linspace(-0.25, 0.25, 100, dtype=np.float32)
    captured = {}

    def fake_conversion(source_audio, sample_rate, reference):
        captured["source"] = source_audio
        captured["sample_rate"] = sample_rate
        captured["reference"] = reference
        buffer = io.BytesIO()
        write_wav(buffer, sample_rate, (converted * 32767.0).astype(np.int16))
        buffer.seek(0)
        return buffer, sample_rate

    plugin.do_conversion = fake_conversion
    original = torch.zeros((1, 100), dtype=torch.float32)
    result = plugin.on_plugin_tts_after_audio_call(
        {"audio": original, "sample_rate": 16000}
    )

    assert captured["source"].shape == (100,)
    assert captured["source"].dtype == np.float32
    assert captured["sample_rate"] == 16000
    assert captured["reference"] == "target.wav"
    assert result["audio"].shape == (1, 100)
    assert result["audio"].dtype == torch.float32
    np.testing.assert_allclose(
        result["audio"].numpy().reshape(-1), converted, atol=2.0 / 32768.0
    )

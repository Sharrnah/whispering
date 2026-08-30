import inspect
import queue
import threading
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np

import audio_processing_recording
import audioprocessor
import audio_routes
import audio_tools
from Models.STS.AudioEnhancer import (
    IncrementalAudioEnhancer,
    pcm16_bytes_to_float32,
)


class _RecordingEnhancer:
    def __init__(self):
        self.calls = []

    def enhance_audio(self, audio, **kwargs):
        samples = np.frombuffer(audio, dtype=np.int16)
        self.calls.append((samples.copy(), kwargs.copy()))
        return samples.copy()


class _BlockingEnhancer:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def enhance_audio(self, audio, **_kwargs):
        self.entered.set()
        self.release.wait(2.0)
        return np.frombuffer(audio, dtype=np.int16).copy()


class _TriggerSettings:
    values = {
        "mic_passthrough_routing": False,
        "stt_enabled": True,
        "phrase_time_limit": 0,
        "pause": 0.5,
        "energy": 100,
        "vad_confidence_threshold": 0.5,
        "vad_smart_turn_enabled": False,
        "silence_cutting_enabled": False,
        "normalize_enabled": False,
        "speaker_diarization": False,
        "denoise_audio": "noise_reduce",
        "denoise_audio_before_trigger": True,
        "denoise_strength": 1.0,
        "realtime": False,
    }

    @classmethod
    def GetOption(cls, name):
        return cls.values.get(name)


def test_capture_paths_do_not_call_the_denoiser_directly():
    callback_source = inspect.getsource(
        audio_processing_recording.AudioProcessor._callback_locked
    )
    legacy_recorder_source = (
        Path(__file__).resolve().parents[1] / "audioWhisper.py"
    ).read_text(encoding="utf-8")

    assert ".enhance_audio(" not in callback_source
    assert ".enhance_prefix(" not in callback_source
    assert "audio_enhancer.enhance_audio(" not in legacy_recorder_source


def test_pcm16_conversion_preserves_absolute_quiet_signal_level():
    samples = np.array([0, 1000, -1000], dtype=np.int16)
    converted = pcm16_bytes_to_float32(samples.tobytes())

    assert converted.dtype == np.float32
    np.testing.assert_allclose(
        converted,
        samples.astype(np.float32) / 32768.0,
    )
    assert np.max(np.abs(converted)) < 0.04


def test_recording_peak_uses_the_full_pcm16_amplitude_range():
    samples = np.array([-32768, 0, 32767], dtype=np.int16)

    _, peak = audio_processing_recording.process_audio_chunk(
        samples.tobytes(),
        sample_rate=16000,
        vad_model=None,
    )

    assert peak == 32768


def test_incremental_enhancer_processes_only_new_tail_and_bounded_context():
    enhancer = _RecordingEnhancer()
    stream = IncrementalAudioEnhancer(
        enhancer,
        sample_rate=1000,
        context_seconds=0.2,
        crossfade_ms=10,
    )
    first = np.arange(1000, dtype=np.int16)
    second = np.arange(1500, dtype=np.int16)

    first_output = stream.enhance_prefix(first, strength=0.8)
    second_output = stream.enhance_prefix(second, strength=0.8)
    repeated_output = stream.enhance_prefix(second, strength=0.8)

    assert [call[0].size for call in enhancer.calls] == [1000, 700]
    assert first_output.size == first.size
    assert second_output.size == second.size
    np.testing.assert_array_equal(repeated_output, second_output)


def test_incremental_enhancer_restarts_when_prefix_or_strength_changes():
    enhancer = _RecordingEnhancer()
    stream = IncrementalAudioEnhancer(enhancer, sample_rate=1000)
    original = np.arange(500, dtype=np.int16)
    changed = original.copy()
    changed[0] = 99

    stream.enhance_prefix(original, strength=0.5)
    stream.enhance_prefix(changed, strength=0.5)
    stream.enhance_prefix(changed, strength=0.75)

    assert [call[0].size for call in enhancer.calls] == [500, 500, 500]


def test_denoised_trigger_never_blocks_the_recording_callback():
    enhancer = _BlockingEnhancer()
    processor = audio_processing_recording.AudioProcessor(
        default_sample_rate=1000,
        recorded_sample_rate=1000,
        input_channel_num=1,
        plugins=[],
        audio_enhancer=enhancer,
        settings=_TriggerSettings(),
        enable_mic_passthrough=False,
    )
    audio = np.full(100, 1000, dtype=np.int16).tobytes()
    callback_result = []
    callback_thread = threading.Thread(
        target=lambda: callback_result.append(
            processor.callback(audio, 100, None, None)
        )
    )

    try:
        callback_thread.start()
        callback_thread.join(0.25)

        assert not callback_thread.is_alive()
        assert callback_result
        assert enhancer.entered.wait(1.0)
        assert not processor.start_rec_on_volume_threshold
    finally:
        enhancer.release.set()
        callback_thread.join(1.0)
        processor.close()


def test_final_denoising_is_queued_instead_of_running_on_callback():
    class _FinalSettings(_TriggerSettings):
        values = {
            **_TriggerSettings.values,
            "denoise_audio_before_trigger": False,
            "pause": 0.01,
            "vad_on_full_clip": False,
            "transcription_save_audio_dir": "",
        }

    enhancer = _BlockingEnhancer()
    audio_queue = queue.Queue()
    processor = audio_processing_recording.AudioProcessor(
        default_sample_rate=1000,
        recorded_sample_rate=1000,
        input_channel_num=1,
        plugins=[],
        audio_enhancer=enhancer,
        audio_queue=audio_queue,
        settings=_FinalSettings(),
        enable_mic_passthrough=False,
    )
    recorded = np.full(100, 1000, dtype=np.int16).tobytes()
    processor.frames = [recorded]
    processor.start_time = time.time() - 1.0

    try:
        processor.callback(recorded, 100, None, None)

        assert not enhancer.entered.is_set()
        queued = audio_queue.get_nowait()
        assert queued["final"] is True
        assert queued["denoise_pcm"] == recorded
        assert queued["run_final_audio_consumers"] is True
    finally:
        enhancer.release.set()
        processor.close()


def test_denoised_trigger_retains_onset_while_worker_is_running():
    enhancer = _BlockingEnhancer()
    processor = audio_processing_recording.AudioProcessor(
        default_sample_rate=1000,
        recorded_sample_rate=1000,
        input_channel_num=1,
        plugins=[],
        audio_enhancer=enhancer,
        settings=_TriggerSettings(),
        enable_mic_passthrough=False,
    )
    chunks = [
        np.full(20, value, dtype=np.int16).tobytes()
        for value in (100, 1000, 1100, 1200)
    ]

    try:
        processor._set_trigger_filter_enabled(True)
        processor._track_trigger_filter_audio(chunks[0], chunks[0])
        processor._track_trigger_filter_audio(chunks[1], chunks[1])
        assert processor._submit_trigger_filter_request(
            chunk_samples=20,
            energy=100,
            raw_confidence=0.9,
            raw_peak=1000,
            strength=1.0,
        )
        assert enhancer.entered.wait(1.0)

        # These arrive after inference started and must still become part of
        # the recording if the denoised volume confirms the trigger.
        processor._track_trigger_filter_audio(chunks[2], chunks[2])
        processor._track_trigger_filter_audio(chunks[3], chunks[3])
        enhancer.release.set()

        result = None
        deadline = time.monotonic() + 1.0
        while result is None and time.monotonic() < deadline:
            result = processor._poll_trigger_filter_result()
            if result is None:
                time.sleep(0.01)

        assert result is not None
        assert result["accepted"]
        assert processor._consume_trigger_candidate_frames() == chunks
    finally:
        enhancer.release.set()
        processor.close()


def test_stt_worker_enhances_realtime_prefixes_with_one_source_local_stream():
    enhancer = _RecordingEnhancer()

    class _EnhancerFacade:
        @staticmethod
        def create_stream(sample_rate, **kwargs):
            return IncrementalAudioEnhancer(
                enhancer,
                sample_rate,
                context_seconds=1.0,
                **kwargs,
            )

    class _Settings:
        values = {
            "denoise_audio": "deepfilter",
            "denoise_audio_post_filter": False,
            "denoise_strength": 0.75,
            "normalize_enabled": False,
            "silence_cutting_enabled": False,
            "verbose": False,
        }

        @classmethod
        def GetOption(cls, name):
            return cls.values.get(name)

    streams = {}
    prefixes = [
        np.arange(16000, dtype=np.int16),
        np.arange(24000, dtype=np.int16),
        np.arange(32000, dtype=np.int16),
    ]
    with mock.patch.object(
        audio_routes,
        "get_audio_enhancer",
        return_value=_EnhancerFacade(),
    ) as resolver:
        outputs = [
            audioprocessor.enhance_queued_realtime_audio(
                {
                    "data": b"fallback",
                    "denoise_pcm": prefix.tobytes(),
                    "settings": _Settings(),
                    "source_id": "microphone",
                },
                streams,
            )
            for prefix in prefixes
        ]

    resolver.assert_called_once_with("deepfilter", post_filter=False)
    assert [call[0].size for call in enhancer.calls] == [16000, 24000, 24000]
    assert [audio_tools.wav_bytes_to_numpy_array(output).size for output in outputs] == [
        16000,
        24000,
        32000,
    ]

    audioprocessor.reset_queued_realtime_denoiser(streams, "microphone")
    assert streams == {}


def test_worker_delivers_enhanced_final_audio_to_plugins_and_wav_export():
    captured = []

    class _Plugin:
        @staticmethod
        def is_enabled(_default):
            return True

        @staticmethod
        def sts(audio, sample_rate):
            captured.append((audio, sample_rate))

    samples = np.array([0, 1000, -1000, 500], dtype=np.int16)
    audio = audio_tools.audio_bytes_to_wav(
        samples.tobytes(),
        channels=1,
        sample_rate=16000,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        class _Settings:
            @staticmethod
            def GetOption(name):
                if name == "transcription_save_audio_dir":
                    return temp_dir
                return None

        audioprocessor.run_queued_final_audio_consumers(
            audio,
            {
                "run_final_audio_consumers": True,
                "settings": _Settings(),
                "plugins": [_Plugin()],
                "time": time.time_ns(),
            },
        )

        assert captured == [(samples.tobytes(), 16000)]
        saved_files = list(Path(temp_dir).glob("audio_transcript_*.wav"))
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes() == audio


def test_realtime_enhancement_does_not_restore_audio_removed_as_silence():
    class _EnhancerFacade:
        @staticmethod
        def create_stream(sample_rate, **kwargs):
            return IncrementalAudioEnhancer(
                _RecordingEnhancer(),
                sample_rate,
                **kwargs,
            )

    class _Settings:
        @staticmethod
        def GetOption(name):
            return {
                "denoise_audio": "noise_reduce",
                "denoise_audio_post_filter": False,
                "denoise_strength": 1.0,
            }.get(name)

    with mock.patch.object(
        audio_routes,
        "get_audio_enhancer",
        return_value=_EnhancerFacade(),
    ), mock.patch.object(
        audioprocessor,
        "_postprocess_denoised_pcm",
        return_value=b"",
    ):
        output = audioprocessor.enhance_queued_realtime_audio(
            {
                "data": b"fallback",
                "denoise_pcm": np.ones(100, dtype=np.int16).tobytes(),
                "settings": _Settings(),
                "source_id": "microphone",
            },
            {},
        )

    assert output == b""


def test_spectral_gate_receives_fixed_scale_audio_instead_of_peak_normalized_audio():
    from Models.STS import Noisereduce as noisereduce_module

    instance = object.__new__(noisereduce_module.Noisereduce)
    source = np.array([0, 1000, -1000, 500], dtype=np.int16)
    captured = {}

    def fake_reduce_noise(*, y, **_kwargs):
        captured["audio"] = np.asarray(y).copy()
        return y

    with mock.patch.object(
        noisereduce_module.nr,
        "reduce_noise",
        side_effect=fake_reduce_noise,
    ):
        output = instance.enhance_audio(source.tobytes(), strength=0.8)

    np.testing.assert_allclose(
        captured["audio"],
        source.astype(np.float32) / 32768.0,
    )
    assert np.max(np.abs(captured["audio"])) < 0.04
    np.testing.assert_allclose(output, source, atol=1)


def test_zero_strength_is_an_exact_pcm16_bypass_for_both_backends():
    from Models.STS import DeepFilterNet as deepfilter_module
    from Models.STS import Noisereduce as noisereduce_module

    source = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    deepfilter = object.__new__(deepfilter_module.DeepFilterNet)
    spectral_gate = object.__new__(noisereduce_module.Noisereduce)

    np.testing.assert_array_equal(
        deepfilter.enhance_audio(source.tobytes(), strength=0.0),
        source,
    )
    np.testing.assert_array_equal(
        spectral_gate.enhance_audio(source.tobytes(), strength=0.0),
        source,
    )


def test_deepfilter_uses_a_fresh_libdf_state_for_each_independent_window():
    from Models.STS import DeepFilterNet as deepfilter_module

    instance = object.__new__(deepfilter_module.DeepFilterNet)
    instance._inference_lock = threading.RLock()
    instance._df_state_kwargs = {
        "sr": 16000,
        "fft_size": 960,
        "hop_size": 480,
        "nb_bands": 32,
        "min_nb_erb_freqs": 2,
    }
    instance.df_model = object()
    states = []
    observed = []

    def fake_state(**kwargs):
        state = {"kwargs": kwargs}
        states.append(state)
        return state

    def fake_enhance(_model, state, audio, **_kwargs):
        observed.append((state, audio.detach().clone()))
        return audio

    source = np.array([0, 1000, -1000, 500], dtype=np.int16)
    with mock.patch.object(deepfilter_module, "DF", side_effect=fake_state), mock.patch.object(
        deepfilter_module,
        "enhance",
        side_effect=fake_enhance,
    ):
        first = instance.enhance_audio(source.tobytes(), strength=1.0)
        second = instance.enhance_audio(source.tobytes(), strength=1.0)

    assert len(states) == 2
    assert states[0] is not states[1]
    assert observed[0][0] is states[0]
    assert observed[1][0] is states[1]
    np.testing.assert_allclose(
        observed[0][1].squeeze().numpy(),
        source.astype(np.float32) / 32768.0,
    )
    np.testing.assert_allclose(first, source, atol=1)
    np.testing.assert_array_equal(first, second)

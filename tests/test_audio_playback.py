import io
import threading
import unittest
import wave
from unittest import mock

import numpy as np
import torch
import audio_tools


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        linux_patch = mock.patch("audio_tools.platform.system", return_value="Linux")
        linux_patch.start()
        self.addCleanup(linux_patch.stop)
        self.audio = mock.Mock()
        self.audio.get_device_info_by_index.return_value = {
            "index": 2, "name": "Default Sink", "maxOutputChannels": 32,
            "defaultSampleRate": 44100,
        }
        self.audio.get_default_output_device_info.return_value = self.audio.get_device_info_by_index.return_value
        self.pool_patch = mock.patch.object(audio_tools, "pyaudio_pool")
        self.pool = self.pool_patch.start()
        self.addCleanup(self.pool_patch.stop)
        self.pool.acquire.return_value = self.audio

    def played_samples(self, dtype):
        return np.frombuffer(b"".join(call.args[0] for call in self.audio.open.return_value.write.call_args_list), dtype=dtype)

    def test_tts_tensor_reaches_device_as_exact_float_pcm_in_mono(self):
        samples = np.linspace(-0.5, 0.5, 4096, dtype=np.float32)
        audio_tools.play_audio(torch.from_numpy(samples).unsqueeze(0), 2,
                              source_sample_rate=44100, audio_device_channel_num=1,
                              target_channels=1, input_channels=1, dtype="float32", tag="test-pcm")
        self.assertEqual(self.audio.open.call_args.kwargs["channels"], 1)
        self.assertEqual(self.audio.open.call_args.kwargs["format"], audio_tools.pyaudio.paFloat32)
        np.testing.assert_array_equal(self.played_samples(np.float32), samples)
        self.audio.open.return_value.close.assert_called_once()
        self.pool.release.assert_called_once_with(self.audio)

    def test_stereo_wav_uses_header_rate_channels_and_pcm_format(self):
        samples = np.arange(-4000, 4000, dtype=np.int16).reshape(-1, 2)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(samples.tobytes())
        # Deliberately leave the old caller hints mismatched: the WAV is authoritative.
        audio_tools.play_audio(buffer.getvalue(), 2, source_sample_rate=16000, tag="test-wav")
        self.assertEqual(self.audio.open.call_args.kwargs["channels"], 2)
        self.assertEqual(self.audio.open.call_args.kwargs["rate"], 44100)
        self.assertEqual(self.audio.open.call_args.kwargs["format"], audio_tools.pyaudio.paInt16)
        np.testing.assert_array_equal(self.played_samples(np.int16), samples.ravel())

    def test_default_output_is_resolved_before_opening(self):
        audio_tools.play_audio(np.zeros(20, dtype=np.int16), None, tag="test-default")
        self.assertEqual(self.audio.open.call_args.kwargs["output_device_index"], 2)
        self.assertEqual(self.audio.open.call_args.kwargs["channels"], 2)

    def test_stereo_is_downmixed_for_mono_output(self):
        self.audio.get_device_info_by_index.return_value["maxOutputChannels"] = 1
        samples = np.array([[0.5, 0.1], [-0.2, -0.4]], dtype=np.float32)
        audio_tools.play_audio(samples, 2, source_sample_rate=44100,
                              input_channels=2, target_channels=2, dtype="float32", tag="test-mono")
        self.assertEqual(self.audio.open.call_args.kwargs["channels"], 1)
        np.testing.assert_allclose(self.played_samples(np.float32), samples.mean(axis=1))

    @mock.patch("audio_tools.platform.system", return_value="Windows")
    def test_windows_keeps_stereo_speaker_layout_for_mono_tts(self, _platform):
        self.audio.get_device_info_by_index.return_value["maxOutputChannels"] = 2
        samples = np.array([0.25, -0.25], dtype=np.float32)
        audio_tools.play_audio(torch.from_numpy(samples), 2, source_sample_rate=44100,
                              input_channels=1, target_channels=1, dtype="float32", tag="test-windows")
        self.assertEqual(self.audio.open.call_args.kwargs["channels"], 2)
        np.testing.assert_array_equal(self.played_samples(np.float32), np.repeat(samples, 2))

    def test_failed_preparation_releases_pool_lease(self):
        self.audio.get_device_info_by_index.side_effect = OSError("device disconnected")
        with self.assertRaisesRegex(OSError, "disconnected"):
            audio_tools.play_audio(np.zeros(10, dtype=np.int16), 2, tag="test-fail")
        self.pool.release.assert_called_once_with(self.audio)

    def test_write_failure_closes_stream_and_releases_pool_lease(self):
        self.audio.open.return_value.write.side_effect = OSError("output disconnected")
        with mock.patch.object(audio_tools.traceback, "print_exc"):
            audio_tools.play_audio(np.zeros(10, dtype=np.int16), 2, tag="test-write-fail")
        self.audio.open.return_value.close.assert_called_once()
        self.pool.release.assert_called_once_with(self.audio)

    def test_each_output_gets_its_own_sample_rate(self):
        def info(index):
            return {"index": index, "name": str(index), "maxOutputChannels": 2,
                    "defaultSampleRate": 44100 if index == 2 else 48000}
        self.audio.get_device_info_by_index.side_effect = info
        audio_tools.play_audio(np.zeros(4410, dtype=np.float32), 2,
                              source_sample_rate=44100, dtype="float32", secondary_device=3, tag="test-secondary")
        self.assertEqual({(c.kwargs["output_device_index"], c.kwargs["rate"]) for c in self.audio.open.call_args_list},
                         {(2, 44100), (3, 48000)})

    def test_retained_stream_uses_float_format(self):
        player = audio_tools.AudioStreamer.__new__(audio_tools.AudioStreamer)
        player.p = None
        player.device_index = 2
        player.playback_channels = 1
        player.dtype = "float32"
        player._device_lock = threading.RLock()
        player.init_stream(44100)
        self.assertEqual(self.audio.open.call_args.kwargs["format"], audio_tools.pyaudio.paFloat32)


class InputFallbackTests(unittest.TestCase):
    @mock.patch("audio_tools.get_default_audio_device_index_by_api", return_value=None)
    @mock.patch("audio_tools.get_audio_api_index_by_name", return_value=(0, "ALSA"))
    def test_missing_alsa_default_has_actionable_error(self, _api, _default):
        with self.assertRaisesRegex(ValueError, "No default audio input device.*ALSA"):
            audio_tools.resolve_audio_input_configuration({"audio_api": "ALSA", "audio_input_device": "Default"})

    def test_none_input_resolves_default_before_rate_fallback(self):
        audio = mock.Mock()
        audio.get_default_input_device_info.return_value = {"index": 7}
        audio.get_device_info_by_index.return_value = {"defaultSampleRate": 48000, "maxInputChannels": 2}
        stream = mock.Mock()
        audio.open.side_effect = [OSError("unsupported rate"), stream]
        actual, converted, rate, channels = audio_tools.start_recording_audio_stream(
            None, audio_tools.pyaudio.paInt16, 16000, 1, py_audio=audio)
        self.assertIs(actual, stream)
        self.assertTrue(converted)
        self.assertEqual((rate, channels), (48000, 2))
        audio.get_device_info_by_index.assert_called_once_with(7)
        for call in audio.open.call_args_list:
            self.assertEqual(call.kwargs["input_device_index"], 7)

    def test_no_system_input_fails_before_open(self):
        audio = mock.Mock()
        audio.get_default_input_device_info.side_effect = OSError("No Default Input Device Available")
        with self.assertRaisesRegex(ValueError, "No default audio input"):
            audio_tools.start_recording_audio_stream(None, audio_tools.pyaudio.paInt16, 16000, 1, py_audio=audio)
        audio.open.assert_not_called()
        audio.get_device_info_by_index.assert_not_called()

    @mock.patch("audio_tools.platform.system", return_value="Linux")
    @mock.patch("audio_tools.get_audio_api_index_by_name", return_value=(1, "PulseAudio"))
    @mock.patch("audio_tools.main_app_py_audio")
    def test_linux_default_uses_recording_interface_device_table(self, audio, _api, _platform):
        audio.get_host_api_info_by_index.return_value = {"defaultInputDevice": 3, "defaultOutputDevice": 2}
        with mock.patch.object(audio_tools.sd, "query_devices", side_effect=AssertionError("different device table")):
            self.assertEqual(audio_tools.get_default_audio_device_index_by_api("PulseAudio"), 3)
            self.assertEqual(audio_tools.get_default_audio_device_index_by_api("PulseAudio", False), 2)
        audio.get_host_api_info_by_index.return_value["defaultInputDevice"] = -1
        self.assertIsNone(audio_tools.get_default_audio_device_index_by_api("PulseAudio"))


if __name__ == "__main__":
    unittest.main()

import queue
import threading
import unittest
from unittest import mock

import audio_tools
import audio_processing_recording


class _FakeAudioProcessor:
    def __init__(self):
        self.reset_count = 0

    def reset_for_audio_input_switch(self):
        self.reset_count += 1


class _FakeStream:
    def __init__(self, name):
        self.name = name
        self.active = False
        self.stop_count = 0
        self.close_count = 0

    def start_stream(self):
        self.active = True

    def stop_stream(self):
        self.stop_count += 1
        self.active = False

    def close(self):
        self.close_count += 1
        self.active = False

    def is_active(self):
        return self.active


def _resolved_configuration(configuration):
    name = configuration["audio_input_device"]
    return {
        "audio_api": "WASAPI",
        "audio_input_device": name,
        "audio_input_process": configuration.get("audio_input_process", ""),
        "audio_input_process_id": configuration.get("audio_input_process_id", 0),
        "device_index": configuration.get("device_index", 1),
        "device_default_in_index": 4,
        "_stream_device_index": name,
    }


class AudioInputStreamControllerTests(unittest.TestCase):
    def make_controller(self, failing_names=None):
        failing_names = set(failing_names or ())
        processor = _FakeAudioProcessor()
        streams = []

        def stream_factory(device_index, **_kwargs):
            if device_index in failing_names:
                raise OSError(f"cannot open {device_index}")
            stream = _FakeStream(device_index)
            streams.append(stream)
            return stream, False, 16000, 1

        controller = audio_tools.AudioInputStreamController(
            sample_format=8,
            sample_rate=16000,
            channels=1,
            chunk=512,
            py_audio=object(),
            audio_processor=processor,
            stream_factory=stream_factory,
            configuration_resolver=_resolved_configuration,
        )
        return controller, processor, streams

    def test_switch_replaces_only_the_stream_and_resets_source_state(self):
        controller, processor, streams = self.make_controller()
        controller.start({"audio_input_device": "Microphone A"})

        result = controller.switch({"audio_input_device": "Microphone B"})

        self.assertEqual(result["audio_input_device"], "Microphone B")
        self.assertEqual(processor.reset_count, 1)
        self.assertEqual(len(streams), 2)
        self.assertEqual(streams[0].stop_count, 1)
        self.assertEqual(streams[0].close_count, 1)
        self.assertTrue(streams[1].is_active())
        self.assertTrue(controller.is_active())
        controller.close()

    def test_failed_switch_restores_the_previous_stream(self):
        controller, processor, streams = self.make_controller({"Missing Device"})
        controller.start({"audio_input_device": "Microphone A"})

        with self.assertRaisesRegex(RuntimeError, "previous input was restored"):
            controller.switch({"audio_input_device": "Missing Device"})

        self.assertEqual(processor.reset_count, 2)
        self.assertEqual([stream.name for stream in streams], ["Microphone A", "Microphone A"])
        self.assertTrue(streams[-1].is_active())
        self.assertTrue(controller.is_active())
        controller.close()

    def test_selecting_the_current_input_is_a_noop(self):
        controller, processor, streams = self.make_controller()
        controller.start({"audio_input_device": "Microphone A"})

        controller.switch({"audio_input_device": "Microphone A"})

        self.assertEqual(len(streams), 1)
        self.assertEqual(processor.reset_count, 0)
        controller.close()

    def test_activity_check_cannot_observe_the_gap_between_streams(self):
        processor = _FakeAudioProcessor()
        replacement_opening = threading.Event()
        allow_replacement = threading.Event()

        def stream_factory(device_index, **_kwargs):
            if device_index == "Microphone B":
                replacement_opening.set()
                allow_replacement.wait(1.0)
            return _FakeStream(device_index), False, 16000, 1

        controller = audio_tools.AudioInputStreamController(
            sample_format=8,
            sample_rate=16000,
            channels=1,
            chunk=512,
            py_audio=object(),
            audio_processor=processor,
            stream_factory=stream_factory,
            configuration_resolver=_resolved_configuration,
        )
        controller.start({"audio_input_device": "Microphone A"})
        switch_thread = threading.Thread(
            target=controller.switch,
            args=({"audio_input_device": "Microphone B"},),
        )
        activity_result = []
        activity_thread = threading.Thread(
            target=lambda: activity_result.append(controller.is_active())
        )
        switch_thread.start()
        self.assertTrue(replacement_opening.wait(1.0))
        activity_thread.start()
        activity_thread.join(0.05)
        self.assertTrue(activity_thread.is_alive())
        allow_replacement.set()
        switch_thread.join(1.0)
        activity_thread.join(1.0)
        self.assertEqual(activity_result, [True])
        controller.close()


class AudioInputConfigurationTests(unittest.TestCase):
    @mock.patch("audio_tools.get_default_audio_device_index_by_api", return_value=4)
    @mock.patch("audio_tools.get_audio_api_index_by_name", return_value=(3, "Windows WASAPI"))
    def test_default_device_keeps_stable_profile_value(self, _api, _default):
        result = audio_tools.resolve_audio_input_configuration({
            "audio_api": "WASAPI",
            "audio_input_device": "Default",
        })

        self.assertEqual(result["device_index"], -1)
        self.assertEqual(result["_stream_device_index"], 4)
        self.assertEqual(result["audio_input_process"], "")

    @mock.patch("audio_tools.get_audio_device_index_by_name_and_api", return_value=12)
    @mock.patch("audio_tools.get_default_audio_device_index_by_api", return_value=4)
    @mock.patch("audio_tools.get_audio_api_index_by_name", return_value=(3, "Windows WASAPI"))
    def test_named_device_is_resolved_in_selected_api(self, _api, _default, by_name):
        result = audio_tools.resolve_audio_input_configuration({
            "audio_api": "WASAPI",
            "audio_input_device": "USB microphone",
        })

        self.assertEqual(result["device_index"], 12)
        self.assertEqual(result["_stream_device_index"], 12)
        by_name.assert_called_once_with("USB microphone", 3, True, default=None)

    @mock.patch("audio_tools.platform.system", return_value="Windows")
    @mock.patch("audio_tools.get_default_audio_device_index_by_api", return_value=4)
    @mock.patch("audio_tools.get_audio_api_index_by_name", return_value=(3, "Windows WASAPI"))
    def test_application_capture_retains_executable_and_pid(self, _api, _default, _platform):
        result = audio_tools.resolve_audio_input_configuration({
            "audio_api": "WASAPI",
            "audio_input_device": "player.exe - Music (PID 99)",
            "audio_input_process": "player.exe",
            "audio_input_process_id": 99,
        })

        self.assertEqual(result["audio_input_process"], "player.exe")
        self.assertEqual(result["audio_input_process_id"], 99)
        self.assertEqual(result["device_index"], -1)
        self.assertIsNone(result["_stream_device_index"])


class AudioOutputConfigurationTests(unittest.TestCase):
    @mock.patch("audio_tools.get_default_audio_device_index_by_api", return_value=8)
    @mock.patch("audio_tools.get_audio_api_index_by_name", return_value=(3, "Windows WASAPI"))
    def test_default_output_keeps_none_as_profile_index(self, _api, _default):
        result = audio_tools.resolve_audio_output_configuration({
            "audio_api": "WASAPI",
            "audio_output_device": "Default",
        })

        self.assertIsNone(result["device_out_index"])
        self.assertEqual(result["_stream_device_index"], 8)
        self.assertEqual(result["device_default_out_index"], 8)

    @mock.patch("audio_tools.get_audio_device_index_by_name_and_api", return_value=15)
    @mock.patch("audio_tools.get_default_audio_device_index_by_api", return_value=8)
    @mock.patch("audio_tools.get_audio_api_index_by_name", return_value=(3, "Windows WASAPI"))
    def test_named_output_is_resolved_in_selected_api(self, _api, _default, by_name):
        result = audio_tools.resolve_audio_output_configuration({
            "audio_api": "WASAPI",
            "audio_output_device": "USB speakers",
        })

        self.assertEqual(result["device_out_index"], 15)
        self.assertEqual(result["_stream_device_index"], 15)
        by_name.assert_called_once_with("USB speakers", 3, False, default=None)

    @mock.patch("audio_tools.switch_registered_audio_streamers")
    @mock.patch("audio_tools.validate_audio_output_configuration")
    @mock.patch("audio_tools.resolve_audio_output_configuration")
    def test_switch_validates_and_retargets_retained_streamers(self, resolve, validate, switch):
        resolve.return_value = {
            "audio_api": "WASAPI",
            "audio_output_device": "USB speakers",
            "device_out_index": 15,
            "device_default_out_index": 8,
            "_stream_device_index": 15,
        }

        result = audio_tools.switch_main_app_audio_output({"audio_api": "WASAPI"})

        validate.assert_called_once_with(resolve.return_value)
        switch.assert_called_once_with(15)
        self.assertNotIn("_stream_device_index", result)
        self.assertEqual(result["audio_output_device"], "USB speakers")


class AudioStreamerOutputSwitchTests(unittest.TestCase):
    def make_streamer(self):
        streamer = audio_tools.AudioStreamer.__new__(audio_tools.AudioStreamer)
        streamer._device_lock = threading.RLock()
        streamer.device_index = 1
        streamer.stream = object()
        streamer.source_sample_rate = 24000
        streamer._stop_locked = mock.Mock(side_effect=lambda: setattr(streamer, "stream", None))
        streamer.init_stream = mock.Mock(side_effect=lambda _rate: setattr(streamer, "stream", object()))
        return streamer

    def test_retained_stream_is_reopened_on_the_new_device(self):
        streamer = self.make_streamer()

        streamer.switch_output_device(2)

        self.assertEqual(streamer.device_index, 2)
        streamer._stop_locked.assert_called_once_with()
        streamer.init_stream.assert_called_once_with(24000)

    def test_failed_reopen_restores_the_previous_device(self):
        streamer = self.make_streamer()
        streamer.init_stream.side_effect = [OSError("new device failed"), None]

        with self.assertRaisesRegex(RuntimeError, "previous device was restored"):
            streamer.switch_output_device(2)

        self.assertEqual(streamer.device_index, 1)
        self.assertEqual(streamer.init_stream.call_count, 2)


class CallbackStreamStartupTests(unittest.TestCase):
    def test_callback_stream_starts_only_after_processor_format_is_updated(self):
        py_audio = mock.Mock()
        stream = mock.Mock()
        py_audio.open.return_value = stream
        processor = mock.Mock()
        processor.callback = mock.Mock()

        result = audio_tools.start_recording_audio_stream(
            device_index=5,
            sample_format=8,
            sample_rate=16000,
            channels=1,
            chunk=512,
            py_audio=py_audio,
            audio_processor=processor,
        )

        self.assertEqual(result, (stream, False, 16000, 1))
        self.assertFalse(py_audio.open.call_args.kwargs["start"])
        self.assertEqual(processor.recorded_sample_rate, 16000)
        self.assertEqual(processor.input_channel_num, 1)


class AudioProcessorResetTests(unittest.TestCase):
    def test_reset_waits_for_an_in_flight_callback(self):
        processor = audio_processing_recording.AudioProcessor.__new__(
            audio_processing_recording.AudioProcessor
        )
        processor._audio_input_state_lock = threading.RLock()
        callback_entered = threading.Event()
        release_callback = threading.Event()
        reset_called = threading.Event()

        def callback_locked(*_args):
            callback_entered.set()
            release_callback.wait(1.0)
            return None, 0

        processor._callback_locked = callback_locked
        processor._reset_for_audio_input_switch_locked = reset_called.set

        callback_thread = threading.Thread(
            target=processor.callback, args=(b"", 0, None, None)
        )
        reset_thread = threading.Thread(target=processor.reset_for_audio_input_switch)
        callback_thread.start()
        self.assertTrue(callback_entered.wait(1.0))
        reset_thread.start()
        self.assertFalse(reset_called.wait(0.05))
        release_callback.set()
        callback_thread.join(1.0)
        reset_thread.join(1.0)
        self.assertTrue(reset_called.is_set())

    def test_reset_discards_audio_from_the_previous_source(self):
        processor = audio_processing_recording.AudioProcessor.__new__(
            audio_processing_recording.AudioProcessor
        )
        processor._audio_input_state_lock = threading.RLock()
        processor.frames = [b"old"]
        processor.previous_audio_chunk = b"old"
        processor.start_rec_on_volume_threshold = True
        processor.keyboard_rec_force_stop = True
        processor.start_time = 0
        processor.pause_time = 0
        processor.intermediate_time_start = 0
        processor.last_callback_time = 0
        processor.last_recorded_chunk_time = 0
        processor._new_speaker = True
        processor.new_speaker_audio = b"old"
        processor.speaker_turn_detected = True
        processor.audio_filter_buffer = object()
        processor.mic_passthrough_queue = queue.Queue()
        processor.mic_passthrough_queue.put(b"old")
        processor.turn_model = mock.Mock()
        processor.vad_model = mock.Mock()
        processor.vad_model._vad_model.reset_states = mock.Mock()

        processor.reset_for_audio_input_switch()

        self.assertEqual(processor.frames, [])
        self.assertIsNone(processor.previous_audio_chunk)
        self.assertFalse(processor.start_rec_on_volume_threshold)
        self.assertFalse(processor.keyboard_rec_force_stop)
        self.assertFalse(processor._new_speaker)
        self.assertIsNone(processor.new_speaker_audio)
        self.assertFalse(processor.speaker_turn_detected)
        self.assertIsNone(processor.audio_filter_buffer)
        self.assertTrue(processor.mic_passthrough_queue.empty())
        processor.turn_model.clear_session.assert_called_once_with()
        processor.vad_model._vad_model.reset_states.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

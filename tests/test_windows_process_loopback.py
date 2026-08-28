import ctypes
import os
import sys
import threading
import time
import unittest
from unittest import mock

from Utilities import windows_process_loopback as process_loopback


class ProcessSelectionTests(unittest.TestCase):
    def setUp(self):
        self.processes = [
            process_loopback.ProcessInfo(100, 10, "browser.exe"),
            process_loopback.ProcessInfo(101, 100, "browser.exe"),
            process_loopback.ProcessInfo(200, 10, "music.exe"),
        ]

    def test_preferred_child_pid_is_promoted_to_process_tree_root(self):
        resolved = process_loopback.choose_process_id(
            "browser.exe", 101, self.processes
        )
        self.assertEqual(resolved, 100)

    def test_stale_pid_falls_back_to_root_of_matching_process_tree(self):
        resolved = process_loopback.choose_process_id(
            r"C:\Program Files\Browser\browser.exe", 999, self.processes
        )
        self.assertEqual(resolved, 100)

    def test_reused_pid_with_wrong_executable_is_not_accepted(self):
        resolved = process_loopback.choose_process_id(
            "browser.exe", 200, self.processes
        )
        self.assertEqual(resolved, 100)

    def test_missing_saved_application_has_clear_error(self):
        with self.assertRaisesRegex(
            process_loopback.ProcessLoopbackError, "Start the application"
        ):
            process_loopback.choose_process_id("missing.exe", 0, self.processes)

    def test_backend_process_is_never_selected_as_fallback(self):
        processes = [
            process_loopback.ProcessInfo(os.getpid(), 10, "python.exe"),
            process_loopback.ProcessInfo(300, 10, "python.exe"),
        ]
        self.assertEqual(
            process_loopback.choose_process_id("python.exe", 0, processes), 300
        )

    def test_selected_child_is_not_promoted_to_backend_process(self):
        processes = [
            process_loopback.ProcessInfo(os.getpid(), 10, "python.exe"),
            process_loopback.ProcessInfo(300, os.getpid(), "python.exe"),
        ]
        self.assertEqual(
            process_loopback.choose_process_id("python.exe", 300, processes), 300
        )

    def test_stale_pid_does_not_guess_between_independent_instances(self):
        processes = [
            process_loopback.ProcessInfo(300, 10, "player.exe"),
            process_loopback.ProcessInfo(400, 20, "player.exe"),
        ]
        with self.assertRaisesRegex(
            process_loopback.ProcessLoopbackError, "Multiple independent processes"
        ):
            process_loopback.choose_process_id("player.exe", 999, processes)


class ChunkDispatcherTests(unittest.TestCase):
    def test_arbitrary_packets_are_reframed_for_audio_processor(self):
        calls = []

        def callback(data, frame_count, time_info, status):
            calls.append((data, frame_count, time_info, status))
            return None, 0

        dispatcher = process_loopback._ChunkDispatcher(4, 1, callback)
        self.assertTrue(dispatcher.feed(b"\x01\x00" * 3))
        self.assertEqual(calls, [])
        self.assertTrue(dispatcher.feed(b"\x02\x00" * 6))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1] == 4 for call in calls))
        self.assertEqual(len(dispatcher.pending), 2)

    def test_non_continue_callback_status_stops_dispatch(self):
        dispatcher = process_loopback._ChunkDispatcher(
            2, 1, lambda *_args: (None, 1)
        )
        self.assertFalse(dispatcher.feed(bytes(4)))

    def test_silence_has_expected_pcm16_size(self):
        chunks = []
        dispatcher = process_loopback._ChunkDispatcher(
            8, 1, lambda data, *_args: (chunks.append(data) or (None, 0))
        )
        self.assertTrue(dispatcher.emit_silence())
        self.assertEqual(chunks, [bytes(16)])

    def test_silence_pads_a_partial_packet_without_reordering_it(self):
        chunks = []
        dispatcher = process_loopback._ChunkDispatcher(
            4, 1, lambda data, *_args: (chunks.append(data) or (None, 0))
        )
        dispatcher.feed(b"\x01\x00" * 2)
        dispatcher.emit_silence()
        self.assertEqual(chunks, [b"\x01\x00" * 2 + bytes(4)])
        self.assertEqual(dispatcher.pending, bytearray())


class _FakeNativeProcessStream:
    instances = []

    def __init__(self, **kwargs):
        self.process_id = kwargs["process_id"]
        self._active = False
        self._error = None
        self._stop_requested = threading.Event()
        self.__class__.instances.append(self)

    def start_stream(self):
        self._stop_requested.clear()
        self._active = True

    def stop_stream(self):
        self._stop_requested.set()
        self._active = False

    def close(self):
        self.stop_stream()

    def is_active(self):
        return self._active


class ReconnectingStreamTests(unittest.TestCase):
    def setUp(self):
        _FakeNativeProcessStream.instances = []

    def _wait_until(self, predicate, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_reconnects_to_new_pid_after_application_restart(self):
        running_pid = {"value": 100}

        def resolver(_executable, _preferred_pid):
            return running_pid["value"]

        stream = process_loopback.ReconnectingProcessLoopbackStream(
            executable="player.exe",
            preferred_pid=100,
            callback=lambda *_args: (None, 0),
            frames_per_buffer=16,
            activation_timeout=0.2,
            process_check_interval=0.02,
            _process_resolver=resolver,
            _stream_factory=_FakeNativeProcessStream,
        )
        try:
            stream.start_stream()
            running_pid["value"] = 200
            self.assertTrue(
                self._wait_until(lambda: len(_FakeNativeProcessStream.instances) == 2)
            )
            self.assertEqual(
                [instance.process_id for instance in _FakeNativeProcessStream.instances],
                [100, 200],
            )
            self.assertFalse(_FakeNativeProcessStream.instances[0].is_active())
            self.assertTrue(stream.is_active())
        finally:
            stream.close()

    def test_emits_silence_while_waiting_then_reconnects(self):
        running_pid = {"value": 100}
        received_silence = threading.Event()

        def resolver(_executable, _preferred_pid):
            if running_pid["value"] is None:
                raise process_loopback.ProcessLoopbackError("not running")
            return running_pid["value"]

        def callback(data, *_args):
            if data == bytes(32):
                received_silence.set()
            return None, 0

        stream = process_loopback.ReconnectingProcessLoopbackStream(
            executable="player.exe",
            preferred_pid=100,
            callback=callback,
            sample_rate=16000,
            frames_per_buffer=16,
            activation_timeout=0.2,
            process_check_interval=0.02,
            _process_resolver=resolver,
            _stream_factory=_FakeNativeProcessStream,
        )
        try:
            stream.start_stream()
            running_pid["value"] = None
            self.assertTrue(received_silence.wait(1.0))
            self.assertTrue(stream.is_active())

            running_pid["value"] = 300
            self.assertTrue(
                self._wait_until(lambda: len(_FakeNativeProcessStream.instances) == 2)
            )
            self.assertEqual(_FakeNativeProcessStream.instances[1].process_id, 300)
            self.assertTrue(stream.is_active())
        finally:
            stream.close()


@unittest.skipUnless(sys.platform == "win32", "Windows ABI layout")
class WindowsStructureLayoutTests(unittest.TestCase):
    def test_activation_structures_match_windows_sdk_layout(self):
        self.assertEqual(ctypes.sizeof(process_loopback.AUDIOCLIENT_ACTIVATION_PARAMS), 12)
        expected_propvariant_size = 24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16
        self.assertEqual(ctypes.sizeof(process_loopback.PROPVARIANT), expected_propvariant_size)


class AudioToolsRoutingTests(unittest.TestCase):
    def test_process_capture_bypasses_pyaudio_device_open(self):
        # audio_tools is intentionally imported here so the pure resolver tests
        # remain runnable in lightweight/non-audio environments.
        import audio_tools

        processor = mock.Mock()
        processor.callback = mock.Mock()
        py_audio = mock.Mock()
        expected_stream = object()

        with mock.patch(
            "Utilities.windows_process_loopback.create_process_loopback_stream",
            return_value=expected_stream,
        ) as create_stream:
            result = audio_tools.start_recording_audio_stream(
                device_index=7,
                sample_rate=16000,
                channels=1,
                chunk=512,
                py_audio=py_audio,
                audio_processor=processor,
                process_executable="music.exe",
                process_id=200,
            )

        self.assertEqual(result, (expected_stream, False, 16000, 1))
        py_audio.open.assert_not_called()
        create_stream.assert_called_once_with(
            executable="music.exe",
            preferred_pid=200,
            callback=processor.callback,
            sample_rate=16000,
            channels=1,
            frames_per_buffer=512,
        )
        self.assertFalse(processor.needs_sample_rate_conversion)
        self.assertEqual(processor.recorded_sample_rate, 16000)
        self.assertEqual(processor.input_channel_num, 1)


if __name__ == "__main__":
    unittest.main()

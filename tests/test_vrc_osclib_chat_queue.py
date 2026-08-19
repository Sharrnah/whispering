import contextlib
import io
import threading
import time
import unittest

import VRC_OSCLib as osc


class OscChatQueueTests(unittest.TestCase):
    def setUp(self):
        osc.osc_queue.join()
        self.original_sender = osc._direct_osc_send
        self.original_interval = osc.min_time_between_messages
        self.sent = []
        osc.set_min_time_between_messages(0)
        with osc._timing_lock:
            osc.last_message_sent_time = 0.0
        osc._direct_osc_send = lambda **message: self.sent.append(message["data"])

    def tearDown(self):
        osc.osc_queue.join()
        osc._direct_osc_send = self.original_sender
        osc.set_min_time_between_messages(self.original_interval)
        osc.set_chat_debug_logging(False)

    def test_two_normal_messages_are_fifo(self):
        osc.Chat("A")
        osc.Chat("B")

        osc.osc_queue.join()

        self.assertEqual(self.sent, ["A", "B"])

    def test_all_chunks_of_one_message_are_sent(self):
        osc.Chat_chunks("A" * 16, chunk_size=8, delay=0, initial_delay=0)

        osc.osc_queue.join()

        self.assertEqual(self.sent, ["A" * 8, "A" * 8])

    def test_second_final_sequence_cannot_interrupt_first(self):
        first_send_started = threading.Event()
        release_first_send = threading.Event()

        def blocking_sender(**message):
            self.sent.append(message["data"])
            if len(self.sent) == 1:
                first_send_started.set()
                self.assertTrue(release_first_send.wait(2))

        osc._direct_osc_send = blocking_sender
        osc.Chat_chunks("A" * 16, chunk_size=8, delay=0, initial_delay=0)
        self.assertTrue(first_send_started.wait(2))
        osc.Chat_chunks("B" * 16, chunk_size=8, delay=0, initial_delay=0)
        release_first_send.set()

        osc.osc_queue.join()

        self.assertEqual(self.sent, ["A" * 8, "A" * 8, "B" * 8, "B" * 8])

    def test_waiting_final_fast_forwards_old_sequence_display_delay(self):
        first_chunk_sent = threading.Event()
        remaining_chunks_sent = threading.Event()

        def recording_sender(**message):
            self.sent.append(message["data"])
            if len(self.sent) == 1:
                first_chunk_sent.set()
            if len(self.sent) == 3:
                remaining_chunks_sent.set()

        osc._direct_osc_send = recording_sender
        osc.Chat_chunks("A" * 16, chunk_size=8, delay=5, initial_delay=5)
        self.assertTrue(first_chunk_sent.wait(1))
        osc.Chat("B")

        self.assertTrue(
            remaining_chunks_sent.wait(1),
            "A waiting final did not bypass the old sequence's display delay",
        )
        osc.osc_queue.join()
        self.assertEqual(self.sent, ["A" * 8, "A" * 8, "B"])

    def test_prioritize_latest_skips_active_remainder_and_queued_finals(self):
        first_send_started = threading.Event()
        release_first_send = threading.Event()

        def blocking_sender(**message):
            self.sent.append(message["data"])
            if len(self.sent) == 1:
                first_send_started.set()
                self.assertTrue(release_first_send.wait(2))

        osc._direct_osc_send = blocking_sender
        osc.Chat_chunks("A" * 16, chunk_size=8, delay=5, initial_delay=5)
        self.assertTrue(first_send_started.wait(2))
        osc.Chat("B")
        osc.Chat("C", prioritize_latest=True)
        release_first_send.set()

        osc.osc_queue.join()

        self.assertEqual(self.sent, ["A" * 8, "C"])

    def test_many_final_messages_are_never_coalesced(self):
        expected = [f"final-{index}" for index in range(100)]
        for message in expected:
            osc.Chat(message)

        osc.osc_queue.join()

        self.assertEqual(self.sent, expected)

    def test_many_prioritized_finals_keep_only_the_newest_pending_message(self):
        first_send_started = threading.Event()
        release_first_send = threading.Event()

        def blocking_sender(**message):
            self.sent.append(message["data"])
            if message["data"] == "already-sending":
                first_send_started.set()
                self.assertTrue(release_first_send.wait(2))

        osc._direct_osc_send = blocking_sender
        osc.Chat("already-sending")
        self.assertTrue(first_send_started.wait(2))
        for index in range(100):
            osc.Chat(f"new-{index}", prioritize_latest=True)
        release_first_send.set()

        osc.osc_queue.join()

        self.assertEqual(self.sent, ["already-sending", "new-99"])

    def test_realtime_previews_coalesce_without_removing_finals(self):
        first_send_started = threading.Event()
        release_first_send = threading.Event()

        def blocking_sender(**message):
            self.sent.append(message["data"])
            if message["data"] == "final-1":
                first_send_started.set()
                self.assertTrue(release_first_send.wait(2))

        osc._direct_osc_send = blocking_sender
        osc.Chat("final-1")
        self.assertTrue(first_send_started.wait(2))
        osc.Chat("final-2")
        osc.Chat("preview-1", replaceable=True)
        osc.Chat("preview-2", replaceable=True)
        release_first_send.set()

        osc.osc_queue.join()

        self.assertEqual(self.sent, ["final-1", "final-2", "preview-2"])

    def test_new_preview_stops_unsent_chunks_of_active_preview(self):
        first_send_started = threading.Event()
        release_first_send = threading.Event()

        def blocking_sender(**message):
            self.sent.append(message["data"])
            if len(self.sent) == 1:
                first_send_started.set()
                self.assertTrue(release_first_send.wait(2))

        osc._direct_osc_send = blocking_sender
        osc.Chat_chunks(
            "A" * 16, chunk_size=8, delay=0, initial_delay=0, replaceable=True
        )
        self.assertTrue(first_send_started.wait(2))
        osc.Chat_chunks(
            "B" * 16, chunk_size=8, delay=0, initial_delay=0, replaceable=True
        )
        release_first_send.set()

        osc.osc_queue.join()

        self.assertEqual(self.sent, ["A" * 8, "B" * 8, "B" * 8])

    def test_unicode_chunks_and_markers_stay_within_utf16_limit(self):
        samples = ["Hello 👋", "日本語", "äöü", "😀😀😀"]
        for sample in samples:
            text = (sample + " \n") * 100
            payloads = osc.split_words_preserve_whitespace(text, 144)
            self.assertEqual("".join(payloads), text)

            for chunk in osc._marked_chunks(text, 144):
                self.assertLessEqual(osc.count_utf16_code_units(chunk), 144)
            for chunk in osc._scrolling_chunks(text, 144, 3):
                self.assertLessEqual(osc.count_utf16_code_units(chunk), 144)

    def test_queue_drain_and_new_transcription_do_not_deadlock(self):
        for index in range(20):
            osc.Chat_chunks(
                f"message-{index}-" * 20,
                chunk_size=32,
                delay=0,
                initial_delay=0,
            )

        drained = threading.Event()

        def join_queue():
            osc.osc_queue.join()
            drained.set()

        join_thread = threading.Thread(target=join_queue)
        join_thread.start()
        self.assertTrue(drained.wait(3), "OSC queue did not drain")
        join_thread.join(1)
        self.assertFalse(join_thread.is_alive())
        self.assertTrue(osc.osc_sender_thread.is_alive())

    def test_global_rate_limit_applies_to_fifo_messages(self):
        send_times = []
        osc.set_min_time_between_messages(0.02)
        osc._direct_osc_send = lambda **message: send_times.append(time.monotonic())

        osc.Chat("A")
        osc.Chat("B")
        osc.Chat("C")
        osc.osc_queue.join()

        self.assertEqual(len(send_times), 3)
        self.assertGreaterEqual(send_times[1] - send_times[0], 0.015)
        self.assertGreaterEqual(send_times[2] - send_times[1], 0.015)

    def test_direct_sender_reuses_one_udp_client_per_destination(self):
        clients = []

        class FakeUdpClient:
            def __init__(self, ip, port):
                self.destination = (ip, port)
                self.messages = []
                clients.append(self)

            def send(self, message):
                self.messages.append(message)

        original_factory = osc.udp_client.UDPClient
        destination = ("127.0.0.1", 59999)
        with osc._chat_clients_lock:
            osc._chat_clients.pop(destination, None)
        osc.udp_client.UDPClient = FakeUdpClient
        try:
            self.original_sender("first", IP=destination[0], PORT=destination[1])
            self.original_sender("second", IP=destination[0], PORT=destination[1])
        finally:
            osc.udp_client.UDPClient = original_factory
            with osc._chat_clients_lock:
                osc._chat_clients.pop(destination, None)

        self.assertEqual(len(clients), 1)
        self.assertEqual(len(clients[0].messages), 2)

    def test_debug_logging_uses_metadata_not_transcript_text(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            osc.set_chat_debug_logging(True)
            osc.Chat("sensitive transcript")
            osc.osc_queue.join()

        log = output.getvalue()
        self.assertNotIn("sensitive transcript", log)
        self.assertIn("[OSC CHAT] enqueue id=", log)
        self.assertIn("[OSC CHAT] send id=", log)
        self.assertLess(log.index("enqueue"), log.index("send"))

    def test_final_send_failure_is_retried_and_reported(self):
        attempts = []

        def failing_sender(**message):
            attempts.append(message["data"])
            raise OSError("simulated UDP failure")

        osc._direct_osc_send = failing_sender
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            osc.Chat("final")
            osc.osc_queue.join()

        self.assertEqual(attempts, ["final", "final", "final"])
        self.assertIn("[OSC CHAT] send error id=", output.getvalue())
        self.assertIn("[OSC CHAT] sequence failed id=", output.getvalue())

if __name__ == "__main__":
    unittest.main()

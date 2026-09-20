import threading
import unittest

from streaming_display import StableDisplayScheduler, live_block, display_options
from streaming_text import utf16_length


def event(text, revision=1, final=False, stream="a", source="main"):
    return dict(text=text, streaming=True, stream_id=stream, stream_revision=revision,
                final=final, audio_source_id=source)


class LiveBlockTests(unittest.TestCase):
    def test_wrap_keeps_old_line_until_it_is_full_then_drops_a_whole_line(self):
        first = live_block("one two three four", limit=25, line_length=12)
        self.assertEqual(first, "one two\nthree four")
        self.assertEqual(live_block("one two three four five", 25, 12), "three four\nfive")
        self.assertEqual(live_block("one two three four five six", 25, 12), "three four\nfive six")

    def test_initial_large_snapshot_immediately_shows_newest_text(self):
        self.assertEqual(live_block("one two three four five six seven eight nine ten", 12), "nine ten")
        self.assertTrue(live_block("old words " * 100 + "latest words", 144).endswith("latest words"))

    def test_unicode_and_prefix_stay_within_chatbox_budget(self):
        for size in range(1, 180):
            result = live_block("old words \U0001f600 \u65e5\u672c\u8a9e \U0001f600 newest", size, prefix="[Game]")
            self.assertLessEqual(utf16_length(result), size)
            result.encode("utf-16-le")
        self.assertEqual(live_block("\U0001f600", 1), "\u2026")
        self.assertEqual(live_block("", 144, prefix="Game"), "")

    def test_legacy_reading_time_cannot_delay_any_mode(self):
        class Settings:
            def GetOption(self, name):
                return {"special_settings": {"stt_vibevoice_streaming": {"block_min_seconds": 10}}}.get(name)
        self.assertEqual(display_options(Settings()), {"mode": "blocks", "line_length": 42})


class Receipt:
    def __init__(self):
        self.done = threading.Event()
        self.sent = self.cancelled = False
        self.sent_at = 0
    def cancel(self):
        self.cancelled = True


class LiveSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.scheduler = StableDisplayScheduler(clock=lambda: self.now, threaded=False)
        self.output = []
    def emit(self, text, metadata):
        self.output.append((self.now, text, metadata))
    def submit(self, text, revision=1, final=False, stream="a", source="main", lane="caption", emit=None):
        self.scheduler.submit(lane, event(text, revision, final, stream, source), text, emit or self.emit, limit=60)

    def test_burst_coalesces_to_one_newest_view_including_finals(self):
        for i in range(100):
            self.submit("older words " * i + "latest %d" % i, final=True, stream=str(i))
        self.assertEqual(len(self.scheduler.lanes), 1)
        self.scheduler.tick()
        self.assertEqual(len(self.output), 1)
        self.assertTrue(self.output[0][1].endswith("latest 99"))
        self.assertEqual(self.output[0][0], 0)

    def test_new_utterance_replaces_idle_final_immediately(self):
        self.submit("first words", final=True)
        self.scheduler.tick()
        self.now = 0.1
        self.submit("new words", stream="b")
        self.scheduler.tick()
        self.assertEqual([x[1] for x in self.output], ["first words", "new words"])
        self.now = 100
        self.scheduler.tick()
        self.assertEqual(len(self.output), 2)  # Old final timer cannot clear new speech.

    def test_updates_cancel_a_blocked_osc_receipt_without_waiting_for_it(self):
        receipts = []
        def send(text, metadata):
            self.emit(text, metadata)
            receipt = Receipt()
            receipts.append(receipt)
            return receipt
        for i in range(25):
            self.now = i * 0.1
            self.submit("new %d" % i, revision=i, emit=send)
            self.scheduler.tick()
        self.assertEqual(len(self.output), 25)
        self.assertTrue(all(r.cancelled for r in receipts[:-1]))
        self.assertFalse(receipts[-1].cancelled)
        self.assertEqual(self.output[-1][1], "new 24")

    def test_sources_can_alternate_on_one_osc_destination(self):
        for revision, source in [(1, "mic"), (1, "game"), (2, "mic"), (2, "game")]:
            self.submit(source + str(revision), revision, source=source, lane="osc")
            self.scheduler.tick()
        self.assertEqual([x[1] for x in self.output], ["mic1", "game1", "mic2", "game2"])
        self.submit("stale mic", 1, source="mic", lane="osc")
        self.scheduler.tick()
        self.assertEqual(len(self.output), 4)

    def test_final_expires_only_when_idle_and_sources_stay_independent(self):
        self.submit("mic", final=True, source="mic", lane="mic")
        self.submit("game", source="game", lane="game")
        self.scheduler.tick()
        self.now = 5.1
        self.scheduler.tick()
        self.assertNotIn("mic", self.scheduler.lanes)
        self.assertIn("game", self.scheduler.lanes)
        self.assertTrue(self.output[-1][2]["display_done"])
        self.submit("duplicate", 99, True, source="mic", lane="mic")
        self.scheduler.tick()
        self.assertNotIn("duplicate", [x[1] for x in self.output])

    def test_translation_correction_replaces_view_immediately(self):
        self.submit("first interpretation")
        self.scheduler.tick()
        self.now = 0.1
        self.submit("corrected translation", 2)
        self.scheduler.tick()
        self.assertEqual(self.output[-1][1], "corrected translation")

    def test_cancel_retires_pending_send_and_rejects_late_update(self):
        receipt = Receipt()
        self.submit("text", emit=lambda *_: receipt)
        self.scheduler.tick()
        self.scheduler.cancel("caption", "a")
        self.assertTrue(receipt.cancelled)
        self.submit("later", 2)
        self.assertFalse(self.scheduler.lanes)

    def test_switching_mode_mid_utterance_can_return_to_line_view(self):
        self.submit("first")
        self.scheduler.tick()
        first_revision = self.output[-1][2]["display_revision"]
        self.scheduler.cancel("caption", retire=False)
        self.submit("returning to lines", 2)
        self.scheduler.tick()
        self.assertEqual(self.output[-1][1], "returning to lines")
        self.assertGreater(self.output[-1][2]["display_revision"], first_revision)

    def test_osc_supersession_by_another_producer_does_not_mute_the_speaker(self):
        receipt = Receipt()
        self.submit("first", emit=lambda *_: receipt)
        self.scheduler.tick()
        receipt.done.set()  # OSC queue dropped this draft after another producer sent.
        self.scheduler.tick()
        self.submit("speaker continues", 2)
        self.scheduler.tick()
        self.assertEqual(self.output[-1][1], "speaker continues")


if __name__ == "__main__":
    unittest.main()

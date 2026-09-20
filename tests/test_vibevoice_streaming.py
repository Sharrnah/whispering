import unittest
from unittest import mock

import numpy as np

from Models.STT import vibevoice_asr_streaming as vibe
from streaming_text import StreamRevisionTracker, rolling_text, utf16_length
from transcription_queue import RouteTranscriptionQueue


class DisplayTests(unittest.TestCase):
    def test_rolling_window_keeps_newest_words_and_unicode(self):
        self.assertEqual(rolling_text("one two three four", 10), "three four")
        for limit in range(1, 20):
            text = rolling_text("older words 日本語 😀😀 newest", limit)
            self.assertLessEqual(utf16_length(text), limit)
            text.encode("utf-16-le")
        self.assertEqual(rolling_text("abc😀def", 5), "😀def")

    def test_prefix_and_lines_count_towards_chat_budget(self):
        text = rolling_text("older\nwords\nnewest", 15, prefix="EN:")
        self.assertEqual(text, "EN: newest")
        self.assertLessEqual(utf16_length(text), 15)
        self.assertNotIn("\n", text)

    def test_stream_revisions_and_sources(self):
        tracker = StreamRevisionTracker()
        event = dict(streaming=True, stream_id="a", stream_revision=1, audio_source_id="main")
        self.assertTrue(tracker.accept(event))
        self.assertFalse(tracker.accept(event))
        self.assertTrue(tracker.accept(dict(event, audio_source_id="game")))
        self.assertTrue(tracker.accept(dict(event, stream_revision=2, final=True)))
        self.assertFalse(tracker.accept(dict(event, stream_revision=3)))
        self.assertTrue(tracker.accept(dict(event, stream_id="b")))

    def test_speaker_headers_and_words_across_chunks(self):
        text, segments = vibe.transcript_from_chunks(" \n Speaker 0:Hello wor" + "ld.\nSpeaker 1:Hi!")
        self.assertEqual(text, "Hello world. Hi!")
        self.assertEqual([x["speaker"] for x in segments], [0, 1])
        self.assertEqual(vibe.transcript_from_chunks("Speaker 0:Hello.\nSpeaker ")[0], "Hello.")
        self.assertEqual(vibe.transcript_from_chunks(" \n Speak")[0], "")

    def test_partial_json_escapes_and_speaker_metadata(self):
        raw = '[{"speaker":0,"content":"Hello \\"there\\" and \\u65'
        text, _ = vibe.transcript_from_chunks(raw)
        self.assertEqual(text, 'Hello "there" and')
        text, segments = vibe.transcript_from_chunks(raw + 'e5"}]')
        self.assertEqual(text, 'Hello "there" and 日')
        self.assertEqual(segments[0]["speaker"], 0)


class _Runtime:
    def __init__(self):
        self.windows = []
        self.contexts = []

    def init_streaming_state(self, tokenizer, context_info=None):
        self.contexts.append(context_info)
        return {"past_key_values": None, "step": 0}

    def encode_speech(self, window):
        self.windows.append(window.cpu().numpy().copy())
        return window

    def streaming_generate_step(self, features, state, tokenizer, **kwargs):
        state["step"] += 1
        return ("\nSpeaker 0:hello" if state["step"] == 1 else " world"), state


class StreamingSessionTests(unittest.TestCase):
    def setUp(self):
        self.adapter = vibe.VibeVoiceStreamingASR()
        self.runtime = self.adapter.model = _Runtime()
        self.adapter.tokenizer = object()
        self.audio = np.linspace(-0.1, 0.1, 16000 * 12, dtype=np.float32)

    def run_snapshot(self, samples, **kwargs):
        return list(self.adapter.process_snapshot(self.audio[:samples], **kwargs))

    def test_lookahead_waits_then_reuses_cache_and_only_new_audio(self):
        self.assertEqual(self.run_snapshot(16000 * 3), [])
        first = self.run_snapshot(16000 * 4, context="VRChat")
        self.assertEqual(first[0]["text"], "hello")
        self.assertFalse(first[0]["final"])
        self.assertEqual(self.run_snapshot(16000 * 4), [])
        later = self.run_snapshot(16000 * 7)
        self.assertEqual(later[0]["text"], "hello world")
        self.assertEqual(later[0]["text_delta"], " world")
        self.assertEqual(len(self.runtime.contexts), 1)
        self.assertEqual(self.runtime.contexts[0], "VRChat")
        self.assertEqual(self.runtime.windows[0].shape, (1, 83200))
        self.assertGreater(self.runtime.windows[1][0, 100], self.runtime.windows[0][0, 100])

    def test_final_flushes_short_tail_and_cleans_session(self):
        events = self.run_snapshot(100, final=True)
        self.assertEqual(len(self.runtime.windows), 1)
        self.assertTrue(events[-1]["final"])
        self.assertEqual(events[-1]["text"], "hello")
        self.assertFalse(self.adapter.sessions)

    def test_final_with_no_new_text_is_still_delivered(self):
        self.run_snapshot(64000)
        events = self.run_snapshot(64000, final=True)
        self.assertEqual(sum(x["final"] for x in events), 1)
        self.assertFalse(self.adapter.sessions)

    def test_new_recording_and_sources_have_independent_caches(self):
        self.run_snapshot(64000, source_id="main", stream_id="a")
        self.run_snapshot(64000, source_id="game", stream_id="b")
        result = self.run_snapshot(64000, source_id="main", stream_id="c")
        self.assertEqual(result[0]["text"], "hello")
        self.assertEqual(len(self.runtime.contexts), 3)
        self.assertEqual(self.adapter.sessions["game"].stream_id, "b")

    def test_shortened_snapshot_is_rejected_and_cache_released(self):
        self.run_snapshot(64000)
        with self.assertRaisesRegex(ValueError, "shortened"):
            self.run_snapshot(60000)
        self.assertFalse(self.adapter.sessions)

    def test_failure_releases_state_instead_of_reusing_partial_cache(self):
        with mock.patch.object(self.runtime, "streaming_generate_step", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                self.run_snapshot(64000)
        self.assertFalse(self.adapter.sessions)
        self.assertEqual(self.run_snapshot(112000), [])
        self.assertTrue(self.run_snapshot(64000, stream_id="new-recording"))

    def test_coalesced_snapshots_produce_identical_windows(self):
        for count in (64000, 112000, 160000):
            self.run_snapshot(count)
        baseline = self.runtime.windows.copy()
        self.adapter.reset_stream("main")
        self.runtime.windows.clear()
        queue = RouteTranscriptionQueue()
        for count in (64000, 112000, 160000):
            queue.put({"source_id": "main", "final": False, "data": self.audio[:count]})
        item = queue.get_nowait()
        list(self.adapter.process_snapshot(item["data"]))
        queue.task_done()
        self.assertEqual(len(self.runtime.windows), len(baseline))
        for first, second in zip(baseline, self.runtime.windows):
            np.testing.assert_array_equal(first, second)

    def test_unpublished_archive_fails_before_network(self):
        with mock.patch.object(vibe, "needs_download", return_value=True), mock.patch.object(vibe.downloader, "download_model") as download:
            for name in vibe.MODEL_LINKS:
                with self.assertRaisesRegex(RuntimeError, "not been published"):
                    vibe.download_model(name)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()

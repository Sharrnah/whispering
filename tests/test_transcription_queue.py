import unittest

from transcription_queue import RouteTranscriptionQueue


def _item(source, label, final=False):
    return {
        "source_id": source,
        "data": label.encode(),
        "label": label,
        "final": final,
    }


class RouteTranscriptionQueueTests(unittest.TestCase):
    def test_realtime_drafts_are_replaced_per_source(self):
        work = RouteTranscriptionQueue()
        work.put(_item("game", "draft 1"))
        work.put(_item("game", "draft 2"))

        self.assertEqual(work.qsize(), 1)
        self.assertEqual(work.get_nowait()["label"], "draft 2")
        work.task_done()
        self.assertEqual(work.replaced_intermediates, 1)

    def test_final_supersedes_its_source_draft_but_not_another_source(self):
        work = RouteTranscriptionQueue()
        work.put(_item("mic", "mic draft"))
        work.put(_item("game", "game draft"))
        work.put(_item("game", "game final", final=True))

        received = [work.get_nowait(), work.get_nowait()]
        for _ in received:
            work.task_done()

        self.assertEqual(
            {(item["source_id"], item["label"]) for item in received},
            {("mic", "mic draft"), ("game", "game final")},
        )

    def test_sources_are_consumed_round_robin(self):
        work = RouteTranscriptionQueue()
        work.put(_item("mic", "mic 1", final=True))
        work.put(_item("mic", "mic 2", final=True))
        work.put(_item("game", "game 1", final=True))
        work.put(_item("game", "game 2", final=True))

        labels = []
        while not work.empty():
            labels.append(work.get_nowait()["label"])
            work.task_done()

        self.assertEqual(labels, ["mic 1", "game 1", "mic 2", "game 2"])

    def test_completed_utterances_are_never_coalesced(self):
        work = RouteTranscriptionQueue()
        for index in range(4):
            work.put(_item("game", f"final {index}", final=True))

        self.assertEqual(work.qsize(), 4)
        labels = []
        for _ in range(4):
            labels.append(work.get_nowait()["label"])
            work.task_done()
        self.assertEqual(labels, ["final 0", "final 1", "final 2", "final 3"])


if __name__ == "__main__":
    unittest.main()


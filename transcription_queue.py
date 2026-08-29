import queue
import threading
import time
from collections import deque


class RouteTranscriptionQueue:
    """A fair queue that preserves finals and coalesces realtime drafts.

    Every audio source owns a small FIFO lane. Lanes are consumed round-robin,
    so a busy microphone cannot indefinitely hide completed game audio. A lane
    may have any number of final utterances, but only its newest non-final draft
    is useful; replacing older drafts prevents realtime mode from building a
    large audio backlog while the shared model is busy.
    """

    def __init__(self):
        self.maxsize = 0
        self._condition = threading.Condition(threading.RLock())
        self._lanes = {}
        self._ready_sources = deque()
        self._ready_source_set = set()
        self._unfinished_tasks = 0
        self.replaced_intermediates = 0

    @staticmethod
    def _source_id(item):
        return str(item.get("source_id") or "main")

    @staticmethod
    def _is_final(item):
        return bool(item.get("final"))

    def _mark_ready(self, source_id):
        if source_id not in self._ready_source_set:
            self._ready_sources.append(source_id)
            self._ready_source_set.add(source_id)

    def put(self, item, block=True, timeout=None):
        del block, timeout
        if not isinstance(item, dict):
            raise TypeError("Transcription queue entries must be dictionaries.")

        source_id = self._source_id(item)
        with self._condition:
            lane = self._lanes.setdefault(source_id, deque())

            if self._is_final(item):
                # A completed utterance supersedes every draft for that source,
                # including a draft that may sit behind an older final.
                retained = deque()
                removed = 0
                while lane:
                    queued = lane.popleft()
                    if self._is_final(queued):
                        retained.append(queued)
                    else:
                        removed += 1
                lane.extend(retained)
                self._unfinished_tasks -= removed
                self.replaced_intermediates += removed
                lane.append(item)
                self._unfinished_tasks += 1
            else:
                replacement_index = None
                for index in range(len(lane) - 1, -1, -1):
                    if not self._is_final(lane[index]):
                        replacement_index = index
                        break
                if replacement_index is None:
                    lane.append(item)
                    self._unfinished_tasks += 1
                else:
                    lane[replacement_index] = item
                    self.replaced_intermediates += 1

            self._mark_ready(source_id)
            self._condition.notify()

    def put_nowait(self, item):
        self.put(item, block=False)

    def get(self, block=True, timeout=None):
        with self._condition:
            if not block and not self._ready_sources:
                raise queue.Empty

            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._ready_sources:
                if not block:
                    raise queue.Empty
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._condition.wait(remaining)

            while self._ready_sources:
                source_id = self._ready_sources.popleft()
                self._ready_source_set.discard(source_id)
                lane = self._lanes.get(source_id)
                if not lane:
                    self._lanes.pop(source_id, None)
                    continue

                item = lane.popleft()
                if lane:
                    self._mark_ready(source_id)
                else:
                    self._lanes.pop(source_id, None)
                return item

            raise queue.Empty

    def get_nowait(self):
        return self.get(block=False)

    def task_done(self):
        with self._condition:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self._condition.notify_all()

    def join(self):
        with self._condition:
            while self._unfinished_tasks:
                self._condition.wait()

    def qsize(self):
        with self._condition:
            return sum(len(lane) for lane in self._lanes.values())

    def empty(self):
        return self.qsize() == 0

    def full(self):
        return False


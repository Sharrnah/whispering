import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from Models.STT import nemo_canary


class _ConcurrentCallDetector:
    def __init__(self):
        self._state_lock = threading.Lock()
        self.active_calls = 0
        self.maximum_active_calls = 0

    def transcribe(self, audio, verbose=False):
        del audio, verbose
        with self._state_lock:
            self.active_calls += 1
            self.maximum_active_calls = max(self.maximum_active_calls, self.active_calls)
        try:
            # Long enough for unsynchronized worker calls to overlap reliably.
            time.sleep(0.05)
            return [SimpleNamespace(text="serialized Parakeet result")]
        finally:
            with self._state_lock:
                self.active_calls -= 1


class NemoCanaryConcurrencyTests(unittest.TestCase):
    def test_parakeet_cuda_graph_decoder_is_persistently_disabled(self):
        adapter = object.__new__(nemo_canary.NemoCanary)
        adapter.model = SimpleNamespace(
            cfg=SimpleNamespace(
                decoding=OmegaConf.create({
                    "strategy": "greedy_batch",
                    "greedy": {"use_cuda_graph_decoder": True},
                }),
            ),
            change_decoding_strategy=mock.Mock(),
        )

        adapter._configure_parakeet_cuda_decoder()

        configured_decoder = adapter.model.change_decoding_strategy.call_args.args[0]
        self.assertFalse(configured_decoder.greedy.use_cuda_graph_decoder)
        # The source config remains untouched until the real NeMo method adopts
        # the copied decoder config.
        self.assertTrue(adapter.model.cfg.decoding.greedy.use_cuda_graph_decoder)

    def test_parakeet_transcriptions_are_serialized(self):
        adapter = object.__new__(nemo_canary.NemoCanary)
        detector = _ConcurrentCallDetector()
        adapter.model = detector
        adapter.compute_type = "float32"
        adapter.compute_device = "cuda"
        adapter.load_model = mock.Mock()

        def transcribe():
            return adapter.transcribe(
                np.zeros(1600, dtype=np.float32),
                task="transcribe",
                source_lang="auto",
                target_lang="en",
                without_timestamps=True,
                model="parakeet-tdt-0_6b-v3",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: transcribe(), range(2)))

        self.assertEqual(detector.maximum_active_calls, 1)
        self.assertEqual([result["text"] for result in results], [
            "serialized Parakeet result",
            "serialized Parakeet result",
        ])
        self.assertEqual(adapter.load_model.call_count, 2)


if __name__ == "__main__":
    unittest.main()

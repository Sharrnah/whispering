import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from Models import cuda_inference


class _ConcurrentCallDetector:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def enter(self):
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)

    def leave(self):
        with self._lock:
            self.active -= 1


class _Runtime:
    def __init__(self, detector):
        self.detector = detector
        self.compute_device = "cuda"

    def infer(self):
        self.detector.enter()
        try:
            time.sleep(0.05)
            return "done"
        finally:
            self.detector.leave()

    def reload(self):
        self.detector.enter()
        try:
            time.sleep(0.05)
            return "reloaded"
        finally:
            self.detector.leave()


class _NestedRuntime:
    compute_device = "cuda"

    def outer(self):
        return self.inner()

    def inner(self):
        return "nested result"


class _ParallelRuntime:
    compute_device = "cuda"

    def __init__(self, detector, barrier):
        self.detector = detector
        self.barrier = barrier

    def infer(self):
        self.detector.enter()
        try:
            self.barrier.wait(timeout=1.0)
            return "done"
        finally:
            self.detector.leave()


class CudaInferenceGateTests(unittest.TestCase):
    def setUp(self):
        self.synchronize = mock.patch.object(cuda_inference, "_synchronize_cuda")
        self.synchronize.start()

    def tearDown(self):
        self.synchronize.stop()

    def test_cuda_calls_from_different_runtimes_are_serialized(self):
        detector = _ConcurrentCallDetector()
        runtimes = [_Runtime(detector), _Runtime(detector)]
        for index, runtime in enumerate(runtimes):
            cuda_inference.guard_cuda_model_methods(
                runtime,
                ("infer",),
                device=lambda runtime=runtime: runtime.compute_device,
                runtime_label=f"runtime-{index}",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda runtime: runtime.infer(), runtimes))

        self.assertEqual(results, ["done", "done"])
        self.assertEqual(detector.maximum_active, 1)

    def test_two_calls_on_the_same_runtime_can_run_in_parallel(self):
        detector = _ConcurrentCallDetector()
        runtime = _ParallelRuntime(detector, threading.Barrier(2))
        cuda_inference.guard_cuda_model_methods(
            runtime,
            ("infer",),
            device=lambda: runtime.compute_device,
            runtime_label="shared-runtime",
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: runtime.infer(), range(2)))

        self.assertEqual(results, ["done", "done"])
        self.assertEqual(detector.maximum_active, 2)
        cuda_inference._synchronize_cuda.assert_called_once_with("cuda")

    def test_lifecycle_method_is_exclusive_from_same_runtime_inference(self):
        detector = _ConcurrentCallDetector()
        runtime = _Runtime(detector)
        cuda_inference.guard_cuda_model_methods(
            runtime,
            ("infer",),
            device="cuda",
            runtime_label="shared-runtime",
        )
        cuda_inference.guard_cuda_model_methods(
            runtime,
            ("reload",),
            device="cuda",
            runtime_label="shared-runtime",
            parallel_same_runtime=False,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            infer_result = executor.submit(runtime.infer)
            reload_result = executor.submit(runtime.reload)

        self.assertEqual(infer_result.result(), "done")
        self.assertEqual(reload_result.result(), "reloaded")
        self.assertEqual(detector.maximum_active, 1)

    def test_cpu_calls_are_not_serialized(self):
        detector = _ConcurrentCallDetector()
        barrier = threading.Barrier(2)

        class CpuRuntime:
            def infer(self):
                detector.enter()
                try:
                    barrier.wait(timeout=1.0)
                    return "done"
                finally:
                    detector.leave()

        runtimes = [CpuRuntime(), CpuRuntime()]
        for runtime in runtimes:
            cuda_inference.guard_cuda_model_methods(
                runtime,
                ("infer",),
                device="cpu",
                runtime_label="cpu-runtime",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda runtime: runtime.infer(), runtimes))

        self.assertEqual(results, ["done", "done"])
        self.assertEqual(detector.maximum_active, 2)

    def test_nested_guarded_methods_are_reentrant_and_synchronize_once(self):
        runtime = _NestedRuntime()
        cuda_inference.guard_cuda_model_methods(
            runtime,
            ("outer", "inner"),
            device="cuda",
            runtime_label="nested-runtime",
            parallel_same_runtime=False,
        )

        self.assertEqual(runtime.outer(), "nested result")
        cuda_inference._synchronize_cuda.assert_called_once_with("cuda")

    def test_exception_releases_gate_for_the_next_runtime(self):
        with self.assertRaisesRegex(RuntimeError, "failed inference"):
            with cuda_inference.cuda_inference_guard("cuda", "failing-runtime"):
                raise RuntimeError("failed inference")

        with cuda_inference.cuda_inference_guard("cuda", "next-runtime"):
            pass

        self.assertEqual(cuda_inference._synchronize_cuda.call_count, 2)

    def test_loader_guards_loading_and_transcription(self):
        runtime = _Runtime(_ConcurrentCallDetector())

        @cuda_inference.guard_cuda_model_loader("STT", ("infer",))
        def load(model, device):
            self.assertEqual((model, device), ("test-model", "cuda"))
            return runtime

        loaded = load("test-model", "cuda")
        self.assertIs(loaded, runtime)
        self.assertEqual(loaded.infer(), "done")
        self.assertEqual(cuda_inference._synchronize_cuda.call_count, 2)


if __name__ == "__main__":
    unittest.main()

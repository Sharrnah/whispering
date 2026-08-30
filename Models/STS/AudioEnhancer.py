"""Shared helpers for offline and incremental noise suppression."""

from __future__ import annotations

import threading

import numpy as np


def pcm16_bytes_to_float32(audio_bytes) -> np.ndarray:
    """Decode raw little-endian PCM16 without changing its signal level."""
    if isinstance(audio_bytes, bytes):
        if len(audio_bytes) % np.dtype(np.int16).itemsize:
            raise ValueError("PCM16 audio must contain complete samples.")
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
    else:
        samples = np.asarray(audio_bytes)
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
    return np.ascontiguousarray(samples.reshape(-1), dtype=np.float32) / 32768.0


def float32_to_pcm16(samples) -> np.ndarray:
    """Encode normalized floating-point audio as clipped signed PCM16."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    return np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def as_pcm16_array(audio) -> np.ndarray:
    """Normalize a raw PCM result from an enhancer to a mono int16 array."""
    if isinstance(audio, bytes):
        if len(audio) % np.dtype(np.int16).itemsize:
            raise ValueError("PCM16 audio must contain complete samples.")
        return np.frombuffer(audio, dtype=np.int16).copy()

    samples = np.asarray(audio).reshape(-1)
    if np.issubdtype(samples.dtype, np.floating):
        return float32_to_pcm16(samples)
    if samples.dtype == np.int16:
        return np.ascontiguousarray(samples)

    info = np.iinfo(samples.dtype)
    scale = float(max(abs(info.min), info.max))
    return float32_to_pcm16(samples.astype(np.float32) / scale)


class IncrementalAudioEnhancer:
    """Cache an enhanced growing recording and process only its new tail.

    Realtime transcription repeatedly submits the complete utterance-so-far.
    Running an offline denoiser on that complete prefix every time makes the
    total work grow quadratically.  This adapter retains the already enhanced
    prefix, reprocesses a bounded amount of left context plus the appended
    samples, and crossfades the one replaceable seam.

    The wrapped enhancer remains responsible for model inference.  Its public
    contract is the existing ``enhance_audio`` PCM16 API, so both spectral
    gating and DeepFilterNet use the same incremental path.
    """

    def __init__(
        self,
        enhancer,
        sample_rate: int,
        *,
        input_channels: int = 1,
        output_channels: int = 1,
        context_seconds: float = 1.0,
        crossfade_ms: float = 20.0,
    ):
        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be positive.")
        if int(input_channels) != 1 or int(output_channels) != 1:
            raise ValueError("Incremental transcription denoising requires mono audio.")

        self._enhancer = enhancer
        self.sample_rate = int(sample_rate)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.context_samples = max(0, int(round(context_seconds * self.sample_rate)))
        self.crossfade_samples = max(0, int(round(crossfade_ms * self.sample_rate / 1000.0)))
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self._input = np.empty(0, dtype=np.int16)
            self._output = np.empty(0, dtype=np.int16)
            self._strength = None

    @staticmethod
    def _fit_length(enhanced: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Keep the PCM stream length stable despite resampler rounding."""
        target_length = source.size
        if enhanced.size == target_length:
            return enhanced
        if enhanced.size > target_length:
            return enhanced[:target_length]

        result = source.copy()
        result[:enhanced.size] = enhanced
        return result

    def enhance_prefix(self, audio, *, strength: float = 1.0) -> np.ndarray:
        samples = as_pcm16_array(audio)
        strength = float(strength)

        with self._lock:
            old_length = self._input.size
            extends_cached_prefix = (
                samples.size >= old_length
                and np.array_equal(samples[:old_length], self._input)
            )
            if not extends_cached_prefix or self._strength != strength:
                self.reset()
                old_length = 0

            if samples.size == old_length:
                return self._output.copy()

            window_start = max(0, old_length - self.context_samples)
            source_window = samples[window_start:]
            enhanced_window = self._enhancer.enhance_audio(
                source_window.tobytes(),
                sample_rate=self.sample_rate,
                output_sample_rate=self.sample_rate,
                input_channels=self.input_channels,
                output_channels=self.output_channels,
                strength=strength,
            )
            enhanced_window = self._fit_length(
                as_pcm16_array(enhanced_window), source_window
            )

            if old_length == 0:
                output = enhanced_window
            else:
                seam_start = max(
                    window_start,
                    old_length - self.crossfade_samples,
                )
                overlap_length = old_length - seam_start
                replacement = enhanced_window[seam_start - window_start:]

                if overlap_length:
                    phase = np.linspace(
                        0.0,
                        np.pi / 2.0,
                        overlap_length,
                        endpoint=True,
                        dtype=np.float32,
                    )
                    fade_in = np.sin(phase) ** 2
                    fade_out = 1.0 - fade_in
                    old_overlap = self._output[seam_start:old_length].astype(np.float32)
                    new_overlap = replacement[:overlap_length].astype(np.float32)
                    mixed = np.rint(
                        old_overlap * fade_out + new_overlap * fade_in
                    ).clip(-32768, 32767).astype(np.int16)
                else:
                    mixed = np.empty(0, dtype=np.int16)

                output = np.concatenate(
                    (
                        self._output[:seam_start],
                        mixed,
                        replacement[overlap_length:],
                    )
                )

            self._input = samples.copy()
            self._output = self._fit_length(output, samples)
            self._strength = strength
            return self._output.copy()

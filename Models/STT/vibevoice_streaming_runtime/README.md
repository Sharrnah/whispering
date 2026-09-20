# VibeVoice streaming ASR runtime

Adapted from Microsoft VibeVoice revision
`1541f590c7099820f10ea012f48d2399282df69f` (MIT; see LICENSE).
Source: https://github.com/microsoft/VibeVoice

This package preserves the published checkpoint names, acoustic/semantic
encoders, connectors, prompt and stateful chunk decoding. It uses the installed
Transformers Qwen2 implementation, without installing Microsoft's package or
its Transformers 4 requirement. The TTS, diffusion, server and training paths
are not imported. Speech tokenizer names are not registered globally.

Compatibility adaptations:

- Transformers 5 text-config lookup and tied-weight mapping, including the
  tied 1.5B decoder and untied 7B decoder.
- CPU construction of the encoder's static drop-path schedule during meta
  loading; convolutions use eager execution while Qwen uses FA2 or SDPA.
- Local-directory-only model loading; tokenizer loading is owned by the adapter.
- Last-position logits only, retaining Qwen's KV cache between incoming windows.
- Explicit error if a chunk exceeds its token limit, rather than silently
  returning a truncated transcript.

Do not change trained chunk sizes to suggest lower latency. The adapter reads
the checkpoint's 22-frame chunk and 4-frame lookahead at 24 kHz / 3200 samples
per frame, resamples aligned windows from 16 kHz input, and flushes the remaining
audio at end of speech. See `documentation/VIBEVOICE_ASR_STREAMING.md`.

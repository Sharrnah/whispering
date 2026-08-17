# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only Qwen3-TTS 12 Hz runtime exports.

The upstream package also imports its legacy 25 Hz tokenizer eagerly. The
released Qwen3-TTS checkpoints used by Whispering Tiger all use the 12 Hz
tokenizer, so keeping the old runtime would add an unused SoX/ONNX path and an
external SoX executable requirement on Windows.
"""

from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

__all__ = ["Qwen3TTSTokenizerV2Config", "Qwen3TTSTokenizerV2Model"]

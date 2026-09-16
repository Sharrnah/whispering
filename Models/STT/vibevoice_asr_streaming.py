"""Stateful VibeVoice ASR: consume new audio while the recorder keeps running.

Capture sends cumulative, unedited 16 kHz PCM snapshots. Coalescing a queued
snapshot is lossless: each session's cursor advances only after inference.
Only a trained window is resampled/encoded, and the language KV cache survives
between windows. No model or tokenizer may access the Hub at runtime.
"""

import gc
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

import downloader
from Models.STT.vibevoice_streaming_manifest import MODEL_LINKS
from Models.transformers_attention import get_preferred_attention_implementation, load_with_attention_fallback

DEFAULT_MODEL = "VibeVoice-ASR-Streaming-1.5B"
MODEL_CACHE_PATH = Path(".cache/vibevoice-asr-streaming")
DOWNLOAD_STATE = {"is_downloading": False}
LANGUAGES = {
    "zh": "Chinese", "en": "English", "fr": "French", "de": "German", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "es": "Spanish",
}


def get_languages():
    # Recognition is automatic; this model has no forced-language prefill.
    return ({"code": "", "name": "Auto"},)


def get_model_path(model_name):
    if model_name in {"custom", "custom-streaming"}:
        return MODEL_CACHE_PATH / "custom"
    if model_name not in MODEL_LINKS:
        raise ValueError(f"Unknown VibeVoice streaming checkpoint: {model_name}")
    return MODEL_CACHE_PATH / MODEL_LINKS[model_name]["path"]


def needs_download(model_name):
    if model_name in {"custom", "custom-streaming"}:
        return False
    return downloader.model_needs_download(get_model_path(model_name), MODEL_LINKS[model_name]["file_checksums"])


def download_model(model_name, force_non_ui_dl=False):
    if model_name in {"custom", "custom-streaming"}:
        return get_model_path(model_name).is_dir()
    if not needs_download(model_name):
        return True
    if MODEL_LINKS[model_name]["checksum"] == "0" * 64:
        raise RuntimeError(
            f"The hosted {model_name} archive has not been published. "
            f"Prepare the verified local files in {get_model_path(model_name)}; "
            "see documentation/VIBEVOICE_ASR_STREAMING.md."
        )
    return downloader.download_model({
        "model_path": MODEL_CACHE_PATH, "model_link_dict": MODEL_LINKS,
        "model_name": model_name, "title": f"Speech to Text (VibeVoice streaming) - {model_name}",
        "alt_fallback": False, "force_non_ui_dl": force_non_ui_dl, "extract_format": "zip",
    }, DOWNLOAD_STATE)


def transcript_from_chunks(raw_text):
    """Extract speaker content without exposing model formatting.

    A content string can span several audio chunks. Accept its complete escape
    sequences but withhold an unfinished escape until the next chunk arrives.
    Never pass speaker IDs or unfinished JSON keys into translation/TTS/OSC.
    """
    segments = []
    if not raw_text.lstrip().startswith(("{", "[")):
        # Published streaming checkpoints emit line-start `Speaker 0:` headers.
        # A header may itself straddle two model windows.
        raw_text = re.sub(r"(?:^|\n)\s*(?:S(?:p(?:e(?:a(?:k(?:e(?:r(?:\s+\d*)?)?)?)?)?)?)?)\s*$", "", raw_text)
        headers = list(re.finditer(r"(?:^|\n)\s*Speaker\s+(\d+)\s*:\s*", raw_text, re.IGNORECASE))
        if headers:
            for index, header in enumerate(headers):
                end = headers[index + 1].start() if index + 1 < len(headers) else len(raw_text)
                content = raw_text[header.end():end].strip()
                if content:
                    segments.append({"speaker": int(header.group(1)), "text": content})
        elif raw_text.strip():
            segments.append({"speaker": None, "text": raw_text.strip()})
        return " ".join(segment["text"] for segment in segments), segments
    for match in re.finditer(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)', raw_text, re.DOTALL):
        escaped = match.group(1)
        while escaped:
            try:
                content = json.loads('"' + escaped + '"')
                break
            except (ValueError, json.JSONDecodeError):
                escaped = escaped[:-1]
        else:
            content = ""
        before = raw_text[:match.start()]
        speakers = list(re.finditer(r'"speaker"\s*:\s*("[^"\\]*"|\d+)', before))
        speaker = json.loads(speakers[-1].group(1)) if speakers else None
        if content:
            segments.append({"speaker": speaker, "text": content})
    return " ".join(segment["text"].strip() for segment in segments).strip(), segments


@dataclass
class _Session:
    stream_id: str
    state: dict | None = None
    cursor: int = 0
    received: int = 0
    raw_text: str = ""
    text: str = ""
    revision: int = 0
    touched: float = field(default_factory=time.monotonic)


class VibeVoiceStreamingASR:
    def __init__(self, compute_type="bfloat16", device="cpu"):
        self.compute_type = compute_type
        self.compute_device = str(device)
        self.model = None
        self.tokenizer = None
        self.loaded_configuration = None
        self.sessions = {}
        self.failed_streams = {}
        self._lock = threading.RLock()
        # Both published models: 22 frames + 4 lookahead, 3200 samples/frame
        # at 24 kHz. Read and validate actual values at load time.
        self.frame_config = None

    def load_model(self, model=DEFAULT_MODEL, compute_type=None, device=None):
        with self._lock:
            from transformers import Qwen2TokenizerFast
            from .vibevoice_streaming_runtime.configuration_vibevoice import VibeVoiceASRConfig
            from .vibevoice_streaming_runtime.modeling_vibevoice import VibeVoiceASRForConditionalGeneration

            precision = compute_type or self.compute_type
            device = str(device or self.compute_device or "cpu")
            if device in {"None", "auto", "cuda"}:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "cpu" and not device.startswith("cuda"):
                raise ValueError("VibeVoice streaming supports CUDA and CPU; DirectML is not supported.")
            if precision not in {"float32", "bfloat16", "float16"}:
                raise ValueError("VibeVoice streaming supports float32 and bfloat16 precision.")
            # Use the official BF16 path for legacy FP16 requests as well.
            bf16 = False
            if device.startswith("cuda"):
                with torch.cuda.device(torch.device(device)):
                    bf16 = torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if precision != "float32" and bf16 else torch.float32
            configuration = (model, dtype, device)
            if self.model is not None and configuration == self.loaded_configuration:
                return
            path = get_model_path(model)
            if not download_model(model):
                raise RuntimeError(f"Could not prepare VibeVoice streaming model: {model}")
            frame = json.loads((path / "preprocessor_config.json").read_text(encoding="utf-8"))
            if (frame.get("target_sample_rate") != 24000 or frame.get("speech_tok_compress_ratio") != 3200
                    or frame.get("chunk_frames") != 22 or frame.get("lookahead_frames") != 4
                    or frame.get("normalize_audio", False)):
                raise ValueError("Expected the VibeVoice streaming 24 kHz / 22+4 frame checkpoint configuration.")
            tokenizer = Qwen2TokenizerFast.from_pretrained(str(path.resolve()), local_files_only=True)
            # ASR reuses Qwen's object-reference delimiters (TTS uses vision
            # delimiters). These are the upstream ASR tokenizer's exact IDs.
            for attr, token in (("speech_start_id", "<|object_ref_start|>"),
                                ("speech_end_id", "<|object_ref_end|>"),
                                ("text_chunk_end_id", "<|text_chunk_end|>")):
                ids = tokenizer.encode(token, add_special_tokens=False)
                if len(ids) != 1 or token not in tokenizer.get_vocab():
                    raise ValueError(f"Checkpoint tokenizer is missing streaming token {token}")
                setattr(tokenizer, attr, ids[0])
            self.release_model()
            config = VibeVoiceASRConfig.from_pretrained(str(path.resolve()), local_files_only=True)
            attention = get_preferred_attention_implementation(torch.device(device), dtype)

            def load(attn):
                # Speech encoders are convolutions; only Qwen uses attention.
                config.decoder_config._attn_implementation = attn
                return VibeVoiceASRForConditionalGeneration.from_pretrained(
                    str(path.resolve()), config=config, dtype=dtype,
                    device_map=device, attn_implementation=attn, local_files_only=True,
                ).eval()

            loaded, attention = load_with_attention_fallback(load, attention, f"VibeVoice streaming {model}")
            self.model, self.tokenizer = loaded, tokenizer
            self.frame_config = frame
            self.compute_type, self.compute_device = precision, device
            self.loaded_configuration = configuration
            if device.startswith("cuda"):
                try:
                    self._warmup()
                except Exception:
                    self.release_model()
                    raise
            print(f"Loaded {model} on {device}, {dtype}, {attention}; streaming every 2.93 s with 0.53 s lookahead.")

    def _warmup(self):
        """Pay CUDA's first convolution/kernel cost before reporting ready."""
        device = torch.device(self.compute_device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.inference_mode(), torch.random.fork_rng(devices=[index]):
            features = self.model.encode_speech(torch.zeros(1, 83200, device=device))
            state = self.model.init_streaming_state(self.tokenizer)
            output = self.model(inputs_embeds=torch.cat(
                [state["sp_start_embed"], features, state["sp_end_embed"]], dim=1,
            ), past_key_values=state["past_key_values"])
            next_embed = state["embed_tokens"](output.logits[:, -1].argmax(-1).unsqueeze(1))
            self.model(inputs_embeds=next_embed, past_key_values=output.past_key_values)
            torch.cuda.synchronize(device)

    def reset_stream(self, source_id):
        with self._lock:
            self.sessions.pop(str(source_id), None)

    def release_model(self):
        with self._lock:
            self.sessions.clear()
            self.failed_streams.clear()
            self.model = self.tokenizer = self.loaded_configuration = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_snapshot(self, audio, *, source_id="main", stream_id="default", final=False,
                         context="", max_new_tokens=256):
        """Yield cumulative plain-text results in capture order, then one final.

        Audio is a cumulative mono float32 16 kHz utterance, never a delta.
        Integer positions derive from the 24 kHz timeline to avoid accumulating
        rounding error at the 22-frame (non-integral at 16 kHz) boundary.
        """
        with self._lock, torch.inference_mode():
            if self.model is None:
                raise RuntimeError("VibeVoice streaming model has not been loaded.")
            now = time.monotonic()
            for key, stale in list(self.sessions.items()):
                if now - stale.touched > 120:
                    del self.sessions[key]
            source_id, stream_id = str(source_id), str(stream_id)
            if self.failed_streams.get(source_id) == stream_id:
                return
            self.failed_streams.pop(source_id, None)
            session = self.sessions.get(source_id)
            if session is None or session.stream_id != stream_id:
                session = self.sessions[source_id] = _Session(stream_id)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if len(audio) < session.received:
                self.reset_stream(source_id)
                self.failed_streams[source_id] = stream_id
                raise ValueError("VibeVoice received a shortened audio snapshot within one stream.")
            session.received, session.touched = len(audio), now
            chunk_24k, window_24k = 22 * 3200, 26 * 3200
            max_new_tokens = min(1024, max(1, int(max_new_tokens)))
            try:
                while True:
                    start_24k = session.cursor * chunk_24k
                    # Align resampling to its 3:2 phase, with a small FIR halo.
                    start_16k = start_24k * 2 // 3
                    end_16k = (start_24k + window_24k) * 2 // 3 + 32
                    if (not final and len(audio) < end_16k) or start_16k >= len(audio):
                        break
                    halo_start = max(0, (start_16k - 32) // 2 * 2)
                    samples = audio[halo_start:min(len(audio), end_16k)]
                    resampled = resample_poly(samples, 3, 2).astype(np.float32)
                    offset = start_24k - halo_start * 3 // 2
                    window = resampled[offset:offset + window_24k]
                    if len(window) < window_24k:
                        window = np.pad(window, (0, window_24k - len(window)))
                    if session.state is None:
                        session.state = self.model.init_streaming_state(self.tokenizer, context_info=context or None)
                    # Bound KV memory independently of recording length. Keep
                    # text/cursor, reset only model context at a chunk boundary.
                    cache = session.state.get("past_key_values")
                    if cache is not None and cache.get_seq_length() > 8192:
                        session.state = self.model.init_streaming_state(self.tokenizer, context_info=context or None)
                    features = self.model.encode_speech(torch.from_numpy(window).to(self.compute_device).unsqueeze(0))
                    chunk, session.state = self.model.streaming_generate_step(
                        features, session.state, self.tokenizer, max_new_tokens=max_new_tokens, temperature=0.0,
                    )
                    session.cursor += 1
                    session.raw_text += chunk
                    text, segments = transcript_from_chunks(session.raw_text)
                    if text != session.text:
                        delta = text[len(session.text):] if text.startswith(session.text) else ""
                        session.text = text
                        session.revision += 1
                        yield self._result(session, segments, delta, False)
                if final:
                    _, segments = transcript_from_chunks(session.raw_text)
                    session.revision += 1
                    yield self._result(session, segments, "", True)
            except Exception:
                self.reset_stream(source_id)
                self.failed_streams[source_id] = stream_id
                raise
            finally:
                if final:
                    self.reset_stream(source_id)

    @staticmethod
    def _result(session, segments, delta, final):
        return {"text": session.text, "type": "transcript", "language": "",
                "segments": segments, "streaming": True, "stream_id": session.stream_id,
                "stream_revision": session.revision, "text_delta": delta, "final": final}

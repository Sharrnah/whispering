import os
import json
from unicodedata import category
import importlib
import re
import sys

# Reduce ONNX Runtime logging via environment vars BEFORE importing it
os.environ.setdefault('ORT_LOG_SEVERITY_LEVEL', '4')  # 0=VERBOSE,1=INFO,2=WARNING,3=ERROR,4=FATAL
os.environ.setdefault('ORT_LOG_VERBOSITY_LEVEL', '0')

import librosa
import numpy as np
import onnxruntime
from transformers import AutoTokenizer

# Define module-level constants and globals
S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
_kakasi = None
_dicta = None
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

# Further reduce ORT logging via runtime API if available
try:
    if hasattr(onnxruntime, 'set_default_logger_severity'):
        onnxruntime.set_default_logger_severity(4)
    if hasattr(onnxruntime, 'set_default_logger_verbosity'):
        onnxruntime.set_default_logger_verbosity(0)
except Exception:
    pass

# Context manager to sanitize stderr (remove ANSI/control chars) and optionally drop ORT warnings
_ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ctrl_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

class _SanitizeStderr:
    def __init__(self, drop_ort_warnings: bool = True, pure_text: bool = True):
        self._orig = sys.stderr
        self._drop = drop_ort_warnings
        self._pure = pure_text
    def __enter__(self):
        class _Wrapper:
            def __init__(self, orig, drop, pure):
                self._orig = orig
                self._drop = drop
                self._pure = pure
            def write(self, s):
                if not isinstance(s, str):
                    try:
                        s = s.decode('utf-8', errors='ignore')
                    except Exception:
                        s = str(s)
                sl = s.lower()
                if self._drop and ("onnx" in sl or "cudnn" in sl or "cudaexecutionprovider" in sl or "ep error" in sl):
                    # Drop ONNX/CUDA/cuDNN noisy logs entirely
                    return 0
                if self._pure:
                    s = _ansi_re.sub('', s)
                    s = _ctrl_re.sub('', s)
                return self._orig.write(s)
            def flush(self):
                return self._orig.flush()
            def writelines(self, lines):
                for line in lines:
                    self.write(line)
            @property
            def encoding(self):
                return getattr(self._orig, 'encoding', 'utf-8')
        self._wrapper = _Wrapper(self._orig, self._drop, self._pure)
        sys.stderr = self._wrapper
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._orig
        return False

class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""

    def __init__(self, mapping_path=None):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self.mapping_path = mapping_path
        self._load_cangjie_mapping()
        self._init_segmenter()

    def _load_cangjie_mapping(self):
        """Load Cangjie mapping from a local file."""
        try:
            if not self.mapping_path or not os.path.isfile(self.mapping_path):
                # Not fatal; Chinese segmentation will be limited
                return

            with open(self.mapping_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)

        except Exception:
            # Mapping load failures are non-fatal
            pass

    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            pkuseg_mod = importlib.import_module('pkuseg')
            self.segmenter = pkuseg_mod.pkuseg()
        except Exception:
            self.segmenter = None

    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        # Guard to satisfy static analysis
        if code not in self.cj2word:
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return f"{code}{index}"

    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text

        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = []
                for c in cangjie:
                    code.append(f"[cj_{c}]")
                code.append("[cj_.]")
                code = "".join(code)
                output.append(code)
            else:
                output.append(t)
        return "".join(output)


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese text normalization: converts kanji to hiragana; katakana remains the same."""
    global _kakasi

    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()

        result = _kakasi.convert(text)
        out = []

        for r in result:
            inp = r['orig']
            hira = r["hira"]

            # Any kanji in the phrase
            if any([is_kanji(c) for c in inp]):
                if hira and hira[0] in ["は", "へ"]:  # Safety check for empty hira
                    hira = " " + hira
                out.append(hira)

            # All katakana
            elif all([is_katakana(c) for c in inp]) if inp else False:  # Safety check for empty inp
                out.append(r['orig'])

            else:
                out.append(inp)

        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', "".join(out))
        return normalized_text

    except Exception:
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta

    try:
        if _dicta is None:
            dicta_mod = importlib.import_module('dicta_onnx')
            _dicta = getattr(dicta_mod, 'Dicta')()
        return _dicta.add_diacritics(text)
    except Exception:
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""

    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        return initial + medial + final

    result = ''.join(decompose_hangul(char) for char in text)
    return result.strip()


def prepare_language(txt, language_id, mapping_path=None):
    # Language-specific text processing
    cangjie_converter = ChineseCangjieConverter(mapping_path)
    if language_id == 'zh':
        txt = cangjie_converter(txt)
    elif language_id == 'ja':
        txt = hiragana_normalize(txt)
    elif language_id == 'he':
        txt = add_hebrew_diacritics(txt)
    elif language_id == 'ko':
        txt = korean_normalize(txt)

    # Prepend language token
    if language_id:
        txt = f"[{language_id.lower()}]{txt}"
    return txt


class ChatterboxOnnxTTS:
    """Reusable ONNX TTS engine that exposes generate and streaming APIs."""

    def __init__(self, model_dir: str, providers: list[str] | None = None):
        self.model_dir = os.path.abspath(model_dir)
        self.onnx_dir = os.path.join(self.model_dir, "onnx")
        self.mapping_path = os.path.join(self.model_dir, "Cangjie5_TC.json")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)

        # Resolve providers
        if providers is None:
            try:
                avail = onnxruntime.get_available_providers()
                if "CUDAExecutionProvider" in avail:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
            except Exception:
                providers = ["CPUExecutionProvider"]
        self._providers_gpu = providers
        self._providers_cpu = ["CPUExecutionProvider"]

        sess_opts = onnxruntime.SessionOptions()
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_cpu_mem_arena = True
        # Reduce session log verbosity as an extra guard
        try:
            sess_opts.log_severity_level = 4  # FATAL only
            sess_opts.log_verbosity_level = 0
        except Exception:
            pass
        self._sess_opts = sess_opts
        # Required ONNX files
        speech_encoder_path = os.path.join(self.onnx_dir, "speech_encoder.onnx")
        embed_tokens_path = os.path.join(self.onnx_dir, "embed_tokens.onnx")
        conditional_decoder_path = os.path.join(self.onnx_dir, "conditional_decoder.onnx")
        language_model_path = os.path.join(self.onnx_dir, "language_model_q4.onnx")
        self._conditional_decoder_path = conditional_decoder_path
        required_files = [speech_encoder_path, embed_tokens_path, conditional_decoder_path, language_model_path]
        missing = [p for p in required_files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Missing ONNX files: {missing}")
        # Create sessions with stderr sanitization
        with _SanitizeStderr(drop_ort_warnings=True, pure_text=True):
            # Run speech encoder and conditional decoder on CPU to avoid cuDNN frontend conv issues
            self.sess_speech = onnxruntime.InferenceSession(speech_encoder_path, sess_opts, providers=self._providers_cpu)
            # Keep token embedder and LM on GPU if available
            self.sess_embed = onnxruntime.InferenceSession(embed_tokens_path, sess_opts, providers=self._providers_gpu)
            self.sess_lm = onnxruntime.InferenceSession(language_model_path, sess_opts, providers=self._providers_gpu)
            self.sess_cond = onnxruntime.InferenceSession(conditional_decoder_path, sess_opts, providers=self._providers_cpu)

        # LM cache config (from export)
        self.num_hidden_layers = 30
        self.num_key_value_heads = 16
        self.head_dim = 64

    def _prepare_text(self, text: str, language_id: str):
        prep = prepare_language(text, language_id, self.mapping_path)
        tok_out = self.tokenizer(prep, return_tensors="np")
        return tok_out["input_ids"].astype(np.int64)

    def _encode_voice(self, target_voice_path: str):
        # Sanitize any provider-level stderr noise during run
        with _SanitizeStderr(drop_ort_warnings=True, pure_text=True):
            audio_values, _ = librosa.load(target_voice_path, sr=S3GEN_SR)
            audio_values = audio_values[np.newaxis, :].astype(np.float32)
            outputs = self.sess_speech.run(None, {"audio_values": audio_values})
        cond_emb, prompt_token, ref_x_vector, prompt_feat = outputs
        return cond_emb, prompt_token, ref_x_vector, prompt_feat

    def _lm_step(self, inputs_embeds, attention_mask, past_key_values):
        with _SanitizeStderr(drop_ort_warnings=True, pure_text=True):
            run_inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, **past_key_values}
            outputs = self.sess_lm.run(None, run_inputs)
        logits = outputs[0]
        present_key_values = outputs[1:]
        return logits, present_key_values

    def generate(self,
                 text: str,
                 language_id: str,
                 target_voice_path: str,
                 max_new_tokens: int = 256,
                 exaggeration: float = 0.5,
                 repetition_penalty: float = 1.2,
                 progress: bool = False,
                 ) -> np.ndarray:
        # Wrap entire generation to sanitize any stray logs
        with _SanitizeStderr(drop_ort_warnings=True, pure_text=True):
            # Prepare input ids and positions
            input_ids = self._prepare_text(text, language_id)
            position_ids = np.where(
                input_ids >= START_SPEECH_TOKEN,
                0,
                np.arange(input_ids.shape[1])[np.newaxis, :] - 1,
            ).astype(np.int64)
            embed_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "exaggeration": np.array([exaggeration], dtype=np.float32),
            }
            rep_penalty = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

            # Initial embeds and voice conditions
            inputs_embeds = self.sess_embed.run(None, embed_inputs)[0]
            cond_emb, prompt_token, ref_x_vector, prompt_feat = self._encode_voice(target_voice_path)
            inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = {
                f"past_key_values.{layer}.{kv}": np.zeros([batch_size, self.num_key_value_heads, 0, self.head_dim], dtype=np.float32)
                for layer in range(self.num_hidden_layers)
                for kv in ("key", "value")
            }
            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
            generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

            for i in range(max_new_tokens):
                logits, present_key_values = self._lm_step(inputs_embeds, attention_mask, past_key_values)
                logits_last = logits[:, -1, :]
                next_token_logits = rep_penalty(generate_tokens, logits_last)
                next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
                generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)
                if np.all(next_token.flatten() == STOP_SPEECH_TOKEN):
                    break
                # Update embedding for the new token
                position_ids = np.full((input_ids.shape[0], 1), i + 1, dtype=np.int64)
                embed_inputs["input_ids"] = next_token
                embed_inputs["position_ids"] = position_ids
                inputs_embeds = self.sess_embed.run(None, embed_inputs)[0]
                # Extend attention and cache
                attention_mask = np.concatenate([attention_mask, np.ones((attention_mask.shape[0], 1), dtype=np.int64)], axis=1)
                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]

            # Decode full waveform
            speech_tokens = generate_tokens[:, 1:-1]
            speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
            cond_inputs = {
                "speech_tokens": speech_tokens,
                "speaker_embeddings": ref_x_vector,
                "speaker_features": prompt_feat,
            }
            wav = self.sess_cond.run(None, cond_inputs)[0]
            return np.squeeze(wav, axis=0)

    def stream_audio(self, *args, **kwargs):
        # Wrap streaming generator to sanitize logs during iteration
        def _wrapped():
            with _SanitizeStderr(drop_ort_warnings=True, pure_text=True):
                for chunk in self._stream_audio_impl(*args, **kwargs):
                    yield chunk
        return _wrapped()

    def _stream_audio_impl(self,
                     text: str,
                     language_id: str,
                     target_voice_path: str,
                     max_new_tokens: int = 256,
                     exaggeration: float = 0.5,
                     repetition_penalty: float = 1.2,
                     emit_every_tokens: int = 12,
                     progress: bool = False,
                     ):
        """Yield incremental audio chunks (np.float32 mono) as tokens are generated.
        Implementation re-decodes with cond decoder each step and emits only new samples.
        """
        input_ids = self._prepare_text(text, language_id)
        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1,
        ).astype(np.int64)
        embed_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "exaggeration": np.array([exaggeration], dtype=np.float32),
        }
        rep_penalty = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        inputs_embeds = self.sess_embed.run(None, embed_inputs)[0]
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self._encode_voice(target_voice_path)
        inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
        batch_size, seq_len, _ = inputs_embeds.shape
        past_key_values = {
            f"past_key_values.{layer}.{kv}": np.zeros([batch_size, self.num_key_value_heads, 0, self.head_dim], dtype=np.float32)
            for layer in range(self.num_hidden_layers)
            for kv in ("key", "value")
        }
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

        emitted_samples = 0
        tok_since_emit = 0

        for i in range(max_new_tokens):
            logits, present_key_values = self._lm_step(inputs_embeds, attention_mask, past_key_values)
            logits_last = logits[:, -1, :]
            next_token_logits = rep_penalty(generate_tokens, logits_last)
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)
            tok_since_emit += 1

            stop = bool(np.all(next_token.flatten() == STOP_SPEECH_TOKEN))

            # Prepare next step embedding
            position_ids = np.full((input_ids.shape[0], 1), i + 1, dtype=np.int64)
            embed_inputs["input_ids"] = next_token
            embed_inputs["position_ids"] = position_ids
            inputs_embeds = self.sess_embed.run(None, embed_inputs)[0]

            # Update attention and KV cache
            attention_mask = np.concatenate([attention_mask, np.ones((attention_mask.shape[0], 1), dtype=np.int64)], axis=1)
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

            # Emit chunk if token batch collected or on stop
            if tok_since_emit >= emit_every_tokens or stop:
                # Decode using current tokens
                speech_tokens = generate_tokens[:, 1:-1]
                speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
                cond_inputs = {
                    "speech_tokens": speech_tokens,
                    "speaker_embeddings": ref_x_vector,
                    "speaker_features": prompt_feat,
                }
                wav = self.sess_cond.run(None, cond_inputs)[0]
                wav = np.squeeze(wav, axis=0)
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                if wav.shape[0] > emitted_samples:
                    new_chunk = wav[emitted_samples:]
                    emitted_samples = wav.shape[0]
                    if new_chunk.size > 0:
                        yield new_chunk.astype(np.float32, copy=False)
                tok_since_emit = 0

            if stop:
                break


# Backward-compatible function (no file writing); returns wav np.ndarray
def run_inference(
        text="The Lord of the Rings is the greatest work of literature.",
        language_id="en",
        target_voice_path=None,
        max_new_tokens=256,
        exaggeration=0.5,
        temperature=0.8,  # unused placeholder for compatibility
        cfg_weight=0.5,   # unused placeholder for compatibility
        output_dir="converted",
        output_file_name="output.wav",  # unused
        model_dir=None,
        tokenizer_dir=None,  # ignored, we use model_dir
        mapping_path=None,   # ignored, we use model_dir/Cangjie5_TC.json
):
    if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(f"Unsupported language_id '{language_id}'. Supported: {supported_langs}")

    if model_dir is None:
        model_dir = output_dir
    if not target_voice_path:
        target_voice_path = os.path.join(model_dir, "default_voice.wav")
    if not os.path.isfile(target_voice_path):
        raise FileNotFoundError(f"Missing voice sample: {target_voice_path}")

    engine = ChatterboxOnnxTTS(model_dir)
    wav = engine.generate(text=text,
                          language_id=language_id,
                          target_voice_path=target_voice_path,
                          max_new_tokens=int(max_new_tokens),
                          exaggeration=float(exaggeration))
    return wav, S3GEN_SR

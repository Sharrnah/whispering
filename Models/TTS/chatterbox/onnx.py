import onnxruntime

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
from unicodedata import category
import json

S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
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
                print("Cangjie mapping not found locally; Chinese segmentation will be limited")
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

        except Exception as e:
            print(f"Could not load Cangjie mapping: {e}")

    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            print("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None

    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)



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

        normalized_text = "".join(out)

        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', normalized_text)

        return normalized_text

    except ImportError:
        print("pykakasi not available - Japanese text processing skipped")
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta

    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            _dicta = Dicta()

        return _dicta.add_diacritics(text)

    except ImportError:
        print("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        print(f"Hebrew diacritization failed: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""

    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char

        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''

        return initial + medial + final

    # Decompose syllables and normalize punctuation
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

def run_inference(
        text="The Lord of the Rings is the greatest work of literature.",
        language_id="en",
        target_voice_path=None,
        max_new_tokens = 256,
        exaggeration=0.5,
        temperature=0.8,
        cfg_weight=0.5,
        output_dir="converted",
        output_file_name="output.wav",
        model_dir=None,
        tokenizer_dir=None,
        mapping_path=None,
):
    # Validate language_id
    if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(
            f"Unsupported language_id '{language_id}'. "
            f"Supported languages: {supported_langs}"
        )

    # Resolve local directories (no downloads)
    if model_dir is None:
        model_dir = output_dir
    if tokenizer_dir is None:
        tokenizer_dir = model_dir
    if mapping_path is None:
        mapping_path = os.path.join(model_dir, "Cangjie5_TC.json")
    if not target_voice_path:
        target_voice_path = os.path.join(model_dir, "default_voice.wav")

    # Validate required local files
    if not os.path.isfile(target_voice_path):
        raise FileNotFoundError(f"Missing voice sample: {target_voice_path}")

    onnx_dir = os.path.join(model_dir, "onnx")
    speech_encoder_path = os.path.join(onnx_dir, "speech_encoder.onnx")
    embed_tokens_path = os.path.join(onnx_dir, "embed_tokens.onnx")
    conditional_decoder_path = os.path.join(onnx_dir, "conditional_decoder.onnx")
    language_model_path = os.path.join(onnx_dir, "language_model_q4.onnx")

    required_files = [
        speech_encoder_path,
        embed_tokens_path,
        conditional_decoder_path,
        language_model_path,
    ]
    missing = [p for p in required_files if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing ONNX files: {missing}")

    # Start inference sessions strictly from local files
    speech_encoder_session = onnxruntime.InferenceSession(speech_encoder_path)
    embed_tokens_session = onnxruntime.InferenceSession(embed_tokens_path)
    llama_with_past_session = onnxruntime.InferenceSession(language_model_path)
    cond_decoder_session = onnxruntime.InferenceSession(conditional_decoder_path)

    def execute_text_to_audio_inference(text):
        print("Start inference script...")

        audio_values, _ = librosa.load(target_voice_path, sr=S3GEN_SR)
        audio_values = audio_values[np.newaxis, :].astype(np.float32)

        ## Prepare input
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        text = prepare_language(text, language_id, mapping_path)
        input_ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)

        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1
        )

        ort_embed_tokens_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids.astype(np.int64),
            "exaggeration": np.array([exaggeration],dtype=np.float32),
            #"temperature": np.array([temperature],dtype=np.float32),
            #"cfg": np.array([cfg_weight],dtype=np.float32)
        }

        ## Instantiate the logits processors.
        repetition_penalty = 1.2
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        num_hidden_layers = 30
        num_key_value_heads = 16
        head_dim = 64

        generate_tokens = np.array([[START_SPEECH_TOKEN]])

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):

            inputs_embeds = embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]
            if i == 0:
                ort_speech_encoder_input = {
                    "audio_values": audio_values,
                }
                cond_emb, prompt_token, ref_x_vector, prompt_feat = speech_encoder_session.run(None, ort_speech_encoder_input)
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)

                ## Prepare llm inputs
                batch_size, seq_len, _ = inputs_embeds.shape
                past_key_values = {
                    f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
                    for layer in range(num_hidden_layers)
                    for kv in ("key", "value")
                }
                attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
            logits, *present_key_values = llama_with_past_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **past_key_values,
            ))

            logits = logits[:, -1, :]
            next_token_logits = repetition_penalty_processor(generate_tokens, logits)

            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)
            if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                break

            # Get embedding for the new token.
            position_ids = np.full(
                (input_ids.shape[0], 1),
                i + 1,
                dtype=np.int64,
                )
            ort_embed_tokens_inputs["input_ids"] = next_token
            ort_embed_tokens_inputs["position_ids"] = position_ids

            ## Update values for next generation loop
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

        speech_tokens = generate_tokens[:, 1:-1]
        speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
        return speech_tokens, ref_x_vector, prompt_feat

    speech_tokens, speaker_embeddings, speaker_features = execute_text_to_audio_inference(text)
    cond_incoder_input = {
        "speech_tokens": speech_tokens,
        "speaker_embeddings": speaker_embeddings,
        "speaker_features": speaker_features,
    }
    wav = cond_decoder_session.run(None, cond_incoder_input)[0]
    wav = np.squeeze(wav, axis=0)

    sf.write(output_file_name, wav, S3GEN_SR)
    print(f"{output_file_name} was successfully saved")

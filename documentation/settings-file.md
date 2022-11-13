# Settings File
All possible options of the settings file.

_Default name is `settings.yaml`, but can be customized with the `--config` option._

```yaml
# whisper settings
ai_device: null  # can be null (auto), "cuda" or "cpu".
whisper_task: translate  # Whisper A.I. Can do "transcribe" or "translate"
current_language: null  # can be null (auto) or any Whisper supported language.
model: small  # Whisper model size. Can be "tiny", "base", "small", "medium" or "large"
condition_on_previous_text: true  # if enabled, Whisper will condition on previous text. (more prone to loops or getting stuck)

# text translate settings
txt_translate: false  # if enabled, pipes whisper A.I. results through text translator
src_lang: auto  # source language for text translator
trg_lang: fra_Latn  # target language for text translator
txt_ascii: false  # if enabled, text translator will convert text to romaji.
txt_translator: NLLB200  # can be "NLLB200", "M2M100" or "ARGOS"
txt_translator_size: small  # for M2M100 model size: Can be "small" or "large", for NLLB200 model size: Can be "small", "medium", "large".

# websocket settings
websocket_ip: 127.0.0.1
websocket_port: 5000

# OSC settings
osc_ip: '0'
osc_port: 9000
osc_address: /chatbox/input
osc_typing_indicator: true
osc_convert_ascii: false

# OCR settings
ocr_lang: en  # language for OCR image to text recognition.
ocr_window_name: VRChat  # window name for OCR image to text recognition.

# FLAN-T5 settings
flan_enabled: false  # Enable FLAN A.I.
flan_size: large  # FLAN model size. Can be "small", "base", "large", "xl" or "xxl"
flan_bits: 32  # precision can be set to 32 (float), 16 (float) or 8 (int) bits. 8 bits is the fastest but least precise
flan_device: cpu  # can be "cpu", "cuda" or "auto". ("cuda" and "auto" doing the same)
flan_whisper_answer: true  # if True, the FLAN A.I. will answer to results from the Whisper A.I.
flan_process_only_questions: true  # if True, the FLAN A.I. will only answer to questions
flan_osc_prefix: 'AI: '  # prefix for OSC messages
flan_translate_to_speaker_language: false  # Translate from english to speaker language
```

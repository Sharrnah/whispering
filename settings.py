# noinspection PyPackageRequirements
import yaml
import os
from pathlib import Path
from click import core
from whisper import available_models

SETTINGS_PATH = Path(Path.cwd() / 'settings.yaml')

TRANSLATE_SETTINGS = {
    # text translate settings
    "txt_translate": False,  # if enabled, pipes whisper A.I. results through text translator
    "txt_translator_device": "auto",  # auto, cuda, cpu
    "src_lang": "auto",  # source language for text translator (Whisper A.I. in translation mode always translates to "en")
    "trg_lang": "fra_Latn",  # target language for text translator
    "txt_ascii": False,  # if enabled, text translator will convert text to romaji.
    "txt_translator": "NLLB200",  # can be "NLLB200", "M2M100" or "ARGOS"
    "txt_translator_size": "small",  # for M2M100 model size: Can be "small" or "large", for NLLB200 model size: Can be "small", "medium", "large".

    # ocr settings
    "ocr_lang": "en",  # language for OCR image to text recognition.
    "ocr_window_name": "VRChat",  # window name for OCR image to text recognition.

    # whisper settings
    "ai_device": None,  # can be None (auto), "cuda" or "cpu".
    "whisper_task": "transcribe",  # Whisper A.I. Can do "transcribe" or "translate"
    "current_language": None,  # can be None (auto) or any Whisper supported language.
    "model": "small",  # Whisper model size. Can be "tiny", "base", "small", "medium" or "large"
    "condition_on_previous_text": False,  # if enabled, Whisper will condition on previous text. (more prone to loops or getting stuck)
    "energy": 300,  # energy of audio volume to start whisper processing. Can be 0-1000
    "phrase_time_limit": 0,  # time limit for Whisper to generate a phrase. (0 = no limit)
    "pause": 0.8,  # pause between phrases.

    # OSC settings
    "osc_ip": "127.0.0.1",  # OSC IP address. set to "0" to disable.
    "osc_port": 9000,
    "osc_address": "/chatbox/input",
    "osc_typing_indicator": True,
    "osc_convert_ascii": False,
    "osc_auto_processing_enabled": True,  # Toggle auto sending of OSC messages on WhisperAI or Flan results. (not saved)

    # websocket settings
    "websocket_ip": "127.0.0.1",
    "websocket_port": 5000,

    # TTS settings
    "tts_enabled": True,  # enable TTS
    "tts_ai_device": "cuda",  # can be "cuda" or "cpu".
    "tts_answer": True,  # answer to whisper results
    "device_out_index": None,  # output device index for TTS
    "tts_model": ["en", "v3_en"],  # TTS language and model to use
    "tts_voice": "en_0",  # TTS voice (one of silero tts voices, or "last" to use last used voice)

    # FLAN settings
    "flan_enabled": False,  # Enable FLAN A.I.
    "flan_size": "large",  # FLAN model size. Can be "small", "base", "large", "xl" or "xxl"
    "flan_bits": 32,  # precision can be set to 32 (float), 16 (float) or 8 (int) bits. 8 bits is the fastest but least precise
    "flan_device": "cpu",  # can be "cpu", "cuda" or "auto". ("cuda" and "auto" doing the same)
    "flan_whisper_answer": True,  # if True, the FLAN A.I. will answer to results from the Whisper A.I.
    "flan_process_only_questions": True,  # if True, the FLAN A.I. will only answer to questions.
    "flan_osc_prefix": "AI: ",  # prefix for OSC messages.
    "flan_translate_to_speaker_language": False,  # Translate from english to speaker language
    "flan_prompt": "",  # text for prompts or wraps prompt around input text if ?? (two question-marks) is present in the string. Otherwise, it is added to the end of the string.
    "flan_memory": "",  # longer term memory for FLAN A.I.
    "flan_conditioning_history": 0  # Number of previous messages to condition on. 0 for no conditioning.
}


def SetOption(setting, value):
    if setting in TRANSLATE_SETTINGS:
        if TRANSLATE_SETTINGS[setting] != value:
            TRANSLATE_SETTINGS[setting] = value
            # Save settings
            SaveYaml(SETTINGS_PATH)
    else:
        TRANSLATE_SETTINGS[setting] = value
        # Save settings
        SaveYaml(SETTINGS_PATH)
    return value

def GetOption(setting):
    return TRANSLATE_SETTINGS[setting]


def LoadYaml(path):
    print(path)
    if os.path.exists(path):
        with open(path, "r") as f:
            TRANSLATE_SETTINGS.update(yaml.safe_load(f))


def SaveYaml(path):
    to_save_settings = TRANSLATE_SETTINGS.copy()

    # Remove settings that are not saved
    if "whisper_languages" in to_save_settings:
        del to_save_settings['whisper_languages']
    if "lang_swap" in to_save_settings:
        del to_save_settings['lang_swap']
    if "verbose" in to_save_settings:
        del to_save_settings['verbose']
    if "transl_result_textarea_savetts_voice" in to_save_settings:
        del to_save_settings['transl_result_textarea_savetts_voice']
    if "transl_result_textarea_sendtts_download" in to_save_settings:
        del to_save_settings['transl_result_textarea_sendtts_download']

    with open(path, "w") as f:
        yaml.dump(to_save_settings, f)


def IsArgumentSetting(ctx, argument_name):
    return ctx.get_parameter_source(argument_name) == core.ParameterSource.COMMANDLINE


# Get Setting from argument if it is set, otherwise get setting from settings file
def GetArgumentSettingFallback(ctx, argument_name, fallback_setting_name):
    if IsArgumentSetting(ctx, argument_name):
        return ctx.params[argument_name]
    else:
        return GetOption(fallback_setting_name)


def GetAvailableSettingValues():
    possible_settings = {
        "ai_device": ["None", "cuda", "cpu"],
        "model": available_models(),
        "whisper_task": ["transcribe", "translate"],
        "tts_ai_device": ["cuda", "cpu"],
        "txt_translator": ["NLLB200", "M2M100", "ARGOS"],
        "txt_translator_size": ["small", "medium", "large"],
        "flan_device": ["cpu", "cuda"],
        "flan_bits": ["32", "16", "8"],
        "flan_size": ["small", "base", "large", "xl", "xxl"],
    }

    return possible_settings

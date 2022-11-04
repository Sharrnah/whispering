# noinspection PyPackageRequirements
import yaml
import os
from pathlib import Path
from click import core

SETTINGS_PATH = Path(Path.cwd() / 'settings.yaml')

TRANSLATE_SETTINGS = {
    # text translate settings
    "txt_translate": False,  # if enabled, pipes whisper A.I. results through text translator
    "src_lang": "en",  # source language for text translator (Whisper A.I. in translation mode always translates to "en")
    "trg_lang": "fr",  # target language for text translator
    "txt_ascii": False,  # if enabled, text translator will convert text to romaji.
    "txt_translator": "M2M100",  # can be "M2M100" or "ARGOS"
    "m2m100_size": "small",  # M2M100 model size. Can be "small" or "large"

    # ocr settings
    "ocr_lang": "en",  # language for OCR image to text recognition.
    "ocr_window_name": "VRChat",  # window name for OCR image to text recognition.

    # whisper settings
    "ai_device": None,  # can be None (auto), "cuda" or "cpu".
    "whisper_task": "transcribe",  # Whisper A.I. Can do "transcribe" or "translate"
    "current_language": None,  # can be None (auto) or any Whisper supported language.
    "model": "small",  # Whisper model size. Can be "tiny", "base", "small", "medium" or "large"
    "condition_on_previous_text": False,  # if enabled, Whisper will condition on previous text. (more prone to loops or getting stuck)

    # OSC settings
    "osc_ip": "0",
    "osc_port": 9000,
    "osc_address": "/chatbox/input",
    "osc_typing_indicator": True,
    "osc_convert_ascii": "False",

    # websocket settings
    "websocket_ip": "0",
    "websocket_port": 5000,

    # FLAN settings
    "flan_enabled": False,  # Enable FLAN A.I.
    "flan_size": "xl",  # FLAN model size. Can be "small", "base", "large", "xl" or "xxl"
    "flan_bits": 32,  # precision can be set to 32 (float), 16 (float) or 8 (int) bits. 8 bits is the fastest but least precise
    "flan_device": "cpu",  # can be "cpu", "cuda" or "auto". ("cuda" and "auto" doing the same)
    "flan_whisper_answer": True,  # if True, the FLAN A.I. will answer to results from the Whisper A.I.
    "flan_process_only_questions": True,  # if True, the FLAN A.I. will only answer to questions
    "flan_osc_prefix": "AI: "  # prefix for OSC messages
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


def GetOption(setting):
    return TRANSLATE_SETTINGS[setting]


def LoadYaml(path):
    print(path)
    if os.path.exists(path):
        with open(path, "r") as f:
            TRANSLATE_SETTINGS.update(yaml.safe_load(f))


def SaveYaml(path):
    to_save_settings = TRANSLATE_SETTINGS.copy()
    if "whisper_languages" in to_save_settings:
        del to_save_settings['whisper_languages']
    if "lang_swap" in to_save_settings:
        del to_save_settings['lang_swap']
    if "verbose" in to_save_settings:
        del to_save_settings['verbose']

    with open(path, "w") as f:
        yaml.dump(to_save_settings, f)


def IsArgumentSetting(ctx, argument_name):
    return ctx.get_parameter_source(argument_name) == core.ParameterSource.COMMANDLINE

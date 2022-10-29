TRANSLATE_SETTINGS = {
    # argostranslate settings
    "txt_translate": False,
    "src_lang": "en",
    "trg_lang": "fr",
    "txt_ascii": False,
    "txt_translator": "M2M100",  # can be "M2M100" or "ARGOS"

    # whisper settings
    "whisper_task": "transcribe"
}


def SetOption(setting, value):
    TRANSLATE_SETTINGS[setting] = value


def GetOption(setting):
    return TRANSLATE_SETTINGS[setting]

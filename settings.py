TRANSLATE_SETTINGS = {
    # argostranslate settings
    "txt_translate": False,
    "src_lang": "en",
    "trg_lang": "fr",
    "txt_ascii": False,

    # whisper settings
    "whisper_task": "transcribe"
}

def SetOption(setting, value):
    TRANSLATE_SETTINGS[setting] = value

def GetOption(setting):
    return TRANSLATE_SETTINGS[setting]

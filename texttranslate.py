import settings
import pykakasi
import texttranslateARGOS
#import texttranslateM2M100
import texttranslateM2M100_CTranslate2


def get_current_translator():
    return settings.GetOption("txt_translator")


def convert_to_romaji(text):
    # Convert Hiragana, Katakana, Japanese to romaji (ascii compatible)
    kks = pykakasi.kakasi()
    converted_text = kks.convert(text)
    full_converted_text = []
    for converted_text_item in converted_text:
        full_converted_text.append(converted_text_item['hepburn'])
    return ' '.join(full_converted_text)


# Download and install Translate packages
def InstallLanguages():
    match get_current_translator():
        case "ARGOS":
            texttranslateARGOS.InstallLanguages()
        case "M2M100":
            texttranslateM2M100_CTranslate2.load_model(settings.GetOption("m2m100_size"))


def GetInstalledLanguageNames():
    match get_current_translator():
        case "ARGOS":
            return texttranslateARGOS.GetInstalledLanguageNames()
        case "M2M100":
            return texttranslateM2M100_CTranslate2.get_installed_language_names()


def TranslateLanguage(text, from_code, to_code, to_romaji=False):
    translation_text = ""
    match get_current_translator():
        case "ARGOS":
            translation_text = texttranslateARGOS.TranslateLanguage(text, from_code, to_code)
        case "M2M100":
            translation_text = texttranslateM2M100_CTranslate2.translate_language(text, from_code, to_code)
    if to_romaji:
        translation_text = convert_to_romaji(translation_text)

    return translation_text.strip()


def SetDevice(option):
    texttranslateM2M100_CTranslate2.set_device(option)

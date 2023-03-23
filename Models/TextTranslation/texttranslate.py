import settings
import pykakasi
# import texttranslateM2M100
from Models.TextTranslation import texttranslateM2M100_CTranslate2
from Models.TextTranslation import texttranslateNLLB200
from Models.TextTranslation import texttranslateNLLB200_CTranslate2


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
        case "M2M100":
            texttranslateM2M100_CTranslate2.load_model(settings.GetOption("txt_translator_size"))
        case "NLLB200":
            texttranslateNLLB200.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))
        case "NLLB200_CT2":
            texttranslateNLLB200_CTranslate2.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))


def GetInstalledLanguageNames():
    match get_current_translator():
        case "M2M100":
            return texttranslateM2M100_CTranslate2.get_installed_language_names()
        case "NLLB200":
            return texttranslateNLLB200.get_installed_language_names()
        case "NLLB200_CT2":
            return texttranslateNLLB200_CTranslate2.get_installed_language_names()


def TranslateLanguage(text, from_code, to_code, to_romaji=False, as_iso1=False):
    translation_text = ""
    match get_current_translator():
        case "M2M100":
            try:
                translation_text = texttranslateM2M100_CTranslate2.translate_language(text, from_code, to_code)
            except Exception as e:
                print("Error: " + str(e))
        case "NLLB200":
            try:
                translation_text, from_code, to_code = texttranslateNLLB200.translate_language(text, from_code, to_code, as_iso1)
            except Exception as e:
                print("Error: " + str(e))
        case "NLLB200_CT2":
            try:
                translation_text, from_code, to_code = texttranslateNLLB200_CTranslate2.translate_language(text, from_code, to_code, as_iso1)
            except Exception as e:
                print("Error: " + str(e))
    if to_romaji:
        translation_text = convert_to_romaji(translation_text)

    return translation_text.strip(), from_code, to_code


def SetDevice(option):
    texttranslateNLLB200.set_device(option)
    texttranslateNLLB200_CTranslate2.set_device(option)
    texttranslateM2M100_CTranslate2.set_device(option)

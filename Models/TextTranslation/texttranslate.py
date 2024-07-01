import settings
import pykakasi
# import texttranslateM2M100
from Models.TextTranslation import texttranslateM2M100_CTranslate2
from Models.TextTranslation import texttranslateNLLB200
from Models.TextTranslation import texttranslateNLLB200_CTranslate2
from Models.Multi.seamless_m4t import SeamlessM4T

import Plugins

txt_translator_instance = None


def get_current_translator():
    return settings.GetOption("txt_translator")


def iso3_to_iso1(iso3_code):
    for iso1, iso3_codes in texttranslateNLLB200_CTranslate2.LANGUAGES_ISO1_TO_ISO3.items():
        if iso3_code in iso3_codes:
            return iso1
    return None


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
    global txt_translator_instance
    match get_current_translator():
        case "M2M100":
            texttranslateM2M100_CTranslate2.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))
        case "NLLB200":
            texttranslateNLLB200.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))
        case "NLLB200_CT2":
            texttranslateNLLB200_CTranslate2.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))
        case "Seamless_M4T":
            txt_translator_instance = SeamlessM4T(
                model=settings.GetOption("txt_translator_size"),
                compute_type=settings.GetOption("txt_translator_precision"),
                device=settings.GetOption("txt_translator_device")
            )


def GetInstalledLanguageNames():
    global txt_translator_instance
    match get_current_translator():
        case "M2M100":
            return texttranslateM2M100_CTranslate2.get_installed_language_names()
        case "NLLB200":
            return texttranslateNLLB200.get_installed_language_names()
        case "NLLB200_CT2":
            return texttranslateNLLB200_CTranslate2.get_installed_language_names()
        case "Seamless_M4T":
            return SeamlessM4T.get_languages()
        case _:
            try:
                # call custom plugin event method
                plugin_translation = Plugins.plugin_custom_event_call('plugin_get_languages', {})
                if plugin_translation is not None and 'languages' in plugin_translation and plugin_translation['languages'] is not None:
                    return plugin_translation['languages']
            except Exception as e:
                print("Error: " + str(e))


def TranslateLanguage(text, from_code, to_code, to_romaji=False, as_iso1=False):
    global txt_translator_instance
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
        case "Seamless_M4T":
            try:
                translation_text, from_code, to_code = txt_translator_instance.text_translate(text, from_code, to_code)
            except Exception as e:
                print("Error: " + str(e))
        case _:
            for plugin_inst in Plugins.plugins:
                try:
                    if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'text_translate'):
                        translation_text, from_code, to_code = plugin_inst.text_translate(text, from_code, to_code)
                except Exception as e:
                    print(f"Error in Plugin {plugin_inst.__class__.__name__}: " + str(e))

    if to_romaji:
        translation_text = convert_to_romaji(translation_text)

    return translation_text.strip(), from_code, to_code


def SetDevice(option):
    global txt_translator_instance
    texttranslateNLLB200.set_device(option)
    texttranslateNLLB200_CTranslate2.set_device(option)
    texttranslateM2M100_CTranslate2.set_device(option)
    if txt_translator_instance is not None:
        txt_translator_instance.set_device(option)

import traceback

import settings
import pykakasi
# import texttranslateM2M100
from Models.TextTranslation import texttranslateM2M100_CTranslate2
from Models.TextTranslation import texttranslateNLLB200
from Models.TextTranslation import texttranslateNLLB200_CTranslate2
from Models.Multi.seamless_m4t import SeamlessM4T
from Models.Multi.phi4 import Phi4
from Models.Multi.voxtral import Voxtral

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
        case "seamless_m4t":
            txt_translator_instance = SeamlessM4T(
                model=settings.GetOption("txt_translator_size"),
                compute_type=settings.GetOption("txt_translator_precision"),
                device=settings.GetOption("txt_translator_device")
            )
        case "phi4":
            txt_translator_instance = Phi4(
                compute_type=settings.GetOption("txt_translator_precision"),
                device=settings.GetOption("txt_translator_device")
            )
        case "voxtral":
            txt_translator_instance = Voxtral(
                compute_type=settings.GetOption("txt_translator_precision"),
                device=settings.GetOption("txt_translator_device")
            )


def GetInstalledLanguageNames():
    match get_current_translator():
        case "M2M100":
            return texttranslateM2M100_CTranslate2.get_installed_language_names()
        case "NLLB200":
            return texttranslateNLLB200.get_installed_language_names()
        case "NLLB200_CT2":
            return texttranslateNLLB200_CTranslate2.get_installed_language_names()
        case "seamless_m4t":
            return SeamlessM4T.get_languages()
        case "phi4":
            return Phi4.get_languages()
        case "voxtral":
            return Voxtral.get_languages()
        case _:
            try:
                # call custom plugin event method
                plugin_translation = Plugins.plugin_custom_event_call('plugin_get_languages', {})
                if plugin_translation is not None and 'languages' in plugin_translation and plugin_translation['languages'] is not None:
                    return plugin_translation['languages']
            except Exception as e:
                print("Error: " + str(e))
                traceback.print_exc()


def TranslateLanguage(text, from_code, to_code, to_romaji=False, as_iso1=False):
    global txt_translator_instance
    translation_text = text
    match get_current_translator():
        case "M2M100":
            try:
                translation_text = texttranslateM2M100_CTranslate2.translate_language(text, from_code, to_code)
            except Exception as e:
                print("Error: " + str(e))
                traceback.print_exc()
        case "NLLB200":
            try:
                translation_text, from_code, to_code = texttranslateNLLB200.translate_language(text, from_code, to_code, as_iso1)
            except Exception as e:
                print("Error: " + str(e))
                traceback.print_exc()
        case "NLLB200_CT2":
            try:
                translation_text, from_code, to_code = texttranslateNLLB200_CTranslate2.translate_language(text, from_code, to_code, as_iso1)
            except Exception as e:
                print("Error: " + str(e))
                traceback.print_exc()
        case "seamless_m4t":
            try:
                translation_text, from_code, to_code = txt_translator_instance.text_translate(text, from_code, to_code)
            except Exception as e:
                print("Error: " + str(e))
                traceback.print_exc()
        case "phi4":
            try:
                response_dict = txt_translator_instance.transcribe(
                    None,
                    task='text_translate',
                    chat_message=text,
                    language=to_code,
                )
                translation_text, from_code, to_code = response_dict['text'], '', response_dict['language']
            except Exception as e:
                print("Error: " + str(e))
                traceback.print_exc()
        case "voxtral":
            try:
                response_dict = txt_translator_instance.transcribe(
                    None,
                    task='text_translate',
                    chat_message=text,
                    language=to_code,
                )
                translation_text, from_code, to_code = response_dict['text'], '', response_dict['language']
            except Exception as e:
                print("Error: " + str(e))
                traceback.print_exc()
        case _:
            for plugin_inst in Plugins.plugins:
                try:
                    if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'text_translate'):
                        translation_text, from_code, to_code = plugin_inst.text_translate(text, from_code, to_code)
                except Exception as e:
                    print(f"Error in Plugin {plugin_inst.__class__.__name__}: " + str(e))
                    traceback.print_exc()

    if to_romaji:
        translation_text = convert_to_romaji(translation_text)

    return translation_text.strip(), from_code, to_code


def SetDevice(option):
    global txt_translator_instance
    texttranslateNLLB200.set_device(option)
    texttranslateNLLB200_CTranslate2.set_device(option)
    texttranslateM2M100_CTranslate2.set_device(option)
    if txt_translator_instance is not None and hasattr(txt_translator_instance, 'set_device'):
        txt_translator_instance.set_device(option)

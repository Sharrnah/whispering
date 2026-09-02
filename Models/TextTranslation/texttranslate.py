import threading
import traceback

import settings
import pykakasi
from Models.ai_device import (
    get_ctranslate2_device,
    get_device,
    resolve_device,
    split_ctranslate2_device,
)
from Models.cuda_inference import cuda_inference_guarded
# import texttranslateM2M100
from Models.TextTranslation import texttranslateM2M100_CTranslate2
from Models.TextTranslation import texttranslateNLLB200
from Models.TextTranslation import texttranslateNLLB200_CTranslate2
from Models.TextTranslation import texttranslate_hunyuan
from Models.TextTranslation import texttranslate_milmmt
from Models.Multi.seamless_m4t import SeamlessM4T
from Models.Multi.phi4 import Phi4
from Models.Multi.voxtral import Voxtral

import Plugins

txt_translator_instance = None
_active_translator_configuration = None
_translation_lock = threading.RLock()
_BUILTIN_TRANSLATORS = {
    "M2M100",
    "NLLB200",
    "NLLB200_CT2",
    "hunyuan_mt",
    "milmmt",
    "seamless_m4t",
    "phi4",
    "voxtral",
}


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
def _translator_configuration(translator):
    return (
        translator,
        settings.GetOption("txt_translator_size"),
        settings.GetOption("txt_translator_precision"),
        get_device("txt_translator_device", "txt_translator_device_index"),
    )


def _translator_is_ready(translator):
    if translator == "hunyuan_mt":
        return texttranslate_hunyuan.is_model_loaded()
    if translator == "milmmt":
        return texttranslate_milmmt.is_model_loaded()
    if translator in {"seamless_m4t", "phi4", "voxtral"}:
        return txt_translator_instance is not None
    return True


def _release_inactive_transformers_models(translator):
    if translator != "hunyuan_mt" and texttranslate_hunyuan.is_model_loaded():
        texttranslate_hunyuan.release_model()
    if translator != "milmmt" and texttranslate_milmmt.is_model_loaded():
        texttranslate_milmmt.release_model()


def _install_languages(translator):
    global txt_translator_instance
    global _active_translator_configuration

    configuration = _translator_configuration(translator)
    if (
        _active_translator_configuration == configuration
        and _translator_is_ready(translator)
    ):
        return

    _active_translator_configuration = None
    _release_inactive_transformers_models(translator)
    selected_device = get_device("txt_translator_device", "txt_translator_device_index")
    ctranslate_device, ctranslate_device_index = get_ctranslate2_device(
        "txt_translator_device", "txt_translator_device_index"
    )

    match translator:
        case "M2M100":
            texttranslateM2M100_CTranslate2.set_device(ctranslate_device, ctranslate_device_index)
            texttranslateM2M100_CTranslate2.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))
        case "NLLB200":
            texttranslateNLLB200.set_device(selected_device)
            texttranslateNLLB200.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))
        case "NLLB200_CT2":
            texttranslateNLLB200_CTranslate2.set_device(ctranslate_device, ctranslate_device_index)
            texttranslateNLLB200_CTranslate2.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))
        case "hunyuan_mt":
            texttranslate_hunyuan.set_device(selected_device)
            texttranslate_hunyuan.load_model(settings.GetOption("txt_translator_size"), compute_type=settings.GetOption("txt_translator_precision"))
        case "milmmt":
            texttranslate_milmmt.set_device(selected_device)
            texttranslate_milmmt.load_model(
                settings.GetOption("txt_translator_size"),
                compute_type=settings.GetOption("txt_translator_precision"),
            )
        case "seamless_m4t":
            txt_translator_instance = SeamlessM4T(
                model=settings.GetOption("txt_translator_size"),
                compute_type=settings.GetOption("txt_translator_precision"),
                device=selected_device
            )
        case "phi4":
            txt_translator_instance = Phi4(
                compute_type=settings.GetOption("txt_translator_precision"),
                device=selected_device
            )
            txt_translator_instance.load_model()
        case "voxtral":
            txt_translator_instance = Voxtral(
                compute_type=settings.GetOption("txt_translator_precision"),
                device=selected_device
            )
            txt_translator_instance.load_model()

    _active_translator_configuration = configuration


@cuda_inference_guarded(
    lambda: get_device("txt_translator_device", "txt_translator_device_index"),
    lambda: f"Text translation/{get_current_translator()}.load",
    runtime_key=lambda: ("text_translation", get_current_translator()),
)
def InstallLanguages():
    with _translation_lock:
        _install_languages(get_current_translator())


def GetInstalledLanguageNames():
    match get_current_translator():
        case "M2M100":
            return texttranslateM2M100_CTranslate2.get_installed_language_names()
        case "NLLB200":
            return texttranslateNLLB200.get_installed_language_names()
        case "NLLB200_CT2":
            return texttranslateNLLB200_CTranslate2.get_installed_language_names()
        case "hunyuan_mt":
            return texttranslate_hunyuan.get_installed_language_names()
        case "milmmt":
            return texttranslate_milmmt.get_installed_language_names()
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


@cuda_inference_guarded(
    lambda: get_device("txt_translator_device", "txt_translator_device_index"),
    lambda: f"Text translation/{get_current_translator()}.translate",
    runtime_key=lambda: ("text_translation", get_current_translator()),
)
def TranslateLanguage(text, from_code, to_code, to_romaji=False, as_iso1=False):
    global txt_translator_instance
    translation_text = text
    translator = get_current_translator()

    if translator in _BUILTIN_TRANSLATORS:
        try:
            with _translation_lock:
                _install_languages(translator)
                match translator:
                    case "M2M100":
                        translation_text = texttranslateM2M100_CTranslate2.translate_language(text, from_code, to_code)
                    case "NLLB200":
                        translation_text, from_code, to_code = texttranslateNLLB200.translate_language(text, from_code, to_code, as_iso1)
                    case "NLLB200_CT2":
                        translation_text, from_code, to_code = texttranslateNLLB200_CTranslate2.translate_language(text, from_code, to_code, as_iso1)
                    case "hunyuan_mt":
                        translation_text, from_code, to_code = texttranslate_hunyuan.translate_language(text, from_code, to_code, as_iso1)
                    case "milmmt":
                        translation_text, from_code, to_code = texttranslate_milmmt.translate_language(
                            text,
                            from_code,
                            to_code,
                            as_iso1,
                        )
                    case "seamless_m4t":
                        translation_text, from_code, to_code = txt_translator_instance.text_translate(text, from_code, to_code)
                    case "phi4":
                        response_dict = txt_translator_instance.transcribe(
                            None,
                            task='text_translate',
                            chat_message=text,
                            language=to_code,
                        )
                        translation_text, from_code, to_code = response_dict['text'], '', response_dict['language']
                    case "voxtral":
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
    else:
        for plugin_inst in Plugins.plugins:
            try:
                if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'text_translate'):
                    translation_text, from_code, to_code = plugin_inst.text_translate(text, from_code, to_code)
            except Exception as e:
                print(f"Error in Plugin {plugin_inst.__class__.__name__}: " + str(e))
                traceback.print_exc()

    if str(text or "").strip() and not str(translation_text or "").strip():
        print(f"Error: {translator or 'text translator'} returned an empty translation; using the source text.")
        translation_text = text

    if to_romaji:
        translation_text = convert_to_romaji(translation_text)

    return str(translation_text or "").strip(), from_code, to_code


def SetDevice(option):
    global txt_translator_instance

    selected_device = resolve_device(
        option,
        settings.GetOption("txt_translator_device_index"),
    )
    ctranslate_device, ctranslate_device_index = split_ctranslate2_device(selected_device)

    match get_current_translator():
        case "NLLB200":
            texttranslateNLLB200.set_device(selected_device)
        case "NLLB200_CT2":
            texttranslateNLLB200_CTranslate2.set_device(ctranslate_device, ctranslate_device_index)
        case "M2M100":
            texttranslateM2M100_CTranslate2.set_device(ctranslate_device, ctranslate_device_index)
        case "hunyuan_mt":
            texttranslate_hunyuan.set_device(selected_device)
        case "milmmt":
            texttranslate_milmmt.set_device(selected_device)
        case _:
            if txt_translator_instance is not None and hasattr(txt_translator_instance, 'set_device'):
                txt_translator_instance.set_device(selected_device)

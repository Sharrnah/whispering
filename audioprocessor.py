import sys
import threading
import queue
import time
import traceback

import whisper

import Utilities
import audio_tools
#import settings
from settings import SETTINGS as main_settings
import VRC_OSCLib
from Models import sentence_split
from Models.TextTranslation import texttranslate
import websocket
import json
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
import io
from Models.TTS import tts

# from faster_whisper import WhisperModel
import Models.STT.faster_whisper as faster_whisper
import Models.STT.whisper_audio_markers as whisper_audio_markers
import Models.STT.speecht5 as speech_t5
#import Models.STT.whisper_cpp as whisper_cpp
import Models.STT.tansformer_whisper as transformer_whisper
import Models.STT.medusa_whisper as medusa_whisper
#import Models.STT.tensorrt_whisper as tensorrt_whisper
import Models.STT.wav2vec_bert as wav2vec_bert
import Models.STT.nemo_canary as nemo_canary
import Models.Multi.seamless_m4t as seamless_m4t
import Models.Multi.mms as mms
# import Models.STT.whisperx as whisperx
import csv

# Plugins
import Plugins

ignore_list = []




# some regular mistakenly recognized words/sentences on mostly silence audio, which are ignored in processing
def load_ignore_list(filename):
    global ignore_list
    if Path(Path.cwd() / filename).is_file():
        with open(str(Path(Path.cwd() / filename).resolve()), "rb") as ignore_list_file:
            content = ignore_list_file.read()
            decoded_content = Utilities.safe_decode(content)
            ignore_list.extend(decoded_content.splitlines())
    else:
        with open(str(Path(Path.cwd() / filename).resolve()), "wb") as ignore_list_file:
            ignore_list_file.write(b"")


load_ignore_list("ignorelist.txt")
load_ignore_list("ignorelist.custom.txt")

# make all list entries lowercase and strip space, tab, CR, LF etc. for later comparison
ignore_list = list((map(lambda x: x.lower().rstrip(), ignore_list)))

max_queue_size = 10
queue_timeout = 5

last_audio_timestamp = 0

q = queue.Queue(maxsize=max_queue_size)

#final_audio = False
queue_data = None
audio = None
audio_timestamp = None


def whisper_get_languages_list_keys():
    return sorted(LANGUAGES.keys())


def whisper_get_languages_list():
    return sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()])


def whisper_get_languages():
    languages = {
        "": "Auto",
        **LANGUAGES
    }

    return tuple([{"code": code, "name": language} for code, language in languages.items()])


def seamless_m4t_get_languages():
    languages = {
        "": "Auto",
        **seamless_m4t.LANGUAGES
    }
    return tuple([{"code": code, "name": language} for code, language in languages.items()])


def mms_get_languages():
    languages = {
        "": "Auto",
        **mms.LANGUAGES
    }
    return tuple([{"code": code, "name": language} for code, language in languages.items()])


def wav2vec_bert_get_languages():
    wav2vec_bert_model = wav2vec_bert.Wav2VecBert()
    return wav2vec_bert_model.get_languages()


def nemo_canary_get_languages():
    return nemo_canary.NemoCanary.get_languages()


def remove_repetitions(text, language='english', settings=main_settings):
    do_txt_translate = settings.GetOption("txt_translate")
    src_lang = settings.GetOption("src_lang")
    if src_lang is not None:
        src_lang = src_lang.lower()
    # Try to prevent sentence repetition
    max_sentence_repetition = int(settings.GetOption("max_sentence_repetition"))
    if max_sentence_repetition > -1 and text != "":
        sentence_split_language = ""
        if language is not None and language != "":
            sentence_split_language = language
        if sentence_split_language == "" and do_txt_translate and src_lang is not None and src_lang != "auto":
            sentence_split_language = src_lang
        if sentence_split_language == "":
            sentence_split_language = "english"
        return sentence_split.remove_repeated_sentences(text, language=sentence_split_language,
                                                        max_repeat=max_sentence_repetition)
    return text


def whisper_result_handling(result, audio_timestamp, final_audio, settings, plugins):
    global last_audio_timestamp
    verbose = settings.GetOption("verbose")
    osc_ip = settings.GetOption("osc_ip")
    do_txt_translate = settings.GetOption("txt_translate")
    transcription_auto_save_file = settings.GetOption("transcription_auto_save_file")
    transcription_auto_save_continuous_text = settings.GetOption("transcription_auto_save_continuous_text")

    predicted_text = result.get('text').strip()
    result["type"] = "transcript"

    # Try to prevent sentence repetition
    sentence_split_language = "english"
    if "language" in result:
        sentence_split_language = result["language"]
    predicted_text = remove_repetitions(predicted_text, language=sentence_split_language, settings=settings)
    if "text" in result:
        result["text"] = predicted_text

    original_text = predicted_text

    if not predicted_text.lower() in ignore_list and \
            (final_audio or (not final_audio and audio_timestamp > last_audio_timestamp)):

        if final_audio:
            if not verbose:
                try:
                    print("Transcribe" + (" (OSC)" if osc_ip != "0" else "") + ": " + predicted_text.encode('utf-8',
                                                                                                            'ignore').decode(
                        'utf-8', 'ignore'))
                except:
                    print("Transcribe" + (" (OSC)" if osc_ip != "0" else "") + ": ???")

            else:
                try:
                    print(result.encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
                except:
                    print("???")

        # translate using text translator if enabled
        # translate text realtime or after audio is finished
        if do_txt_translate and settings.GetOption("txt_translate_realtime") or \
                do_txt_translate and not settings.GetOption("txt_translate_realtime") and final_audio:
            from_lang = settings.GetOption("src_lang")
            to_lang = settings.GetOption("trg_lang")
            to_romaji = settings.GetOption("txt_romaji")
            second_translation_enabled = settings.GetOption("txt_second_translation_enabled")
            second_translation_languages = settings.GetOption("txt_second_translation_languages")

            # main translation
            predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(original_text, from_lang,
                                                                                         to_lang, to_romaji)

            # split second_translation language codes at comma with trim if enabled
            second_translation_texts = {}
            if second_translation_enabled and second_translation_languages!= "":
                second_translation_split_codes = [st.strip() for st in second_translation_languages.split(",")]
                for split_code in second_translation_split_codes:
                    if split_code != "":
                        second_translation_text, second_txt_from_lang, second_txt_to_lang = texttranslate.TranslateLanguage(
                            original_text, from_lang, split_code, False)
                        second_translation_texts[second_txt_to_lang] = second_translation_text

            result["txt_translation"] = predicted_text
            result["txt_translation_source"] = txt_from_lang
            result["txt_translation_target"] = to_lang

            # combine all translations second_translation_texts to result with wrap
            if second_translation_enabled and second_translation_texts:
                result["txt_second_translation"] = second_translation_texts

        if final_audio:
            if "txt_translation" in result:
                translation_text = predicted_text
            else:
                translation_text = ""

            Utilities.add_transcription(audio_timestamp, time.time_ns(), result["text"], translation_text,
                                        transcription_auto_save_continuous_text, transcription_auto_save_file
                              )

        # send realtime processing data to websocket
        if not final_audio and predicted_text.strip() != "" and settings.GetOption("websocket_ip") != "0" and settings.GetOption("websocket_ip") != "":
            websocket.BroadcastMessage(json.dumps({"type": "processing_data", "data": predicted_text}))
            # threading.Thread(
            #    target=websocket.BroadcastMessage,
            #    args=(json.dumps({"type": "processing_data", "data": predicted_text}),)
            # ).start()

        # send regular message
        send_message(predicted_text, result, final_audio, settings, plugins)

        last_audio_timestamp = audio_timestamp


def plugin_process(plugins, predicted_text, result_obj, final_audio, settings):
    for plugin_inst in plugins:
        if final_audio:
            if hasattr(plugin_inst, 'stt'):
                try:
                    plugin_inst.stt(predicted_text, result_obj)
                except Exception as e:
                    print(f"Error while processing plugin stt in Plugin {plugin_inst.__class__.__name__}: " + str(e))
                    traceback.print_exc()
        else:
            if hasattr(plugin_inst, 'stt_intermediate'):
                try:
                    plugin_inst.stt_intermediate(predicted_text, result_obj)
                except Exception as e:
                    print(f"Error while processing plugin stt_intermediate in Plugin {plugin_inst.__class__.__name__}: " + str(e))
                    traceback.print_exc()
    audio_processor_call_name = settings.GetOption("audio_processor_caller")
    if audio_processor_call_name is not None and audio_processor_call_name != "":
        Plugins.internal_plugin_custom_event_call(plugins, "audio_processor_stt_"+audio_processor_call_name, {
            "text": predicted_text,
            "result_obj": result_obj,
            "final_audio": final_audio
        })

def plugin_process_stt_processing(current_audio_timestamp, audio_data, sample_rate, final_audio, settings, plugins):
    for plugin_inst in plugins:
        if hasattr(plugin_inst, 'stt_processing'):
            try:
                result_obj = plugin_inst.stt_processing(audio_data, sample_rate, final_audio)
                if result_obj is not None:
                    whisper_result_thread(result_obj, current_audio_timestamp, final_audio, settings, plugins)
            except Exception as e:
                print(f"Error while processing plugin stt_result in Plugin {plugin_inst.__class__.__name__}: " + str(e))
                traceback.print_exc()

def replace_osc_placeholders(text, result_obj, settings):
    txt_translate_enabled = settings.GetOption("txt_translate")
    whisper_task = settings.GetOption("whisper_task")

    # replace \n with new line
    text = text.replace("\\n", "\n")

    # replace {src} with source language
    if "language" in result_obj and result_obj["language"] is not None:
        text = text.replace("{src}", result_obj["language"])
    elif "language" in result_obj and result_obj["language"] is None:
        text = text.replace("{src}", "?")

    if txt_translate_enabled and "txt_translation" in result_obj and "txt_translation_target" in result_obj:
        # replace {trg} with target language
        target_language = texttranslate.iso3_to_iso1(result_obj["txt_translation_target"])
        if target_language is None:
            target_language = result_obj["txt_translation_target"]
        if target_language is not None:
            text = text.replace("{trg}", target_language)
    else:
        if "target_lang" in result_obj and result_obj["target_lang"] is not None:
            # replace {trg} with target language of whisper
            text = text.replace("{trg}", result_obj["target_lang"])
        elif whisper_task == "transcribe":
            # replace {trg} with target language of whisper
            if "language" in result_obj and result_obj["language"] is not None:
                text = text.replace("{trg}", result_obj["language"])
        elif whisper_task == "translate":
            # replace {trg} with target language of whisper
            text = text.replace("{trg}", "en")
        else:
            text = text.replace("{trg}", "?")
    return text


# replace {src} and {trg} with source and target language in osc prefix
def build_whisper_translation_osc_prefix(result_obj, settings):
    prefix = settings.GetOption("osc_chat_prefix")

    return replace_osc_placeholders(prefix, result_obj, settings)


def send_message(predicted_text, result_obj, final_audio, settings, plugins):
    osc_ip = settings.GetOption("osc_ip")
    osc_address = settings.GetOption("osc_address")
    osc_port = settings.GetOption("osc_port")
    websocket_ip = settings.GetOption("websocket_ip")

    second_translation_enabled = settings.GetOption("txt_second_translation_enabled")
    second_translation_wrap = settings.GetOption("txt_second_translation_wrap")
    second_translation_wrap = second_translation_wrap.replace("\\n", "\n")
    second_translations = None
    if "txt_second_translation" in result_obj:
        second_translations = result_obj["txt_second_translation"]

    # Update osc_min_time_between_messages option
    VRC_OSCLib.set_min_time_between_messages(settings.GetOption("osc_min_time_between_messages"))

    # WORKAROUND: prevent it from outputting the initial prompt.
    if predicted_text == settings.GetOption("initial_prompt"):
        return

    # process plugins
    if ((final_audio and not settings.GetOption("realtime")) or settings.GetOption("realtime")) and plugins is not None:
        plugin_thread = threading.Thread(target=plugin_process, args=(plugins, predicted_text, result_obj, final_audio, settings,))
        plugin_thread.start()

    # Send over OSC
    if osc_ip != "0" and settings.GetOption("osc_auto_processing_enabled") and predicted_text != "":
        osc_notify = final_audio and settings.GetOption("osc_typing_indicator")

        osc_send_type = settings.GetOption("osc_send_type")
        osc_chat_limit = settings.GetOption("osc_chat_limit")
        osc_time_limit = settings.GetOption("osc_time_limit")
        osc_scroll_time_limit = settings.GetOption("osc_scroll_time_limit")
        osc_initial_time_limit = settings.GetOption("osc_initial_time_limit")
        osc_scroll_size = settings.GetOption("osc_scroll_size")
        osc_max_scroll_size = settings.GetOption("osc_max_scroll_size")
        osc_type_transfer_split = settings.GetOption("osc_type_transfer_split")
        osc_type_transfer_split = replace_osc_placeholders(osc_type_transfer_split, result_obj, settings)

        osc_text = predicted_text
        if "text" in result_obj and result_obj["text"] is not None and result_obj["text"] != "":
            if settings.GetOption("osc_type_transfer") == "source":
                osc_text = result_obj["text"]
            elif settings.GetOption("osc_type_transfer") == "both":
                if predicted_text != result_obj["text"]:
                    osc_text = result_obj["text"] + osc_type_transfer_split + predicted_text
            elif settings.GetOption("osc_type_transfer") == "both_inverted":
                if predicted_text != result_obj["text"]:
                    osc_text = predicted_text + osc_type_transfer_split + result_obj["text"]

        message = build_whisper_translation_osc_prefix(result_obj, settings) + osc_text

        if second_translation_enabled and second_translations:
            for lang, text in second_translations.items():
                message += second_translation_wrap + text
                result_obj["txt_translation"] += second_translation_wrap + text
                result_obj["txt_translation_target"] += "|"+lang

        # delay sending message if it is the final audio and until TTS starts playing
        if final_audio and settings.GetOption("osc_delay_until_audio_playback"):
            # wait until is_audio_playing is True or timeout is reached
            delay_timeout = time.time() + settings.GetOption("osc_delay_timeout")
            tag = settings.GetOption("osc_delay_until_audio_playback_tag")
            tts_answer = settings.GetOption("tts_answer")
            if tag == "tts" and tts_answer:
                while not audio_tools.is_audio_playing(tag=tag) and time.time() < delay_timeout:
                    time.sleep(0.05)

        if osc_send_type == "full":
            VRC_OSCLib.Chat(message, True, osc_notify, osc_address,
                            IP=osc_ip, PORT=osc_port,
                            convert_ascii=settings.GetOption("osc_convert_ascii"))
        elif osc_send_type == "chunks":
            VRC_OSCLib.Chat_chunks(message,
                                   nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                   chunk_size=osc_chat_limit, delay=osc_time_limit,
                                   initial_delay=osc_initial_time_limit,
                                   convert_ascii=settings.GetOption("osc_convert_ascii"))
        elif osc_send_type == "scroll":
            VRC_OSCLib.Chat_scrolling_chunks(message,
                                             nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                             chunk_size=osc_max_scroll_size, delay=osc_scroll_time_limit,
                                             initial_delay=osc_initial_time_limit,
                                             scroll_size=osc_scroll_size,
                                             convert_ascii=settings.GetOption("osc_convert_ascii"))
        elif osc_send_type == "full_or_scroll":
            # send full if message fits in osc_chat_limit, otherwise send scrolling chunks
            if len(message.encode('utf-16le')) <= osc_chat_limit * 2:
                VRC_OSCLib.Chat(message, True, osc_notify, osc_address,
                                IP=osc_ip, PORT=osc_port,
                                convert_ascii=settings.GetOption("osc_convert_ascii"))
            else:
                VRC_OSCLib.Chat_scrolling_chunks(message,
                                                 nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                                 chunk_size=osc_chat_limit, delay=osc_scroll_time_limit,
                                                 initial_delay=osc_initial_time_limit,
                                                 scroll_size=osc_scroll_size,
                                                 convert_ascii=settings.GetOption("osc_convert_ascii"))

        settings.SetOption("plugin_timer_stopped", True)

    # Send to Websocket
    if settings.GetOption("websocket_final_messages") and websocket_ip != "0" and websocket_ip != "" and final_audio:
        websocket.BroadcastMessage(json.dumps(result_obj))
        # threading.Thread(
        #    target=websocket.BroadcastMessage,
        #    args=(json.dumps(result_obj),)
        # ).start()

    # Send to TTS on final audio
    if final_audio:
        streamed_playback = settings.GetOption("tts_streamed_playback")
        if settings.GetOption("tts_answer") and predicted_text != "" and tts.init():
            try:
                if streamed_playback and hasattr(tts.tts, "tts_streaming"):
                    tts.tts.tts_streaming(predicted_text)
                else:
                    tts_wav, sample_rate = tts.tts.tts(predicted_text)
                    tts.tts.play_audio(tts_wav, settings.GetOption("device_out_index"))
            except Exception as e:
                print("Error while playing TTS audio: " + str(e))


def load_whisper(model, ai_device):
    cpu_threads = main_settings.GetOption("whisper_cpu_threads")
    num_workers = main_settings.GetOption("whisper_num_workers")
    stt_type = main_settings.GetOption("stt_type")
    if stt_type == "original_whisper":
        try:
            set_ai_device = ai_device

            if ai_device.startswith("direct-ml"):
                device_id = 0
                device_id_split = ai_device.split(":")
                if len(device_id_split) > 1:
                    device_id = int(device_id_split[1])
                import torch_directml
                set_ai_device = torch_directml.device(device_id)
            return whisper.load_model(model, download_root=".cache/whisper", device=set_ai_device)
        except Exception as e:
            print("Failed to load whisper model. Application exits. " + str(e))
            sys.exit(1)
    elif stt_type == "faster_whisper":
        compute_dtype = main_settings.GetOption("whisper_precision")

        return faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                            cpu_threads=cpu_threads, num_workers=num_workers)
        # return whisperx.WhisperX(model, device=ai_device, compute_type=compute_dtype,
        #                                    cpu_threads=cpu_threads, num_workers=num_workers)
    elif stt_type == "seamless_m4t":
        compute_dtype = main_settings.GetOption("whisper_precision")
        try:
            return seamless_m4t.SeamlessM4T(model=model, compute_type=compute_dtype, device=ai_device)
        except Exception as e:
            print("Failed to load Seamless M4T model. Application exits. " + str(e))
    elif stt_type == "mms":
        compute_dtype = main_settings.GetOption("whisper_precision")
        try:
            return mms.Mms(model=model, compute_type=compute_dtype, device=ai_device)
        except Exception as e:
            print("Failed to load MMS model. Application exits. " + str(e))
    elif stt_type == "speech_t5":
        try:
            return speech_t5.SpeechT5STT(device=ai_device)
        except Exception as e:
            print("Failed to load speech t5 model. Application exits. " + str(e))
    elif stt_type == "transformer_whisper":
        compute_dtype = main_settings.GetOption("whisper_precision")
        try:
            return transformer_whisper.TransformerWhisper(compute_type=compute_dtype, device=ai_device)
        except Exception as e:
            print("Failed to load transformer_whisper model. Application exits. " + str(e))
    elif stt_type == "medusa_whisper":
        compute_dtype = main_settings.GetOption("whisper_precision")
        try:
            return medusa_whisper.MedusaWhisper(compute_type=compute_dtype, device=ai_device)
        except Exception as e:
            print("Failed to load medusa_whisper model. Application exits. " + str(e))
    #elif stt_type == "tensorrt_whisper":
    #    try:
    #        return tensorrt_whisper.TensorRTWhisper(model=model)
    #    except Exception as e:
    #        print("Failed to load tensorrt_whisper model. Application exits. " + str(e))
    #elif stt_type == "whisper_cpp":
    #    try:
    #        return whisper_cpp.WhisperCpp(model=model)
    #    except Exception as e:
    #        print("Failed to load whisper_cpp model. Application exits. " + str(e))
    elif stt_type == "wav2vec_bert":
        compute_dtype = main_settings.GetOption("whisper_precision")
        try:
            return wav2vec_bert.Wav2VecBert(compute_type=compute_dtype, device=ai_device)
        except Exception as e:
            print("Failed to load Wav2VecBert model. Application exits. " + str(e))
    elif stt_type == "nemo_canary":
        compute_dtype = main_settings.GetOption("whisper_precision")
        try:
            return nemo_canary.NemoCanary(compute_type=compute_dtype, device=ai_device)
        except Exception as e:
            print("Failed to load Nemo Canary model. Application exits. " + str(e))

    # return None if no stt model is loaded
    return None


def load_realtime_whisper(model, ai_device):
    cpu_threads = main_settings.GetOption("whisper_cpu_threads")
    num_workers = main_settings.GetOption("whisper_num_workers")
    if main_settings.GetOption("stt_type") == "original_whisper":
        return whisper.load_model(model, download_root=".cache/whisper", device=ai_device)
    elif main_settings.GetOption("stt_type") == "faster_whisper":
        compute_dtype = main_settings.GetOption("realtime_whisper_precision")

        return faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                            cpu_threads=cpu_threads, num_workers=num_workers)
    elif main_settings.GetOption("stt_type") == "seamless_m4t":
        compute_dtype = main_settings.GetOption("realtime_whisper_precision")
        return seamless_m4t.SeamlessM4T(model=model, compute_type=compute_dtype, device=ai_device)
    elif main_settings.GetOption("stt_type") == "mms":
        compute_dtype = main_settings.GetOption("realtime_whisper_precision")
        return mms.Mms(model=model, compute_type=compute_dtype, device=ai_device)
    elif main_settings.GetOption("stt_type") == "speech_t5":
        return speech_t5.SpeechT5STT(device=ai_device)
    elif main_settings.GetOption("stt_type") == "transformer_whisper":
        compute_dtype = main_settings.GetOption("realtime_whisper_precision")
        return transformer_whisper.TransformerWhisper(compute_type=compute_dtype, device=ai_device)
    elif main_settings.GetOption("stt_type") == "medusa_whisper":
        compute_dtype = main_settings.GetOption("realtime_whisper_precision")
        return medusa_whisper.MedusaWhisper(compute_type=compute_dtype, device=ai_device)
    #elif settings.GetOption("stt_type") == "tensorrt_whisper":
    #    return tensorrt_whisper.TensorRTWhisper(model=model)
    #elif settings.GetOption("stt_type") == "whisper_cpp":
    #    return whisper_cpp.WhisperCpp(model=model)
    elif main_settings.GetOption("stt_type") == "wav2vec_bert":
        compute_dtype = main_settings.GetOption("realtime_whisper_precision")
        return wav2vec_bert.Wav2VecBert(compute_type=compute_dtype, device=ai_device)
    elif main_settings.GetOption("stt_type") == "nemo_canary":
        compute_dtype = main_settings.GetOption("realtime_whisper_precision")
        return nemo_canary.NemoCanary(compute_type=compute_dtype, device=ai_device)


def convert_audio(audio_bytes: bytes):
    audio_data = io.BytesIO(audio_bytes)
    audio_clip = AudioSegment.from_file(audio_data)

    audio_clip = audio_clip.set_frame_rate(whisper.audio.SAMPLE_RATE)
    audio_clip = audio_clip.set_channels(1)
    # audio_clip = audio_clip.set_sample_width(2)

    return np.frombuffer(audio_clip.get_array_of_samples(), np.int16).flatten().astype(np.float32) / 32768.0


def whisper_result_thread(result, audio_timestamp, final_audio, settings, plugins):
    whisper_result_handling(result, audio_timestamp, final_audio, settings, plugins)

    # send stop info for processing indicator in websocket client
    if settings.GetOption("websocket_ip") != "0" and settings.GetOption("websocket_ip") != "" and not settings.GetOption("realtime") and final_audio:
        threading.Thread(
            target=websocket.BroadcastMessage,
            args=(json.dumps({"type": "processing_start", "data": False}),)
        ).start()


def whisper_ai_thread(audio_data, current_audio_timestamp, audio_model, audio_model_realtime, last_whisper_result,
                      final_audio, settings, plugins):
    whisper_task = settings.GetOption("whisper_task")
    whisper_language = settings.GetOption("current_language")
    stt_target_language = settings.GetOption("target_language")
    whisper_condition_on_previous_text = settings.GetOption("condition_on_previous_text")
    whisper_logprob_threshold = settings.GetOption("logprob_threshold")
    whisper_no_speech_threshold = settings.GetOption("no_speech_threshold")
    whisper_beam_size = settings.GetOption("beam_size")
    whisper_beam_size_realtime = settings.GetOption("realtime_whisper_beam_size")
    whisper_word_timestamps = settings.GetOption("word_timestamps")
    whisper_faster_without_timestamps = settings.GetOption("faster_without_timestamps")

    whisper_faster_length_penalty = settings.GetOption("length_penalty")
    whisper_faster_beam_search_patience = settings.GetOption("beam_search_patience")

    whisper_temperature_fallback = settings.GetOption("temperature_fallback")
    whisper_temperature_fallback_option = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    if not whisper_temperature_fallback:
        whisper_temperature_fallback_option = 0

    whisper_temperature_fallback_realtime = settings.GetOption("realtime_temperature_fallback")
    whisper_temperature_fallback_realtime_option = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    if not whisper_temperature_fallback_realtime:
        whisper_temperature_fallback_realtime_option = 0

    whisper_initial_prompt = settings.GetOption("initial_prompt").strip()
    if whisper_initial_prompt is None or whisper_initial_prompt == "" or whisper_initial_prompt.lower() == "none":
        whisper_initial_prompt = None

    # some fix for invalid whisper language configs
    if whisper_language is None or whisper_language == "" or whisper_language.lower() == "auto" or whisper_language.lower() == "null":
        whisper_language = None

    if whisper_logprob_threshold is None or whisper_logprob_threshold == "" or whisper_logprob_threshold.lower() == "none" or whisper_logprob_threshold.lower() == "null":
        whisper_logprob_threshold = None
    else:
        whisper_logprob_threshold = float(whisper_logprob_threshold)

    if whisper_no_speech_threshold is None or whisper_no_speech_threshold == "" or whisper_no_speech_threshold.lower() == "none" or whisper_no_speech_threshold.lower() == "null":
        whisper_no_speech_threshold = None
    else:
        whisper_no_speech_threshold = float(whisper_no_speech_threshold)

    # use realtime settings if realtime is enabled but no realtime model is set and its not the final audio clip
    if settings.GetOption("realtime") and audio_model_realtime is None and not final_audio:
        whisper_beam_size = whisper_beam_size_realtime
        whisper_temperature_fallback_option = whisper_temperature_fallback_realtime_option

    prompt_reset_on_temperature = settings.GetOption("prompt_reset_on_temperature")
    repetition_penalty = settings.GetOption("repetition_penalty")
    no_repeat_ngram_size = settings.GetOption("no_repeat_ngram_size")

    # do not process audio if it is older than the last result
    if not final_audio and current_audio_timestamp < last_audio_timestamp:
        print("Audio is older than last result. Skipping...")
        return

    result = None
    try:
        audio_data_numpy = convert_audio(audio_data)

        if settings.GetOption("stt_type") == "original_whisper":
            # official whisper model
            whisper_fp16 = False
            if settings.GetOption("whisper_precision") == "float16":  # set precision
                whisper_fp16 = True

            if settings.GetOption("realtime") and audio_model_realtime is not None and not final_audio:
                realtime_whisper_fp16 = False
                if settings.GetOption("realtime_whisper_precision") == "float16":  # set precision
                    realtime_whisper_fp16 = True
                result = audio_model_realtime.transcribe(audio_data_numpy, task=whisper_task,
                                                         language=whisper_language,
                                                         condition_on_previous_text=whisper_condition_on_previous_text,
                                                         initial_prompt=whisper_initial_prompt,
                                                         logprob_threshold=whisper_logprob_threshold,
                                                         no_speech_threshold=whisper_no_speech_threshold,
                                                         fp16=realtime_whisper_fp16,
                                                         temperature=whisper_temperature_fallback_realtime_option,
                                                         beam_size=whisper_beam_size_realtime,
                                                         word_timestamps=whisper_word_timestamps)
            else:
                result = audio_model.transcribe(audio_data_numpy, task=whisper_task, language=whisper_language,
                                                condition_on_previous_text=whisper_condition_on_previous_text,
                                                initial_prompt=whisper_initial_prompt,
                                                logprob_threshold=whisper_logprob_threshold,
                                                no_speech_threshold=whisper_no_speech_threshold,
                                                fp16=whisper_fp16,
                                                temperature=whisper_temperature_fallback_option,
                                                beam_size=whisper_beam_size,
                                                word_timestamps=whisper_word_timestamps)
        elif settings.GetOption("stt_type") == "faster_whisper":

            # faster whisper
            if settings.GetOption("realtime") and audio_model_realtime is not None and not final_audio:
                if not settings.GetOption("whisper_apply_voice_markers"):
                    result = audio_model_realtime.transcribe(audio_data_numpy, task=whisper_task,
                                                             language=whisper_language,
                                                             condition_on_previous_text=whisper_condition_on_previous_text,
                                                             prompt_reset_on_temperature=prompt_reset_on_temperature,
                                                             initial_prompt=whisper_initial_prompt,
                                                             logprob_threshold=whisper_logprob_threshold,
                                                             no_speech_threshold=whisper_no_speech_threshold,
                                                             temperature=whisper_temperature_fallback_realtime_option,
                                                             beam_size=whisper_beam_size_realtime,
                                                             word_timestamps=whisper_word_timestamps,
                                                             without_timestamps=whisper_faster_without_timestamps,
                                                             patience=whisper_faster_beam_search_patience,
                                                             length_penalty=whisper_faster_length_penalty,
                                                             repetition_penalty=repetition_penalty,
                                                             no_repeat_ngram_size=no_repeat_ngram_size,
                                                             multilingual=settings.GetOption("language_detection_on_each_segment"))
                else:
                    marker_audio_tool = whisper_audio_markers.WhisperVoiceMarker(audio_model)
                    result = marker_audio_tool.voice_marker_transcribe(audio=audio_data_numpy,
                                                                       stt_model=settings.GetOption("stt_type"),
                                                                       task=whisper_task,
                                                                       language=whisper_language,
                                                                       condition_on_previous_text=whisper_condition_on_previous_text,
                                                                       prompt_reset_on_temperature=prompt_reset_on_temperature,
                                                                       initial_prompt=whisper_initial_prompt,
                                                                       logprob_threshold=whisper_logprob_threshold,
                                                                       no_speech_threshold=whisper_no_speech_threshold,
                                                                       temperature=whisper_temperature_fallback_realtime_option,
                                                                       beam_size=whisper_beam_size_realtime,
                                                                       word_timestamps=whisper_word_timestamps,
                                                                       without_timestamps=whisper_faster_without_timestamps,
                                                                       patience=whisper_faster_beam_search_patience,
                                                                       length_penalty=whisper_faster_length_penalty,
                                                                       repetition_penalty=repetition_penalty,
                                                                       no_repeat_ngram_size=no_repeat_ngram_size,
                                                                       multilingual=settings.GetOption("language_detection_on_each_segment"))
                    del marker_audio_tool

            else:
                if not settings.GetOption("whisper_apply_voice_markers"):
                    result = audio_model.transcribe(audio_data_numpy, task=whisper_task,
                                                    language=whisper_language,
                                                    condition_on_previous_text=whisper_condition_on_previous_text,
                                                    prompt_reset_on_temperature=prompt_reset_on_temperature,
                                                    initial_prompt=whisper_initial_prompt,
                                                    logprob_threshold=whisper_logprob_threshold,
                                                    no_speech_threshold=whisper_no_speech_threshold,
                                                    temperature=whisper_temperature_fallback_option,
                                                    beam_size=whisper_beam_size,
                                                    word_timestamps=whisper_word_timestamps,
                                                    without_timestamps=whisper_faster_without_timestamps,
                                                    patience=whisper_faster_beam_search_patience,
                                                    length_penalty=whisper_faster_length_penalty,
                                                    repetition_penalty=repetition_penalty,
                                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                                    multilingual=settings.GetOption("language_detection_on_each_segment"))
                else:
                    print("Applying voice markers.")
                    marker_audio_tool = whisper_audio_markers.WhisperVoiceMarker(audio_model)
                    result = marker_audio_tool.voice_marker_transcribe(audio=audio_data_numpy,
                                                                       stt_model=settings.GetOption("stt_type"),
                                                                       task=whisper_task,
                                                                       language=whisper_language,
                                                                       condition_on_previous_text=whisper_condition_on_previous_text,
                                                                       prompt_reset_on_temperature=prompt_reset_on_temperature,
                                                                       initial_prompt=whisper_initial_prompt,
                                                                       logprob_threshold=whisper_logprob_threshold,
                                                                       no_speech_threshold=whisper_no_speech_threshold,
                                                                       temperature=whisper_temperature_fallback_option,
                                                                       beam_size=whisper_beam_size,
                                                                       word_timestamps=whisper_word_timestamps,
                                                                       without_timestamps=whisper_faster_without_timestamps,
                                                                       patience=whisper_faster_beam_search_patience,
                                                                       length_penalty=whisper_faster_length_penalty,
                                                                       repetition_penalty=repetition_penalty,
                                                                       no_repeat_ngram_size=no_repeat_ngram_size,
                                                                       multilingual=settings.GetOption("language_detection_on_each_segment"))
                    del marker_audio_tool
        elif settings.GetOption("stt_type") == "seamless_m4t":
            # facebook seamless M4T
            if settings.GetOption("realtime") and audio_model_realtime is not None and not final_audio:
                if not settings.GetOption("whisper_apply_voice_markers"):
                    result = audio_model_realtime.transcribe(audio_data_numpy,
                                                             source_lang=whisper_language,
                                                             target_lang=stt_target_language,
                                                             beam_size=whisper_beam_size_realtime,
                                                             repetition_penalty=repetition_penalty,
                                                             length_penalty=whisper_faster_length_penalty,
                                                             no_repeat_ngram_size=no_repeat_ngram_size,
                                                             )
                else:
                    print("Applying voice markers.")
                    marker_audio_tool = whisper_audio_markers.WhisperVoiceMarker(audio_model)
                    result = marker_audio_tool.voice_marker_transcribe(audio=audio_data_numpy,
                                                                       stt_model=settings.GetOption("stt_type"),
                                                                       task=whisper_task,
                                                                       language=whisper_language,
                                                                       target_lang=stt_target_language,
                                                                       condition_on_previous_text=whisper_condition_on_previous_text,
                                                                       prompt_reset_on_temperature=prompt_reset_on_temperature,
                                                                       initial_prompt=whisper_initial_prompt,
                                                                       logprob_threshold=whisper_logprob_threshold,
                                                                       no_speech_threshold=whisper_no_speech_threshold,
                                                                       temperature=whisper_temperature_fallback_option,
                                                                       beam_size=whisper_beam_size,
                                                                       word_timestamps=whisper_word_timestamps,
                                                                       without_timestamps=whisper_faster_without_timestamps,
                                                                       patience=whisper_faster_beam_search_patience,
                                                                       length_penalty=whisper_faster_length_penalty,
                                                                       repetition_penalty=repetition_penalty,
                                                                       no_repeat_ngram_size=no_repeat_ngram_size)
                    del marker_audio_tool
            else:
                if not settings.GetOption("whisper_apply_voice_markers"):
                    result = audio_model.transcribe(audio_data_numpy,
                                                    source_lang=whisper_language,
                                                    target_lang=stt_target_language,
                                                    beam_size=whisper_beam_size,
                                                    repetition_penalty=repetition_penalty,
                                                    length_penalty=whisper_faster_length_penalty,
                                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                                    )
                else:
                    print("Applying voice markers.")
                    marker_audio_tool = whisper_audio_markers.WhisperVoiceMarker(audio_model)
                    result = marker_audio_tool.voice_marker_transcribe(audio=audio_data_numpy,
                                                                       stt_model=settings.GetOption("stt_type"),
                                                                       task=whisper_task,
                                                                       language=whisper_language,
                                                                       target_lang=stt_target_language,
                                                                       condition_on_previous_text=whisper_condition_on_previous_text,
                                                                       prompt_reset_on_temperature=prompt_reset_on_temperature,
                                                                       initial_prompt=whisper_initial_prompt,
                                                                       logprob_threshold=whisper_logprob_threshold,
                                                                       no_speech_threshold=whisper_no_speech_threshold,
                                                                       temperature=whisper_temperature_fallback_option,
                                                                       beam_size=whisper_beam_size,
                                                                       word_timestamps=whisper_word_timestamps,
                                                                       without_timestamps=whisper_faster_without_timestamps,
                                                                       patience=whisper_faster_beam_search_patience,
                                                                       length_penalty=whisper_faster_length_penalty,
                                                                       repetition_penalty=repetition_penalty,
                                                                       no_repeat_ngram_size=no_repeat_ngram_size)

        elif settings.GetOption("stt_type") == "mms":
            result = audio_model.transcribe(audio_data_numpy,
                                            source_lang=whisper_language,
                                            )

        elif settings.GetOption("stt_type") == "speech_t5":
            # microsoft SpeechT5
            result = audio_model.transcribe(audio_data_numpy)

        elif settings.GetOption("stt_type") == "transformer_whisper":
            # Whisper Huggingface Transformer
            audio_model.set_compute_type(settings.GetOption("whisper_precision"))
            audio_model.set_compute_device(settings.GetOption("ai_device"))
            result = audio_model.transcribe(audio_data_numpy, model=settings.GetOption("model"), task=whisper_task,
                                            language=whisper_language, return_timestamps=False,
                                            beam_size=whisper_beam_size)

        elif settings.GetOption("stt_type") == "medusa_whisper":
            # Whisper Huggingface Transformer
            audio_model.set_compute_type(settings.GetOption("whisper_precision"))
            audio_model.set_compute_device(settings.GetOption("ai_device"))
            whisper_num_workers = int(settings.GetOption("whisper_num_workers"))
            result = audio_model.transcribe(audio_data_numpy, model=settings.GetOption("model"), task=whisper_task,
                                            language=whisper_language, return_timestamps=False,
                                            beam_size=whisper_beam_size)

        #elif settings.GetOption("stt_type") == "tensorrt_whisper":
        #    result = audio_model.transcribe(audio_data_numpy, model=settings.GetOption("model"), task=whisper_task,
        #                                    language=whisper_language)

        #elif settings.GetOption("stt_type") == "whisper_cpp":
        #    # WhisperCPP
        #    result = audio_model.transcribe(audio_sample, task=whisper_task,
        #                                    language=whisper_language,
        #                                    condition_on_previous_text=whisper_condition_on_previous_text,
        #                                    prompt_reset_on_temperature=prompt_reset_on_temperature,
        #                                    initial_prompt=whisper_initial_prompt,
        #                                    logprob_threshold=whisper_logprob_threshold,
        #                                    no_speech_threshold=whisper_no_speech_threshold,
        #                                    temperature=whisper_temperature_fallback_option,
        #                                    beam_size=whisper_beam_size,
        #                                    word_timestamps=whisper_word_timestamps,
        #                                    without_timestamps=whisper_faster_without_timestamps,
        #                                    patience=whisper_faster_beam_search_patience,
        #                                    length_penalty=whisper_faster_length_penalty,
        #                                    repetition_penalty=repetition_penalty,
        #                                    no_repeat_ngram_size=no_repeat_ngram_size)

        elif settings.GetOption("stt_type") == "wav2vec_bert":
            # Wav2VecBert
            audio_model.set_compute_type(settings.GetOption("whisper_precision"))
            audio_model.set_compute_device(settings.GetOption("ai_device"))
            result = audio_model.transcribe(audio_data_numpy, task=whisper_task,
                                            language=whisper_language)

        elif settings.GetOption("stt_type") == "nemo_canary":
            # Nemo Canary
            audio_model.set_compute_type(settings.GetOption("whisper_precision"))
            audio_model.set_compute_device(settings.GetOption("ai_device"))
            result = audio_model.transcribe(audio_data_numpy, task=whisper_task,
                                            source_lang=whisper_language,
                                            target_lang=stt_target_language,
                                            beam_size=whisper_beam_size,
                                            length_penalty=whisper_faster_length_penalty,
                                            temperature=1.0,)
        else:
            # process audio by plugin for Speech-to-Text
            threading.Thread(target=plugin_process_stt_processing, args=(
                current_audio_timestamp, audio_data, whisper.audio.SAMPLE_RATE, final_audio, settings, plugins),
                             daemon=True).start()
            return

        if result is None or (last_whisper_result == result.get('text').strip() and not final_audio):
            print("skipping... result: ", result)
            return

        whisper_result_thread(result, current_audio_timestamp, final_audio, settings, plugins)

    except Exception as e:
        print("Error while processing audio: " + str(e))
        traceback.print_exc()


def whisper_worker():
    #global final_audio
    #global queue_data
    #global audio
    #global audio_timestamp

    whisper_model = main_settings.GetOption("model")

    whisper_ai_device = main_settings.GetOption("ai_device")
    websocket.set_loading_state("speech2text_loading", True)
    audio_model = load_whisper(whisper_model, whisper_ai_device)
    # load realtime whisper model
    audio_model_realtime = None
    if main_settings.GetOption("realtime") and main_settings.GetOption("realtime_whisper_model") != "" and main_settings.GetOption(
            "realtime_whisper_model") is not None:
        audio_model_realtime = load_realtime_whisper(main_settings.GetOption("realtime_whisper_model"), whisper_ai_device)
    websocket.set_loading_state("speech2text_loading", False)

    last_audio_time = 0

    last_whisper_result = ""

    print("Whispering Tiger is now ready!")

    while True:
        final_audio = False
        queue_data = None
        audio = None
        audio_timestamp = None
        plugins = None
        realtime_mode = main_settings.GetOption("realtime")
        try:
            queue_data = q.get(timeout=queue_timeout)
            audio = queue_data["data"]
            final_audio = queue_data["final"]
            audio_timestamp = queue_data["time"]
            settings = queue_data["settings"]
            plugins = queue_data["plugins"]
            realtime_mode = settings.GetOption("realtime")
        except queue.Empty:
            # print("Queue processing timed out. Skipping...")
            continue
        except queue.Full:
            print("Queue is full. Skipping...")
            continue

        q.task_done()

        # skip if no audio data is available
        if audio is None or len(audio) == 0:
            continue

        # skip if queue is full
        if realtime_mode and q.qsize() >= max_queue_size and not final_audio or \
                not realtime_mode and q.qsize() >= max_queue_size:
            continue

        # skip if audio is too old, except if it's the final audio
        if audio_timestamp < last_audio_time and not final_audio:
            continue

        if main_settings.GetOption("thread_per_transcription"):
            threading.Thread(target=whisper_ai_thread, args=(
                audio, audio_timestamp, audio_model, audio_model_realtime, last_whisper_result, final_audio, settings, plugins),
                             daemon=True).start()
        else:
            whisper_ai_thread(audio, audio_timestamp, audio_model, audio_model_realtime, last_whisper_result,
                              final_audio, settings, plugins)


def start_whisper_thread():
    # Turn-on the worker thread.
    threading.Thread(target=whisper_worker, daemon=True).start()

import sys
import threading
import queue
import time

import whisper

import Utilities
import audio_tools
import settings
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
from Models.TTS import silero

# from faster_whisper import WhisperModel
import Models.STT.faster_whisper as faster_whisper
import Models.STT.whisper_audio_markers as whisper_audio_markers
import Models.STT.speecht5 as speech_t5
import Models.Multi.seamless_m4t as seamless_m4t
import csv
from datetime import datetime

# Plugins
import Plugins

ignore_list = []

transcriptions_list = {}
# Lock for thread-safe dictionary update
transcriptions_list_lock = threading.Lock()


def ns_to_datetime(ns):
    # Convert nanoseconds to seconds
    seconds = ns / 1_000_000_000
    # Create a datetime object
    dt_object = datetime.fromtimestamp(seconds)
    # Format the datetime object as a string
    return dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # trimming microseconds to milliseconds


def add_transcription(start_time, end_time, transcription, translation, file_path=None):
    global transcriptions_list

    start_time_str = ns_to_datetime(start_time)
    end_time_str = ns_to_datetime(end_time)

    # Update the dictionary
    with transcriptions_list_lock:
        transcriptions_list[(start_time, end_time)] = {"transcription": transcription, "translation": translation}

        # Add the new entry to the CSV file
        if file_path is not None and isinstance(file_path, str) and file_path != "":
            with open(file_path, "a", newline='') as transcription_file:
                csv_writer = csv.writer(transcription_file, quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([start_time_str, end_time_str, transcription, translation])


def save_transcriptions(file_path: str):
    global transcriptions_list

    with open(file_path, "w", newline='') as transcription_file:
        csv_writer = csv.writer(transcription_file, quoting=csv.QUOTE_MINIMAL)

        # Write headers if you want (optional)
        #csv_writer.writerow(["Start Time", "End Time", "Transcription", "Translation"])

        for (start_time, end_time), entry in transcriptions_list.items():
            transcription = entry["transcription"]
            translation = entry["translation"]
            start_time_str = ns_to_datetime(start_time)
            end_time_str = ns_to_datetime(end_time)
            csv_writer.writerow([start_time_str, end_time_str, transcription, translation])
        transcription_file.close()


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

final_audio = False
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


def remove_repetitions(text, language='english'):
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
        return sentence_split.remove_repeated_sentences(text, language=sentence_split_language, max_repeat=max_sentence_repetition)
    return text


def whisper_result_handling(result, audio_timestamp, final_audio):
    global last_audio_timestamp
    verbose = settings.GetOption("verbose")
    osc_ip = settings.GetOption("osc_ip")
    do_txt_translate = settings.GetOption("txt_translate")
    transcription_auto_save_file = settings.GetOption("transcription_auto_save_file")

    predicted_text = result.get('text').strip()
    result["type"] = "transcript"

    # Try to prevent sentence repetition
    sentence_split_language = "english"
    if "language" in result:
        sentence_split_language = result["language"]
    predicted_text = remove_repetitions(predicted_text, language=sentence_split_language)
    if "text" in result:
        result["text"] = predicted_text

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
            predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text, from_lang,
                                                                                         to_lang, to_romaji)
            result["txt_translation"] = predicted_text
            result["txt_translation_source"] = txt_from_lang
            result["txt_translation_target"] = to_lang

        if final_audio:
            if "txt_translation" in result:
                translation_text = predicted_text
            else:
                translation_text = ""

            add_transcription(audio_timestamp, time.time_ns(), result["text"], translation_text, transcription_auto_save_file)

        # send realtime processing data to websocket
        if not final_audio and predicted_text.strip() != "":
            threading.Thread(
                target=websocket.BroadcastMessage,
                args=(json.dumps({"type": "processing_data", "data": predicted_text}),)
            ).start()

        # send regular message
        send_message(predicted_text, result, final_audio)

        last_audio_timestamp = audio_timestamp


def plugin_process(predicted_text, result_obj, final_audio):
    for plugin_inst in Plugins.plugins:
        if final_audio:
            if hasattr(plugin_inst, 'stt'):
                plugin_inst.stt(predicted_text, result_obj)
        else:
            if hasattr(plugin_inst, 'stt_intermediate'):
                plugin_inst.stt_intermediate(predicted_text, result_obj)


# replace {src} and {trg} with source and target language in osc prefix
def build_whisper_translation_osc_prefix(result_obj):
    prefix = settings.GetOption("osc_chat_prefix")
    txt_translate_enabled = settings.GetOption("txt_translate")
    whisper_task = settings.GetOption("whisper_task")

    # replace {src} with source language
    if "language" in result_obj and result_obj["language"] is not None:
        prefix = prefix.replace("{src}", result_obj["language"])
    elif "language" in result_obj and result_obj["language"] is None:
        prefix = prefix.replace("{src}", "?")

    if txt_translate_enabled and "txt_translation" in result_obj and "txt_translation_target" in result_obj:
        # replace {trg} with target language
        target_language = texttranslate.iso3_to_iso1(result_obj["txt_translation_target"])
        if target_language is None:
            target_language = result_obj["txt_translation_target"]
        if target_language is not None:
            prefix = prefix.replace("{trg}", target_language)
    else:
        if "target_lang" in result_obj and result_obj["target_lang"] is not None:
            # replace {trg} with target language of whisper
            prefix = prefix.replace("{trg}", result_obj["target_lang"])
        elif whisper_task == "transcribe":
            # replace {trg} with target language of whisper
            if "language" in result_obj and result_obj["language"] is not None:
                prefix = prefix.replace("{trg}", result_obj["language"])
        elif whisper_task == "translate":
            # replace {trg} with target language of whisper
            prefix = prefix.replace("{trg}", "en")
        else:
            prefix = prefix.replace("{trg}", "?")
    return prefix


def send_message(predicted_text, result_obj, final_audio):
    osc_ip = settings.GetOption("osc_ip")
    osc_address = settings.GetOption("osc_address")
    osc_port = settings.GetOption("osc_port")
    websocket_ip = settings.GetOption("websocket_ip")

    # Update osc_min_time_between_messages option
    VRC_OSCLib.set_min_time_between_messages(settings.GetOption("osc_min_time_between_messages"))

    # WORKAROUND: prevent it from outputting the initial prompt.
    if predicted_text == settings.GetOption("initial_prompt"):
        return

    # process plugins
    if final_audio and not settings.GetOption("realtime") or settings.GetOption("realtime"):
        plugin_thread = threading.Thread(target=plugin_process, args=(predicted_text, result_obj, final_audio))
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

        osc_text = predicted_text
        if settings.GetOption("osc_type_transfer") == "source":
            osc_text = result_obj["text"]
        elif settings.GetOption("osc_type_transfer") == "both":
            osc_text = result_obj["text"] + settings.GetOption("osc_type_transfer_split") + predicted_text
        elif settings.GetOption("osc_type_transfer") == "both_inverted":
            osc_text = predicted_text + settings.GetOption("osc_type_transfer_split") + result_obj["text"]

        message = build_whisper_translation_osc_prefix(result_obj) + osc_text

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
    if settings.GetOption("websocket_final_messages") and websocket_ip != "0" and final_audio:
        threading.Thread(
            target=websocket.BroadcastMessage,
            args=(json.dumps(result_obj),)
        ).start()

    # Send to TTS on final audio
    if final_audio:
        if settings.GetOption("tts_answer") and predicted_text != "" and silero.init():
            try:
                silero_wav, sample_rate = silero.tts.tts(predicted_text)
                silero.tts.play_audio(silero_wav, settings.GetOption("device_out_index"))
            except Exception as e:
                print("Error while playing TTS audio: " + str(e))


def load_whisper(model, ai_device):
    cpu_threads = settings.GetOption("whisper_cpu_threads")
    num_workers = settings.GetOption("whisper_num_workers")
    stt_type = settings.GetOption("stt_type")
    if stt_type == "original_whisper":
        try:
            return whisper.load_model(model, download_root=".cache/whisper", device=ai_device)
        except Exception as e:
            print("Failed to load whisper model. Application exits. " + str(e))
            sys.exit(1)
    elif stt_type == "faster_whisper":
        compute_dtype = settings.GetOption("whisper_precision")

        return faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                            cpu_threads=cpu_threads, num_workers=num_workers)
        # return whisperx.WhisperX(model, device=ai_device, compute_type=compute_dtype,
        #                                    cpu_threads=cpu_threads, num_workers=num_workers)
    elif stt_type == "seamless_m4t":
        compute_dtype = settings.GetOption("whisper_precision")
        try:
            return seamless_m4t.SeamlessM4T(model=model, compute_type=compute_dtype, device=ai_device)
        except Exception as e:
            print("Failed to load Seamless M4T model. Application exits. " + str(e))
    elif stt_type == "speech_t5":
        try:
            return speech_t5.SpeechT5STT(device=ai_device)
        except Exception as e:
            print("Failed to load speech t5 model. Application exits. " + str(e))

    # return None if no stt model is loaded
    return None


def load_realtime_whisper(model, ai_device):
    cpu_threads = settings.GetOption("whisper_cpu_threads")
    num_workers = settings.GetOption("whisper_num_workers")
    if settings.GetOption("stt_type") == "original_whisper":
        return whisper.load_model(model, download_root=".cache/whisper", device=ai_device)
    elif settings.GetOption("stt_type") == "faster_whisper":
        compute_dtype = settings.GetOption("realtime_whisper_precision")

        return faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                            cpu_threads=cpu_threads, num_workers=num_workers)
    elif settings.GetOption("stt_type") == "seamless_m4t":
        compute_dtype = settings.GetOption("realtime_whisper_precision")
        return seamless_m4t.SeamlessM4T(model=model, compute_type=compute_dtype, device=ai_device)
    elif settings.GetOption("stt_type") == "speech_t5":
        return speech_t5.SpeechT5STT(device=ai_device)


def convert_audio(audio_bytes: bytes):
    audio_data = io.BytesIO(audio_bytes)
    audio_clip = AudioSegment.from_file(audio_data)

    audio_clip = audio_clip.set_frame_rate(whisper.audio.SAMPLE_RATE)
    audio_clip = audio_clip.set_channels(1)

    return np.frombuffer(audio_clip.get_array_of_samples(), np.int16).flatten().astype(np.float32) / 32768.0


def whisper_result_thread(result, audio_timestamp, final_audio):
    whisper_result_handling(result, audio_timestamp, final_audio)

    # send stop info for processing indicator in websocket client
    if settings.GetOption("websocket_ip") != "0" and not settings.GetOption("realtime") and final_audio:
        threading.Thread(
            target=websocket.BroadcastMessage,
            args=(json.dumps({"type": "processing_start", "data": False}),)
        ).start()


def whisper_ai_thread(audio_data, current_audio_timestamp, audio_model, audio_model_realtime, last_whisper_result,
                      final_audio):
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

    # convert audio to numpy array, change channels to 1 etc.
    audio_sample = convert_audio(audio_data)

    # do not process audio if it is older than the last result
    if not final_audio and current_audio_timestamp < last_audio_timestamp:
        print("Audio is older than last result. Skipping...")
        return

    result = None
    try:
        if settings.GetOption("stt_type") == "original_whisper":
            # official whisper model
            whisper_fp16 = False
            if settings.GetOption("whisper_precision") == "float16":  # set precision
                whisper_fp16 = True

            if settings.GetOption("realtime") and audio_model_realtime is not None and not final_audio:
                realtime_whisper_fp16 = False
                if settings.GetOption("realtime_whisper_precision") == "float16":  # set precision
                    realtime_whisper_fp16 = True
                result = audio_model_realtime.transcribe(audio_sample, task=whisper_task,
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
                result = audio_model.transcribe(audio_sample, task=whisper_task, language=whisper_language,
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
                    result = audio_model_realtime.transcribe(audio_sample, task=whisper_task,
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
                                                             no_repeat_ngram_size=no_repeat_ngram_size)
                else:
                    marker_audio_tool = whisper_audio_markers.WhisperVoiceMarker(audio_sample, audio_model)
                    result = marker_audio_tool.voice_marker_transcribe(stt_model=settings.GetOption("stt_type"),
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
                                                                       no_repeat_ngram_size=no_repeat_ngram_size)
                    del marker_audio_tool

            else:
                if not settings.GetOption("whisper_apply_voice_markers"):
                    result = audio_model.transcribe(audio_sample, task=whisper_task,
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
                                                    no_repeat_ngram_size=no_repeat_ngram_size)
                else:
                    print("Applying voice markers.")
                    marker_audio_tool = whisper_audio_markers.WhisperVoiceMarker(audio_sample, audio_model)
                    result = marker_audio_tool.voice_marker_transcribe(stt_model=settings.GetOption("stt_type"),
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
                                                                       no_repeat_ngram_size=no_repeat_ngram_size)
                    del marker_audio_tool
        elif settings.GetOption("stt_type") == "seamless_m4t":
            # facebook seamless M4T
            if settings.GetOption("realtime") and audio_model_realtime is not None and not final_audio:
                if not settings.GetOption("whisper_apply_voice_markers"):
                    result = audio_model_realtime.transcribe(audio_sample,
                                                             source_lang=whisper_language,
                                                             target_lang=stt_target_language,
                                                             beam_size=whisper_beam_size_realtime,
                                                             repetition_penalty=repetition_penalty,
                                                             length_penalty=whisper_faster_length_penalty,
                                                             no_repeat_ngram_size=no_repeat_ngram_size,
                                                             )
                else:
                    print("Applying voice markers.")
                    marker_audio_tool = whisper_audio_markers.WhisperVoiceMarker(audio_sample, audio_model)
                    result = marker_audio_tool.voice_marker_transcribe(stt_model=settings.GetOption("stt_type"),
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
                    result = audio_model.transcribe(audio_sample,
                                                    source_lang=whisper_language,
                                                    target_lang=stt_target_language,
                                                    beam_size=whisper_beam_size,
                                                    repetition_penalty=repetition_penalty,
                                                    length_penalty=whisper_faster_length_penalty,
                                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                                    )
                else:
                    print("Applying voice markers.")
                    marker_audio_tool = whisper_audio_markers.WhisperVoiceMarker(audio_sample, audio_model)
                    result = marker_audio_tool.voice_marker_transcribe(stt_model=settings.GetOption("stt_type"),
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


        elif settings.GetOption("stt_type") == "speech_t5":
            # microsoft SpeechT5
            result = audio_model.transcribe(audio_sample)

        if result is None or (last_whisper_result == result.get('text').strip() and not final_audio):
            print("skipping... result: ", result)
            return

        whisper_result_thread(result, current_audio_timestamp, final_audio)

    except Exception as e:
        print("Error while processing audio: " + str(e))


def whisper_worker():
    global final_audio
    global queue_data
    global audio
    global audio_timestamp

    whisper_model = settings.GetOption("model")

    whisper_ai_device = settings.GetOption("ai_device")
    websocket.set_loading_state("speech2text_loading", True)
    audio_model = load_whisper(whisper_model, whisper_ai_device)
    # load realtime whisper model
    audio_model_realtime = None
    if settings.GetOption("realtime") and settings.GetOption("realtime_whisper_model") != "" and settings.GetOption(
            "realtime_whisper_model") is not None:
        audio_model_realtime = load_realtime_whisper(settings.GetOption("realtime_whisper_model"), whisper_ai_device)
    websocket.set_loading_state("speech2text_loading", False)

    last_audio_time = 0

    last_whisper_result = ""

    print("Speech2Text AI Ready. You can now say something!")

    while True:
        final_audio = False
        queue_data = None
        audio = None
        audio_timestamp = None
        try:
            queue_data = q.get(timeout=queue_timeout)
            audio = queue_data["data"]
            final_audio = queue_data["final"]
            audio_timestamp = queue_data["time"]
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
        if settings.GetOption("realtime") and q.qsize() >= max_queue_size and not final_audio or \
                not settings.GetOption("realtime") and q.qsize() >= max_queue_size:
            continue

        # skip if audio is too old, except if it's the final audio
        if audio_timestamp < last_audio_time and not final_audio:
            continue

        # start processing audio thread if audio_model is not None
        if audio_model is not None:
            threading.Thread(target=whisper_ai_thread, args=(
                audio, audio_timestamp, audio_model, audio_model_realtime, last_whisper_result, final_audio),
                             daemon=True).start()


def start_whisper_thread():
    # Turn-on the worker thread.
    threading.Thread(target=whisper_worker, daemon=True).start()

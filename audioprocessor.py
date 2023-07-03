import sys
import threading
import queue

import whisper
import settings
import VRC_OSCLib
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
import Models.STT.speecht5 as speech_t5

# Plugins
import Plugins


# some regular mistakenly recognized words/sentences on mostly silence audio, which are ignored in processing
ignore_list_file = open(str(Path(Path.cwd() / "ignorelist.txt").resolve()), "r", encoding="utf-8")
ignore_list = ignore_list_file.readlines()
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


def whisper_result_handling(result, audio_timestamp, final_audio):
    global last_audio_timestamp
    verbose = settings.GetOption("verbose")
    osc_ip = settings.GetOption("osc_ip")

    predicted_text = result.get('text').strip()
    result["type"] = "transcript"

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
        do_txt_translate = settings.GetOption("txt_translate")
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

        # send realtime processing data to websocket
        if not final_audio and predicted_text.strip() != "":
            websocket.BroadcastMessage(json.dumps({"type": "processing_data", "data": predicted_text}))

        # send regular message
        send_message(predicted_text, result, final_audio)

        last_audio_timestamp = audio_timestamp


def plugin_process(predicted_text, result_obj, final_audio):
    for plugin_inst in Plugins.plugins:
        if final_audio:
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
    prefix = prefix.replace("{src}", result_obj["language"])

    if txt_translate_enabled and "txt_translation" in result_obj and "txt_translation_target" in result_obj:
        # replace {trg} with target language
        target_language = texttranslate.iso3_to_iso1(result_obj["txt_translation_target"])
        if target_language is None:
            target_language = result_obj["txt_translation_target"]
        prefix = prefix.replace("{trg}", target_language)
    else:
        if whisper_task == "transcribe":
            # replace {trg} with target language of whisper
            prefix = prefix.replace("{trg}", result_obj["language"])
        else:
            # replace {trg} with target language of whisper
            prefix = prefix.replace("{trg}", "en")
    return prefix


def send_message(predicted_text, result_obj, final_audio):
    osc_ip = settings.GetOption("osc_ip")
    osc_address = settings.GetOption("osc_address")
    osc_port = settings.GetOption("osc_port")
    websocket_ip = settings.GetOption("websocket_ip")

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

        osc_text = predicted_text
        if settings.GetOption("osc_type_transfer") == "source":
            osc_text = result_obj["text"]
        elif settings.GetOption("osc_type_transfer") == "both":
            osc_text = result_obj["text"] + settings.GetOption("osc_type_transfer_split") + predicted_text

        VRC_OSCLib.Chat(build_whisper_translation_osc_prefix(result_obj) + osc_text, True, osc_notify, osc_address,
                        IP=osc_ip, PORT=osc_port,
                        convert_ascii=settings.GetOption("osc_convert_ascii"))
        settings.SetOption("plugin_timer_stopped", True)

    # Send to Websocket
    if settings.GetOption("websocket_final_messages") and websocket_ip != "0" and final_audio:
        websocket.BroadcastMessage(json.dumps(result_obj))

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
    if settings.GetOption("stt_type") == "original_whisper":
        try:
            return whisper.load_model(model, download_root=".cache/whisper", device=ai_device)
        except Exception as e:
            print("Failed to load whisper model. Application exits. " + str(e))
            sys.exit(1)
    elif settings.GetOption("stt_type") == "faster_whisper":
        compute_dtype = settings.GetOption("whisper_precision")

        return faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                            cpu_threads=cpu_threads, num_workers=num_workers)
        #return whisperx.WhisperX(model, device=ai_device, compute_type=compute_dtype,
        #                                    cpu_threads=cpu_threads, num_workers=num_workers)
    elif settings.GetOption("stt_type") == "speech_t5":
        try:
            return speech_t5.SpeechT5STT(device=ai_device)
        except Exception as e:
            print("Failed to load speech t5 model. Application exits. " + str(e))


def load_realtime_whisper(model, ai_device):
    cpu_threads = settings.GetOption("whisper_cpu_threads")
    num_workers = settings.GetOption("whisper_num_workers")
    if settings.GetOption("stt_type") == "original_whisper":
        return whisper.load_model(model, download_root=".cache/whisper", device=ai_device)
    elif settings.GetOption("stt_type") == "faster_whisper":
        compute_dtype = settings.GetOption("realtime_whisper_precision")

        return faster_whisper.FasterWhisper(model, device=ai_device, compute_type=compute_dtype,
                                            cpu_threads=cpu_threads, num_workers=num_workers)
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
        websocket.BroadcastMessage(json.dumps({"type": "processing_start", "data": False}))


def whisper_ai_thread(audio_data, current_audio_timestamp, audio_model, audio_model_realtime, last_whisper_result, final_audio):
    whisper_task = settings.GetOption("whisper_task")
    whisper_language = settings.GetOption("current_language")
    whisper_condition_on_previous_text = settings.GetOption("condition_on_previous_text")
    whisper_logprob_threshold = settings.GetOption("logprob_threshold")
    whisper_no_speech_threshold = settings.GetOption("no_speech_threshold")
    whisper_beam_size = settings.GetOption("beam_size")
    whisper_beam_size_realtime = settings.GetOption("realtime_whisper_beam_size")

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

    audio_sample = convert_audio(audio_data)

    # do not process audio if it is older than the last result
    if not final_audio and current_audio_timestamp < last_audio_timestamp:
        print("Audio is older than last result. Skipping...")
        return

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
                                                         beam_size=whisper_beam_size_realtime
                                                         )
            else:
                result = audio_model.transcribe(audio_sample, task=whisper_task, language=whisper_language,
                                                condition_on_previous_text=whisper_condition_on_previous_text,
                                                initial_prompt=whisper_initial_prompt,
                                                logprob_threshold=whisper_logprob_threshold,
                                                no_speech_threshold=whisper_no_speech_threshold,
                                                fp16=whisper_fp16,
                                                temperature=whisper_temperature_fallback_option,
                                                beam_size=whisper_beam_size
                                                )
        elif settings.GetOption("stt_type") == "faster_whisper":
            # faster whisper
            if settings.GetOption("realtime") and audio_model_realtime is not None and not final_audio:
                result = audio_model_realtime.transcribe(audio_sample, task=whisper_task,
                                                         language=whisper_language,
                                                         condition_on_previous_text=whisper_condition_on_previous_text,
                                                         initial_prompt=whisper_initial_prompt,
                                                         logprob_threshold=whisper_logprob_threshold,
                                                         no_speech_threshold=whisper_no_speech_threshold,
                                                         temperature=whisper_temperature_fallback_realtime_option,
                                                         beam_size=whisper_beam_size_realtime
                                                         )

            else:
                result = audio_model.transcribe(audio_sample, task=whisper_task,
                                                language=whisper_language,
                                                condition_on_previous_text=whisper_condition_on_previous_text,
                                                initial_prompt=whisper_initial_prompt,
                                                logprob_threshold=whisper_logprob_threshold,
                                                no_speech_threshold=whisper_no_speech_threshold,
                                                temperature=whisper_temperature_fallback_option,
                                                beam_size=whisper_beam_size)
        elif settings.GetOption("stt_type") == "speech_t5":
            # microsoft SpeechT5
            result = audio_model.transcribe(audio_sample)

        if last_whisper_result == result.get('text').strip() and not final_audio:
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

        # start processing audio thread
        threading.Thread(target=whisper_ai_thread, args=(
            audio, audio_timestamp, audio_model, audio_model_realtime, last_whisper_result, final_audio),
                         daemon=True).start()


def start_whisper_thread():
    # Turn-on the worker thread.
    threading.Thread(target=whisper_worker, daemon=True).start()

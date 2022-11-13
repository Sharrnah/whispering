import threading
import queue
import whisper
import settings
import VRC_OSCLib
import texttranslate
import websocket
import json
import numpy as np
from pydub import AudioSegment
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
import io
import flanLanguageModel

# some regular mistakenly recognized words/sentences on mostly silence audio, which are ignored in processing
blacklist = [
    "",
    "Thanks for watching!",
    "Thank you for watching!",
    "Thanks for watching.",
    "Thank you for watching.",
    "you"
]
# make all list entries lowercase for later comparison
blacklist = list((map(lambda x: x.lower(), blacklist)))

max_queue_size = 5
queue_timeout = 5

q = queue.Queue(maxsize=max_queue_size)


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


def whisper_result_handling(result):
    verbose = settings.GetOption("verbose")
    osc_ip = settings.GetOption("osc_ip")
    flan_whisper_answer = settings.GetOption("flan_whisper_answer")

    predicted_text = result.get('text').strip()
    result["type"] = "transcript"

    if not predicted_text.lower() in blacklist:
        if not verbose:
            print("Transcribe" + (" (OSC)" if osc_ip != "0" else "") + ": " + predicted_text)
        else:
            print(result)

        # translate using text translator if enabled
        do_txt_translate = settings.GetOption("txt_translate")
        if do_txt_translate:
            from_lang = settings.GetOption("src_lang")
            to_lang = settings.GetOption("trg_lang")
            to_romaji = settings.GetOption("txt_ascii")
            predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text, from_lang, to_lang, to_romaji)
            result["txt_translation"] = predicted_text
            result["txt_translation_source"] = txt_from_lang
            result["txt_translation_target"] = to_lang

        # replace predicted_text with FLAN response
        flan_loaded = False
        # check if FLAN is enabled
        if flan_whisper_answer and flanLanguageModel.init():
            flan_osc_prefix = settings.GetOption("flan_osc_prefix")
            flan_loaded = True
            result["type"] = "flan_answer"
            # Only process using FLAN if question is asked
            if settings.GetOption("flan_process_only_questions"):
                prompted_text, prompt_change = flanLanguageModel.flan.whisper_result_prompter(predicted_text)
                if prompt_change:
                    predicted_text = flanLanguageModel.flan.encode(prompted_text)

                    # translate from auto-detected language to speaker language
                    if settings.GetOption("flan_translate_to_speaker_language"):
                        predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text, "auto", result['language'], False, True)
                    result['flan_answer'] = predicted_text
                    print("FLAN question: " + prompted_text)
                    print("FLAN result: " + predicted_text)
                    send_message(flan_osc_prefix + predicted_text, result)
            # otherwise process every text with FLAN
            else:
                print("flan general processing")
                predicted_text = flanLanguageModel.flan.encode(predicted_text)
                result['flan_answer'] = predicted_text
                print("FLAN result: " + predicted_text)
                send_message(flan_osc_prefix + predicted_text, result)

        # send regular message if flan was not loaded
        if not flan_loaded:
            send_message(predicted_text, result)


def send_message(predicted_text, result_obj):
    osc_ip = settings.GetOption("osc_ip")
    osc_address = settings.GetOption("osc_address")
    osc_port = settings.GetOption("osc_port")
    websocket_ip = settings.GetOption("websocket_ip")

    # Send over OSC
    if osc_ip != "0":
        VRC_OSCLib.Chat(predicted_text, True, True, osc_address, IP=osc_ip, PORT=osc_port,
                        convert_ascii=settings.GetOption("osc_convert_ascii"))
    # Send to Websocket
    if websocket_ip != "0":
        websocket.BroadcastMessage(json.dumps(result_obj))


def load_whisper(model, ai_device):
    return whisper.load_model(model, download_root=".cache/whisper", device=ai_device)


def convert_audio(audio_bytes: bytes):
    audio_data = io.BytesIO(audio_bytes)
    audio_clip = AudioSegment.from_file(audio_data)

    audio_clip = audio_clip.set_frame_rate(whisper.audio.SAMPLE_RATE)
    #audio_clip = audio_clip.set_channels(1)

    return np.frombuffer(audio_clip.get_array_of_samples(), np.int16).flatten().astype(np.float32) / 32768.0


def whisper_worker():
    whisper_model = settings.GetOption("model")

    whisper_ai_device = settings.GetOption("ai_device")
    audio_model = load_whisper(whisper_model, whisper_ai_device)

    print("Whisper AI Ready. You can now say something!")

    while True:
        try:
            audio = q.get(timeout=queue_timeout)
        except queue.Empty:
            # print("Queue processing timed out. Skipping...")
            continue
        except queue.Full:
            print("Queue is full. Skipping...")
            continue

        # skip if queue is full
        if q.qsize() >= max_queue_size:
            q.task_done()
            continue

        audio_sample = convert_audio(audio)

        whisper_task = settings.GetOption("whisper_task")

        whisper_language = settings.GetOption("current_language")

        whisper_condition_on_previous_text = settings.GetOption("condition_on_previous_text")

        result = audio_model.transcribe(audio_sample, task=whisper_task, language=whisper_language,
                                        condition_on_previous_text=whisper_condition_on_previous_text)

        whisper_result_handling(result)

        q.task_done()


def start_whisper_thread():
    # Turn-on the worker thread.
    threading.Thread(target=whisper_worker, daemon=True).start()

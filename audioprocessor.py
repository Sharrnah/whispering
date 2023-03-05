import threading
import queue
import whisper
import settings
import VRC_OSCLib
from Models.TextTranslation import texttranslate
import websocket
import json
import numpy as np
from pydub import AudioSegment
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
import io
from Models.LLM import LLM
from Models.TTS import silero
import loading_state

# Plugins
from Plugins import Base

# load plugins into array
plugins = []
for plugin in Base.plugins:
    plugins.append(plugin())
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
        if do_txt_translate:
            from_lang = settings.GetOption("src_lang")
            to_lang = settings.GetOption("trg_lang")
            to_romaji = settings.GetOption("txt_ascii")
            predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text, from_lang,
                                                                                         to_lang, to_romaji)
            result["txt_translation"] = predicted_text
            result["txt_translation_source"] = txt_from_lang
            result["txt_translation_target"] = to_lang

        # replace predicted_text with FLAN response
        flan_loaded = False
        # check if FLAN is enabled
        if flan_whisper_answer and LLM.init():
            flan_osc_prefix = settings.GetOption("flan_osc_prefix")
            flan_loaded = True
            result["type"] = "flan_answer"
            # Only process using FLAN if question is asked
            if settings.GetOption("flan_process_only_questions"):
                prompted_text, prompt_change = LLM.llm.whisper_result_prompter(predicted_text)
                if prompt_change:
                    predicted_text = LLM.llm.encode(prompted_text)

                    # translate from auto-detected language to speaker language
                    if settings.GetOption("flan_translate_to_speaker_language"):
                        predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text,
                                                                                                     "auto",
                                                                                                     result['language'],
                                                                                                     False, True)
                    result['flan_answer'] = predicted_text
                    try:
                        print("FLAN question: " + prompted_text)
                        print("FLAN result: " + predicted_text)
                    except:
                        print("FLAN question: ???")
                        print("FLAN result: ???")

                    if predicted_text != prompted_text:
                        send_message(flan_osc_prefix + predicted_text, result)

            # otherwise process every text with FLAN
            else:
                orig_predicted_text = predicted_text
                predicted_text = LLM.llm.encode(predicted_text)
                # translate from auto-detected language to speaker language
                if settings.GetOption("flan_translate_to_speaker_language"):
                    predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text, "auto",
                                                                                                 result['language'],
                                                                                                 False, True)
                result['flan_answer'] = predicted_text
                try:
                    print("FLAN result: " + predicted_text)
                except:
                    print("FLAN result: ???")

                if predicted_text != orig_predicted_text:
                    send_message(flan_osc_prefix + predicted_text, result)

        # send regular message if flan was not loaded
        if not flan_loaded:
            send_message(predicted_text, result)


def plugin_process(predicted_text, result_obj):
    for plugin_inst in plugins:
        plugin_inst.stt(predicted_text, result_obj)


def send_message(predicted_text, result_obj):
    osc_ip = settings.GetOption("osc_ip")
    osc_address = settings.GetOption("osc_address")
    osc_port = settings.GetOption("osc_port")
    websocket_ip = settings.GetOption("websocket_ip")

    # process plugins
    plugin_thread = threading.Thread(target=plugin_process, args=(predicted_text, result_obj))
    plugin_thread.start()

    # Send over OSC
    if osc_ip != "0" and settings.GetOption("osc_auto_processing_enabled") and predicted_text != "":
        VRC_OSCLib.Chat(predicted_text, True, True, osc_address, IP=osc_ip, PORT=osc_port,
                        convert_ascii=settings.GetOption("osc_convert_ascii"))
        settings.SetOption("plugin_timer_stopped", True)

    # Send to Websocket
    if websocket_ip != "0":
        websocket.BroadcastMessage(json.dumps(result_obj))

    # Send to TTS
    if settings.GetOption("flan_whisper_answer"):
        # remove osc prefix from message
        predicted_text = predicted_text.removeprefix(settings.GetOption("flan_osc_prefix")).strip()
    if settings.GetOption("tts_answer") and predicted_text != "" and silero.init():
        try:
            silero_wav, sample_rate = silero.tts.tts(predicted_text)
            silero.tts.play_audio(silero_wav, settings.GetOption("device_out_index"))
        except Exception as e:
            print("Error while playing TTS audio: " + str(e))


def load_whisper(model, ai_device):
    return whisper.load_model(model, download_root=".cache/whisper", device=ai_device)


def convert_audio(audio_bytes: bytes):
    audio_data = io.BytesIO(audio_bytes)
    audio_clip = AudioSegment.from_file(audio_data)

    audio_clip = audio_clip.set_frame_rate(whisper.audio.SAMPLE_RATE)
    # audio_clip = audio_clip.set_channels(1)

    return np.frombuffer(audio_clip.get_array_of_samples(), np.int16).flatten().astype(np.float32) / 32768.0


def whisper_worker():
    whisper_model = settings.GetOption("model")

    whisper_ai_device = settings.GetOption("ai_device")
    loading_state.set_loading_state("whisper_loading", True)
    audio_model = load_whisper(whisper_model, whisper_ai_device)
    loading_state.set_loading_state("whisper_loading", False)

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
        whisper_logprob_threshold = settings.GetOption("logprob_threshold")
        whisper_no_speech_threshold = settings.GetOption("no_speech_threshold")
        whisper_fp16 = settings.GetOption("fp16")

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

        try:

            result = audio_model.transcribe(audio_sample, task=whisper_task, language=whisper_language,
                                            condition_on_previous_text=whisper_condition_on_previous_text,
                                            initial_prompt=whisper_initial_prompt,
                                            logprob_threshold=whisper_logprob_threshold,
                                            no_speech_threshold=whisper_no_speech_threshold,
                                            fp16=whisper_fp16
                                            )

            whisper_result_handling(result)
        except Exception as e:
            print("Error while processing audio: " + str(e))

        q.task_done()
        # send stop info for processing indicator in websocket client
        websocket.BroadcastMessage(json.dumps({"type": "processing_start", "data": False}))


def start_whisper_thread():
    # Turn-on the worker thread.
    threading.Thread(target=whisper_worker, daemon=True).start()

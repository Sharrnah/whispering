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
import io

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

q = queue.Queue()


def whisper_result_handling(result):
    verbose = settings.GetOption("verbose")
    osc_ip = settings.GetOption("osc_ip")
    osc_address = settings.GetOption("osc_address")
    osc_port = settings.GetOption("osc_port")
    websocket_ip = settings.GetOption("websocket_ip")

    predicted_text = result.get('text').strip()

    if not predicted_text.lower() in blacklist:
        if not verbose:
            print("Transcribe" + (" (OSC)" if osc_ip != "0" else "") + ": " + predicted_text)
        else:
            print(result)

        do_txt_translate = settings.GetOption("txt_translate")
        if do_txt_translate:
            from_lang = settings.GetOption("src_lang")
            to_lang = settings.GetOption("trg_lang")
            to_romaji = settings.GetOption("txt_ascii")
            predicted_text = texttranslate.TranslateLanguage(predicted_text, from_lang, to_lang, to_romaji)
            result["txt_translation"] = predicted_text
            result["txt_translation_target"] = to_lang

        # Send to VRChat
        if osc_ip != "0":
            VRC_OSCLib.Chat(predicted_text, True, osc_address, IP=osc_ip, PORT=osc_port,
                            convert_ascii=settings.GetOption("osc_convert_ascii"))
        # Send to Websocket
        if websocket_ip != "0":
            websocket.BroadcastMessage(json.dumps(result))


def load_whisper(model, english, ai_device):
    # there are no english models for large
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model, download_root=".cache/whisper", device=ai_device)
    return audio_model


def convert_audio(audio_bytes: bytes):
    audio_data = io.BytesIO(audio_bytes)
    audio_clip = AudioSegment.from_file(audio_data)

    audio_clip = audio_clip.set_frame_rate(whisper.audio.SAMPLE_RATE)

    return np.frombuffer(audio_clip.get_array_of_samples(), np.int16).flatten().astype(np.float32) / 32768.0


def whisper_worker():
    whisper_model = settings.GetOption("model")
    whisper_english = settings.GetOption("english")
    whisper_ai_device = settings.GetOption("ai_device")
    audio_model = load_whisper(whisper_model, whisper_english, whisper_ai_device)

    print("Say something!")

    while True:
        audio_sample = convert_audio(q.get())

        whisper_task = settings.GetOption("whisper_task")
        whisper_condition_on_previous_text = settings.GetOption("condition_on_previous_text")

        if whisper_english:
            result = audio_model.transcribe(audio_sample, task=whisper_task, language='english',
                                            condition_on_previous_text=whisper_condition_on_previous_text)
        else:
            result = audio_model.transcribe(audio_sample, task=whisper_task,
                                            condition_on_previous_text=whisper_condition_on_previous_text)

        whisper_result_handling(result)
        q.task_done()


def start_whisper_thread():
    # Turn-on the worker thread.
    threading.Thread(target=whisper_worker, daemon=True).start()

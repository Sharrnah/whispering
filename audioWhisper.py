import io
import json
import signal
import sys
import time
import threading

# import speech_recognition_patch as sr  # this is a patched version of speech_recognition. (disabled for now because of freeze issues)
import speech_recognition as sr
import audioprocessor
import os
from pathlib import Path
import click
import VRC_OSCLib
import websocket
import settings
import remote_opener
from Models.STT import faster_whisper
from Models.TextTranslation import texttranslate
from Models import languageClassification
from Models.LLM import LLM
import pyaudiowpatch as pyaudio
from whisper import available_models, audio as whisper_audio

import numpy as np
import torch
import torchaudio
import wave

import Plugins

torchaudio.set_audio_backend("soundfile")
py_audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = whisper_audio.SAMPLE_RATE
CHUNK = int(SAMPLE_RATE / 10)

cache_vad_path = Path(Path.cwd() / ".cache" / "silero-vad")
os.makedirs(cache_vad_path, exist_ok=True)


def sigterm_handler(_signo, _stack_frame):
    # it raises SystemExit(0):
    print('Process died')
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


# Taken from utils_vad.py
def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs


# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound


def call_plugin_timer():
    # Call the method every x seconds
    timer = threading.Timer(settings.GetOption("plugin_timer"), call_plugin_timer)
    timer.start()
    if not settings.GetOption("plugin_timer_stopped"):
        for plugin_inst in Plugins.plugins:
            plugin_inst.timer()
    else:
        if settings.GetOption("plugin_current_timer") <= 0.0:
            settings.SetOption("plugin_current_timer", settings.GetOption("plugin_timer_timeout"))
        else:
            settings.SetOption("plugin_current_timer",
                               settings.GetOption("plugin_current_timer") - settings.GetOption("plugin_timer"))
            if settings.GetOption("plugin_current_timer") <= 0.0:
                settings.SetOption("plugin_timer_stopped", False)
                settings.SetOption("plugin_current_timer", 0.0)


@click.command()
@click.option('--devices', default='False', help='print all available devices id', type=str)
@click.option('--device_index', default=-1, help='the id of the input device (-1 = default active Mic)', type=int)
@click.option('--device_out_index', default=-1, help='the id of the output device (-1 = default active Speaker)',
              type=int)
@click.option('--sample_rate', default=whisper_audio.SAMPLE_RATE, help='sample rate of recording', type=int)
@click.option("--task", default="transcribe",
              help="task for the model whether to only transcribe the audio or translate the audio to english",
              type=click.Choice(["transcribe", "translate"]))
@click.option("--model", default="small", help="Model to use", type=click.Choice(available_models()))
@click.option("--language", default=None,
              help="language spoken in the audio, specify None to perform language detection",
              type=click.Choice(audioprocessor.whisper_get_languages_list_keys()))
@click.option("--condition_on_previous_text", default=False,
              help="Feed it the previous result to keep it consistent across recognition windows, but makes it more prone to getting stuck in a failure loop",
              is_flag=True,
              type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--phrase_time_limit", default=None,
              help="phrase time limit before entry ends to break up long recognitions.", type=float)
@click.option("--osc_ip", default="127.0.0.1", help="IP to send OSC message to. Set to '0' to disable", type=str)
@click.option("--osc_port", default=9000, help="Port to send OSC message to. ('9000' as default for VRChat)", type=int)
@click.option("--osc_address", default="/chatbox/input",
              help="The Address the OSC messages are send to. ('/chatbox/input' as default for VRChat)", type=str)
@click.option("--osc_convert_ascii", default='False', help="Convert Text to ASCII compatible when sending over OSC.",
              type=str)
@click.option("--websocket_ip", default="0", help="IP where Websocket Server listens on. Set to '0' to disable",
              type=str)
@click.option("--websocket_port", default=5000, help="Port where Websocket Server listens on. ('5000' as default)",
              type=int)
@click.option("--ai_device", default=None,
              help="The Device the AI is loaded on. can be 'cuda' or 'cpu'. default does autodetect",
              type=click.Choice(["cuda", "cpu"]))
@click.option("--txt_translator", default="NLLB200",
              help="The Model the AI is loading for text translations. can be 'NLLB200', 'M2M100', 'ARGOS' or 'None'. default is M2M100",
              type=click.Choice(["NLLB200", "M2M100", "ARGOS"]))
@click.option("--txt_translator_size", default="small",
              help="The Model size if M2M100 or NLLB200 text translator is used. can be 'small', 'medium' or 'large' for NLLB200 or 'small' or 'large' for M2M100. default is small. (has no effect with ARGOS)",
              type=click.Choice(["small", "medium", "large"]))
@click.option("--txt_translator_device", default="auto",
              help="The device used for M2M100 translation. (has no effect with ARGOS or NLLB200)",
              type=click.Choice(["auto", "cuda", "cpu"]))
@click.option("--ocr_window_name", default="VRChat",
              help="Window name of the application for OCR translations. (Default: 'VRChat')", type=str)
@click.option("--flan_enabled", default=False,
              help="Enable FLAN-T5 A.I. (General A.I. which can be used for Question Answering.)", type=bool)
@click.option("--open_browser", default=False,
              help="Open default Browser with websocket-remote on start. (requires --websocket_ip to be set as well)",
              is_flag=True, type=bool)
@click.option("--config", default=None,
              help="Use the specified config file instead of the default 'settings.yaml' (relative to the current path) [overwrites without asking!!!]",
              type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
@click.pass_context
def main(ctx, devices, device_index, sample_rate, dynamic_energy, open_browser, config, verbose, **kwargs):
    # Load settings from file
    if config is not None:
        settings.SETTINGS_PATH = Path(Path.cwd() / config)
    settings.LoadYaml(settings.SETTINGS_PATH)

    for plugin_inst in Plugins.plugins:
        plugin_inst.init()

    if str2bool(devices):
        audio = pyaudio.PyAudio()
        print("-------------------------------------------------------------------")
        print("                           Input Devices                           ")
        print(" In form of: DEVICE_NAME [Sample Rate=?] [Loopback?] (Index=INDEX) ")
        print("-------------------------------------------------------------------")
        for device in audio.get_device_info_generator():
            device_list_index = device["index"]
            # device_list_api = device["hostApi"]
            device_list_name = device["name"]
            device_list_sample_rate = int(device["defaultSampleRate"])
            device_list_max_channels = audio.get_device_info_by_index(device_list_index)['maxInputChannels']
            if device_list_max_channels >= 1:
                print(f"{device_list_name} [Sample Rate={device_list_sample_rate}] (Index={device_list_index})")
        print("")
        print("-------------------------------------------------------------------")
        print("                          Output Devices                           ")
        print("-------------------------------------------------------------------")
        for device in audio.get_device_info_generator():
            device_list_index = device["index"]
            device_list_name = device["name"]
            device_list_sample_rate = int(device["defaultSampleRate"])
            device_list_max_channels = audio.get_device_info_by_index(device_list_index)['maxOutputChannels']
            if device_list_max_channels >= 1:
                print(f"{device_list_name} [Sample Rate={device_list_sample_rate}] (Index={device_list_index})")
        return

    print("###################################")
    print("# Whispering Tiger is starting... #")
    print("###################################")

    # set initial settings
    settings.SetOption("whisper_task", settings.GetArgumentSettingFallback(ctx, "task", "whisper_task"))
    device_out_index = settings.GetArgumentSettingFallback(ctx, "device_out_index", "device_out_index")
    settings.SetOption("device_out_index",
                       (device_out_index if device_out_index is None or device_out_index > -1 else None))

    settings.SetOption("condition_on_previous_text",
                       settings.GetArgumentSettingFallback(ctx, "condition_on_previous_text",
                                                           "condition_on_previous_text"))
    model = settings.SetOption("model", settings.GetArgumentSettingFallback(ctx, "model", "model"))

    language = settings.SetOption("current_language",
                                  settings.GetArgumentSettingFallback(ctx, "language", "current_language"))

    phrase_time_limit = settings.SetOption("phrase_time_limit",
                                           settings.GetArgumentSettingFallback(ctx, "phrase_time_limit",
                                                                               "phrase_time_limit"))
    if phrase_time_limit == 0:
        phrase_time_limit = None

    pause = settings.SetOption("pause", settings.GetArgumentSettingFallback(ctx, "pause", "pause"))

    energy = settings.SetOption("energy", settings.GetArgumentSettingFallback(ctx, "energy", "energy"))

    # check if english only model is loaded, and configure whisper languages accordingly.
    if model.endswith(".en") and language not in {"en", "English"}:
        if language is not None:
            print(f"{model} is an English-only model but received '{language}' as language; using English instead.")

        print(f"{model} is an English-only model. only English speech is supported.")
        settings.SetOption("current_language", "en")
        settings.SetOption("whisper_languages", [{"code": "en", "name": "English"}])
    else:
        settings.SetOption("whisper_languages", audioprocessor.whisper_get_languages())

    settings.SetOption("ai_device", settings.GetArgumentSettingFallback(ctx, "ai_device", "ai_device"))
    settings.SetOption("verbose", verbose)

    osc_ip = settings.SetOption("osc_ip", settings.GetArgumentSettingFallback(ctx, "osc_ip", "osc_ip"))
    osc_port = settings.SetOption("osc_port", settings.GetArgumentSettingFallback(ctx, "osc_port", "osc_port"))
    settings.SetOption("osc_address", settings.GetArgumentSettingFallback(ctx, "osc_address", "osc_address"))
    settings.SetOption("osc_convert_ascii",
                       str2bool(settings.GetArgumentSettingFallback(ctx, "osc_convert_ascii", "osc_convert_ascii")))

    websocket_ip = settings.SetOption("websocket_ip",
                                      settings.GetArgumentSettingFallback(ctx, "websocket_ip", "websocket_ip"))
    websocket_port = settings.SetOption("websocket_port",
                                        settings.GetArgumentSettingFallback(ctx, "websocket_port", "websocket_port"))

    txt_translator = settings.SetOption("txt_translator",
                                        settings.GetArgumentSettingFallback(ctx, "txt_translator", "txt_translator"))
    settings.SetOption("txt_translator_size",
                       settings.GetArgumentSettingFallback(ctx, "txt_translator_size", "txt_translator_size"))

    txt_translator_device = settings.SetOption("txt_translator_device",
                                               settings.GetArgumentSettingFallback(ctx, "txt_translator_device",
                                                                                   "txt_translator_device"))
    texttranslate.SetDevice(txt_translator_device)

    settings.SetOption("ocr_window_name",
                       settings.GetArgumentSettingFallback(ctx, "ocr_window_name", "ocr_window_name"))

    settings.SetOption("flan_enabled", settings.GetArgumentSettingFallback(ctx, "flan_enabled", "flan_enabled"))

    if websocket_ip != "0":
        websocket.StartWebsocketServer(websocket_ip, websocket_port)
        if open_browser:
            open_url = 'file://' + os.getcwd() + '/websocket_clients/websocket-remote/index.html' + '?ws_server=ws://' + (
                "127.0.0.1" if websocket_ip == "0.0.0.0" else websocket_ip) + ':' + str(websocket_port)
            remote_opener.openBrowser(open_url)

    if websocket_ip == "0" and open_browser:
        print("--open_browser flag requres --websocket_ip to be set.")

    # Load textual translation dependencies
    if txt_translator.lower() != "none":
        websocket.set_loading_state("txt_transl_loading", True)
        try:
            texttranslate.InstallLanguages()
        except Exception as e:
            print(e)
            pass
        websocket.set_loading_state("txt_transl_loading", False)

    # Load language identification dependencies
    languageClassification.download_model()

    # Load FLAN-T5 dependencies
    LLM.init()

    # Load faster-whisper model
    if settings.GetOption("faster_whisper"):
        whisper_model = settings.GetOption("model")
        whisper_precision = settings.GetOption("whisper_precision")
        # download the model here since its only possible in the main thread
        websocket.set_loading_state("downloading_whisper_model", True)
        faster_whisper.download_model(whisper_model, whisper_precision)
        websocket.set_loading_state("downloading_whisper_model", False)

    # prepare the plugin timer calls
    call_plugin_timer()

    vad_enabled = settings.SetOption("vad_enabled",
                                     settings.GetArgumentSettingFallback(ctx, "vad_enabled", "vad_enabled"))
    vad_thread_num = settings.SetOption("vad_thread_num",
                                        settings.GetArgumentSettingFallback(ctx, "vad_thread_num", "vad_thread_num"))

    if vad_enabled:
        torch.hub.set_dir(str(Path(cache_vad_path).resolve()))
        torch.set_num_threads(vad_thread_num)
        try:
            vad_model, vad_utils = torch.hub.load(trust_repo=True, skip_validation=True,
                                                  repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
                                                  )
        except:
            try:
                vad_model, vad_utils = torch.hub.load(trust_repo=True, skip_validation=True,
                                                      source="local", model="silero_vad", onnx=False,
                                                      repo_or_dir=str(Path(
                                                          cache_vad_path / "snakers4_silero-vad_master").resolve())
                                                      )
            except:
                print("Error loading vad model")
                return False

        # num_samples = 1536
        num_samples = int(settings.SetOption("vad_num_samples",
                                             settings.GetArgumentSettingFallback(ctx, "vad_num_samples",
                                                                                 "vad_num_samples")))
        # clip_duration = 4
        clip_duration = phrase_time_limit

        frames = []
        stream = py_audio.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=SAMPLE_RATE,
                               input=True,
                               input_device_index=(device_index if device_index > -1 else None),
                               frames_per_buffer=CHUNK)

        audioprocessor.start_whisper_thread()

        fps = 0
        if clip_duration is not None:
            fps = int(SAMPLE_RATE / CHUNK * clip_duration)

        start_time = time.time()
        pause_time = time.time()
        previous_audio_chunk = None

        start_rec_on_volume_threshold = False

        continue_recording = True
        while continue_recording:
            audio_chunk = stream.read(num_samples)

            audio_int16 = np.frombuffer(audio_chunk, np.int16)

            audio_float32 = int2float(audio_int16)

            # rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))  # audio volume

            # Calculate the peak amplitude of the audio samples
            # peak_amplitude = np.max(np.abs(audio_int16.astype(np.float32)))
            peak_amplitude = np.max(np.abs(audio_int16))

            # get the confidences and add them to the list to plot them later
            new_confidence = vad_model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()

            end_time = time.time()
            elapsed_time = end_time - start_time

            confidence_threshold = float(settings.GetOption("vad_confidence_threshold"))

            # put frames with recognized speech into a list and send to whisper
            # if (clip_duration is not None and len(frames) > fps) or (elapsed_time > 3 and len(frames) > 0):
            if (clip_duration is not None and len(frames) > fps) or (elapsed_time > pause and len(frames) > 0):
                clip = []
                # for i in range(0, fps):
                for i in range(0, len(frames)):
                    clip.append(frames[i])

                wavefiledata = b''.join(clip)

                # check if the full audio clip is above the confidence threshold
                vad_clip_test = settings.GetOption("vad_on_full_clip")
                full_audio_confidence = 0.
                if vad_clip_test:
                    audio_full_int16 = np.frombuffer(wavefiledata, np.int16)
                    audio_full_float32 = int2float(audio_full_int16)
                    full_audio_confidence = vad_model(torch.from_numpy(audio_full_float32), SAMPLE_RATE).item()
                    print(full_audio_confidence)

                if (not vad_clip_test) or (vad_clip_test and full_audio_confidence >= confidence_threshold):
                    finalwavfile = io.BytesIO()
                    wavefile = wave.open(finalwavfile, 'wb')
                    wavefile.setnchannels(CHANNELS)
                    wavefile.setsampwidth(2)
                    wavefile.setframerate(SAMPLE_RATE)
                    wavefile.writeframes(wavefiledata)

                    finalwavfile.seek(0)
                    audioprocessor.q.put(finalwavfile.read())
                    # vad_iterator.reset_states()  # reset model states after each audio

                    wavefile.close()

                    # set typing indicator for VRChat
                    if osc_ip != "0" and settings.GetOption("osc_auto_processing_enabled") and settings.GetOption(
                            "osc_typing_indicator"):
                        VRC_OSCLib.Bool(True, "/chatbox/typing", IP=osc_ip, PORT=osc_port)
                    # send start info for processing indicator in websocket client
                    websocket.BroadcastMessage(json.dumps({"type": "processing_start", "data": True}))

                frames = []
                start_time = time.time()

            # set start recording variable to true if the volume and voice confidence is above the threshold
            if peak_amplitude >= energy and new_confidence >= confidence_threshold:
                start_rec_on_volume_threshold = True
                pause_time = time.time()

            # append audio frame to the list if the recording var is set and voice confidence is above the threshold (So it only adds the audio parts with speech)
            if start_rec_on_volume_threshold and new_confidence >= confidence_threshold:
                # append previous audio chunk to improve recognition on too late audio recording starts
                if previous_audio_chunk is not None:
                    frames.append(previous_audio_chunk)

                frames.append(audio_chunk)
                start_time = time.time()

            # stop recording if no speech is detected for pause seconds
            if start_rec_on_volume_threshold and (
                    new_confidence < confidence_threshold or confidence_threshold == 0.0) and peak_amplitude < energy and (
                    time.time() - pause_time) > pause:
                start_rec_on_volume_threshold = False

            # save chunk as previous audio chunk to reuse later
            if not start_rec_on_volume_threshold and (
                    new_confidence < confidence_threshold or confidence_threshold == 0.0):
                previous_audio_chunk = audio_chunk
            else:
                previous_audio_chunk = None

    else:
        # load the speech recognizer and set the initial energy threshold and pause threshold
        r = sr.Recognizer()
        r.energy_threshold = energy
        r.pause_threshold = pause
        r.dynamic_energy_threshold = dynamic_energy

        with sr.Microphone(sample_rate=sample_rate,
                           device_index=(device_index if device_index > -1 else None)) as source:

            audioprocessor.start_whisper_thread()

            while True:
                # get and save audio to wav file
                audio = r.listen(source, phrase_time_limit=phrase_time_limit)

                audio_data = audio.get_wav_data()

                # add audio data to the queue
                audioprocessor.q.put(audio_data)

                # set typing indicator for VRChat
                if osc_ip != "0" and settings.GetOption("osc_auto_processing_enabled") and settings.GetOption(
                        "osc_typing_indicator"):
                    VRC_OSCLib.Bool(True, "/chatbox/typing", IP=osc_ip, PORT=osc_port)
                # send start info for processing indicator in websocket client
                websocket.BroadcastMessage(json.dumps({"type": "processing_start", "data": True}))


def str2bool(string):
    if type(string) == str:
        str2val = {"true": True, "false": False}
        if string.lower() in str2val:
            return str2val[string.lower()]
        else:
            raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")
    else:
        return bool(string)


main()

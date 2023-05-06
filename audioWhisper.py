import os
import sys
import json
import traceback

# set environment variable CT2_CUDA_ALLOW_FP16 to 1 (before ctranslate2 is imported)
# to allow using FP16 computation on GPU even if the device does not have efficient FP16 support.
os.environ["CT2_CUDA_ALLOW_FP16"] = "1"


def handle_exception(exc_type, exc_value, exc_traceback):
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    print(error_msg, file=sys.stderr)  # print to standard error stream

    # Format the traceback and error message as a JSON string
    error_dict = {
        'type': "error",
        'message': str(exc_value),
        'traceback': traceback.format_tb(exc_traceback)
    }
    error_json = json.dumps(error_dict)
    print(error_json, file=sys.stderr)  # print to standard error stream


sys.excepthook = handle_exception

import io
import signal
import time
import threading

# import speech_recognition_patch as sr  # this is a patched version of speech_recognition. (disabled for now because of freeze issues)
import speech_recognition as sr
import audioprocessor
from pathlib import Path
import click
import VRC_OSCLib
import websocket
import settings
import remote_opener
from Models.STT import faster_whisper
from Models.TextTranslation import texttranslate
from Models import languageClassification
import pyaudiowpatch as pyaudio
from whisper import available_models, audio as whisper_audio

import numpy as np
import torch
import torchaudio
import audio_tools
import sounddevice as sd

import wave


def save_to_wav(data, filename, sample_rate, channels=1):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # Assuming 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(data)


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
    # reset process id
    settings.SetOption("process_id", 0)

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


def audio_bytes_to_wav(audio_bytes):
    final_wavfile = io.BytesIO()
    wavefile = wave.open(final_wavfile, 'wb')
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(2)
    wavefile.setframerate(SAMPLE_RATE)
    wavefile.writeframes(audio_bytes)

    final_wavfile.seek(0)
    return_data = final_wavfile.read()
    wavefile.close()
    return return_data


def typing_indicator_function(osc_ip, osc_port, send_websocket=True):
    if osc_ip != "0" and settings.GetOption("osc_auto_processing_enabled") and settings.GetOption(
            "osc_typing_indicator"):
        VRC_OSCLib.Bool(True, "/chatbox/typing", IP=osc_ip, PORT=osc_port)
    if send_websocket and settings.GetOption("websocket_ip") != "0":
        websocket.BroadcastMessage(json.dumps({"type": "processing_start", "data": True}))


def process_audio_chunk(audio_chunk, vad_model, sample_rate):
    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)
    new_confidence = vad_model(torch.from_numpy(audio_float32), sample_rate).item()
    peak_amplitude = np.max(np.abs(audio_int16))
    return new_confidence, peak_amplitude


def should_start_recording(peak_amplitude, energy, new_confidence, confidence_threshold):
    return peak_amplitude >= energy and new_confidence >= confidence_threshold


def should_stop_recording(new_confidence, confidence_threshold, peak_amplitude, energy, pause_time, pause):
    return (new_confidence < confidence_threshold or confidence_threshold == 0.0) and peak_amplitude < energy and (
            time.time() - pause_time) > pause


def get_host_audio_api_names():
    audio = pyaudio.PyAudio()
    host_api_count = audio.get_host_api_count()
    host_api_names = {}
    for i in range(host_api_count):
        host_api_info = audio.get_host_api_info_by_index(i)
        host_api_names[i] = host_api_info["name"]
    return host_api_names


def get_default_audio_device_index_by_api(api, is_input=True):
    devices = sd.query_devices()
    api_info = sd.query_hostapis()
    host_api_index = None

    for i, host_api in enumerate(api_info):
        if api.lower() in host_api['name'].lower():
            host_api_index = i
            break

    if host_api_index is None:
        return None

    api_pyaudio_index, _ = get_audio_api_index_by_name(api)

    default_device_index = api_info[host_api_index]['default_input_device' if is_input else 'default_output_device']
    default_device_name = devices[default_device_index]['name']
    return get_audio_device_index_by_name_and_api(default_device_name, api_pyaudio_index, is_input)


def get_audio_device_index_by_name_and_api(name, api, is_input=True, default=None):
    audio = pyaudio.PyAudio()
    device_count = audio.get_device_count()
    for i in range(device_count):
        device_info = audio.get_device_info_by_index(i)
        device_name = device_info["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode('utf-8')

        if device_info["hostApi"] == api and device_info[
            "maxInputChannels" if is_input else "maxOutputChannels"] > 0 and name in device_name:
            return i
    return default


def get_audio_api_index_by_name(name):
    audio = pyaudio.PyAudio()
    host_api_count = audio.get_host_api_count()
    for i in range(host_api_count):
        host_api_info = audio.get_host_api_info_by_index(i)
        if name.lower() in host_api_info["name"].lower():
            return i, host_api_info["name"]
    return 0, ""


@click.command()
@click.option('--devices', default='False', help='print all available devices id', type=str)
@click.option('--device_index', default=-1, help='the id of the input device (-1 = default active Mic)', type=int)
@click.option('--device_out_index', default=-1, help='the id of the output device (-1 = default active Speaker)',
              type=int)
@click.option('--audio_api', default='MME', help='the name of the audio API. ("MME", "DirectSound", "WASAPI")',
              type=str)
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
              help="The Model the AI is loading for text translations. can be 'NLLB200', 'M2M100' or 'None'. default is NLLB200",
              type=click.Choice(["NLLB200", "M2M100"]))
@click.option("--txt_translator_size", default="small",
              help="The Model size if M2M100 or NLLB200 text translator is used. can be 'small', 'medium' or 'large' for NLLB200 or 'small' or 'large' for M2M100. default is small.",
              type=click.Choice(["small", "medium", "large"]))
@click.option("--txt_translator_device", default="auto",
              help="The device used for text translation.",
              type=click.Choice(["auto", "cuda", "cpu"]))
@click.option("--ocr_window_name", default="VRChat",
              help="Window name of the application for OCR translations. (Default: 'VRChat')", type=str)
@click.option("--open_browser", default=False,
              help="Open default Browser with websocket-remote on start. (requires --websocket_ip to be set as well)",
              is_flag=True, type=bool)
@click.option("--config", default=None,
              help="Use the specified config file instead of the default 'settings.yaml' (relative to the current path) [overwrites without asking!!!]",
              type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
@click.pass_context
def main(ctx, devices, sample_rate, dynamic_energy, open_browser, config, verbose, **kwargs):
    if str2bool(devices):
        host_audio_api_names = get_host_audio_api_names()
        audio = pyaudio.PyAudio()
        # print all available host apis
        print("-------------------------------------------------------------------")
        print("                           Host APIs                               ")
        print("-------------------------------------------------------------------")
        for i in range(audio.get_host_api_count()):
            print(f"Host API {i}: {audio.get_host_api_info_by_index(i)['name']}")
        print("")
        print("-------------------------------------------------------------------")
        print("                           Input Devices                           ")
        print(" In form of: DEVICE_NAME [Sample Rate=?] [Loopback?] (Index=INDEX) ")
        print("-------------------------------------------------------------------")
        for device in audio.get_device_info_generator():
            device_list_index = device["index"]
            device_list_api = host_audio_api_names[device["hostApi"]]
            device_list_name = device["name"]
            device_list_sample_rate = int(device["defaultSampleRate"])
            device_list_max_channels = audio.get_device_info_by_index(device_list_index)['maxInputChannels']
            if device_list_max_channels >= 1:
                print(
                    f"{device_list_name} [Sample Rate={device_list_sample_rate}, API={device_list_api}] (Index={device_list_index})")
        print("")
        print("-------------------------------------------------------------------")
        print("                          Output Devices                           ")
        print("-------------------------------------------------------------------")
        for device in audio.get_device_info_generator():
            device_list_index = device["index"]
            device_list_api = host_audio_api_names[device["hostApi"]]
            device_list_name = device["name"]
            device_list_sample_rate = int(device["defaultSampleRate"])
            device_list_max_channels = audio.get_device_info_by_index(device_list_index)['maxOutputChannels']
            if device_list_max_channels >= 1:
                print(
                    f"{device_list_name} [Sample Rate={device_list_sample_rate}, API={device_list_api}] (Index={device_list_index})")
        return

    # Load settings from file
    if config is not None:
        settings.SETTINGS_PATH = Path(Path.cwd() / config)
    settings.LoadYaml(settings.SETTINGS_PATH)

    # set process id
    settings.SetOption("process_id", os.getpid())

    for plugin_inst in Plugins.plugins:
        plugin_inst.init()

    print("###################################")
    print("# Whispering Tiger is starting... #")
    print("###################################")

    # set initial settings
    settings.SetOption("whisper_task", settings.GetArgumentSettingFallback(ctx, "task", "whisper_task"))

    # set audio settings
    device_index = settings.GetArgumentSettingFallback(ctx, "device_index", "device_index")
    settings.SetOption("device_index",
                       (device_index if device_index is None or device_index > -1 else None))
    device_out_index = settings.GetArgumentSettingFallback(ctx, "device_out_index", "device_out_index")
    settings.SetOption("device_out_index",
                       (device_out_index if device_out_index is None or device_out_index > -1 else None))

    audio_api = settings.SetOption("audio_api", settings.GetArgumentSettingFallback(ctx, "audio_api", "audio_api"))
    audio_api_index, audio_api_name = get_audio_api_index_by_name(audio_api)
    print("using Audio API: " + audio_api_name)

    audio_input_device = settings.GetOption("audio_input_device")
    if audio_input_device is not None and audio_input_device != "":
        if audio_input_device.lower() == "Default".lower():
            device_index = None
        else:
            device_index = get_audio_device_index_by_name_and_api(audio_input_device, audio_api_index, True,
                                                                  device_index)
    settings.SetOption("device_index", device_index)

    audio_output_device = settings.GetOption("audio_output_device")
    if audio_output_device is not None and audio_output_device != "":
        if audio_output_device.lower() == "Default".lower():
            device_out_index = None
        else:
            device_out_index = get_audio_device_index_by_name_and_api(audio_output_device, audio_api_index, False,
                                                                      device_out_index)
    settings.SetOption("device_out_index", device_out_index)

    # set default devices:
    device_default_in_index = get_default_audio_device_index_by_api(audio_api, True)
    device_default_out_index = get_default_audio_device_index_by_api(audio_api, False)
    settings.SetOption("device_default_in_index", device_default_in_index)
    settings.SetOption("device_default_out_index", device_default_out_index)

    settings.SetOption("condition_on_previous_text",
                       settings.GetArgumentSettingFallback(ctx, "condition_on_previous_text",
                                                           "condition_on_previous_text"))
    model = settings.SetOption("model", settings.GetArgumentSettingFallback(ctx, "model", "model"))

    language = settings.SetOption("current_language",
                                  settings.GetArgumentSettingFallback(ctx, "language", "current_language"))

    settings.SetOption("phrase_time_limit", settings.GetArgumentSettingFallback(ctx, "phrase_time_limit",
                                                                                "phrase_time_limit"))

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

    if websocket_ip != "0":
        websocket.StartWebsocketServer(websocket_ip, websocket_port)
        if open_browser:
            open_url = 'file://' + os.getcwd() + '/websocket_clients/websocket-remote/index.html' + '?ws_server=ws://' + (
                "127.0.0.1" if websocket_ip == "0.0.0.0" else websocket_ip) + ':' + str(websocket_port)
            remote_opener.openBrowser(open_url)

    if websocket_ip == "0" and open_browser:
        print("--open_browser flag requres --websocket_ip to be set.")

    # Load textual translation dependencies
    if txt_translator.lower() != "none" and txt_translator != "":
        websocket.set_loading_state("txt_transl_loading", True)
        try:
            texttranslate.InstallLanguages()
        except Exception as e:
            print(e)
            pass
        websocket.set_loading_state("txt_transl_loading", False)

    # Load language identification dependencies
    languageClassification.download_model()

    # Download faster-whisper model
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

        # set default devices if not set
        if device_index is None:
            device_index = device_default_in_index

        frames = []

        default_sample_rate = SAMPLE_RATE
        recorded_sample_rate = SAMPLE_RATE
        needs_sample_rate_conversion = False
        try:
            stream = py_audio.open(format=FORMAT,
                                   channels=CHANNELS,
                                   rate=default_sample_rate,
                                   input=True,
                                   input_device_index=(device_index if device_index > -1 else None),
                                   frames_per_buffer=CHUNK)
        except Exception as e:
            print("opening stream failed, falling back to default sample rate")
            dev_info = py_audio.get_device_info_by_index(device_index)

            recorded_sample_rate = int(dev_info['defaultSampleRate'])
            stream = py_audio.open(format=FORMAT,
                                   channels=2,
                                   rate=int(dev_info['defaultSampleRate']),
                                   input=True,
                                   input_device_index=(device_index if device_index > -1 else None),
                                   frames_per_buffer=CHUNK)
            needs_sample_rate_conversion = True

        audioprocessor.start_whisper_thread()

        start_time = time.time()
        pause_time = time.time()
        intermediate_time_start = time.time()
        previous_audio_chunk = None

        start_rec_on_volume_threshold = False

        continue_recording = True
        while continue_recording:
            phrase_time_limit = settings.GetOption("phrase_time_limit")
            pause = settings.GetOption("pause")
            energy = settings.GetOption("energy")
            if phrase_time_limit == 0:
                phrase_time_limit = None

            clip_duration = phrase_time_limit
            fps = 0
            if clip_duration is not None:
                fps = int(default_sample_rate / CHUNK * clip_duration)

            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_intermediate_time = end_time - intermediate_time_start

            confidence_threshold = float(settings.GetOption("vad_confidence_threshold"))

            audio_chunk = stream.read(num_samples)

            # special case which seems to be needed for WASAPI
            if needs_sample_rate_conversion:
                audio_chunk = audio_tools.resample_audio(audio_chunk, recorded_sample_rate, default_sample_rate, -1,
                                                         is_mono=False).tobytes()

            new_confidence, peak_amplitude = process_audio_chunk(audio_chunk, vad_model, default_sample_rate)
            if verbose:
                print("new_confidence: " + str(new_confidence) + " peak_amplitude: " + str(peak_amplitude))

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
                    full_audio_confidence = vad_model(torch.from_numpy(audio_full_float32), default_sample_rate).item()
                    print(full_audio_confidence)

                if (not vad_clip_test) or (vad_clip_test and full_audio_confidence >= confidence_threshold):
                    # debug save of audio clip
                    # save_to_wav(wavefiledata, "resampled_audio_chunk.wav", default_sample_rate)

                    audioprocessor.q.put(
                        {'time': time.time_ns(), 'data': audio_bytes_to_wav(wavefiledata), 'final': True})
                    # vad_iterator.reset_states()  # reset model states after each audio

                    # set typing indicator for VRChat and Websocket clients
                    typing_indicator_thread = threading.Thread(target=typing_indicator_function,
                                                               args=(osc_ip, osc_port, True))
                    typing_indicator_thread.start()

                frames = []
                start_time = time.time()
                intermediate_time_start = time.time()

            # set start recording variable to true if the volume and voice confidence is above the threshold
            if should_start_recording(peak_amplitude, energy, new_confidence, confidence_threshold):
                if not start_rec_on_volume_threshold:
                    # clear frames on start of new recording
                    # frames = []
                    # start processing_start event
                    typing_indicator_thread = threading.Thread(target=typing_indicator_function,
                                                               args=(osc_ip, osc_port, True))
                    typing_indicator_thread.start()
                start_rec_on_volume_threshold = True
                pause_time = time.time()

            # append audio frame to the list if the recording var is set and voice confidence is above the threshold (So it only adds the audio parts with speech)
            if start_rec_on_volume_threshold and new_confidence >= confidence_threshold:
                # append previous audio chunk to improve recognition on too late audio recording starts
                if previous_audio_chunk is not None:
                    frames.append(previous_audio_chunk)

                frames.append(audio_chunk)
                start_time = time.time()
                if settings.GetOption("realtime"):
                    clip = []
                    frame_count = len(frames)
                    # send realtime intermediate results every x frames and every x seconds (making sure its at least x frame length)
                    if frame_count % settings.GetOption(
                            "realtime_frame_multiply") == 0 and elapsed_intermediate_time > settings.GetOption(
                        "realtime_frequency_time"):
                        # set typing indicator for VRChat but not websocket
                        typing_indicator_thread = threading.Thread(target=typing_indicator_function,
                                                                   args=(osc_ip, osc_port, False))
                        typing_indicator_thread.start()
                        # for i in range(0, fps):
                        for i in range(0, len(frames)):
                            clip.append(frames[i])
                        wavefiledata = b''.join(clip)
                        audioprocessor.q.put(
                            {'time': time.time_ns(), 'data': audio_bytes_to_wav(wavefiledata), 'final': False})

                        intermediate_time_start = time.time()

            # stop recording if no speech is detected for pause seconds
            if should_stop_recording(new_confidence, confidence_threshold, peak_amplitude, energy, pause_time, pause):
                start_rec_on_volume_threshold = False
                intermediate_time_start = time.time()

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
                phrase_time_limit = settings.GetOption("phrase_time_limit")
                if phrase_time_limit == 0:
                    phrase_time_limit = None
                pause = settings.GetOption("pause")
                energy = settings.GetOption("energy")

                r.energy_threshold = energy
                r.pause_threshold = pause

                # get and save audio to wav file
                audio = r.listen(source, phrase_time_limit=phrase_time_limit)

                audio_data = audio.get_wav_data()

                # add audio data to the queue
                audioprocessor.q.put({'time': time.time_ns(), 'data': audio_data, 'final': True})

                # set typing indicator for VRChat and websocket clients
                typing_indicator_thread = threading.Thread(target=typing_indicator_function,
                                                           args=(osc_ip, osc_port, True))
                typing_indicator_thread.start()


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

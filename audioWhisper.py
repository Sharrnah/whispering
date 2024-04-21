# -*- encoding: utf-8 -*-
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    import os
    import platform
    import sys
    import json
    import traceback

    import Utilities
    import downloader
    import processmanager
    import atexit

    from Models.TTS import silero

    # set environment variable CT2_CUDA_ALLOW_FP16 to 1 (before ctranslate2 is imported)
    # to allow using FP16 computation on GPU even if the device does not have efficient FP16 support.
    os.environ["CT2_CUDA_ALLOW_FP16"] = "1"

    # enable fast GPU mode for safetensors (https://huggingface.co/docs/safetensors/speed)
    os.environ["SAFETENSORS_FAST_GPU"] = "1"

    atexit.register(processmanager.cleanup_subprocesses)


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
    from Models.Multi import seamless_m4t
    from Models.TextTranslation import texttranslate
    from Models import languageClassification
    from Models import sentence_split
    import pyaudiowpatch as pyaudio
    from whisper import available_models, audio as whisper_audio

    import keyboard

    import numpy as np
    import torch

    torch.backends.cudnn.benchmark = True

    import audio_tools
    import audio_processing_recording
    import sounddevice as sd

    import wave

    from Models.STS import DeepFilterNet


    def save_to_wav(data, filename, sample_rate, channels=1):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # Assuming 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(data)


    #torchaudio.set_audio_backend("soundfile")
    py_audio = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = whisper_audio.SAMPLE_RATE
    CHUNK = int(SAMPLE_RATE / 10)

    cache_vad_path = Path(Path.cwd() / ".cache" / "silero-vad")
    os.makedirs(cache_vad_path, exist_ok=True)


    def sigterm_handler(_signo, _stack_frame):
        processmanager.cleanup_subprocesses()

        # reset process id
        settings.SetOption("process_id", 0)

        # it raises SystemExit(0):
        print('Process died')
        sys.exit(0)


    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGABRT, sigterm_handler)

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


    def call_plugin_timer(plugins):
        # Call the method every x seconds
        timer = threading.Timer(settings.GetOption("plugin_timer"), call_plugin_timer, args=[plugins])
        timer.start()
        if not settings.GetOption("plugin_timer_stopped"):
            for plugin_inst in plugins.plugins:
                if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'timer'):
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


    def call_plugin_sts(plugins, wavefiledata, sample_rate):
        for plugin_inst in plugins.plugins:
            if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'sts'):
                plugin_inst.sts(wavefiledata, sample_rate)


    #def call_plugin_sts_chunk(plugins, wavefiledata, sample_rate):
    #    for plugin_inst in plugins.plugins:
    #        if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'sts_chunk'):
    #            plugin_inst.sts_chunk(wavefiledata, sample_rate)


    def typing_indicator_function(osc_ip, osc_port, send_websocket=True):
        if osc_ip != "0" and settings.GetOption("osc_auto_processing_enabled") and settings.GetOption(
                "osc_typing_indicator"):
            VRC_OSCLib.Bool(True, "/chatbox/typing", IP=osc_ip, PORT=osc_port)
        if send_websocket and settings.GetOption("websocket_ip") != "0":
            threading.Thread(
                target=websocket.BroadcastMessage,
                args=(json.dumps({"type": "processing_start", "data": True}),)
            ).start()


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
                device_name = Utilities.safe_decode(device_name)
            if isinstance(name, bytes):
                name = Utilities.safe_decode(name)

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


    def record_highest_peak_amplitude(device_index=-1, record_time=10):
        py_audio = pyaudio.PyAudio()

        default_sample_rate = SAMPLE_RATE

        stream, needs_sample_rate_conversion, recorded_sample_rate, is_mono = audio_tools.start_recording_audio_stream(
            device_index,
            sample_format=FORMAT,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            chunk=CHUNK,
            py_audio=py_audio,
        )

        highest_peak_amplitude = 0
        start_time = time.time()

        while time.time() - start_time < record_time:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            # special case which seems to be needed for WASAPI
            if needs_sample_rate_conversion:
                audio_chunk = audio_tools.resample_audio(audio_chunk, recorded_sample_rate, default_sample_rate, target_channels=1,
                                                         input_channels=1).tobytes()

            _, peak_amplitude = audio_processing_recording.process_audio_chunk(audio_chunk, default_sample_rate, None)
            highest_peak_amplitude = max(highest_peak_amplitude, peak_amplitude)

        stream.stop_stream()
        stream.close()

        return highest_peak_amplitude



    @click.command()
    @click.option('--detect_energy', default=False, is_flag=True,
                  help='detect energy level after set time of seconds recording.', type=bool)
    @click.option('--detect_energy_time', default=10, help='detect energy level time it records for.', type=int)
    @click.option('--audio_input_device', default="Default", help='audio input device name. (used for detect_energy',
                  type=str)
    @click.option('--ui_download', default=False, is_flag=True,
                  help='use UI application for downloads.', type=bool)
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
    def main(ctx, detect_energy, detect_energy_time, ui_download, devices, sample_rate, dynamic_energy, open_browser,
             config, verbose,
             **kwargs):
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

        # is set to run energy detection
        if detect_energy:
            # get selected audio api
            audio_api = "MME"
            if settings.IsArgumentSetting(ctx, "audio_api"):
                audio_api = ctx.params["audio_api"]
            audio_api_index, audio_api_name = get_audio_api_index_by_name(audio_api)

            # get selected audio input device
            device_index = None
            if settings.IsArgumentSetting(ctx, "device_index"):
                device_index = ctx.params["device_index"]
            device_default_in_index = get_default_audio_device_index_by_api(audio_api, True)

            # get selected audio input device by name if possible
            if settings.IsArgumentSetting(ctx, "audio_input_device"):
                audio_input_device = ctx.params["audio_input_device"]
                if audio_input_device is not None and audio_input_device != "":
                    if audio_input_device.lower() == "Default".lower():
                        device_index = None
                    else:
                        device_index = get_audio_device_index_by_name_and_api(audio_input_device, audio_api_index, True,
                                                                              device_index)
            if device_index is None or device_index < 0:
                device_index = device_default_in_index

            max_detected_energy = record_highest_peak_amplitude(device_index, detect_energy_time)
            print("detected_energy: " + str(max_detected_energy))
            return

        # Load settings from file
        if config is not None:
            settings.SETTINGS_PATH = Path(Path.cwd() / config)
        settings.LoadYaml(settings.SETTINGS_PATH)

        # set process id
        settings.SetOption("process_id", os.getpid())

        settings.SetOption("ui_download", ui_download)

        # enable stt by default
        settings.SetOption("stt_enabled", True)

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

        print("###################################")
        print("# Whispering Tiger is starting... #")
        print("###################################")

        print("running Python: " + platform.python_implementation() + " / v" + platform.python_version())
        print("using Audio API: " + audio_api_name)
        print("")

        # check if english only model is loaded, and configure STT languages accordingly.
        if model.endswith(".en") and "_whisper" in settings.GetOption("stt_type"):
            if language is not None and language not in {"en", "English"}:
                print(f"{model} is an English-only model but received '{language}' as language; using English instead.")

            print(f"{model} is an English-only model. only English speech is supported.")
            settings.SetOption("whisper_languages", ({"code": "", "name": "Auto"}, {"code": "en", "name": "English"},))
            settings.SetOption("current_language", "en")
        elif "_whisper" in settings.GetOption("stt_type") or "whisper_" in settings.GetOption("stt_type"):
            settings.SetOption("whisper_languages", audioprocessor.whisper_get_languages())
        elif settings.GetOption("stt_type") == "seamless_m4t":
            settings.SetOption("whisper_languages", audioprocessor.seamless_m4t_get_languages())
        elif settings.GetOption("stt_type") == "speech_t5":
            # speech t5 only supports english
            print(f"speechT5 is an English-only model. only English speech is supported.")
            settings.SetOption("whisper_languages", ({"code": "", "name": "Auto"}, {"code": "en", "name": "English"},))
            settings.SetOption("current_language", "en")
        elif settings.GetOption("stt_type") == "wav2vec_bert":
            settings.SetOption("whisper_languages", audioprocessor.wav2vec_bert_get_languages())
        elif settings.GetOption("stt_type") == "nemo_canary":
            settings.SetOption("whisper_languages", audioprocessor.nemo_canary_get_languages())
        else:
            # show no language if unspecified STT type
            settings.SetOption("whisper_languages", ({"code": "", "name": ""},))

        settings.SetOption("ai_device", settings.GetArgumentSettingFallback(ctx, "ai_device", "ai_device"))
        settings.SetOption("verbose", verbose)

        osc_ip = settings.SetOption("osc_ip", settings.GetArgumentSettingFallback(ctx, "osc_ip", "osc_ip"))
        osc_port = settings.SetOption("osc_port", settings.GetArgumentSettingFallback(ctx, "osc_port", "osc_port"))
        settings.SetOption("osc_address", settings.GetArgumentSettingFallback(ctx, "osc_address", "osc_address"))
        settings.SetOption("osc_convert_ascii",
                           str2bool(settings.GetArgumentSettingFallback(ctx, "osc_convert_ascii", "osc_convert_ascii")))
        osc_min_time_between_messages = settings.SetOption("osc_min_time_between_messages", settings.GetArgumentSettingFallback(ctx, "osc_min_time_between_messages", "osc_min_time_between_messages"))
        VRC_OSCLib.set_min_time_between_messages(osc_min_time_between_messages)

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

        # initialize Silero TTS
        try:
            silero.init()
        except Exception as e:
            print(e)

        if ui_download:
            # wait until ui is connected
            print("waiting for ui to connect...")
            max_wait = 15  # wait max 15 seconds for ui to connect
            last_wait_time = time.time()
            while len(websocket.WS_CLIENTS) == 0 and websocket.UI_CONNECTED["value"] is False:
                time.sleep(0.1)
                if time.time() - last_wait_time > max_wait:
                    print("timeout while waiting for ui to connect.")
                    ui_download = False
                    settings.SetOption("ui_download", ui_download)
                    break
            if ui_download:  # still true? then ui did connect
                print("ui connected.")
                time.sleep(0.5)

        # initialize plugins
        import Plugins
        print("initializing plugins...")
        for plugin_inst in Plugins.plugins:
            plugin_inst.init()
            if plugin_inst.is_enabled(False):
                print(plugin_inst.__class__.__name__ + " is enabled")
            else:
                print(plugin_inst.__class__.__name__ + " is disabled")

        # Load textual translation dependencies
        if txt_translator.lower() != "none" and txt_translator != "":
            websocket.set_loading_state("txt_transl_loading", True)
            try:
                texttranslate.InstallLanguages()
            except Exception as e:
                print(e)
                pass
            websocket.set_loading_state("txt_transl_loading", False)

        # load nltk sentence splitting dependency
        sentence_split.load_model()

        # Load language identification dependencies
        languageClassification.download_model()

        # Download faster-whisper model
        if settings.GetOption("stt_type") == "faster_whisper":
            whisper_model = settings.GetOption("model")
            whisper_precision = settings.GetOption("whisper_precision")
            realtime_whisper_model = settings.GetOption("realtime_whisper_model")
            realtime_whisper_precision = settings.GetOption("realtime_whisper_precision")
            # download the model here since its only possible in the main thread
            if faster_whisper.needs_download(whisper_model, whisper_precision):
                websocket.set_loading_state("downloading_whisper_model", True)
                faster_whisper.download_model(whisper_model, whisper_precision)
                websocket.set_loading_state("downloading_whisper_model", False)
            # download possibly needed realtime model
            if realtime_whisper_model != "" and faster_whisper.needs_download(realtime_whisper_model,
                                                                              realtime_whisper_precision):
                websocket.set_loading_state("downloading_whisper_model", True)
                faster_whisper.download_model(realtime_whisper_model, realtime_whisper_precision)
                websocket.set_loading_state("downloading_whisper_model", False)
        if settings.GetOption("stt_type") == "seamless_m4t":
            stt_model_size = settings.GetOption("model")
            if seamless_m4t.SeamlessM4T.needs_download(stt_model_size):
                websocket.set_loading_state("downloading_whisper_model", True)
                seamless_m4t.SeamlessM4T.download_model(stt_model_size)
                websocket.set_loading_state("downloading_whisper_model", False)

        # load audio filter model
        audio_enhancer = None
        if settings.GetOption("denoise_audio"):
            websocket.set_loading_state("loading_denoiser", True)
            post_filter = settings.GetOption("denoise_audio_post_filter")
            audio_enhancer = DeepFilterNet.DeepFilterNet(post_filter=post_filter)
            websocket.set_loading_state("loading_denoiser", False)

        # prepare the plugin timer calls
        call_plugin_timer(Plugins)

        vad_enabled = settings.SetOption("vad_enabled",
                                         settings.GetArgumentSettingFallback(ctx, "vad_enabled", "vad_enabled"))
        try:
            vad_thread_num = int(float(settings.SetOption("vad_thread_num",
                                            settings.GetArgumentSettingFallback(ctx, "vad_thread_num", "vad_thread_num"))))
        except ValueError as e:
            print("Error assigning vad_thread_num. using 1")
            print(e)
            vad_thread_num = int(1)

        if vad_enabled:
            vad_model = None
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
                except Exception as e:
                    print("Error loading vad model trying to load from fallback server...")
                    print(e)

                    vad_fallback_server = {
                        "urls": [
                            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/silero-vad.zip",
                            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/silero-vad.zip",
                            "https://s3.libs.space:9000/ai-models/silero/silero-vad.zip"
                        ],
                        "sha256": "097cfacdc2b2f5b09e0da1273b3e30b0e96c3588445958171a7e339cc5805683",
                    }

                    try:
                        downloader.download_extract(vad_fallback_server["urls"],
                                                    str(Path(cache_vad_path).resolve()),
                                                    vad_fallback_server["sha256"],
                                                    alt_fallback=True,
                                                    fallback_extract_func=downloader.extract_zip,
                                                    fallback_extract_func_args=(
                                                        str(Path(cache_vad_path / "silero-vad.zip")),
                                                        str(Path(cache_vad_path).resolve()),
                                                    ),
                                                    title="Silero VAD", extract_format="zip")

                        vad_model, vad_utils = torch.hub.load(trust_repo=True, skip_validation=True,
                                                              source="local", model="silero_vad", onnx=False,
                                                              repo_or_dir=str(Path(
                                                                  cache_vad_path / "snakers4_silero-vad_master").resolve())
                                                              )

                    except Exception as e:
                        print("Error loading vad model.")
                        print(e)

            # num_samples = 1536
            vad_frames_per_buffer = int(settings.SetOption("vad_frames_per_buffer",
                                                 settings.GetArgumentSettingFallback(ctx, "vad_frames_per_buffer",
                                                                                     "vad_frames_per_buffer")))

            # set default devices if not set
            if device_index is None or device_index < 0:
                device_index = device_default_in_index

            #frames = []

            default_sample_rate = SAMPLE_RATE

            previous_audio_chunk = None

            start_rec_on_volume_threshold = False

            push_to_talk_key = settings.GetOption("push_to_talk_key")
            if push_to_talk_key == "":
                push_to_talk_key = None
            keyboard_rec_force_stop = False

            processor = audio_processing_recording.AudioProcessor(
                default_sample_rate=default_sample_rate,
                previous_audio_chunk=previous_audio_chunk,
                start_rec_on_volume_threshold=start_rec_on_volume_threshold,
                push_to_talk_key=push_to_talk_key,
                keyboard_rec_force_stop=keyboard_rec_force_stop,
                vad_model=vad_model,
                plugins=Plugins,
                audio_enhancer=audio_enhancer,
                osc_ip=osc_ip,
                osc_port=osc_port,
                chunk=vad_frames_per_buffer,
                channels=CHANNELS,
                sample_format=FORMAT,
                audio_queue=audioprocessor.q,
                settings=settings,
                typing_indicator_function=typing_indicator_function,
                verbose=verbose,
            )

            # initialize audio stream
            stream, needs_sample_rate_conversion, recorded_sample_rate, is_mono = audio_tools.start_recording_audio_stream(
                device_index,
                sample_format=FORMAT,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                chunk=vad_frames_per_buffer,
                py_audio=py_audio,
                audio_processor=processor,
            )

            # Start the stream
            stream.start_stream()

            #orig_recorded_sample_rate = recorded_sample_rate

            audioprocessor.start_whisper_thread()

            #continue_recording = True

            while stream.is_active():
                time.sleep(0.1)
                #if not settings.GetOption("stt_enabled"):
                #    time.sleep(0.1)
                #    continue

        else:
            # load the speech recognizer and set the initial energy threshold and pause threshold
            r = sr.Recognizer()
            r.energy_threshold = energy
            r.pause_threshold = pause
            r.dynamic_energy_threshold = dynamic_energy

            with sr.Microphone(sample_rate=whisper_audio.SAMPLE_RATE,
                               device_index=device_index) as source:

                audioprocessor.start_whisper_thread()

                while True:
                    if not settings.GetOption("stt_enabled"):
                        time.sleep(0.1)
                        continue

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

                    silence_cutting_enabled = settings.GetOption("silence_cutting_enabled")
                    silence_offset = settings.GetOption("silence_offset")
                    max_silence_length = settings.GetOption("max_silence_length")
                    keep_silence_length = settings.GetOption("keep_silence_length")

                    normalize_enabled = settings.GetOption("normalize_enabled")
                    normalize_lower_threshold = settings.GetOption("normalize_lower_threshold")
                    normalize_upper_threshold = settings.GetOption("normalize_upper_threshold")
                    normalize_gain_factor = settings.GetOption("normalize_gain_factor")
                    block_size_samples = int(whisper_audio.SAMPLE_RATE * 0.400)
                    # normalize audio (and make sure it's longer or equal the default block size by pyloudnorm)
                    if normalize_enabled and len(audio_data) >= block_size_samples:
                        audio_data = audio_tools.convert_audio_datatype_to_float(np.frombuffer(audio_data, np.int16))
                        audio_data, lufs = audio_tools.normalize_audio_lufs(
                            audio_data, whisper_audio.SAMPLE_RATE, normalize_lower_threshold, normalize_upper_threshold,
                            normalize_gain_factor, verbose=verbose
                        )
                        audio_data = audio_tools.convert_audio_datatype_to_integer(audio_data, np.int16)
                        audio_data = audio_data.tobytes()

                    # remove silence from audio
                    if silence_cutting_enabled:
                        audio_data_np = np.frombuffer(audio_data, np.int16)
                        if len(audio_data_np) >= block_size_samples:
                            audio_data = audio_tools.remove_silence_parts(
                                audio_data_np, whisper_audio.SAMPLE_RATE,
                                silence_offset=silence_offset, max_silence_length=max_silence_length, keep_silence_length=keep_silence_length,
                                verbose=verbose
                            )
                            audio_data = audio_data.tobytes()

                    # denoise audio
                    if settings.GetOption("denoise_audio") and audio_enhancer is not None:
                        audio_data = audio_enhancer.enhance_audio(audio_data).tobytes()

                    # add audio data to the queue
                    wav_audio_bytes = audio_tools.audio_bytes_to_wav(audio_data, channels=CHANNELS, sample_rate=SAMPLE_RATE)
                    audioprocessor.q.put({'time': time.time_ns(), 'data': wav_audio_bytes, 'final': True})

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


    #freeze_support()
    main()

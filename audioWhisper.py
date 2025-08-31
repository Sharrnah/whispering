# -*- encoding: utf-8 -*-

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    import os
    import platform
    import sys
    import json
    import traceback

    import processmanager
    import atexit

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
    import platform
    if platform.system() == 'Windows':
        import pyaudiowpatch as pyaudio
    else:
        import pyaudio
    from whisper import available_models, audio as whisper_audio

    import numpy as np
    import torch

    torch.backends.cudnn.benchmark = True

    import audio_tools
    import audio_processing_recording

    import VRC_OSCServer

    import wave

    from Models.STS import DeepFilterNet
    from Models.STS import Noisereduce
    from Models.STS import VAD

    def save_to_wav(data, filename, sample_rate, channels=1):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # Assuming 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(data)


    #torchaudio.set_audio_backend("soundfile")
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = whisper_audio.SAMPLE_RATE
    CHUNK = int(SAMPLE_RATE / 10)

    def sigterm_handler(_signo, _stack_frame):
        processmanager.cleanup_subprocesses()

        # reset process id
        settings.SETTINGS.SetOption("process_id", 0)

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
        timer = threading.Timer(settings.SETTINGS.GetOption("plugin_timer"), call_plugin_timer, args=[plugins])
        timer.start()
        if not settings.SETTINGS.GetOption("plugin_timer_stopped"):
            for plugin_inst in plugins.plugins:
                if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'timer'):
                    try:
                        plugin_inst.timer()
                    except Exception as e:
                        print(f"Error calling plugin timer for {plugin_inst.__class__.__name__}: {e}")
                        traceback.print_exc()
        else:
            if settings.SETTINGS.GetOption("plugin_current_timer") <= 0.0:
                settings.SETTINGS.SetOption("plugin_current_timer", settings.SETTINGS.GetOption("plugin_timer_timeout"))
            else:
                settings.SETTINGS.SetOption("plugin_current_timer",
                                   settings.SETTINGS.GetOption("plugin_current_timer") - settings.SETTINGS.GetOption("plugin_timer"))
                if settings.SETTINGS.GetOption("plugin_current_timer") <= 0.0:
                    settings.SETTINGS.SetOption("plugin_timer_stopped", False)
                    settings.SETTINGS.SetOption("plugin_current_timer", 0.0)


    def typing_indicator_function(osc_ip, osc_port, send_websocket=True):
        if osc_ip != "0" and settings.SETTINGS.GetOption("osc_auto_processing_enabled") and settings.SETTINGS.GetOption(
                "osc_typing_indicator"):
            VRC_OSCLib.Bool(True, "/chatbox/typing", IP=osc_ip, PORT=osc_port)
        if send_websocket and settings.SETTINGS.GetOption("websocket_ip") != "0":
            threading.Thread(
                target=websocket.BroadcastMessage,
                args=(json.dumps({"type": "processing_start", "data": True}),)
            ).start()



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


    def get_device_info_generator():
        audio = pyaudio.PyAudio()
        if hasattr(audio, "get_device_info_generator"):
            return audio.get_device_info_generator()
        else:
            return (audio.get_device_info_by_index(i) for i in range(audio.get_device_count()))



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
            host_audio_api_names = audio_tools.get_host_audio_api_names()
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
            for device in get_device_info_generator():
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
            for device in get_device_info_generator():
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
            if settings.SETTINGS.is_argument_setting(ctx, "audio_api"):
                audio_api = ctx.params["audio_api"]
            audio_api_index, audio_api_name = audio_tools.get_audio_api_index_by_name(audio_api)

            # get selected audio input device
            device_index = None
            if settings.SETTINGS.is_argument_setting(ctx, "device_index"):
                device_index = ctx.params["device_index"]
            device_default_in_index = audio_tools.get_default_audio_device_index_by_api(audio_api, True)

            # get selected audio input device by name if possible
            if settings.SETTINGS.is_argument_setting(ctx, "audio_input_device"):
                audio_input_device = ctx.params["audio_input_device"]
                if audio_input_device is not None and audio_input_device != "":
                    if audio_input_device.lower() == "Default".lower():
                        device_index = None
                    else:
                        device_index = audio_tools.get_audio_device_index_by_name_and_api(audio_input_device, audio_api_index, True,
                                                                              device_index)
            if device_index is None or device_index < 0:
                device_index = device_default_in_index

            max_detected_energy = record_highest_peak_amplitude(device_index, detect_energy_time)
            print("detected_energy: " + str(max_detected_energy))
            return

        # Load settings from file
        settings_path = settings.DEFAULT_SETTINGS_PATH
        if config is not None:
            settings_path = Path(Path.cwd() / config)
        settings.SETTINGS.load_yaml(settings_path)

        # set process id
        settings.SETTINGS.SetOption("process_id", os.getpid())

        settings.SETTINGS.SetOption("ui_download", ui_download)

        # enable stt by default
        settings.SETTINGS.SetOption("stt_enabled", True)

        # set initial settings
        settings.SETTINGS.SetOption("whisper_task", settings.SETTINGS.get_argument_setting_fallback(ctx, "task", "whisper_task"))

        # set audio settings
        device_index = settings.SETTINGS.get_argument_setting_fallback(ctx, "device_index", "device_index")
        settings.SETTINGS.SetOption("device_index",
                           (device_index if device_index is None or device_index > -1 else None))
        device_out_index = settings.SETTINGS.get_argument_setting_fallback(ctx, "device_out_index", "device_out_index")
        settings.SETTINGS.SetOption("device_out_index",
                           (device_out_index if device_out_index is None or device_out_index > -1 else None))

        audio_api = settings.SETTINGS.SetOption("audio_api", settings.SETTINGS.get_argument_setting_fallback(ctx, "audio_api", "audio_api"))
        audio_api_index, audio_api_name = audio_tools.get_audio_api_index_by_name(audio_api)

        audio_input_device = settings.SETTINGS.GetOption("audio_input_device")
        if audio_input_device is not None and audio_input_device != "":
            if audio_input_device.lower() == "Default".lower():
                device_index = None
            else:
                device_index = audio_tools.get_audio_device_index_by_name_and_api(audio_input_device, audio_api_index, True,
                                                                      device_index)
        settings.SETTINGS.SetOption("device_index", device_index)

        audio_output_device = settings.SETTINGS.GetOption("audio_output_device")
        if audio_output_device is not None and audio_output_device != "":
            if audio_output_device.lower() == "Default".lower():
                device_out_index = None
            else:
                device_out_index = audio_tools.get_audio_device_index_by_name_and_api(audio_output_device, audio_api_index, False,
                                                                          device_out_index)
        settings.SETTINGS.SetOption("device_out_index", device_out_index)

        # set default devices:
        device_default_in_index = audio_tools.get_default_audio_device_index_by_api(audio_api, True)
        device_default_out_index = audio_tools.get_default_audio_device_index_by_api(audio_api, False)
        settings.SETTINGS.SetOption("device_default_in_index", device_default_in_index)
        settings.SETTINGS.SetOption("device_default_out_index", device_default_out_index)

        settings.SETTINGS.SetOption("condition_on_previous_text",
                           settings.SETTINGS.get_argument_setting_fallback(ctx, "condition_on_previous_text",
                                                               "condition_on_previous_text"))
        model = settings.SETTINGS.SetOption("model", settings.SETTINGS.get_argument_setting_fallback(ctx, "model", "model"))

        language = settings.SETTINGS.SetOption("current_language",
                                      settings.SETTINGS.get_argument_setting_fallback(ctx, "language", "current_language"))

        settings.SETTINGS.SetOption("phrase_time_limit", settings.SETTINGS.get_argument_setting_fallback(ctx, "phrase_time_limit",
                                                                                    "phrase_time_limit"))

        pause = settings.SETTINGS.SetOption("pause", settings.SETTINGS.get_argument_setting_fallback(ctx, "pause", "pause"))

        energy = settings.SETTINGS.SetOption("energy", settings.SETTINGS.get_argument_setting_fallback(ctx, "energy", "energy"))

        print("###################################")
        print("# Whispering Tiger is starting... #")
        print("###################################")

        print("running Python: " + platform.python_implementation() + " / v" + platform.python_version())
        print("using Torch: " + torch.__version__)
        print("cuDNN-Version:", torch.backends.cudnn.version())
        print("using Audio API: " + audio_api_name)
        print("")

        # check if english only model is loaded, and configure STT languages accordingly.
        if model.endswith(".en") and "_whisper" in settings.SETTINGS.GetOption("stt_type"):
            if language is not None and language not in {"en", "English"}:
                print(f"{model} is an English-only model but received '{language}' as language; using English instead.")

            print(f"{model} is an English-only model. only English speech is supported.")
            settings.SETTINGS.SetOption("whisper_languages", ({"code": "", "name": "Auto"}, {"code": "en", "name": "English"},))
            settings.SETTINGS.SetOption("current_language", "en")
        elif "_whisper" in settings.SETTINGS.GetOption("stt_type") or "whisper_" in settings.SETTINGS.GetOption("stt_type"):
            settings.SETTINGS.SetOption("whisper_languages", audioprocessor.whisper_get_languages())
        elif settings.SETTINGS.GetOption("stt_type") == "seamless_m4t":
            settings.SETTINGS.SetOption("whisper_languages", audioprocessor.seamless_m4t_get_languages())
        elif settings.SETTINGS.GetOption("stt_type") == "mms":
            settings.SETTINGS.SetOption("whisper_languages", audioprocessor.mms_get_languages())
        elif settings.SETTINGS.GetOption("stt_type") == "speech_t5":
            # speech t5 only supports english
            print(f"speechT5 is an English-only model. only English speech is supported.")
            settings.SETTINGS.SetOption("whisper_languages", ({"code": "", "name": "Auto"}, {"code": "en", "name": "English"},))
            settings.SETTINGS.SetOption("current_language", "en")
        elif settings.SETTINGS.GetOption("stt_type") == "wav2vec_bert":
            settings.SETTINGS.SetOption("whisper_languages", audioprocessor.wav2vec_bert_get_languages())
        elif settings.SETTINGS.GetOption("stt_type") == "nemo_canary":
            settings.SETTINGS.SetOption("whisper_languages", audioprocessor.nemo_canary_get_languages())
        elif settings.SETTINGS.GetOption("stt_type") == "phi4":
            settings.SETTINGS.SetOption("whisper_languages", audioprocessor.phi4_get_languages())
        elif settings.SETTINGS.GetOption("stt_type") == "voxtral":
            settings.SETTINGS.SetOption("whisper_languages", audioprocessor.voxtral_get_languages())
        else:
            # show no language if unspecified STT type
            settings.SETTINGS.SetOption("whisper_languages", ({"code": "", "name": ""},))

        settings.SETTINGS.SetOption("ai_device", settings.SETTINGS.get_argument_setting_fallback(ctx, "ai_device", "ai_device"))
        settings.SETTINGS.SetOption("verbose", verbose)

        osc_ip = settings.SETTINGS.SetOption("osc_ip", settings.SETTINGS.get_argument_setting_fallback(ctx, "osc_ip", "osc_ip"))
        osc_port = settings.SETTINGS.SetOption("osc_port", settings.SETTINGS.get_argument_setting_fallback(ctx, "osc_port", "osc_port"))
        settings.SETTINGS.SetOption("osc_address", settings.SETTINGS.get_argument_setting_fallback(ctx, "osc_address", "osc_address"))
        settings.SETTINGS.SetOption("osc_convert_ascii",
                           str2bool(settings.SETTINGS.get_argument_setting_fallback(ctx, "osc_convert_ascii", "osc_convert_ascii")))
        osc_min_time_between_messages = settings.SETTINGS.SetOption("osc_min_time_between_messages", settings.SETTINGS.get_argument_setting_fallback(ctx, "osc_min_time_between_messages", "osc_min_time_between_messages"))
        VRC_OSCLib.set_min_time_between_messages(osc_min_time_between_messages)

        websocket_ip = settings.SETTINGS.SetOption("websocket_ip",
                                          settings.SETTINGS.get_argument_setting_fallback(ctx, "websocket_ip", "websocket_ip"))
        websocket_port = settings.SETTINGS.SetOption("websocket_port",
                                            settings.SETTINGS.get_argument_setting_fallback(ctx, "websocket_port", "websocket_port"))

        txt_translator = settings.SETTINGS.SetOption("txt_translator",
                                            settings.SETTINGS.get_argument_setting_fallback(ctx, "txt_translator", "txt_translator"))
        settings.SETTINGS.SetOption("txt_translator_size",
                           settings.SETTINGS.get_argument_setting_fallback(ctx, "txt_translator_size", "txt_translator_size"))

        txt_translator_device = settings.SETTINGS.SetOption("txt_translator_device",
                                                   settings.SETTINGS.get_argument_setting_fallback(ctx, "txt_translator_device",
                                                                                       "txt_translator_device"))
        texttranslate.SetDevice(txt_translator_device)

        settings.SETTINGS.SetOption("ocr_window_name",
                           settings.SETTINGS.get_argument_setting_fallback(ctx, "ocr_window_name", "ocr_window_name"))

        if websocket_ip != "0":
            websocket.main_server = websocket.StartWebsocketServer(websocket_ip, websocket_port)
            if open_browser:
                open_url = 'file://' + os.getcwd() + '/websocket_clients/websocket-remote/index.html' + '?ws_server=ws://' + (
                    "127.0.0.1" if websocket_ip == "0.0.0.0" else websocket_ip) + ':' + str(websocket_port)
                remote_opener.openBrowser(open_url)

        if websocket_ip == "0" and open_browser:
            print("--open_browser flag requres --websocket_ip to be set.")

        if ui_download:
            # wait until ui is connected
            print("waiting for ui to connect...")
            max_wait = 15  # wait max 15 seconds for ui to connect
            last_wait_time = time.time()
            while len(websocket.get_connected_clients()) == 0 and websocket.UI_CONNECTED["value"] is False:
                time.sleep(0.1)
                if time.time() - last_wait_time > max_wait:
                    print("timeout while waiting for ui to connect.")
                    ui_download = False
                    settings.SETTINGS.SetOption("ui_download", ui_download)
                    break
            if ui_download:  # still true? then ui did connect
                print("ui connected.")
                time.sleep(0.5)

        # initialize Integrated TTS
        try:
            from Models.TTS import tts
            tts.init()
            if tts.tts is not None and not tts.failed:
                available_tts_models = tts.tts.list_models_indexed()
                threading.Thread(
                   target=websocket.BroadcastMessage,
                   args=(json.dumps({"type": "available_tts_models", "data": available_tts_models}),)
                ).start()
                tts.tts.load()
                threading.Thread(
                   target=websocket.BroadcastMessage,
                   args=(json.dumps({"type": "available_tts_voices", "data": tts.tts.list_voices()}),)
                ).start()
        except Exception as e:
            print(e)

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
        if settings.SETTINGS.GetOption("stt_type") == "faster_whisper":
            whisper_model = settings.SETTINGS.GetOption("model")
            whisper_precision = settings.SETTINGS.GetOption("whisper_precision")
            realtime_whisper_model = settings.SETTINGS.GetOption("realtime_whisper_model")
            realtime_whisper_precision = settings.SETTINGS.GetOption("realtime_whisper_precision")
            # download the model here since its only possible in the main thread
            if faster_whisper.needs_download(whisper_model, whisper_precision):
                faster_whisper.download_model(whisper_model, whisper_precision)
            # download possibly needed realtime model
            if realtime_whisper_model != "" and faster_whisper.needs_download(realtime_whisper_model,
                                                                              realtime_whisper_precision):
                faster_whisper.download_model(realtime_whisper_model, realtime_whisper_precision)
        if settings.SETTINGS.GetOption("stt_type") == "seamless_m4t":
            stt_model_size = settings.SETTINGS.GetOption("model")
            if seamless_m4t.SeamlessM4T.needs_download(stt_model_size):
                seamless_m4t.SeamlessM4T.download_model(stt_model_size)

        # load audio filter model
        audio_enhancer = None
        if settings.SETTINGS.GetOption("denoise_audio") == "deepfilter":
            websocket.set_loading_state("loading_denoiser", True)
            post_filter = settings.SETTINGS.GetOption("denoise_audio_post_filter")
            audio_enhancer = DeepFilterNet.DeepFilterNet(post_filter=post_filter)
            websocket.set_loading_state("loading_denoiser", False)
        elif settings.SETTINGS.GetOption("denoise_audio") == "noise_reduce":
            websocket.set_loading_state("loading_denoiser", True)
            audio_enhancer = Noisereduce.Noisereduce()
            websocket.set_loading_state("loading_denoiser", False)

        # Initialize VAD model
        vad_enabled = settings.SETTINGS.SetOption("vad_enabled",
                                         settings.SETTINGS.get_argument_setting_fallback(ctx, "vad_enabled", "vad_enabled"))
        try:
            vad_thread_num = int(float(settings.SETTINGS.SetOption("vad_thread_num",
                                            settings.SETTINGS.get_argument_setting_fallback(ctx, "vad_thread_num", "vad_thread_num"))))
        except ValueError as e:
            print("Error assigning vad_thread_num. using 1")
            print(e)
            vad_thread_num = int(1)

        vad_model = None
        if vad_enabled:
            vad_model = VAD.VAD(vad_thread_num)

        # initialize plugins
        import Plugins
        print("initializing plugins...")
        for plugin_inst in Plugins.plugins:
            try:
                plugin_inst.init()
                if plugin_inst.is_enabled(False):
                    print(plugin_inst.__class__.__name__ + " is enabled")
                else:
                    print(plugin_inst.__class__.__name__ + " is disabled")
            except Exception as e:
                print(f"Error initializing plugin {plugin_inst.__class__.__name__}: {e}")
                traceback.print_exc()

        # prepare the plugin timer calls
        call_plugin_timer(Plugins)

        # start OSC Server
        #if settings.GetOption("osc_sync_mute") or settings.GetOption("osc_sync_afk"):
        if settings.GetOption("osc_server_ip") != "" and settings.GetOption("osc_server_ip") != "0":
            try:
                VRC_OSCServer.start_osc_server()
            except:
                print("Error starting OSC Server. Skipping...")

        if vad_enabled and vad_model is not None:
            # num_samples = 1536
            vad_frames_per_buffer = int(settings.SETTINGS.SetOption("vad_frames_per_buffer",
                                                 settings.SETTINGS.get_argument_setting_fallback(ctx, "vad_frames_per_buffer",
                                                                                     "vad_frames_per_buffer")))

            if vad_frames_per_buffer != 512 and vad_frames_per_buffer != 256:
                print("Warning: vad_frames_per_buffer should be 512 or 256. Using 512.")
                vad_frames_per_buffer = 512
                settings.SETTINGS.SetOption("vad_frames_per_buffer", vad_frames_per_buffer)

            vad_model.set_vad_frames_per_buffer(vad_frames_per_buffer)

            # set default devices if not set
            if device_index is None or device_index < 0:
                device_index = device_default_in_index

            default_sample_rate = SAMPLE_RATE

            start_rec_on_volume_threshold = False

            push_to_talk_key = settings.SETTINGS.GetOption("push_to_talk_key")
            if push_to_talk_key == "":
                push_to_talk_key = None
            keyboard_rec_force_stop = False

            # initialize later plugins
            # for plugin_inst in Plugins.plugins:
            #     if hasattr(plugin_inst, 'late_init'):
            #         try:
            #             plugin_inst.late_init()
            #         except Exception as e:
            #             print(f"Error late initializing plugin {plugin_inst.__class__.__name__}: {e}")

            processor = audio_processing_recording.AudioProcessor(
                default_sample_rate=default_sample_rate,
                start_rec_on_volume_threshold=start_rec_on_volume_threshold,
                push_to_talk_key=push_to_talk_key,
                keyboard_rec_force_stop=keyboard_rec_force_stop,
                vad_model=vad_model,
                plugins=Plugins.plugins,
                audio_enhancer=audio_enhancer,
                osc_ip=osc_ip,
                osc_port=osc_port,
                chunk=vad_frames_per_buffer,
                channels=CHANNELS,
                sample_format=FORMAT,
                audio_queue=audioprocessor.q,
                settings=settings,
                typing_indicator_function=typing_indicator_function,
                before_callback_called_func=audio_processing_recording.main_app_before_callback_called,
                before_recording_send_to_queue_callback_func=audio_processing_recording.main_app_before_recording_send_to_queue_callback,
                before_recording_starts_callback_func=audio_processing_recording.main_app_before_recording_starts_callback,
                before_recording_running_callback_func=audio_processing_recording.main_app_before_recording_running_callback,
                verbose=verbose,
            )

            # initialize audio stream
            stream, needs_sample_rate_conversion, recorded_sample_rate, is_mono = audio_tools.start_recording_audio_stream(
                device_index,
                sample_format=FORMAT,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                chunk=vad_frames_per_buffer,
                py_audio=audio_tools.main_app_py_audio,
                audio_processor=processor,
            )

            # Start the stream
            stream.start_stream()

            #orig_recorded_sample_rate = recorded_sample_rate

            audioprocessor.start_whisper_thread()

            #continue_recording = True

            while stream.is_active():
                time.sleep(0.1)
                #if not settings.SETTINGS.GetOption("stt_enabled"):
                #    time.sleep(0.1)
                #    continue

        else:
            # load the speech recognizer and set the initial energy threshold and pause threshold
            r = sr.Recognizer()
            r.energy_threshold = energy
            r.pause_threshold = pause
            r.dynamic_energy_threshold = dynamic_energy

            # # initialize later plugins
            # for plugin_inst in Plugins.plugins:
            #     if hasattr(plugin_inst, 'late_init'):
            #         try:
            #             plugin_inst.late_init()
            #         except Exception as e:
            #             print(f"Error late initializing plugin {plugin_inst.__class__.__name__}: {e}")

            with sr.Microphone(sample_rate=whisper_audio.SAMPLE_RATE,
                               device_index=device_index) as source:

                audioprocessor.start_whisper_thread()

                while True:
                    if not settings.SETTINGS.GetOption("stt_enabled"):
                        time.sleep(0.1)
                        continue

                    phrase_time_limit = settings.SETTINGS.GetOption("phrase_time_limit")
                    if phrase_time_limit == 0:
                        phrase_time_limit = None
                    pause = settings.SETTINGS.GetOption("pause")
                    energy = settings.SETTINGS.GetOption("energy")

                    r.energy_threshold = energy
                    r.pause_threshold = pause

                    # get and save audio to wav file
                    audio = r.listen(source, phrase_time_limit=phrase_time_limit)

                    audio_data = audio.get_wav_data()

                    silence_cutting_enabled = settings.SETTINGS.GetOption("silence_cutting_enabled")
                    silence_offset = settings.SETTINGS.GetOption("silence_offset")
                    max_silence_length = settings.SETTINGS.GetOption("max_silence_length")
                    keep_silence_length = settings.SETTINGS.GetOption("keep_silence_length")

                    normalize_enabled = settings.SETTINGS.GetOption("normalize_enabled")
                    normalize_lower_threshold = settings.SETTINGS.GetOption("normalize_lower_threshold")
                    normalize_upper_threshold = settings.SETTINGS.GetOption("normalize_upper_threshold")
                    normalize_gain_factor = settings.SETTINGS.GetOption("normalize_gain_factor")
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
                    if settings.SETTINGS.GetOption("denoise_audio") == "deepfilter" and audio_enhancer is not None:
                        audio_data = audio_enhancer.enhance_audio(audio_data).tobytes()

                    # add audio data to the queue
                    wav_audio_bytes = audio_tools.audio_bytes_to_wav(audio_data, channels=CHANNELS, sample_rate=SAMPLE_RATE)
                    audioprocessor.q.put({'time': time.time_ns(), 'data': wav_audio_bytes, 'final': True, 'settings': settings.SETTINGS, 'plugins': Plugins.plugins})

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

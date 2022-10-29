import json
import speech_recognition_patch as sr
import audioprocessor
import os
import click
import VRC_OSCLib
import websocket
import settings
import remote_opener
import texttranslate
import pyaudiowpatch as pyaudio
from whisper import available_models, audio as whisper_audio


@click.command()
@click.option('--devices', default='False', help='print all available devices id', type=str)
@click.option('--device_index', default=-1, help='the id of the device (-1 = default active Mic)', type=int)
@click.option('--sample_rate', default=whisper_audio.SAMPLE_RATE, help='sample rate of recording', type=int)
@click.option("--task", default="transcribe", help="task for the model whether to only transcribe the audio or translate the audio to english",
              type=click.Choice(["transcribe", "translate"]))
@click.option("--model", default="small", help="Model to use", type=click.Choice(available_models()))
@click.option("--language", default=None, help="language spoken in the audio, specify None to perform language detection", type=click.Choice(audioprocessor.whisper_get_languages_list_keys()))
@click.option("--condition_on_previous_text", default=False,
              help="Feed it the previous result to keep it consistent across recognition windows, but makes it more prone to getting stuck in a failure loop", is_flag=True,
              type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--phrase_time_limit", default=None, help="phrase time limit before entry ends to break up long recognitions.", type=float)
@click.option("--osc_ip", default="0", help="IP to send OSC message to. Set to '0' to disable", type=str)
@click.option("--osc_port", default=9000, help="Port to send OSC message to. ('9000' as default for VRChat)", type=int)
@click.option("--osc_address", default="/chatbox/input", help="The Address the OSC messages are send to. ('/chatbox/input' as default for VRChat)", type=str)
@click.option("--osc_convert_ascii", default='True', help="Convert Text to ASCII compatible when sending over OSC.", type=str)
@click.option("--websocket_ip", default="0", help="IP where Websocket Server listens on. Set to '0' to disable", type=str)
@click.option("--websocket_port", default=5000, help="Port where Websocket Server listens on. ('5000' as default)", type=int)
@click.option("--ai_device", default=None, help="The Device the AI is loaded on. can be 'cuda' or 'cpu'. default does autodetect", type=click.Choice(["cuda", "cpu"]))
@click.option("--txt_translator", default="M2M100", help="The Model the AI is loading for text translations. can be 'M2M100', 'ARGOS' or 'None'. default is M2M100", type=click.Choice(["M2M100", "ARGOS"]))
@click.option("--m2m100_size", default="small", help="The Model size if M2M100 text translator is used. can be 'small' or 'large'. default is small. (has no effect with ARGOS)", type=click.Choice(["small", "large"]))
@click.option("--m2m100_device", default="auto", help="The device used for M2M100 translation. (has no effect with ARGOS)", type=click.Choice(["auto", "cuda", "cpu"]))
@click.option("--open_browser", default=False, help="Open default Browser with websocket-remote on start. (requires --websocket_ip to be set as well)", is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
def main(devices, device_index, sample_rate, task, model, language, condition_on_previous_text, energy, pause, dynamic_energy, phrase_time_limit, osc_ip, osc_port,
         osc_address, osc_convert_ascii, websocket_ip, websocket_port, ai_device, txt_translator, m2m100_size, m2m100_device, open_browser, verbose):

    if str2bool(devices):
        audio = pyaudio.PyAudio()
        print("-------------------------------------------------------------------")
        print("                           Input Devices                           ")
        print(" In form of: DEVICE_NAME [Sample Rate=?] [Loopback?] (Index=INDEX) ")
        print("-------------------------------------------------------------------")
        for device in audio.get_device_info_generator():
            device_index = device["index"]
            device_name = device["name"]
            device_sample_rate = int(device["defaultSampleRate"])
            device_max_channels = audio.get_device_info_by_index(device_index)['maxInputChannels']
            if device_max_channels >= 1:
                print(f"{device_name} [Sample Rate={device_sample_rate}] (Index={device_index})")
        return

    print("###################################")
    print("# Whispering Tiger is starting... #")
    print("###################################")

    # set initial settings
    settings.SetOption("whisper_task", task)
    settings.SetOption("condition_on_previous_text", condition_on_previous_text)
    settings.SetOption("model", model)

    settings.SetOption("current_language", language)

    # check if english only model is loaded, and configure whisper languages accordingly.
    if model.endswith(".en") and language not in {"en", "English"}:
        if language is not None:
            print(f"{model} is an English-only model but receipted '{language}'; using English instead.")

        print(f"{model} is an English-only model. only English speech is supported.")
        settings.SetOption("current_language", "en")
        settings.SetOption("whisper_languages", [{"code": "en", "name": "English"}])
    else:
        settings.SetOption("whisper_languages", audioprocessor.whisper_get_languages())

    settings.SetOption("ai_device", ai_device)
    settings.SetOption("verbose", verbose)

    settings.SetOption("osc_ip", osc_ip)
    settings.SetOption("osc_port", osc_port)
    settings.SetOption("osc_address", osc_address)
    settings.SetOption("osc_convert_ascii", str2bool(osc_convert_ascii))

    settings.SetOption("websocket_ip", websocket_ip)
    settings.SetOption("websocket_port", websocket_port)

    settings.SetOption("txt_translator", txt_translator)
    settings.SetOption("m2m100_size", m2m100_size)

    texttranslate.SetDevice(m2m100_device)


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
        try:
            texttranslate.InstallLanguages()
        except Exception as e:
            print(e)
            pass

    # load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=sample_rate, device_index=(device_index if device_index > -1 else None)) as source:

        audioprocessor.start_whisper_thread()

        while True:
            # get and save audio to wav file
            audio = r.listen(source, phrase_time_limit=phrase_time_limit)

            audio_data = audio.get_wav_data()

            # add audio data to the queue
            audioprocessor.q.put(audio_data)

            # set typing indicator for VRChat
            if osc_ip != "0":
                VRC_OSCLib.Bool(True, "/chatbox/typing", IP=osc_ip, PORT=osc_port)
            # send start info for processing indicator in websocket client
            websocket.BroadcastMessage(json.dumps({"type": "processing_start", "data": True}))


def str2bool(string):
    str2val = {"true": True, "false": False}
    if string.lower() in str2val:
        return str2val[string.lower()]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


main()

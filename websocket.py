import sys
import threading
import asyncio
import websockets
import json
import base64

from Models.TextTranslation import texttranslate
from Models.OCR import easyocr
from windowcapture import WindowCapture
import settings
import VRC_OSCLib
from Models.TTS import silero

# Plugins
import Plugins

WS_CLIENTS = set()

LOADING_QUEUE = {}


def tts_request(msgObj, websocket):
    silero_wav, sample_rate = silero.tts.tts(msgObj["value"]["text"])
    if silero_wav is not None:
        if msgObj["value"]["to_device"]:
            if "device_index" in msgObj["value"]:
                silero.tts.play_audio(silero_wav, msgObj["value"]["device_index"])
            else:
                silero.tts.play_audio(silero_wav, settings.GetOption("device_out_index"))
        else:
            AnswerMessage(websocket, json.dumps(
                {"type": "tts_result", "wav_data": silero_wav.tolist(), "sample_rate": sample_rate}))
            if msgObj["value"]["download"]:
                wav_data = silero.tts.return_wav_file_binary(silero_wav)
                wav_data = base64.b64encode(wav_data).decode('utf-8')
                AnswerMessage(websocket, json.dumps({"type": "tts_save", "wav_data": wav_data}))
    else:
        print("TTS failed")


def tts_plugin_process(msgObj, websocket, download=False):
    text = msgObj["value"]["text"]
    device = None
    if msgObj["value"]["to_device"]:
        if "device_index" in msgObj["value"]:
            device = msgObj["value"]["device_index"]
        else:
            device = settings.GetOption("device_out_index")

    for plugin_inst in Plugins.plugins:
        plugin_inst.tts(text, device, websocket, download)


def ocr_req(msgObj, websocket):
    window_name = settings.GetOption("ocr_window_name")
    ocr_result, image, bounding_boxes = easyocr.run_image_processing(window_name, ['en', msgObj["value"]["ocr_lang"]])
    if len(ocr_result) > 0:
        image_data = base64.b64encode(image).decode('utf-8')
        translate_result, txt_from_lang, txt_to_lang = (
            texttranslate.TranslateLanguage(" -- ".join(ocr_result), msgObj["value"]["from_lang"],
                                            msgObj["value"]["to_lang"]))
        AnswerMessage(websocket, json.dumps(
            {"type": "translate_result", "original_text": "\n".join(ocr_result),
             "translate_result": "\n".join(translate_result.split(" -- ")), "txt_from_lang": txt_from_lang,
             "txt_to_lang": txt_to_lang}))
        AnswerMessage(websocket, json.dumps(
            {"type": "ocr_result", "data": {"bounding_boxes": bounding_boxes, "image_data": image_data}}))


def websocketMessageHandler(msgObj, websocket):
    if msgObj["type"] == "setting_change":

        # handle plugin activation / deactivation before setting the option
        if msgObj["name"] == "plugins":
            for plugin_name, is_enabled in msgObj["value"].items():
                for plugin_inst in Plugins.plugins:
                    if plugin_name == type(plugin_inst).__name__:
                        if plugin_name in settings.GetOption("plugins") and is_enabled != settings.GetOption("plugins")[plugin_name]:
                            settings.SetOption(msgObj["name"], msgObj["value"])
                            if is_enabled:
                                if hasattr(plugin_inst, 'on_enable'):
                                    plugin_inst.on_enable()
                            else:
                                if hasattr(plugin_inst, 'on_disable'):
                                    plugin_inst.on_disable()

        settings.SetOption(msgObj["name"], msgObj["value"])
        BroadcastMessage(json.dumps({"type": "translate_settings", "data": settings.TRANSLATE_SETTINGS}),
                         exclude_client=websocket)  # broadcast updated settings to all clients
        # reload tts voices if tts model changed
        if msgObj["name"] == "tts_model":
            silero.tts.load()
            BroadcastMessage(json.dumps({"type": "available_tts_voices", "data": silero.tts.list_voices()}))

    if msgObj["type"] == "setting_update_req":
        AnswerMessage(websocket, json.dumps({"type": "translate_settings", "data": settings.TRANSLATE_SETTINGS}))

    if msgObj["type"] == "translate_req":
        if msgObj["value"]["to_lang"] != "":  # if to_lang is empty, don't translate
            translate_result, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(msgObj["value"]["text"],
                                                                                           msgObj["value"]["from_lang"],
                                                                                           msgObj["value"]["to_lang"])
            AnswerMessage(websocket, json.dumps(
                {"type": "translate_result", "translate_result": translate_result, "txt_from_lang": txt_from_lang,
                 "txt_to_lang": txt_to_lang}))

    if msgObj["type"] == "ocr_req":
        ocr_thread = threading.Thread(target=ocr_req, args=(msgObj, websocket))
        ocr_thread.start()

    if msgObj["type"] == "tts_req":
        if silero.init():
            tts_thread = threading.Thread(target=tts_request, args=(msgObj, websocket))
            tts_thread.start()
        else:
            download = False
            if not msgObj["value"]["to_device"]:
                download = True
            tts_thread = threading.Thread(target=tts_plugin_process, args=(msgObj, websocket, download))
            tts_thread.start()

    if msgObj["type"] == "tts_voice_save_req":
        if silero.init():
            silero.tts.save_voice()
        else:
            tts_thread = threading.Thread(target=tts_plugin_process, args=(msgObj, websocket, True))
            tts_thread.start()

    if msgObj["type"] == "get_windows_list":
        windows_list = WindowCapture.list_window_names()
        AnswerMessage(websocket, json.dumps({"type": "windows_list", "data": windows_list}))

    if msgObj["type"] == "send_osc":
        osc_address = settings.GetOption("osc_address")
        osc_ip = settings.GetOption("osc_ip")
        osc_port = settings.GetOption("osc_port")
        osc_notify = settings.GetOption("osc_typing_indicator")
        if osc_ip != "0":
            VRC_OSCLib.Chat(msgObj["value"], True, osc_notify, osc_address, IP=osc_ip, PORT=osc_port,
                            convert_ascii=settings.GetOption("osc_convert_ascii"))
            settings.SetOption("plugin_timer_stopped", True)

    if msgObj["type"] == "quit":
        print("Received quit command.")
        sys.exit(0)


async def handler(websocket):
    print('Websocket: Client connected.')

    # send all available text translation languages
    available_languages = texttranslate.GetInstalledLanguageNames()
    await send(websocket, json.dumps({"type": "installed_languages", "data": available_languages}))

    # send all available image recognition languages
    available_languages = easyocr.get_installed_language_names()
    await send(websocket, json.dumps({"type": "available_img_languages", "data": available_languages}))

    # send all available TTS models + voices
    if silero.init():
        available_tts_models = silero.tts.list_models_indexed()
        await send(websocket, json.dumps({"type": "available_tts_models", "data": available_tts_models}))
        silero.tts.load()
        await send(websocket, json.dumps({"type": "available_tts_voices", "data": silero.tts.list_voices()}))

    # send all available setting values
    await send(websocket, json.dumps({"type": "settings_values", "data": settings.GetAvailableSettingValues()}))

    # send all current settings
    await send(websocket, json.dumps({"type": "translate_settings", "data": settings.TRANSLATE_SETTINGS}))

    # send loading state
    if len(LOADING_QUEUE) > 0:
        await send(websocket, json.dumps({"type": "loading_state", "data": LOADING_QUEUE}))

    WS_CLIENTS.add(websocket)
    try:
        async for message in websocket:
            msgObj = json.loads(message)
            try:
                print(message.encode('utf-8'))
            except:
                print("???")
            websocketMessageHandler(msgObj, websocket)

        await websocket.wait_closed()
    except websockets.ConnectionClosedError as error:
        print('Websocket: Client connection failed.', error)
    finally:
        WS_CLIENTS.remove(websocket)
        print('Websocket: Client disconnected.')


async def send(websocket, message):
    try:
        await websocket.send(message)
    except websockets.ConnectionClosed:
        pass


async def broadcast(message, exclude_client=None):
    for websocket in WS_CLIENTS:
        if websocket != exclude_client:
            asyncio.create_task(send(websocket, message))


def AnswerMessage(websocket, message):
    # detect if a loop is running and run on existing loop or asyncio.run
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = None

    if loop and loop.is_running():
        loop.create_task(send(websocket, message))
    else:
        asyncio.run(send(websocket, message))


def BroadcastMessage(message, exclude_client=None):
    # detect if a loop is running and run on existing loop or asyncio.run
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = None

    if loop and loop.is_running():
        loop.create_task(broadcast(message, exclude_client))
    else:
        asyncio.run(broadcast(message, exclude_client))


async def server_program(ip, port):
    async with websockets.serve(handler, ip, port):
        print('Websocket: Server started.')
        await asyncio.Future()  # run forever


def StartWebsocketServer(ip, port):
    SocketServerThread(ip, port)


class SocketServerThread(object):
    """ Threading example class
    The run() method will be started, and it will run in the background
    until the application exits.
    """

    def __init__(self, ip, port):
        thread = threading.Thread(target=self.run, args=(ip, port,))
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution

    @staticmethod
    def run(ip, port):
        while True:
            asyncio.run(server_program(ip, port))


def set_loading_state(key, value):
    LOADING_QUEUE[key] = value
    BroadcastMessage(json.dumps({"type": "loading_state", "data": LOADING_QUEUE}))


def get_loading_state(key):
    if key in LOADING_QUEUE:
        return LOADING_QUEUE[key]
    else:
        return None

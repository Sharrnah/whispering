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
from Models.LLM import flanLanguageModel
from Models.TTS import silero


WS_CLIENTS = set()


def websocketMessageHandler(msgObj):

    if msgObj["type"] == "setting_change":
        settings.SetOption(msgObj["name"], msgObj["value"])
        BroadcastMessage(json.dumps({"type": "translate_settings", "data": settings.TRANSLATE_SETTINGS}))  # broadcast updated settings to all clients
        # reload tts voices if tts model changed
        if msgObj["name"] == "tts_model":
            silero.tts.load()
            BroadcastMessage(json.dumps({"type": "available_tts_voices", "data": silero.tts.list_voices()}))

    if msgObj["type"] == "translate_req":
        translate_result, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(msgObj["text"], msgObj["from_lang"], msgObj["to_lang"])
        BroadcastMessage(json.dumps({"type": "translate_result", "translate_result": translate_result, "txt_from_lang": txt_from_lang, "txt_to_lang": txt_to_lang}))

    if msgObj["type"] == "ocr_req":
        window_name = settings.GetOption("ocr_window_name")
        ocr_result = easyocr.run_image_processing(window_name, ['en', msgObj["ocr_lang"]])
        translate_result, txt_from_lang, txt_to_lang = (texttranslate.TranslateLanguage(" -- ".join(ocr_result), msgObj["from_lang"], msgObj["to_lang"]))
        BroadcastMessage(json.dumps({"type": "translate_result", "original_text": "\n".join(ocr_result), "translate_result": "\n".join(translate_result.split(" -- ")), "txt_from_lang": txt_from_lang, "txt_to_lang": txt_to_lang}))

    if msgObj["type"] == "tts_req":
        if silero.init():
            silero_wav, sample_rate = silero.tts.tts(msgObj["text"])
            if silero_wav is not None:
                if msgObj["to_device"]:
                    silero.tts.play_audio(silero_wav, settings.GetOption("device_out_index"))
                else:
                    BroadcastMessage(json.dumps({"type": "tts_result", "wav_data": silero_wav.tolist(), "sample_rate": sample_rate}))
                    if msgObj["download"]:
                        wav_data = silero.tts.return_wav_file_binary(silero_wav)
                        wav_data = base64.b64encode(wav_data).decode('utf-8')
                        BroadcastMessage(json.dumps({"type": "tts_save", "wav_data": wav_data}))
            else:
                print("TTS failed")

    if msgObj["type"] == "tts_voice_save_req":
        if silero.init():
            silero.tts.save_voice()

    if msgObj["type"] == "flan_req":
        if flanLanguageModel.init():
            flan_result = flanLanguageModel.flan.encode(msgObj["text"])
            BroadcastMessage(json.dumps({"type": "flan_result", "flan_result": flan_result}))

    if msgObj["type"] == "get_windows_list":
        windows_list = WindowCapture.list_window_names()
        BroadcastMessage(json.dumps({"type": "windows_list", "data": windows_list}))

    if msgObj["type"] == "send_osc":
        osc_address = settings.GetOption("osc_address")
        osc_ip = settings.GetOption("osc_ip")
        osc_port = settings.GetOption("osc_port")
        if osc_ip != "0":
            VRC_OSCLib.Chat(msgObj["text"], True, osc_address, IP=osc_ip, PORT=osc_port)


async def handler(websocket):
    print('Websocket: Client connected.')

    # send all available text translation languages
    available_languages = texttranslate.GetInstalledLanguageNames()
    await send(websocket, json.dumps({"type": "installed_languages", "data": available_languages}))

    # send all available image recognition languages
    available_languages = easyocr.get_installed_language_names()
    await send(websocket, json.dumps({"type": "available_img_languages", "data": available_languages}))

    # send all available TTS models
    if silero.init():
        available_tts_models = silero.tts.list_models_indexed()
        await send(websocket, json.dumps({"type": "available_tts_models", "data": available_tts_models}))

    # send all available TTS models
    if silero.init():
        available_tts_models = silero.tts.list_models_indexed()
        await send(websocket, json.dumps({"type": "available_tts_models", "data": available_tts_models}))
        silero.tts.load()
        await send(websocket, json.dumps({"type": "available_tts_voices", "data": silero.tts.list_voices()}))

    # send all current text translation settings
    await send(websocket, json.dumps({"type": "translate_settings", "data": settings.TRANSLATE_SETTINGS}))

    WS_CLIENTS.add(websocket)
    try:
        async for message in websocket:
            msgObj = json.loads(message)
            print(message.encode('utf-8'))
            websocketMessageHandler(msgObj)

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


async def broadcast(message):
    for websocket in WS_CLIENTS:
        asyncio.create_task(send(websocket, message))


def BroadcastMessage(message):
    # detect if a loop is running and run on existing loop or asyncio.run
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = None

    if loop and loop.is_running():
        loop.create_task(broadcast(message))
    else:
        asyncio.run(broadcast(message))


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

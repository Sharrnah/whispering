import sys
import threading
import asyncio
import time
import traceback
from datetime import datetime
from pathlib import Path

import websockets
import json
import base64

import Utilities
import audio_tools
import processmanager

from Models.TextTranslation import texttranslate
from Models import OCR
from windowcapture import WindowCapture
import settings
import VRC_OSCLib
from Models.TTS import tts

# Plugins
import Plugins

UI_CONNECTED = {"value": False, "websocket": None}

LOADING_QUEUE = {}


def tts_request(msgObj, websocket):
    text = msgObj["value"]["text"]
    path = ""
    if "path" in msgObj["value"] and msgObj["value"]["path"] != "":
        path = msgObj["value"]["path"]

    streamed_playback = settings.GetOption("tts_streamed_playback")

    tts_wav = None
    if streamed_playback and hasattr(tts.tts, "tts_streaming"):
        tts_wav, sample_rate = tts.tts.tts_streaming(text)
        if tts_wav is not None:
            return

    if tts_wav is None:
        tts_wav, sample_rate = tts.tts.tts(text)

        if tts_wav is not None:
            if msgObj["value"]["to_device"]:
                if "device_index" in msgObj["value"]:
                    tts.tts.play_audio(tts_wav, msgObj["value"]["device_index"])
                else:
                    tts.tts.play_audio(tts_wav, settings.GetOption("device_out_index"))
            else:
                # send raw wav data to non UI clients (like html websocket pages, for backwards compatibility)
                if websocket is not None and websocket != UI_CONNECTED["websocket"]:
                    AnswerMessage(websocket, json.dumps(
                        {"type": "tts_result", "wav_data": tts_wav.tolist(), "sample_rate": sample_rate}))
                if msgObj["value"]["download"]:
                    wav_data = tts.tts.return_wav_file_binary(tts_wav)
                    if path is not None and path != '':
                        # write wav_data to file in path
                        try:
                            with open(path, "wb") as f:
                                f.write(wav_data)
                            print("100% Finished. TTS file saved to:", path)
                        except Exception as e:
                            print("Failed to save TTS file:", e)
                            traceback.print_exc()
                    else:
                        wav_data = base64.b64encode(wav_data).decode('utf-8')
                        AnswerMessage(websocket, json.dumps({"type": "tts_save", "wav_data": wav_data}))
            return

        print("TTS failed")


def tts_request_last(msgObj, websocket):
    path = ""
    if "path" in msgObj["value"] and msgObj["value"]["path"] != "":
        path = msgObj["value"]["path"]

    if hasattr(tts.tts, 'get_last_generation'):
        tts_wav, sample_rate = tts.tts.get_last_generation()
    else:
        print("TTS does not support get_last_generation")
        return

    if tts_wav is not None:
        if msgObj["value"]["to_device"]:
            if "device_index" in msgObj["value"]:
                tts.tts.play_audio(tts_wav, msgObj["value"]["device_index"])
            else:
                tts.tts.play_audio(tts_wav, settings.GetOption("device_out_index"))
        else:
            # send raw wav data to non UI clients (like html websocket pages, for backwards compatibility)
            if websocket is not None and websocket != UI_CONNECTED["websocket"]:
                AnswerMessage(websocket, json.dumps(
                    {"type": "tts_result", "wav_data": tts_wav.tolist(), "sample_rate": sample_rate}))
            if msgObj["value"]["download"]:
                wav_data = tts.tts.return_wav_file_binary(tts_wav)
                if path is not None and path != '':
                    # write wav_data to file in path
                    try:
                        with open(path, "wb") as f:
                            f.write(wav_data)
                        print("100% Finished. TTS file saved to:", path)
                    except Exception as e:
                        print("Failed to save TTS file:", e)
                        traceback.print_exc()
                else:
                    wav_data = base64.b64encode(wav_data).decode('utf-8')
                    AnswerMessage(websocket, json.dumps({"type": "tts_save", "wav_data": wav_data}))
    else:
        print("No TTS generation found")

def tts_request_last_plugin(msgObj, websocket):
    path = ""
    if "path" in msgObj["value"] and msgObj["value"]["path"] != "":
        path = msgObj["value"]["path"]

    tts_wav, sample_rate = None, None

    for plugin_inst in Plugins.plugins:
        if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'tts_get_last_generation'):
            try:
                tts_wav, sample_rate = plugin_inst.tts_get_last_generation()
            except Exception as e:
                print(f"Plugin TTS failed in Plugin {plugin_inst.__class__.__name__}:", e)
                traceback.print_exc()

    if tts_wav is not None:
        # send raw wav data to non UI clients (like html websocket pages, for backwards compatibility)
        if websocket is not None and websocket != UI_CONNECTED["websocket"]:
            AnswerMessage(websocket, json.dumps(
                {"type": "tts_result", "wav_data": tts_wav.tolist(), "sample_rate": sample_rate}))
        if path is not None and path != '':
            # write wav_data to file in path
            try:
                with open(path, "wb") as f:
                    f.write(tts_wav)
                print("100% Finished. TTS file saved to:", path)
            except Exception as e:
                print("Failed to save TTS file:", e)
                traceback.print_exc()
        else:
            wav_data = base64.b64encode(tts_wav).decode('utf-8')
            AnswerMessage(websocket, json.dumps({"type": "tts_save", "wav_data": wav_data}))
    else:
        print("No TTS generation found")


def translate_request(msgObj, websocket):
    def send_osc_request(msg_obj, websocket):
        # delay sending message if it is the final audio and until TTS starts playing
        if settings.GetOption("osc_delay_until_audio_playback"):
            # wait until is_audio_playing is True or timeout is reached
            delay_timeout = time.time() + settings.GetOption("osc_delay_timeout")
            tag = settings.GetOption("osc_delay_until_audio_playback_tag")
            tts_answer = settings.GetOption("tts_answer")
            if tag == "tts" and tts_answer:
                while not audio_tools.is_audio_playing(tag=tag) and time.time() < delay_timeout:
                    time.sleep(0.05)
        osc_request(msg_obj, websocket)

    ignore_send_options = True
    if "ignore_send_options" in msgObj["value"]:
        ignore_send_options = msgObj["value"]["ignore_send_options"]
    orig_text = msgObj["value"]["text"]
    text = orig_text
    if msgObj["value"]["to_lang"] != "":  # if to_lang is empty, don't translate
        to_romaji = False
        if "to_romaji" in msgObj["value"]:
            to_romaji = msgObj["value"]["to_romaji"]
        # main translation
        text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(orig_text,
                                                                           msgObj["value"]["from_lang"],
                                                                           msgObj["value"]["to_lang"],
                                                                           to_romaji)

        # do secondary translations if enabled
        second_translation_enabled = settings.GetOption("txt_second_translation_enabled")
        second_translation_languages = settings.GetOption("txt_second_translation_languages")
        second_translation_wrap = settings.GetOption("txt_second_translation_wrap")
        second_translation_wrap = second_translation_wrap.replace("\\n", "\n")
        if second_translation_enabled and second_translation_languages!= "":
            second_translation_split_codes = [st.strip() for st in second_translation_languages.split(",")]
            for split_code in second_translation_split_codes:
                if split_code != "":
                    second_translation_text, second_txt_from_lang, second_txt_to_lang = texttranslate.TranslateLanguage(
                        orig_text, msgObj["value"]["from_lang"], split_code, False)
                    text += second_translation_wrap + second_translation_text
                    txt_to_lang += "|"+second_txt_to_lang

        AnswerMessage(websocket, json.dumps(
            {"type": "translate_result", "translate_result": text, "txt_from_lang": txt_from_lang,
             "txt_to_lang": txt_to_lang}))

    if not ignore_send_options:
        if settings.GetOption("osc_auto_processing_enabled"):
            msgObj["value"]["text"] = text
            if settings.GetOption("osc_type_transfer") == "source":
                msgObj["value"]["text"] = orig_text
            elif settings.GetOption("osc_type_transfer") == "both":
                msgObj["value"]["text"] = orig_text + settings.GetOption("osc_type_transfer_split") + text
            elif settings.GetOption("osc_type_transfer") == "both_inverted":
                msgObj["value"]["text"] = text + settings.GetOption("osc_type_transfer_split") + orig_text
            threading.Thread(target=send_osc_request, args=(msgObj, websocket,)).start()

        if settings.GetOption("tts_answer"):
            msgObj["value"]["text"] = text
            msgObj["value"]["to_device"] = True
            msgObj["value"]["download"] = False
            if tts.init():
                tts_request(msgObj, websocket)
            else:
                tts_plugin_process(msgObj, websocket)


def osc_request(msgObj, websocket):
    osc_address = settings.GetOption("osc_address")
    osc_ip = settings.GetOption("osc_ip")
    osc_port = settings.GetOption("osc_port")
    osc_notify = settings.GetOption("osc_typing_indicator")

    osc_send_type = settings.GetOption("osc_send_type")
    osc_chat_limit = settings.GetOption("osc_chat_limit")
    osc_time_limit = settings.GetOption("osc_time_limit")
    osc_scroll_time_limit = settings.GetOption("osc_scroll_time_limit")
    osc_initial_time_limit = settings.GetOption("osc_initial_time_limit")
    osc_scroll_size = settings.GetOption("osc_scroll_size")
    osc_max_scroll_size = settings.GetOption("osc_max_scroll_size")

    #VRC_OSCLib.set_min_time_between_messages(settings.GetOption("osc_min_time_between_messages"))

    if osc_ip != "0":
        if osc_send_type == "full":
            VRC_OSCLib.Chat(msgObj["value"]["text"], True, osc_notify, osc_address, IP=osc_ip, PORT=osc_port,
                            convert_ascii=settings.GetOption("osc_convert_ascii"))
        elif osc_send_type == "chunks":
            VRC_OSCLib.Chat_chunks(msgObj["value"]["text"],
                                   nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                   chunk_size=osc_chat_limit, delay=osc_time_limit,
                                   initial_delay=osc_initial_time_limit,
                                   convert_ascii=settings.GetOption("osc_convert_ascii"))
        elif osc_send_type == "scroll":
            VRC_OSCLib.Chat_scrolling_chunks(msgObj["value"]["text"],
                                             nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                             chunk_size=osc_max_scroll_size, delay=osc_scroll_time_limit,
                                             initial_delay=osc_initial_time_limit,
                                             scroll_size=osc_scroll_size,
                                             convert_ascii=settings.GetOption("osc_convert_ascii"))
        elif osc_send_type == "full_or_scroll":
            # send full if message fits in osc_chat_limit, otherwise send scrolling chunks
            if len(msgObj["value"]["text"].encode('utf-16le')) <= osc_chat_limit * 2:
                VRC_OSCLib.Chat(msgObj["value"]["text"], True, osc_notify, osc_address,
                                IP=osc_ip, PORT=osc_port,
                                convert_ascii=settings.GetOption("osc_convert_ascii"))
            else:
                VRC_OSCLib.Chat_scrolling_chunks(msgObj["value"]["text"],
                                                 nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                                 chunk_size=osc_chat_limit, delay=osc_scroll_time_limit,
                                                 initial_delay=osc_initial_time_limit,
                                                 scroll_size=osc_scroll_size,
                                                 convert_ascii=settings.GetOption("osc_convert_ascii"))

        settings.SetOption("plugin_timer_stopped", True)


def tts_plugin_process(msgObj, websocket, download=False):
    text = msgObj["value"]["text"]
    device = None
    if msgObj["value"]["to_device"]:
        if "device_index" in msgObj["value"]:
            device = msgObj["value"]["device_index"]
        else:
            device = settings.GetOption("device_out_index")

    path = ""
    if "path" in msgObj["value"] and msgObj["value"]["path"] != "":
        path = msgObj["value"]["path"]

    for plugin_inst in Plugins.plugins:
        if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'tts'):
            try:
                plugin_inst.tts(text, device, websocket, download, path)
            except Exception as e:
                print(f"Plugin TTS failed in Plugin {plugin_inst.__class__.__name__}:", e)
                traceback.print_exc()


def ocr_req(msgObj, websocket):
    window_name = settings.GetOption("ocr_window_name")
    to_romaji = False
    if "to_romaji" in msgObj["value"]:
        to_romaji = msgObj["value"]["to_romaji"]
    if "image" in msgObj["value"]:
        image = base64.b64decode(msgObj["value"]["image"])
        OCR.init_ocr_model()
        ocr_result, _, bounding_boxes = OCR.run_image_processing_from_image(image,
                                                                                ['en', msgObj["value"]["ocr_lang"]])
    else:
        ocr_result, image, bounding_boxes = OCR.run_image_processing(window_name,
                                                                         ['en', msgObj["value"]["ocr_lang"]])
    if len(ocr_result) > 0:
        image_data = base64.b64encode(image).decode('utf-8')
        translate_result, txt_from_lang, txt_to_lang = (
            texttranslate.TranslateLanguage(" -- ".join(ocr_result), msgObj["value"]["from_lang"],
                                            msgObj["value"]["to_lang"], to_romaji))
        AnswerMessage(websocket, json.dumps(
            {"type": "translate_result", "original_text": "\n".join(ocr_result),
             "translate_result": "\n".join(translate_result.split(" -- ")), "txt_from_lang": txt_from_lang,
             "txt_to_lang": txt_to_lang}))
        AnswerMessage(websocket, json.dumps(
            {"type": "ocr_result", "data": {"bounding_boxes": bounding_boxes, "image_data": image_data}}))

def chat_request(msgObj, websocket):
    system_prompt = ''
    if "system_prompt" in msgObj["value"]:
        system_prompt = msgObj["value"]["system_prompt"]
    chat_message = ''
    if "text" in msgObj["value"]:
        chat_message = msgObj["value"]["text"]
    stt_type = settings.GetOption("stt_type")
    compute_dtype = settings.GetOption("whisper_precision")
    ai_device = settings.GetOption("ai_device")
    if chat_message == '':
        return

    if stt_type == "phi4":
        import Models.Multi.phi4 as phi4
        llm_model = phi4.Phi4(compute_type=compute_dtype, device=ai_device)
        response = llm_model.transcribe(None, task='chat', language='', chat_message=chat_message, system_prompt=system_prompt)
        response['text'] = chat_message
        AnswerMessage(websocket, json.dumps(response))
        del llm_model

def plugin_event_handler(msgObj, websocket):
    pluginClassName = msgObj["name"]
    for plugin_inst in Plugins.plugins:
        if pluginClassName == type(plugin_inst).__name__:
            if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'on_event_received'):
                try:
                    plugin_inst.on_event_received(msgObj, websocket)
                except Exception as e:
                    print(f"Plugin event failed in Plugin {plugin_inst.__class__.__name__}:", e)
                    traceback.print_exc()
                return


# ==============================================================
# Websocket Class
# ==============================================================

class WebSocketServer:
    def __init__(self, ip, port, websocket_message_handler=None, on_connect_handler=None, on_disconnect_handler=None,
                 debug=False):
        self.ip = ip
        self.port = port
        self.websocket_message_handler = websocket_message_handler
        self.on_connect_handler = on_connect_handler
        self.on_disconnect_handler = on_disconnect_handler
        self.debug = debug
        self.ws_clients = set()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run, args=(self.loop,))
        self.thread.start()

    async def handler(self, websocket, path):
        print('Websocket: Client connected.')
        if self.on_connect_handler is not None:
            await self.on_connect_handler(self, websocket)
        self.ws_clients.add(websocket)
        try:
            async for message in websocket:
                msg_obj = json.loads(message)
                if self.debug:
                    try:
                        if "type" in msg_obj:
                            print("Received WS message:", msg_obj["type"])
                    except:
                        print("Received WS message: ???")
                if self.websocket_message_handler is not None:
                    await self.websocket_message_handler(self, msg_obj, websocket)
            await websocket.wait_closed()
        except websockets.ConnectionClosedError as error:
            print('Websocket: Client connection failed.', error)
        finally:
            if self.on_disconnect_handler is not None:
                await self.on_disconnect_handler(self, websocket)
            self.ws_clients.remove(websocket)
            print('Websocket: Client disconnected.')

    async def send(self, websocket, message):
        if self.debug:
            try:
                if "type" in message:
                    print("Send WS message:", message["type"])
            except:
                print("Send WS message: ???")

        try:
            await websocket.send(message)
        except websockets.ConnectionClosed:
            pass

    async def broadcast(self, message, exclude_client=None):
        if self.debug:
            try:
                if "type" in message:
                    print("broadcast WS message:", message["type"])
            except:
                print("broadcast WS message: ???")
        for websocket in self.ws_clients:
            if websocket != exclude_client:
                await self.send(websocket, message)

    def answer_message(self, websocket, message):
        asyncio.run_coroutine_threadsafe(self.send(websocket, message), self.loop)

    def broadcast_message(self, message, exclude_client=None):
        asyncio.run_coroutine_threadsafe(self.broadcast(message, exclude_client), self.loop)

    def get_connected_clients(self):
        return self.ws_clients

    async def server_program(self):
        server = await websockets.serve(self.handler, self.ip, self.port, max_size=1_000_000_000, timeout=120,
                                        close_timeout=120)
        print('Websocket: Server started.')
        await server.wait_closed()

    def run(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.server_program())
        loop.run_forever()


async def custom_message_handler(server_instance, msg_obj, websocket):
    global UI_CONNECTED
    if msg_obj["type"] == "setting_change":

        # handle plugin activation / deactivation before setting the option
        if msg_obj["name"] == "plugins":
            for plugin_name, is_enabled in list(msg_obj["value"].items()):
                for plugin_inst in Plugins.plugins:
                    if plugin_name == type(plugin_inst).__name__:
                        if plugin_name in settings.GetOption("plugins") and is_enabled != settings.GetOption("plugins")[
                            plugin_name]:
                            settings.SetOption(msg_obj["name"], msg_obj["value"])
                            if is_enabled:
                                if hasattr(plugin_inst, 'on_enable'):
                                    try:
                                        plugin_inst.on_enable()
                                    except Exception as e:
                                        print(f"Plugin enable failed for {plugin_name}:", e)
                                        traceback.print_exc()
                            else:
                                if hasattr(plugin_inst, 'on_disable'):
                                    try:
                                        plugin_inst.on_disable()
                                    except Exception as e:
                                        print(f"Plugin disable failed for {plugin_name}:", e)
                                        traceback.print_exc()

        settings.SetOption(msg_obj["name"], msg_obj["value"])
        server_instance.broadcast_message(
            json.dumps({"type": "translate_settings", "data": settings.SETTINGS.get_all_settings()}),
            exclude_client=websocket)  # broadcast updated settings to all clients
        # reload tts voices if tts model changed
        if msg_obj["name"] == "tts_model":
            print("Loading new TTS model. Please wait.")
            def tts_load():
                if hasattr(tts.tts, 'load'):
                    tts.tts.load()
                if hasattr(tts.tts, 'list_voices'):
                    server_instance.broadcast_message(
                        json.dumps({"type": "available_tts_voices", "data": tts.tts.list_voices()}))
            tts_load = threading.Thread(target=tts_load)
            tts_load.start()
        if msg_obj["name"] == "osc_min_time_between_messages":
            VRC_OSCLib.set_min_time_between_messages(msg_obj["value"])

    if msg_obj["type"] == "setting_update_req":
        server_instance.answer_message(websocket, json.dumps(
            {"type": "translate_settings", "data": settings.SETTINGS.get_all_settings()}))

    if msg_obj["type"] == "translate_req":
        translate_thread = threading.Thread(target=translate_request, args=(msg_obj, websocket))
        translate_thread.start()

    if msg_obj["type"] == "ocr_req":
        ocr_thread = threading.Thread(target=ocr_req, args=(msg_obj, websocket))
        ocr_thread.start()

    if msg_obj["type"] == "tts_req_last":
        if tts.init():
            tts_thread = threading.Thread(target=tts_request_last, args=(msg_obj, websocket))
            tts_thread.start()
        else:
            tts_thread = threading.Thread(target=tts_request_last_plugin, args=(msg_obj, websocket))
            tts_thread.start()

    if msg_obj["type"] == "tts_req":
        if tts.init():
            tts_thread = threading.Thread(target=tts_request, args=(msg_obj, websocket))
            tts_thread.start()
        else:
            download = False
            if not msg_obj["value"]["to_device"]:
                download = True
            tts_thread = threading.Thread(target=tts_plugin_process, args=(msg_obj, websocket, download))
            tts_thread.start()

    if msg_obj["type"] == "chat_req":
        translate_thread = threading.Thread(target=chat_request, args=(msg_obj, websocket))
        translate_thread.start()

    if msg_obj["type"] == "tts_setting_special":
        if "value" in msg_obj:
            if tts.init() and hasattr(tts.tts, 'set_special_setting'):
                tts.tts.set_special_setting(msg_obj["value"])

    if msg_obj["type"] == "audio_stop":
        tag = None
        if "value" in msg_obj:
            tag = msg_obj["value"]
        if tag == "tts":
            if tts.init() and hasattr(tts.tts, 'stop'):
                tts.tts.stop()
        audio_tools.stop_audio(tag=tag)

    if msg_obj["type"] == "tts_voice_save_req":
        if hasattr(tts, 'init') and tts.init():
            try:
                if hasattr(tts.tts, 'save_voice'):
                    tts.tts.save_voice()
                else:
                    print("Save voice method not found.")
            except Exception as e:
                print(f"Failed to save voice: {e}")
                traceback.print_exc()
        else:
            tts_thread = threading.Thread(target=tts_plugin_process, args=(msg_obj, websocket, True))
            tts_thread.start()
    if msg_obj["type"] == "tts_voice_reload_req":
        if hasattr(tts, 'init') and tts.init():
            try:
                if hasattr(tts.tts, 'list_voices'):
                    server_instance.broadcast_message(
                        json.dumps({"type": "available_tts_voices", "data": tts.tts.list_voices()}))
                else:
                    print("List voices method not found.")
            except Exception as e:
                print(f"Failed to save voice: {e}")
                traceback.print_exc()
        else:
            tts_thread = threading.Thread(target=tts_plugin_process, args=(msg_obj, websocket, True))
            tts_thread.start()

    if msg_obj["type"] == "get_windows_list":
        windows_list = Utilities.handle_bytes(WindowCapture.list_window_names())
        server_instance.answer_message(websocket, json.dumps({"type": "windows_list", "data": windows_list}))

    if msg_obj["type"] == "send_osc":
        osc_request(msg_obj, websocket)

    if msg_obj["type"] == "ui_connected":
        UI_CONNECTED["value"] = True
        UI_CONNECTED["websocket"] = websocket

    # plugin event handler
    if msg_obj["type"] == "plugin_button_press":
        plugin_event_thread = threading.Thread(target=plugin_event_handler, args=(msg_obj, websocket,))
        plugin_event_thread.start()
        # plugin_event_handler(msgObj)

    if msg_obj["type"] == "plugin_custom_event":
        plugin_event_thread = threading.Thread(target=plugin_event_handler, args=(msg_obj, websocket,))
        plugin_event_thread.start()

    if msg_obj["type"] == "save_transcription":
        # save file with date in filename
        save_file = str(
            Path(Path.cwd() / ("transcriptions_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv")).resolve())
        if "value" in msg_obj:
            save_file = msg_obj["value"]
        Utilities.save_transcriptions(file_path=save_file)

    if msg_obj["type"] == "clear_transcription" and msg_obj["value"]:
        Utilities.clear_transcriptions()

    if msg_obj["type"] == "quit":
        print("Received quit command.")
        processmanager.cleanup_subprocesses()
        sys.exit(0)


async def main_on_connect_handler(server_instance, websocket):
    # send all available text translation languages
    available_languages = texttranslate.GetInstalledLanguageNames()
    if available_languages is not None:
        await server_instance.send(websocket, json.dumps({"type": "installed_languages", "data": available_languages}))

    # send all available image recognition languages
    OCR.init_ocr_model()
    available_languages = OCR.get_installed_language_names()
    if available_languages is not None:
        await server_instance.send(websocket,
                                   json.dumps({"type": "available_img_languages", "data": available_languages}))

    # send all available TTS models + voices
    if tts.tts is not None and not tts.failed:
        available_tts_models = tts.tts.list_models_indexed()
        await server_instance.send(websocket,
                                   json.dumps({"type": "available_tts_models", "data": available_tts_models}))
        tts.tts.load()
        await server_instance.send(websocket,
                                   json.dumps({"type": "available_tts_voices", "data": tts.tts.list_voices()}))

    # send all available setting values
    await server_instance.send(websocket, json.dumps(
        {"type": "settings_values", "data": settings.SETTINGS.get_available_setting_values()}))

    # send all current settings
    await server_instance.send(websocket, json.dumps(
        {"type": "translate_settings", "data": Utilities.handle_bytes(settings.SETTINGS.get_all_settings())}))

    # send loading state
    if len(LOADING_QUEUE) > 0:
        await server_instance.send(websocket, json.dumps({"type": "loading_state", "data": LOADING_QUEUE}))


async def main_on_disconnect_handler(server_instance, websocket):
    if UI_CONNECTED["websocket"] == websocket:
        UI_CONNECTED["value"] = False
        UI_CONNECTED["websocket"] = None


main_server = None


def StartWebsocketServer(ip, port):
    return WebSocketServer(ip, int(port), custom_message_handler, main_on_connect_handler, main_on_disconnect_handler,
                           debug=False)


def get_connected_clients():
    if main_server is not None and isinstance(main_server, WebSocketServer):
        return main_server.get_connected_clients()
    else:
        return set()


# Legacy functions
def AnswerMessage(websocket, message):
    if main_server is not None and isinstance(main_server, WebSocketServer):
        main_server.answer_message(websocket, message)


def BroadcastMessage(message, exclude_client=None):
    if main_server is not None and isinstance(main_server, WebSocketServer):
        main_server.broadcast_message(message, exclude_client=exclude_client)


def set_loading_state(key, value):
    LOADING_QUEUE[key] = value
    #BroadcastMessage(json.dumps({"type": "loading_state", "data": LOADING_QUEUE}))
    print(json.dumps({"type": "loading_state", "data": {"name": key, "value": value}}))


def get_loading_state(key):
    if key in LOADING_QUEUE:
        return LOADING_QUEUE[key]
    else:
        return None

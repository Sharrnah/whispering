import json
import threading

from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio

import Utilities
import settings
import websocket

osc_server = None

class OSC_Server:
    osc_addresses = ["/avatar/parameters/MuteSelf", "/avatar/parameters/AFK"]

    def __init__(self, ip="127.0.0.1", port=9001):
        self.ip = ip
        self.port = port
        self.transport, self.protocol = None, None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run, args=(self.loop,))
        self.thread.start()

    def run(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_server())
        loop.run_forever()

    async def start_server(self):
        print(f"Starting OSC server on {self.ip}:{self.port}")
        dispatcher = Dispatcher()
        dispatcher.map("/avatar/parameters*", self.filter_handler)

        server = AsyncIOOSCUDPServer((self.ip, self.port), dispatcher, self.loop)
        self.transport, self.protocol = await server.create_serve_endpoint()

    def filter_handler(self, address, *args):
        if address not in self.osc_addresses:
            return
        # we expect *args to contain a bool
        if len(args) != 1 or not isinstance(args[0], bool):
            return
        stt_enabled = not args[0]

        if not settings.GetOption("osc_sync_mute") and address.endswith("/MuteSelf"):
            return
        if not settings.GetOption("osc_sync_afk") and address.endswith("/AFK"):
            return

        settings.SetOption("stt_enabled", stt_enabled)

        websocket.AnswerMessage(websocket.UI_CONNECTED["websocket"], json.dumps(
            {"type": "translate_settings", "data": Utilities.handle_bytes(settings.SETTINGS.get_all_settings())}))
        # websocket.AnswerMessage(websocket.UI_CONNECTED["websocket"], json.dumps(
        #     {
        #         "type": "set_translate_setting", "data": {"stt_enabled": stt_enabled}
        #     }
        # ))

def start_osc_server():
    global osc_server
    if osc_server is None:
        osc_server_ip = settings.GetOption("osc_server_ip")
        osc_server_port = settings.GetOption("osc_server_port")
        osc_server = OSC_Server(osc_server_ip, osc_server_port)

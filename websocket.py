import threading
import asyncio
import websockets
import json
import texttranslate
import settings

WS_CLIENTS = set()

async def handler(websocket):
    print('Websocket: Client connected.')

    # send all available text translation languages
    availableLanguages = texttranslate.GetInstalledLanguageNames()
    await send(websocket, json.dumps({"type": "installed_languages", "data": availableLanguages}))

    # send all current text translation settings
    await send(websocket, json.dumps({"type": "translate_settings", "data": settings.TRANSLATE_SETTINGS}))

    WS_CLIENTS.add(websocket)
    try:
        async for message in websocket:
            print("Setting: ", message)
            msgObj = json.loads(message)
            settings.SetOption(msgObj["name"], msgObj["value"])
            if msgObj["name"] == "dl_langs":
                texttranslate.InstallLanguages()
        await websocket.wait_closed()
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
    asyncio.run(broadcast(message))

async def server_program(ip, port):
    async with websockets.serve(handler, ip, port):
        print('Websocket: Server started.')
        await asyncio.Future()  # run forever

def StartWebsocketServer(ip, port):
    SocketServerThread(ip, port)


class SocketServerThread(object):
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, ip, port):
        thread = threading.Thread(target=self.run, args=(ip, port,))
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def run(self, ip, port):
        while True:
            asyncio.run(server_program(ip, port))

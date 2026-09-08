"""Real socket regression without importing the application's model runtimes."""
import ast
import asyncio
import json
from pathlib import Path
import types
import unittest

import websockets


class WebSocketCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_library_starts_and_exchanges_application_messages(self):
        source = Path(__file__).resolve().parents[1] / "websocket.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WebSocketServer")
        started = asyncio.get_running_loop().create_future()

        async def serve(*args, **kwargs):
            try:
                listener = await websockets.serve(*args, **kwargs)
                started.set_result(listener)
                return listener
            except Exception as error:
                started.set_exception(error)
                raise

        namespace = {"asyncio": asyncio, "json": json, "websockets": types.SimpleNamespace(
            serve=serve, ConnectionClosedError=websockets.ConnectionClosedError,
            ConnectionClosed=websockets.ConnectionClosed)}
        exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(source), "exec"), namespace)
        server = namespace["WebSocketServer"].__new__(namespace["WebSocketServer"])
        server.ip, server.port, server.debug = "127.0.0.1", 0, False
        server.ws_clients = set()
        server.on_connect_handler = None
        disconnected = asyncio.Event()

        async def on_disconnect(_server, _socket):
            disconnected.set()

        async def message_handler(current, value, socket):
            await current.send(socket, json.dumps({"type": "reply", "value": value["value"]}))

        server.on_disconnect_handler = on_disconnect
        server.websocket_message_handler = message_handler
        task = asyncio.create_task(server.server_program())
        listener = None
        try:
            listener = await asyncio.wait_for(started, 5)
            port = listener.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as socket:
                await socket.send(json.dumps({"type": "probe", "value": "Linux profile connected"}))
                reply = json.loads(await asyncio.wait_for(socket.recv(), 5))
                self.assertEqual(reply, {"type": "reply", "value": "Linux profile connected"})
                self.assertEqual(len(server.ws_clients), 1)
            await asyncio.wait_for(disconnected.wait(), 5)
            self.assertEqual(server.ws_clients, set())
        finally:
            if listener is not None:
                listener.close()
                await listener.wait_closed()
            else:
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

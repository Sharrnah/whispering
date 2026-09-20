import asyncio
import json
import struct
import threading
import unittest

import websockets

from remote_audio import Server, RemoteSettings
from remote_audio import Discovery
from remote_audio_output import current_destination, destination


class RemoteAudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.frames = []
        self.source_closed = threading.Event()
        test = self

        def factory(session, hello):
            class Source:
                def feed(self, pcm):
                    test.frames.append(pcm)
                    session.result({"text": "hello", "txt_translation": "Hallo"}, True)
                def close(self):
                    test.source_closed.set()
            return Source()

        def synthesize(session, text):
            session.audio(b"\x01\x00" * 100, 24000)

        self.server = await Server(factory, synthesize, "test-key").start("127.0.0.1", 0)
        port = self.server.listener.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def asyncTearDown(self):
        await self.server.close()

    async def connect(self, token="test-key", **hello):
        client = await websockets.connect(self.url)
        await client.send(json.dumps({"version": 1, "token": token, **hello}))
        class ClosingSocket:
            def __getattr__(self, name):
                return getattr(client, name)
            async def __aenter__(self):
                return client
            async def __aexit__(self, *args):
                await client.close()
        return ClosingSocket()

    async def test_pcm_transcript_and_tts_return_to_client(self):
        async with await self.connect(speak=True) as client:
            self.assertEqual(json.loads(await client.recv())["type"], "ready")
            pcm = b"\x12\x34" * 512
            await client.send(pcm)
            result = json.loads(await client.recv())
            self.assertEqual(result["result"]["txt_translation"], "Hallo")
            self.assertEqual(result["final"], True)
            self.assertEqual(json.loads(await client.recv())["type"], "audio_start")
            audio = await client.recv()
            self.assertEqual(struct.unpack("<IHH", audio[:8]), (24000, 1, 1))
            self.assertEqual(audio[8:], b"\x01\x00" * 100)
            self.assertEqual(json.loads(await client.recv())["type"], "audio_end")
            self.assertEqual(self.frames, [pcm])
        await asyncio.wait_for(asyncio.to_thread(self.source_closed.wait), 2)

    async def test_bad_pairing_never_opens_source(self):
        async with await self.connect(token="wrong") as client:
            with self.assertRaises(websockets.ConnectionClosedError) as error:
                await client.recv()
            self.assertEqual(error.exception.code, 1008)
        self.assertEqual(self.frames, [])
        self.assertFalse(self.source_closed.is_set())

    async def test_second_client_is_rejected_and_reconnect_is_clean(self):
        first = await self.connect()
        await first.recv()
        async with await self.connect() as second:
            with self.assertRaises(websockets.ConnectionClosedError) as error:
                await second.recv()
            self.assertEqual(error.exception.code, 1013)
        await first.close()
        await asyncio.wait_for(asyncio.to_thread(self.source_closed.wait), 2)
        async with await self.connect() as third:
            self.assertEqual(json.loads(await third.recv())["type"], "ready")

    async def test_malformed_pcm_closes_source(self):
        async with await self.connect() as client:
            await client.recv()
            await client.send(b"bad")
            with self.assertRaises(websockets.ConnectionClosedError) as error:
                await client.recv()
            self.assertEqual(error.exception.code, 1008)
        self.assertTrue(self.source_closed.is_set())

    async def test_no_administration_commands_on_audio_connection(self):
        async with await self.connect() as client:
            await client.recv()
            await client.send(json.dumps({"type": "setting_change", "name": "tts_type", "value": ""}))
            with self.assertRaises(websockets.ConnectionClosedError):
                await client.recv()


class DestinationTests(unittest.TestCase):
    def test_discovery_echoes_nonce_but_never_pairing_key(self):
        from unittest.mock import Mock
        protocol = Discovery(5001)
        transport = Mock()
        protocol.connection_made(transport)
        protocol.datagram_received(b"WT_AUDIO_DISCOVER " + b"a" * 32, ("127.0.0.1", 1234))
        response = json.loads(transport.sendto.call_args.args[0])
        self.assertEqual(response["nonce"], "a" * 32)
        self.assertEqual(response["port"], 5001)
        self.assertNotIn("token", response)
        protocol.datagram_received(b"WT_AUDIO_DISCOVER " + b"a" * 32, ("127.0.0.1", 1234))
        self.assertEqual(transport.sendto.call_count, 1)

    def test_destination_is_scoped_and_thread_local(self):
        captured = []
        with destination(captured.append):
            self.assertEqual(current_destination(), captured.append)
            thread = threading.Thread(target=lambda: captured.append(current_destination()))
            thread.start()
            thread.join()
        self.assertIsNone(current_destination())
        self.assertEqual(captured, [None])

    def test_remote_settings_snapshots_keep_result_destination(self):
        class Settings:
            def GetOption(self, name):
                return 42
        class Session:
            closed = threading.Event()
            def result(self, value, final):
                self.received = (value, final)
        session = Session()
        snapshot = RemoteSettings(Settings(), session).snapshot()
        snapshot.remote_result({"text": "test"}, True)
        self.assertEqual(session.received, ({"text": "test"}, True))
        session.closed.set()
        self.assertTrue(snapshot.remote_closed())


if __name__ == "__main__":
    unittest.main()

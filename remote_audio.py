"""Version 1 LAN audio service. No platform or model imports in the transport.

Client sends PCM16 LE, mono, 16 kHz, 512 samples per binary message.
Server audio messages: little-endian uint32 sample rate, uint16 channels,
uint16 format (1 = PCM16 LE), then PCM. JSON carries session events.
The separate authenticated listener exposes no settings/files/plugin commands.
"""
import asyncio
import hmac
import json
import secrets
import socket
import struct
import threading
import time
from pathlib import Path

VERSION = 1
INPUT_BYTES = 1024
MAX_MESSAGE = 65536
AUDIO_HEADER = struct.Struct("<IHH")


class Discovery(asyncio.DatagramProtocol):
    def __init__(self, port):
        self.port = port
        self.last_reply = {}

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, address):
        prefix = b"WT_AUDIO_DISCOVER "
        if len(data) != len(prefix) + 32 or not data.startswith(prefix):
            return
        now = time.monotonic()
        if now - self.last_reply.get(address[0], -10) < 1:
            return
        if len(self.last_reply) >= 128:
            self.last_reply.clear()
        self.last_reply[address[0]] = now
        try:
            nonce = data[len(prefix):].decode("ascii")
        except UnicodeDecodeError:
            return
        reply = {"service": "whispering-tiger-audio", "version": VERSION,
                 "nonce": nonce, "port": self.port, "name": socket.gethostname()[:64]}
        self.transport.sendto(json.dumps(reply).encode("utf-8"), address)


class Session:
    def __init__(self, socket, source_factory, synthesize):
        self.socket = socket
        self.loop = asyncio.get_running_loop()
        self.source_factory = source_factory
        self.synthesize = synthesize
        self.closed = threading.Event()
        self.cancel_audio = threading.Event()
        self.outgoing = asyncio.Queue(maxsize=256)
        self.speech = asyncio.Queue(maxsize=8)
        self.source = None
        self.speak = False
        self.attach_token = secrets.token_urlsafe(32)
        self.control = None
        self.follow_profile = False

    def _offer(self, value):
        if self.closed.is_set():
            return
        if isinstance(value, bytes) and self.cancel_audio.is_set():
            return
        try:
            self.outgoing.put_nowait(value)
        except asyncio.QueueFull:
            self.closed.set()
            asyncio.create_task(self.socket.close(1013, "Client is too slow"))

    def send(self, value):
        if not self.closed.is_set():
            self.loop.call_soon_threadsafe(self._offer, value)

    def result(self, result, final):
        self.send(json.dumps({"type": "transcript", "final": bool(final), "result": result}))
        speak = self.speak
        if self.follow_profile:
            import settings
            speak = settings.GetOption("tts_answer")
        if final and speak:
            text = result.get("txt_translation") or result.get("text") or ""
            self.loop.call_soon_threadsafe(self.queue_speech, text)

    def queue_speech(self, text):
        if self.closed.is_set() or not text:
            return
        try:
            self.speech.put_nowait(text)
        except asyncio.QueueFull:
            self.send(json.dumps({"type": "error", "message": "Speech queue is full"}))

    def audio(self, pcm, rate, channels=1):
        if self.closed.is_set() or self.cancel_audio.is_set():
            raise InterruptedError("Remote playback cancelled")
        frame_bytes = channels * 2
        limit = (MAX_MESSAGE - AUDIO_HEADER.size) // frame_bytes * frame_bytes
        for offset in range(0, len(pcm), limit):
            self.send(AUDIO_HEADER.pack(rate, channels, 1) + pcm[offset:offset + limit])

    async def writer(self):
        while True:
            await self.socket.send(await self.outgoing.get())

    async def speaker(self):
        while True:
            text = await self.speech.get()
            self.cancel_audio.clear()
            self.send(json.dumps({"type": "audio_start"}))
            try:
                await asyncio.to_thread(self.synthesize, self, text)
            except InterruptedError:
                pass
            except Exception as exc:
                self.send(json.dumps({"type": "error", "message": str(exc)}))
            finally:
                self.send(json.dumps({"type": "audio_end"}))

    async def run(self, hello):
        self.speak = hello.get("speak") is True
        self.follow_profile = hello.get("follow_profile") is True
        # Load the recorder off the network loop, before accepting PCM.
        self.source = await asyncio.to_thread(self.source_factory, self, hello)
        writer = asyncio.create_task(self.writer())
        speaker = asyncio.create_task(self.speaker())
        self.send(json.dumps({"type": "ready", "version": VERSION,
                              "sample_rate": 16000, "frames": 512, "attach_token": self.attach_token}))
        try:
            async for message in self.socket:
                if isinstance(message, bytes):
                    if len(message) != INPUT_BYTES:
                        raise ValueError("Expected 512 mono PCM16 samples")
                    # Awaiting processing bounds inbound buffering and preserves order.
                    await asyncio.to_thread(self.source.feed, message)
                else:
                    request = json.loads(message)
                    if not isinstance(request, dict):
                        raise ValueError("Expected an object")
                    if request.get("type") == "speak":
                        text = request.get("text", "")
                        if not isinstance(text, str) or len(text) > 4000:
                            raise ValueError("Speech text must be at most 4000 characters")
                        self.queue_speech(text)
                    elif request.get("type") == "stop":
                        self.cancel_audio.set()
                        while not self.speech.empty():
                            self.speech.get_nowait()
                        self.send(json.dumps({"type": "audio_stopped"}))
                    else:
                        raise ValueError("Unsupported remote command")
        finally:
            self.closed.set()
            self.cancel_audio.set()
            writer.cancel()
            speaker.cancel()
            await asyncio.gather(writer, speaker, return_exceptions=True)
            await asyncio.to_thread(self.source.close)


class Server:
    def __init__(self, source_factory, synthesize, token=None):
        self.source_factory = source_factory
        self.synthesize = synthesize
        self.token = token or secrets.token_urlsafe(24)
        self.listener = None
        self.discovery = None
        self.active = False
        self.session = None

    async def handler(self, socket, path=None):
        try:
            raw = await asyncio.wait_for(socket.recv(), 5)
            hello = json.loads(raw) if isinstance(raw, str) else None
            if not isinstance(hello, dict) or hello.get("version") != VERSION \
                    or not isinstance(hello.get("token"), str) \
                    or not hmac.compare_digest(hello["token"], self.token):
                await socket.close(1008, "Pairing failed")
                return
            if self.active:
                await socket.close(1013, "AI host already has an audio client")
                return
            self.active = True
            try:
                self.session = Session(socket, self.source_factory, self.synthesize)
                await self.session.run(hello)
            finally:
                self.session = None
                self.active = False
        except (ValueError, TypeError) as exc:
            await socket.close(1008, str(exc)[:100])
        except asyncio.TimeoutError:
            await socket.close(1008, "Pairing timed out")
        except Exception:
            await socket.close(1011, "Remote audio session failed")

    async def start(self, host="0.0.0.0", port=5001):
        import websockets
        self.listener = await websockets.serve(
            self.handler, host, port, max_size=MAX_MESSAGE, max_queue=16,
            compression=None, ping_interval=10, ping_timeout=20, close_timeout=2,
        )
        actual_port = self.listener.sockets[0].getsockname()[1]
        try:
            self.discovery, _ = await asyncio.get_running_loop().create_datagram_endpoint(
                lambda: Discovery(actual_port), local_addr=(host, actual_port))
        except OSError:
            # Manual connections remain usable when UDP discovery is unavailable.
            pass
        return self

    async def close(self):
        if self.discovery:
            self.discovery.close()
            self.discovery = None
        if self.listener:
            self.listener.close()
            await self.listener.wait_closed()


class RemoteSettings:
    def __init__(self, wrapped, session):
        self.wrapped = wrapped
        self.session = session

    def GetOption(self, name):
        return self.wrapped.GetOption(name)

    get_option = GetOption

    def snapshot(self):
        snapshot = getattr(self.wrapped, "snapshot", lambda: self.wrapped)()
        if hasattr(snapshot, "get_all_settings"):
            import audio_routes
            snapshot = audio_routes.FrozenRouteSettings(snapshot, snapshot.get_all_settings())
        return RemoteSettings(snapshot, self.session)

    def remote_closed(self):
        return self.session.closed.is_set()

    def remote_result(self, result, final):
        self.session.result(result, final)


def create_source(session, hello):
    import audio_routes
    import audioprocessor
    import settings

    config = audio_routes.normalize_route({
        "id": "remote-" + secrets.token_hex(8),
        "name": str(hello.get("name") or "Remote audio")[:64],
        "websocket_enabled": False, "osc_enabled": False, "plugins": [],
        "phrase_time_limit": 30,
        "txt_translate": bool(settings.GetOption("txt_translate")),
        "realtime": bool(settings.GetOption("realtime")),
    }, 0, settings.SETTINGS)
    route = audio_routes.AudioRoute(config, settings.SETTINGS, [], audioprocessor.q)
    if session.follow_profile:
        class LiveSettings(audio_routes.RouteSettings):
            def _overrides(self):
                values = super()._overrides()
                for key in self._DIRECT_OVERRIDES:
                    values.pop(key, None)
                values["phrase_time_limit"] = 30
                return values
        route.settings = LiveSettings(settings.SETTINGS, config)
    route.settings = RemoteSettings(route.settings, session)
    try:
        route.start(capture=False)
    except BaseException:
        route.close()
        raise

    class Source:
        def __init__(self):
            self.buffer = bytearray()
            self.last_frame = time.monotonic()
            self.sample_budget = 32000.0

        def feed(self, pcm):
            if audioprocessor.q.source_size(config["id"]) >= 8:
                raise ValueError("AI processing cannot keep up with this source")
            # A paired sender still cannot submit audio faster than real time.
            now = time.monotonic()
            self.sample_budget = min(32000, self.sample_budget + (now - self.last_frame) * 16000)
            self.last_frame = now
            self.sample_budget -= len(pcm) // 2
            if self.sample_budget < 0:
                raise ValueError("Audio arrived faster than real time")
            self.buffer.extend(pcm)
            size = route.processor.chunk * 2
            while len(self.buffer) >= size:
                frame = bytes(self.buffer[:size])
                del self.buffer[:size]
                route.processor.callback(frame, size // 2, None, None)

        def close(self):
            route.close()
            audioprocessor.q.discard_source(config["id"])

    return Source()


_synthesis_lock = threading.Lock()


def synthesize(session, text):
    import numpy as np
    from Models.TTS import tts
    from remote_audio_output import destination

    sent = False
    request = text if isinstance(text, dict) else {"text": text, "to_device": True}
    text = request.get("text", "")

    def sink(chunk, rate, channels, dtype):
        nonlocal sent
        samples = np.frombuffer(chunk, dtype=dtype)
        if np.issubdtype(samples.dtype, np.floating):
            samples = np.clip(np.nan_to_num(samples), -1, 1) * 32767
        if request.get("to_device", True):
            session.audio(samples.astype("<i2").tobytes(), int(rate), int(channels))
        sent = True

    with _synthesis_lock, destination(sink):
        if session.closed.is_set() or session.cancel_audio.is_set():
            return
        if not tts.init():
            raise ValueError("Select a built-in TTS model on the AI host first")
        adapter = tts.tts
        if request.get("last"):
            waveform, rate = adapter.get_last_generation()
        elif hasattr(adapter, "tts_streaming") and request.get("to_device", True):
            waveform, rate = adapter.tts_streaming(text)
        else:
            waveform, rate = adapter.tts(text)
        if not sent and waveform is not None:
            if hasattr(waveform, "detach"):
                waveform = waveform.detach().float().cpu().numpy()
            wave = np.asarray(waveform).squeeze()
            if wave.ndim != 1:
                raise ValueError("Remote TTS requires mono audio")
            sink(wave.tobytes(), rate, 1, wave.dtype)
        if request.get("download") and waveform is not None and session.control is not None:
            import base64
            import websocket
            websocket.AnswerMessage(session.control, json.dumps({"type": "tts_save", "request_id": request.get("request_id", ""), "wav_data": base64.b64encode(adapter.return_wav_file_binary(waveform)).decode("ascii")}))


host_server = None


def control_session(websocket):
    session = host_server.session if host_server else None
    return session if session and session.control is websocket and not session.closed.is_set() else None


def route_tts(request, websocket, last=False):
    session = control_session(websocket)
    if session is None:
        return False
    value = dict(request.get("value") or {})
    value["last"] = last
    session.loop.call_soon_threadsafe(session.queue_speech, value)
    return True


async def configure_host(enabled, port=5001):
    global host_server
    if host_server is not None and (not enabled or host_server.listener is None):
        await host_server.close()
        host_server = None
    if enabled and host_server is None:
        server = Server(create_source, synthesize, load_pairing_key())
        await server.start(port=port)
        host_server = server
    return {"type": "remote_audio_host", "enabled": host_server is not None,
            "port": port, "token": host_server.token if host_server else ""}


def load_pairing_key():
    """Machine-local secret, deliberately outside the shared profile YAML."""
    path = Path(".cache/remote-audio/pairing-key")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="ascii") as output:
            path.chmod(0o600)
            output.write(secrets.token_urlsafe(24))
    except FileExistsError:
        pass
    token = path.read_text(encoding="ascii").strip()
    if len(token) < 32:
        raise ValueError("Invalid local remote-audio pairing key")
    return token

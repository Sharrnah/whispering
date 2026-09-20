"""Opt-in real CUDA loopback: network TTS -> network capture -> real ASR.

Run from repository root with the developer venv. Uses only already-cached
Qwen checkpoints, no saved profile, microphone, or local playback device.
"""
import asyncio
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import settings
import websockets
from scipy.signal import resample_poly

for key, value in {
    "tts_type": "qwen3_tts", "tts_ai_device": "cuda", "tts_precision": "bfloat16",
    "tts_model": ["Built-in voices and control", "Qwen3-TTS-12Hz-0.6B-CustomVoice"],
    "tts_voice": "Ryan", "stt_type": "qwen3_asr", "model": "Qwen3-ASR-0.6B-hf",
    "ai_device": "cuda", "whisper_precision": "bfloat16", "beam_size": 1,
    "current_language": "en", "realtime": False, "txt_translate": False,
    "osc_ip": "0", "websocket_ip": "0", "denoise_audio": "", "energy": 100,
    "pause": 0.5, "vad_smart_turn_enabled": False,
    "special_settings": {"tts_qwen3_tts": {"language": "en", "seed": 42,
         "streaming_buffer_mode": "fixed", "streaming_mode": "codec"}},
}.items():
    settings.SetOption(key, value)

import audio_tools
import audioprocessor
import remote_audio
import websocket as control_api
from remote_audio import Server, create_source, synthesize, AUDIO_HEADER


async def main():
    server = await Server(create_source, synthesize, "smoke-test").start("127.0.0.1", 0)
    remote_audio.host_server = server
    class Bridge:
        async def send(self, socket, message):
            await socket.send(message)
    async def control_handler(socket, path=None):
        async for raw in socket:
            await control_api.custom_message_handler(Bridge(), json.loads(raw), socket)
    control_listener = await websockets.serve(control_handler, "127.0.0.1", 0)
    control_port = control_listener.sockets[0].getsockname()[1]
    loop = asyncio.get_running_loop()
    answer_patch = patch.object(control_api, "AnswerMessage", side_effect=lambda socket, message: asyncio.run_coroutine_threadsafe(socket.send(message), loop))
    answer_patch.start()
    port = server.listener.sockets[0].getsockname()[1]
    audioprocessor.start_whisper_thread()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client, websockets.connect(f"ws://127.0.0.1:{control_port}") as control:
            await client.send(json.dumps({"version": 1, "token": "smoke-test", "follow_profile": True}))
            ready = json.loads(await asyncio.wait_for(client.recv(), 60))
            print("HANDSHAKE", ready["type"], flush=True)
            await control.send(json.dumps({"type": "remote_audio_attach", "value": {"token": ready["attach_token"]}}))
            await control.send(json.dumps({"type": "tts_req", "value": {"text": "The remote audio connection is working.", "to_device": True, "download": False}}))
            chunks = []
            started = time.monotonic()
            while True:
                message = await asyncio.wait_for(client.recv(), 180)
                if isinstance(message, bytes):
                    rate, channels, fmt = AUDIO_HEADER.unpack(message[:8])
                    assert channels == 1 and fmt == 1
                    if not chunks:
                        print("FIRST_AUDIO_SECONDS", time.monotonic() - started, flush=True)
                    chunks.append(message[8:])
                else:
                    event = json.loads(message)
                    if event["type"] == "error":
                        raise RuntimeError(event["message"])
                    if event["type"] == "audio_end":
                        break
            pcm = b"".join(chunks)
            assert pcm and np.max(np.abs(np.frombuffer(pcm, "<i2"))) > 100
            print("TTS_PACKETS", len(chunks), "SAMPLES", len(pcm) // 2, "RATE", rate, flush=True)
            from Models.TTS import tts
            last, last_rate = tts.tts.get_last_generation()
            if hasattr(last, "detach"):
                last = last.detach().float().cpu().numpy()
            expected = (np.clip(np.asarray(last).squeeze(), -1, 1) * 32767).astype("<i2").tobytes()
            assert pcm == expected, "Network and returned TTS audio differ"
            print("PCM_IDENTITY_OK", flush=True)
            wave = resample_poly(np.frombuffer(pcm, "<i2").astype(np.float32), 16000, rate)
            capture = np.clip(wave, -32768, 32767).astype("<i2").tobytes() + b"\0" * 64000
            capture += b"\0" * (-len(capture) % 1024)
            for offset in range(0, len(capture), 1024):
                await client.send(capture[offset:offset + 1024])
                await asyncio.sleep(0.032)
            while True:
                event = json.loads(await asyncio.wait_for(client.recv(), 120))
                print("RESULT", event, flush=True)
                if event["type"] == "transcript" and event["final"]:
                    assert "remote" in event["result"]["text"].lower()
                    break
            await control.send(json.dumps({"type": "tts_req_last", "value": {"to_device": False, "download": True, "request_id": "client-export"}}))
            exported = json.loads(await asyncio.wait_for(control.recv(), 60))
            import base64
            assert exported["type"] == "tts_save" and exported["request_id"] == "client-export"
            assert base64.b64decode(exported["wav_data"]).startswith(b"RIFF")
            print("CONTROL_TTS_AND_CLIENT_EXPORT_OK", flush=True)
    finally:
        await server.close()
        control_listener.close()
        await control_listener.wait_closed()
        answer_patch.stop()
        remote_audio.host_server = None


if __name__ == "__main__":
    # Any accidental local output open fails this validation.
    with patch.object(audio_tools.pyaudio_pool, "acquire", side_effect=AssertionError("Opened host audio device")):
        asyncio.run(main())

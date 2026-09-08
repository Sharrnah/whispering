"""Docker-only frozen-app capture -> CPU ASR -> WebSocket -> quit check.

Requires PulseAudio, espeak-ng, verified tiny.en-ct2 and lid caches in --work.
The virtual audio device and backend process are owned by this test.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pulsectl
import websockets
import yaml


async def exercise(executable, work, source=None, vad=False):
    profile = work / "Profiles/linux-frozen-smoke.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(yaml.safe_dump({
        "audio_api": "PulseAudio", "audio_input_device": "Monitor of WT_Frozen_Smoke",
        "audio_output_device": "WT_Frozen_Smoke", "device_index": -1, "device_out_index": -1,
        "model": "tiny.en", "stt_type": "faster_whisper", "ai_device": "cpu",
        "whisper_precision": "float32", "current_language": "en", "beam_size": 1,
        "txt_translator": "", "tts_type": "", "ocr_type": "", "tts_answer": False,
        "vad_enabled": vad, "vad_smart_turn_enabled": False, "realtime": False,
        "energy": 200, "pause": 0.7, "phrase_time_limit": 15,
        "osc_ip": "0", "osc_server_ip": "0", "websocket_ip": "127.0.0.1",
        "websocket_port": 5055, "plugins": {}, "run_backend": True,
        "denoise_audio": "", "normalize_enabled": False, "silence_cutting_enabled": False,
    }), encoding="utf-8")
    speech = work / "linux-smoke-speech.wav"
    subprocess.run(["espeak-ng", "-s", "140", "-w", str(speech),
                    "The Linux audio system is recording my voice. This is a test of speech recognition."], check=True)
    log_path = work / "linux-frozen-smoke.log"
    with pulsectl.Pulse("wt-frozen-test") as pulse:
        module = pulse.module_load("module-null-sink", "sink_name=wt_frozen_smoke rate=16000 channels=1 sink_properties=device.description=WT_Frozen_Smoke")
        process = None
        try:
            with log_path.open("w") as log:
                command = [str(executable)] + ([str(source)] if source else [])
                process = subprocess.Popen(command + ["--config", str(profile)], cwd=work,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 120
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Frozen backend exited with {process.returncode}; see {log_path}")
                    try:
                        socket = await websockets.connect("ws://127.0.0.1:5055", open_timeout=2)
                        break
                    except (OSError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Backend did not open WebSocket; see {log_path}")
                        await asyncio.sleep(0.5)
                async with socket:
                    await socket.send(json.dumps({"type": "ui_connected"}))
                    source_index = next(source.index for source in pulse.source_list() if source.name == "wt_frozen_smoke.monitor")
                    while not any(output.source == source_index for output in pulse.source_output_list()):
                        if time.monotonic() >= deadline or process.poll() is not None:
                            raise TimeoutError(f"Backend did not open monitor capture; see {log_path}")
                        await asyncio.sleep(0.25)
                    await asyncio.sleep(1)
                    player = await asyncio.create_subprocess_exec("paplay", "--device=wt_frozen_smoke", str(speech))
                    assert await player.wait() == 0
                    async with asyncio.timeout(60):
                        while True:
                            message = json.loads(await socket.recv())
                            if message.get("type") == "transcript" and message.get("text"):
                                transcript = message["text"]
                                break
                    # Tiny's decoding of synthetic espeak pronunciation varies;
                    # verify both recognizable clauses, not every final noun.
                    assert "recording my voice" in transcript.lower() and "this is a test" in transcript.lower(), transcript
                    await socket.send(json.dumps({"type": "quit", "value": ""}))
                await asyncio.to_thread(process.wait, timeout=15)
                assert process.returncode == 0, process.returncode
                assert yaml.safe_load(profile.read_text())["process_id"] == 0
                print(json.dumps({"status": "passed", "transcript": transcript,
                                  "capture": "PulseAudio monitor", "backend": str(executable),
                                  "shutdown": "WebSocket quit; exit 0; persisted process_id 0"}))
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            pulse.module_unload(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=Path("/out/audioWhisper/audioWhisper"))
    parser.add_argument("--work", type=Path, default=Path("/out"))
    parser.add_argument("--source", type=Path, help="Optional audioWhisper.py path when --executable is Python")
    parser.add_argument("--vad", action="store_true", help="Exercise the default Silero VAD capture path")
    args = parser.parse_args()
    asyncio.run(exercise(args.executable.resolve(), args.work.resolve(), args.source, args.vad))

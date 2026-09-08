"""Docker integration check: named PulseAudio playback and monitor capture.

Requires a running container-local PulseAudio server. Never run on the host.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pulsectl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with pulsectl.Pulse("whispering-tiger-audio-test") as pulse:
    module = pulse.module_load("module-null-sink", "sink_name=wt_smoke rate=16000 channels=1 sink_properties=device.description=WT_Smoke_Speakers")
    try:
        # Import after creating the virtual devices: PortAudio enumerates at init.
        import pyaudio
        import audio_tools

        api, api_name = audio_tools.get_audio_api_index_by_name("PulseAudio")
        assert api_name == "PulseAudio", api_name
        output = audio_tools.get_audio_device_index_by_name_and_api("WT_Smoke_Speakers", api, False)
        source = next(device for device in pulse.source_list() if device.name == "wt_smoke.monitor")
        input_index = audio_tools.get_audio_device_index_by_name_and_api(source.description, api, True)
        captured = []
        audio = pyaudio.PyAudio()

        def record(data, count, timing, status):
            captured.append(data)
            return None, pyaudio.paContinue

        reader = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                            input_device_index=input_index, frames_per_buffer=512, stream_callback=record)
        writer = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True,
                            output_device_index=output, frames_per_buffer=512)
        try:
            time.sleep(0.15)
            signal = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 12000).astype(np.int16)
            writer.write(signal.tobytes())
            time.sleep(0.3)
        finally:
            writer.stop_stream()
            writer.close()
            reader.stop_stream()
            reader.close()
            audio.terminate()
        samples = np.frombuffer(b"".join(captured), dtype=np.int16).astype(np.float32)
        assert len(samples) >= 16000, len(samples)
        assert float(np.max(np.abs(samples))) > 10000, "Recorded only silence or wrong endpoint"
        spectrum = np.abs(np.fft.rfft(samples))
        peak_hz = int(np.argmax(spectrum) * 16000 / len(samples))
        assert abs(peak_hz - 440) < 3, peak_hz
        print(json.dumps({"api": api_name, "input": source.description, "output_index": output,
                          "captured_samples": len(samples), "peak_hz": peak_hz, "status": "passed"}))
    finally:
        pulse.module_unload(module)

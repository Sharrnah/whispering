"""Docker integration check: application playback during PulseAudio capture.

Requires a running container-local PulseAudio server. Never run on the host.
"""
import io
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np
import pulsectl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with pulsectl.Pulse("whispering-tiger-audio-test") as pulse:
    previous = pulse.server_info()
    module = pulse.module_load("module-null-sink", "sink_name=wt_smoke rate=44100 channels=2 sink_properties=device.description=WT_Smoke_Speakers")
    try:
        sink = next(device for device in pulse.sink_list() if device.name == "wt_smoke")
        source = next(device for device in pulse.source_list() if device.name == "wt_smoke.monitor")
        pulse.default_set(sink)
        pulse.default_set(source)
        # Import after creating the virtual devices: PortAudio enumerates at init.
        import audio_tools
        import torch

        api, api_name = audio_tools.get_audio_api_index_by_name("PulseAudio")
        assert api_name == "PulseAudio", api_name
        output = audio_tools.get_audio_device_index_by_name_and_api("WT_Smoke_Speakers", api, False)
        input_index = audio_tools.get_audio_device_index_by_name_and_api(source.description, api, True)
        default_output = audio_tools.get_default_audio_device_index_by_api("PulseAudio", False)
        audio = audio_tools.main_app_py_audio
        # This endpoint reported 32 channels and caused the original failure.
        assert audio.get_device_info_by_index(default_output)["name"] == "Default Sink"
        captured = []
        results = []

        def record(data, count, timing, status):
            captured.append(data)
            return None, audio_tools.pyaudio.paContinue

        reader = audio.open(format=audio_tools.pyaudio.paFloat32, channels=1, rate=44100, input=True,
                            input_device_index=input_index, frames_per_buffer=512, stream_callback=record)

        def check(label, play):
            start = len(captured)
            play()
            time.sleep(0.4)
            samples = np.frombuffer(b"".join(captured[start:]), dtype=np.float32)
            assert len(samples) >= 22050, (label, len(samples))
            assert np.isfinite(samples).all(), label
            peak = float(np.max(np.abs(samples)))
            assert 0.48 < peak < 0.52, (label, peak)
            assert np.count_nonzero(np.abs(samples) > 0.1) > 18000, (label, "missing/truncated audio")
            peak_hz = float(np.argmax(np.abs(np.fft.rfft(samples))) * 44100 / len(samples))
            assert abs(peak_hz - 440) < 3, (label, peak_hz)
            results.append({"case": label, "captured_samples": len(samples), "peak": peak, "peak_hz": peak_hz})

        try:
            signal = (np.sin(2 * np.pi * 440 * np.arange(44100) / 44100) * 0.5).astype(np.float32)
            def play_tensor():
                audio_tools.play_audio(torch.from_numpy(signal).unsqueeze(0), default_output,
                                       source_sample_rate=44100, audio_device_channel_num=1,
                                       input_channels=1, target_channels=1, dtype="float32", tag="tts-smoke")

            check("tts_tensor_default_sink", play_tensor)
            # Reuse the pool and probe devices while capture remains active.
            audio_tools.get_audio_device_index_by_name_and_api("WT_Smoke_Speakers", api, False)
            check("tts_tensor_repeated", play_tensor)

            wav_buffer = io.BytesIO()
            pcm = (signal * 32767).astype(np.int16)
            with wave.open(wav_buffer, "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(44100)
                wav.writeframes(np.column_stack((pcm, pcm)).tobytes())
            check("stereo_wav_named_sink", lambda: audio_tools.play_audio(wav_buffer.getvalue(), output, tag="wav-smoke"))

            def play_streamed():
                streamer = audio_tools.AudioStreamer(device_index=default_output, source_sample_rate=44100,
                                                     input_channels=1, playback_channels=1, dtype="float32")
                try:
                    streamer.stop_playing_timeout = 0.3
                    signal = (np.sin(2 * np.pi * 440 * np.arange(65536) / 44100) * 0.5).astype(np.float32)
                    streamer.add_audio_chunk(signal.tobytes())
                    worker = streamer.playback_thread
                    if worker is not None:
                        worker.join(10)
                        assert not worker.is_alive(), "streamed playback did not finish"
                finally:
                    streamer.stop()
            check("streamed_float_pcm", play_streamed)
        finally:
            reader.stop_stream()
            reader.close()
        print(json.dumps({"api": api_name, "checks": results, "status": "passed"}))
    finally:
        for device in pulse.sink_list():
            if device.name == previous.default_sink_name:
                pulse.default_set(device)
        for device in pulse.source_list():
            if device.name == previous.default_source_name:
                pulse.default_set(device)
        pulse.module_unload(module)

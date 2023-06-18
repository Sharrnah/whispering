import io
import threading
import time
import wave

import resampy
import numpy as np
import pyaudio
import torch
from pydub import AudioSegment


# resample_audio function to resample audio data to a different sample rate and convert it to mono.
# set target_channels to '-1' to average the left and right channels to create mono audio (default)
# set target_channels to '0' to extract the first channel (left channel) data
# set target_channels to '1' to extract the second channel (right channel) data
# set target_channels to '2' to keep stereo channels (or copy the mono channel to both channels if is_mono is True)
# to Convert the int16 numpy array to bytes use .tobytes()
def resample_audio(audio_chunk, recorded_sample_rate, target_sample_rate, target_channels=-1, is_mono=None, dtype="int16"):
    audio_data_dtype = np.int16
    if dtype == "int16":
        audio_data_dtype = np.int16
    elif dtype == "float32":
        audio_data_dtype = np.float32
    audio_data = np.frombuffer(audio_chunk, dtype=audio_data_dtype)

    # try to guess if the audio is mono or stereo
    if is_mono is None:
        is_mono = audio_data.shape[0] % 2 != 0

    if target_channels < 2 and not is_mono:
        # Reshape the array to separate the channels
        audio_data = audio_data.reshape(-1, 2)

    if target_channels == -1 and not is_mono:
        # Average the left and right channels to create mono audio
        audio_data = audio_data.mean(axis=1)
    elif target_channels == 0 or target_channels == 1 and not is_mono:
        # Extract the first channel (left channel) data
        audio_data = audio_data[:, target_channels]
    elif target_channels == 2 and is_mono:
        # Duplicate the mono channel to create left and right channels
        audio_data = np.column_stack((audio_data, audio_data))
        # Flatten the array and convert it back to int16 dtype
        audio_data = audio_data.flatten()

    # Resample the audio data to the desired sample rate
    audio_data = resampy.resample(audio_data, recorded_sample_rate, target_sample_rate)
    # Convert the resampled data back to int16 dtype
    return np.asarray(audio_data, dtype=audio_data_dtype)


def get_closest_sample_rate_of_device(device_index, target_sample_rate, fallback_sample_rate=44100):
    p = pyaudio.PyAudio()
    device_info = p.get_device_info_by_index(device_index if device_index is not None else p.get_default_output_device_info()["index"])
    supported_sample_rates = device_info.get("supportedSampleRates")

    # If supported_sample_rates is empty, use common sample rates as a fallback
    if not supported_sample_rates:
        supported_sample_rates = [device_info.get("defaultSampleRate")]
        if not supported_sample_rates:
            supported_sample_rates = [fallback_sample_rate]

    # Find the closest supported sample rate to the original sample rate
    closest_sample_rate = min(supported_sample_rates, key=lambda x: abs(x - target_sample_rate))
    return closest_sample_rate


# ------------------------
# Audio Playback Functions
# ------------------------
def _tensor_to_buffer(tensor):
    buff = io.BytesIO()
    torch.save(tensor, buff)
    buff.seek(0)
    return buff


def _generate_binary_buffer(audio):
    return io.BytesIO(audio)


def convert_tensor_to_wav_buffer(audio, sample_rate=24000, channels=1, sample_width=4):
    audio = _tensor_to_buffer(audio)

    wav_file = AudioSegment.from_file(audio, format="raw", frame_rate=sample_rate, channels=channels, sample_width=sample_width)

    buff = io.BytesIO()
    wav_file.export(buff, format="wav")

    return buff


def play_stream(p=None, device=None, audio_data=None, chunk=1024, audio_format=2, channels=2, sample_rate=44100):
    try:
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=int(sample_rate),
                        output_device_index=device,
                        output=True)

        for i in range(0, len(audio_data), chunk * channels):
            stream.write(audio_data[i:i + chunk * channels].tobytes())

        stream.close()
    except Exception as e:
        print("Error playing audio: {}".format(e))


# play wav binary audio to device, converting audio sample_rate and channels if necessary
# audio can be bytes (in wav) or tensor
# tensor_sample_with is the sample width of the tensor (if audio is tensor and not bytes) [default is 4 bytes]
# tensor_channels is the number of channels of the tensor (if audio is tensor and not bytes) [default is 1 channel (mono)]
def play_audio(audio, device=None, source_sample_rate=44100, audio_device_channel_num=2, target_channels=2, is_mono=True, dtype="int16", tensor_sample_with=4, tensor_channels=1, secondary_device=None):
    if isinstance(audio, bytes):
        buff = _generate_binary_buffer(audio)
    else:
        buff = convert_tensor_to_wav_buffer(audio, sample_rate=source_sample_rate, channels=tensor_channels, sample_width=tensor_sample_with)

    # Set chunk size of 1024 samples per data frame
    chunk = 1024

    # Open the sound file
    wf = wave.open(buff, 'rb')

    # Create an interface to PortAudio
    p = pyaudio.PyAudio()

    # Find the closest supported sample rate to the original sample rate
    closest_sample_rate = get_closest_sample_rate_of_device(device, wf.getframerate())

    # Read all audio data and resample if necessary
    frame_data = wf.readframes(wf.getnframes())

    # get audio sample width
    audio_sample_width = wf.getsampwidth()

    wf.close()

    # resample audio data
    audio_data = resample_audio(frame_data, source_sample_rate, closest_sample_rate, target_channels=target_channels, is_mono=is_mono, dtype=dtype)

    audio_thread = None
    if secondary_device is not None:
        audio_thread = threading.Thread(target=play_stream, args=(
            p, secondary_device, audio_data, chunk,
            p.get_format_from_width(audio_sample_width),
            audio_device_channel_num,
            closest_sample_rate
        ))
        audio_thread.start()

    # Open a .Stream object to write the WAV file to
    # 'output = True' indicates that the sound will be played rather than recorded
    play_stream(p=p, device=device, audio_data=audio_data, chunk=chunk,
                audio_format=p.get_format_from_width(audio_sample_width),
                channels=audio_device_channel_num,
                sample_rate=closest_sample_rate
                )

    # wait while audio thread is running
    if audio_thread is not None:
        while audio_thread.is_alive():
            time.sleep(0.1)

    p.terminate()


def start_recording_audio_stream(device_index=None, sample_format=pyaudio.paInt16, sample_rate=16000, channels=1, chunk=int(16000/10), py_audio=None):
    if py_audio is None:
        py_audio = pyaudio.PyAudio()

    needs_sample_rate_conversion = False
    is_mono = False

    recorded_sample_rate = sample_rate

    try:
        stream = py_audio.open(format=sample_format,
                               channels=channels,
                               rate=sample_rate,
                               input=True,
                               input_device_index=device_index,
                               frames_per_buffer=chunk)
    except Exception as e:
        print("opening stream failed, falling back to default sample rate")
        dev_info = py_audio.get_device_info_by_index(device_index)

        #channel_number = int(dev_info['maxInputChannels'])
        recorded_sample_rate = int(dev_info['defaultSampleRate'])
        try:
            stream = py_audio.open(format=sample_format,
                                   channels=2,
                                   rate=recorded_sample_rate,
                                   input=True,
                                   input_device_index=device_index,
                                   frames_per_buffer=chunk)
        except Exception as e:
            print("opening stream failed, falling back to mono")
            # try again with mono
            is_mono = True
            stream = py_audio.open(format=sample_format,
                                   channels=1,
                                   rate=recorded_sample_rate,
                                   input=True,
                                   input_device_index=device_index,
                                   frames_per_buffer=chunk)

        needs_sample_rate_conversion = True

    return stream, needs_sample_rate_conversion, recorded_sample_rate, is_mono

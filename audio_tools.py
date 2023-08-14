import io
import threading
import time
import wave

import pyloudnorm
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
# filter can be sync_window, kaiser_fast, kaiser_best
def resample_audio(audio_chunk, recorded_sample_rate, target_sample_rate, target_channels=-1, is_mono=None, dtype="int16", filter="kaiser_best"):
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
    audio_data = resampy.resample(audio_data, recorded_sample_rate, target_sample_rate, filter=filter)
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


stop_flag = threading.Event()
audio_threads = []  # List to manage all audio threads
audio_thread_lock = threading.Lock()
audio_list_lock = threading.Lock()  # Lock to protect the audio_threads list


def play_stream(p=None, device=None, audio_data=None, chunk=1024, audio_format=2, channels=2, sample_rate=44100):
    try:
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=int(sample_rate),
                        output_device_index=device,
                        output=True)

        for i in range(0, len(audio_data), chunk * channels):
            if stop_flag.is_set():
                break
            stream.write(audio_data[i:i + chunk * channels].tobytes())

        stream.close()
    except Exception as e:
        print("Error playing audio: {}".format(e))


# play wav binary audio to device, converting audio sample_rate and channels if necessary
# audio can be bytes (in wav) or tensor
# tensor_sample_with is the sample width of the tensor (if audio is tensor and not bytes) [default is 4 bytes]
# tensor_channels is the number of channels of the tensor (if audio is tensor and not bytes) [default is 1 channel (mono)]
def play_audio(audio, device=None, source_sample_rate=44100, audio_device_channel_num=2, target_channels=2, is_mono=True, dtype="int16", tensor_sample_with=4, tensor_channels=1, secondary_device=None, stop_play=True):
    global audio_threads

    if stop_play:
        stop_audio()

    stop_flag.clear()

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

    channels = wf.getnchannels()

    # get audio sample width
    audio_sample_width = wf.getsampwidth()

    wf.close()

    # resample audio data
    audio_data = resample_audio(frame_data, source_sample_rate, closest_sample_rate, target_channels=target_channels, is_mono=is_mono, dtype=dtype)

    current_threads = []

    if secondary_device is not None:
        secondary_audio_thread = threading.Thread(target=play_stream, args=(
            p, secondary_device, audio_data, chunk,
            p.get_format_from_width(audio_sample_width),
            audio_device_channel_num,
            closest_sample_rate
        ))
        secondary_audio_thread.start()
        current_threads.append(secondary_audio_thread)

    # Open a .Stream object to write the WAV file to
    # 'output = True' indicates that the sound will be played rather than recorded
    main_audio_thread = threading.Thread(target=play_stream, args=(
        p, device, audio_data, chunk,
        p.get_format_from_width(audio_sample_width),
        audio_device_channel_num,
        closest_sample_rate
    ))
    main_audio_thread.start()
    current_threads.append(main_audio_thread)

    # Add the current threads to the global list
    with audio_list_lock:
        audio_threads.extend(current_threads)

    # Wait only for the threads that this invocation of play_audio has started
    for thread in current_threads:
        thread.join()

    # Cleanup: Remove threads that have completed from the global list
    with audio_list_lock:
        for thread in current_threads:
            if thread in audio_threads:
                audio_threads.remove(thread)

    p.terminate()
    stop_flag.clear()

    current_threads.clear()


def stop_audio():
    """
    Stop the audio playback.
    """
    with audio_thread_lock:
        stop_flag.set()
        with audio_list_lock:
            for thread in audio_threads:
                thread.join()  # Wait for each thread to complete its execution
            audio_threads.clear()  # Clear the list after all threads have completed


def start_recording_audio_stream(device_index=None, sample_format=pyaudio.paInt16, sample_rate=16000, channels=1, chunk=int(16000/10), py_audio=None, audio_processor=None):
    if py_audio is None:
        py_audio = pyaudio.PyAudio()

    needs_sample_rate_conversion = False
    is_mono = False

    recorded_sample_rate = sample_rate

    callback = None
    if audio_processor is not None and "callback" in dir(audio_processor):
        callback = audio_processor.callback

    try:
        if callback is not None:
            audio_processor.needs_sample_rate_conversion = needs_sample_rate_conversion
            audio_processor.recorded_sample_rate = recorded_sample_rate
            audio_processor.is_mono = is_mono

        stream = py_audio.open(format=sample_format,
                               channels=channels,
                               rate=sample_rate,
                               input=True,
                               input_device_index=device_index,
                               frames_per_buffer=chunk,
                               stream_callback=callback)
    except Exception as e:
        print("opening stream failed, falling back to default sample rate")
        dev_info = py_audio.get_device_info_by_index(device_index)

        #channel_number = int(dev_info['maxInputChannels'])
        recorded_sample_rate = int(dev_info['defaultSampleRate'])
        print("default sample rate: {}".format(recorded_sample_rate))
        needs_sample_rate_conversion = True
        try:
            if callback is not None:
                audio_processor.needs_sample_rate_conversion = needs_sample_rate_conversion
                audio_processor.recorded_sample_rate = recorded_sample_rate
                audio_processor.is_mono = is_mono

            stream = py_audio.open(format=sample_format,
                                   channels=2,
                                   rate=recorded_sample_rate,
                                   input=True,
                                   input_device_index=device_index,
                                   frames_per_buffer=chunk,
                                   stream_callback=callback)
        except Exception as e:
            print("opening stream failed, falling back to mono")
            # try again with mono
            is_mono = True

            if callback is not None:
                audio_processor.needs_sample_rate_conversion = needs_sample_rate_conversion
                audio_processor.recorded_sample_rate = recorded_sample_rate
                audio_processor.is_mono = is_mono
            stream = py_audio.open(format=sample_format,
                                   channels=1,
                                   rate=recorded_sample_rate,
                                   input=True,
                                   input_device_index=device_index,
                                   frames_per_buffer=chunk,
                                   stream_callback=callback)

    return stream, needs_sample_rate_conversion, recorded_sample_rate, is_mono


# Function to calculate LUFS
def calculate_lufs(audio, sample_rate):
    meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    return loudness


# Function to normalize the audio based on LUFS
def normalize_audio_lufs(audio, sample_rate, lower_threshold=-24.0, upper_threshold=-16.0, gain_factor=2.0, verbose=False):
    block_size_samples = int(sample_rate * 0.400)  # calculate block size in samples. (0.400 is the default block size of pyloudnorm)
    if len(audio) < block_size_samples:
        if verbose:
            print(f"audio is too short to calculate lufs")
        return audio, None

    lufs = calculate_lufs(audio, sample_rate)

    if verbose:
        print(f"LUFS: {lufs}")

    # If LUFS is lower than the lower threshold, increase volume
    if lufs < lower_threshold:
        if verbose:
            print(f"audio is too quiet, increasing volume")
        gain = (lower_threshold - lufs) / gain_factor
        audio = audio * np.power(10.0, gain/20.0)

    # If LUFS is higher than the upper threshold, decrease volume
    elif lufs > upper_threshold:
        if verbose:
            print(f"audio is too loud, decreasing volume")
        gain = (upper_threshold - lufs) * gain_factor
        audio = audio * np.power(10.0, gain/20.0)
    else:
        if verbose:
            print(f"audio is within the desired range")
        return audio, lufs

    # Limit audio values to [-1, 1] (this is important to avoid clipping when converting to 16-bit PCM)
    audio = np.clip(audio, -1, 1)

    return audio, lufs


def convert_audio_datatype_to_float(audio):
    """
    Convert audio data to floating-point representation.

    The function checks if the audio data is an integer type. If it is, the function converts the data to a floating-point
    range between -1.0 and 1.0. If the data is already in floating-point format, it leaves the data unchanged.

    Parameters:
    audio (numpy array): The audio data to be converted.

    Returns:
    audio (numpy array): The audio data in floating-point representation.
    """
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    return audio


def convert_audio_datatype_to_integer(audio, dtype=np.int16):
    """
    Convert audio data to integer representation.

    The function checks if the audio data is in floating-point format. If it is, the function converts the data to the
    specified integer type, scaling it based on the maximum value for that integer type. If the data is already in integer
    format, it leaves the data unchanged.

    Parameters:
    audio (numpy array): The audio data to be converted.
    dtype (numpy dtype, optional): The desired integer data type for the output. Defaults to np.int16.

    Returns:
    audio (numpy array): The audio data in integer representation.
    """
    if np.issubdtype(audio.dtype, np.floating):
        audio = (audio * np.iinfo(dtype).max).astype(dtype)
    return audio


# remove silence parts from audio. Make sure that keep_silence_length is less than or equal to half of
# max_silence_length, or else the entire silent section will be kept.
# fallback_silence_threshold is used if the audio is too short to calculate the LUFS
def remove_silence_parts(audio, sample_rate, silence_offset=-40.0, max_silence_length=30.0, keep_silence_length=0.20, fallback_silence_threshold=0.15, trim_silence_end=True, verbose=False):
    # Store the original data type
    original_dtype = audio.dtype

    # Convert audio to floating-point if necessary
    if np.issubdtype(original_dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(original_dtype).max

    # Calculate LUFS and define silence threshold
    block_size_samples = int(sample_rate * 0.400)  # calculate block size in samples. (0.400 is the default block size of pyloudnorm)
    if len(audio) >= block_size_samples:
        try:
            lufs = calculate_lufs(audio, sample_rate)
            silence_threshold = 10 ** ((lufs + silence_offset) / 20)
        except Exception as e:
            print(f"Could not calculate LUFS due to error: {str(e)}. Falling back to fixed silence threshold.")
            silence_threshold = fallback_silence_threshold
    else:
        silence_threshold = fallback_silence_threshold

    audio_abs = np.abs(audio)
    above_threshold = audio_abs > silence_threshold

    # Convert length parameters to number of samples
    max_silence_samples = int(max_silence_length * sample_rate)
    keep_silence_samples = int((keep_silence_length / 2.0) * sample_rate)

    last_silence_end = 0
    silence_start = None

    chunks = []

    for i, sample in enumerate(above_threshold):
        if not sample:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                silence_duration = i - silence_start
                if silence_duration > max_silence_samples:
                    # Keep silence at the start and end
                    start = max(0, last_silence_end)
                    end = min(len(audio), silence_start + keep_silence_samples)
                    chunks.append(audio[start:end])

                    # Define the start of the next chunk as the end of the current silence minus keep_silence_samples
                    last_silence_end = i - keep_silence_samples
                silence_start = None

    # Append the final chunk of audio after the last silence
    if last_silence_end < len(audio):
        start = last_silence_end
        end = len(audio)
        # If the audio ends in a silent section, trim the silence beyond keep_silence_samples
        if silence_start is not None and trim_silence_end:
            end = min(end, silence_start + keep_silence_samples)
        chunks.append(audio[start:end])

    if len(chunks) == 0:
        if verbose:
            print("No non-silent sections found in audio.")
        return np.array([])
    else:
        if verbose:
            print(f"found {len(chunks)} non-silent sections in audio")
        audio = np.concatenate(chunks)
        # Convert the audio back to the original data type if it was integer
        if np.issubdtype(original_dtype, np.integer):
            audio = (audio * np.iinfo(original_dtype).max).astype(original_dtype)
        return audio


# loads a wav file and resamples it to the target sample rate and converts it to mono if necessary
def load_wav_to_bytes(wav_path, target_sample_rate=16000):
    # Open the existing wav file
    with wave.open(wav_path, 'rb') as wave_file:
        params = wave_file.getparams()
        audio_bytes = wave_file.readframes(params.nframes)
        # get audio sample width
        audio_sample_width = wave_file.getframerate()
        channels = wave_file.getnchannels()

    return resample_audio(audio_bytes, audio_sample_width, target_sample_rate, target_channels=-1, is_mono=channels == 1, dtype="int16")

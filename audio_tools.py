import io
import threading
import traceback
import wave

import numpy
import pyloudnorm
# import resampy
import numpy as np
import platform
if platform.system() == 'Windows':
    import pyaudiowpatch as pyaudio
else:
    import pyaudio
import sounddevice as sd
import torch
from librosa.core.audio import resampy
from pydub import AudioSegment
from threading import Lock
import time
from scipy.io.wavfile import write as write_wav
from io import BytesIO

import Utilities


main_app_py_audio = pyaudio.PyAudio()


class PyAudioPool:
    def __init__(self, min_instances=2, max_unused_time=20):
        # max_unused_time in seconds
        self.pool = []
        self.lock = Lock()
        self.min_instances = min_instances
        self.max_unused_time = max_unused_time

    def acquire(self):
        with self.lock:
            for i, (p, _, in_use) in enumerate(self.pool):
                if not in_use:
                    self.pool[i] = (p, time.time(), True)
                    return p
            p = pyaudio.PyAudio()
            self.pool.append((p, time.time(), True))
            return p

    def release(self, p):
        with self.lock:
            for i, (instance, _, _) in enumerate(self.pool):
                if instance == p:
                    self.pool[i] = (instance, time.time(), False)
                    break

    def manage_unused(self):
        with self.lock:
            current_time = time.time()
            self.pool.sort(key=lambda x: x[1])  # Sort by last used time
            while len(self.pool) > self.min_instances:
                p, last_used_time, in_use = self.pool[0]
                if not in_use and (current_time - last_used_time) > self.max_unused_time:
                    p.terminate()
                    self.pool.pop(0)
                else:
                    break


pyaudio_pool = PyAudioPool()


def get_host_audio_api_names():
    audio = pyaudio.PyAudio()
    host_api_count = audio.get_host_api_count()
    host_api_names = {}
    for i in range(host_api_count):
        host_api_info = audio.get_host_api_info_by_index(i)
        host_api_names[i] = host_api_info["name"]
    return host_api_names


def get_default_audio_device_index_by_api(api, is_input=True):
    devices = sd.query_devices()
    api_info = sd.query_hostapis()
    host_api_index = None

    for i, host_api in enumerate(api_info):
        if api.lower() in host_api['name'].lower():
            host_api_index = i
            break

    if host_api_index is None:
        return None

    api_pyaudio_index, _ = get_audio_api_index_by_name(api)

    default_device_index = api_info[host_api_index]['default_input_device' if is_input else 'default_output_device']
    default_device_name = devices[default_device_index]['name']
    return get_audio_device_index_by_name_and_api(default_device_name, api_pyaudio_index, is_input)


def get_audio_device_index_by_name_and_api(name, api, is_input=True, default=None):
    audio = pyaudio.PyAudio()
    device_count = audio.get_device_count()
    for i in range(device_count):
        device_info = audio.get_device_info_by_index(i)
        device_name = device_info["name"]
        if isinstance(device_name, bytes):
            device_name = Utilities.safe_decode(device_name)
        if isinstance(name, bytes):
            name = Utilities.safe_decode(name)

        if device_info["hostApi"] == api and device_info[
            "maxInputChannels" if is_input else "maxOutputChannels"] > 0 and name in device_name:
            return i
    return default


def get_audio_api_index_by_name(name):
    audio = pyaudio.PyAudio()
    host_api_count = audio.get_host_api_count()
    # replace simple names to correct names
    if name.lower() == "winmm":
        name = "MME"
    if name.lower() == "directsound" or name.lower() == "dsound":
        name = "Windows DirectSound"
    if name.lower() == "wasapi":
        name = "Windows WASAPI"

    for i in range(host_api_count):
        host_api_info = audio.get_host_api_info_by_index(i)
        if name.lower() in host_api_info["name"].lower():
            return i, host_api_info["name"]
    return 0, ""


# resampy_audio function using the resampy library to resample audio data to a different sample rate and convert it to mono. (slower than resample, but less error prone to strange data)
# set target_channels to '-1' to average the left and right channels to create mono audio (default)
# set target_channels to '0' to extract the first channel (left channel) data
# set target_channels to '1' to extract the second channel (right channel) data
# set target_channels to '2' to keep stereo channels (or copy the mono channel to both channels if is_mono is True)
# to Convert the int16 numpy array to bytes use .tobytes()
# filter can be sync_window, kaiser_fast, kaiser_best
def resampy_audio(audio_chunk, recorded_sample_rate, target_sample_rate, target_channels=-1, is_mono=None,
                  dtype="int16", filter="kaiser_best"):
    audio_data_dtype = np.int16
    if dtype == "int16":
        audio_data_dtype = np.int16
    elif dtype == "float32":
        audio_data_dtype = np.float32

    # Convert the audio chunk to a numpy array
    if isinstance(audio_chunk, torch.Tensor):
        audio_chunk = audio_chunk.detach().cpu().numpy()

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
        # Also flatten the array and convert it back to int16 dtype
        audio_data = np.column_stack((audio_data, audio_data)).flatten()

    # Resample the audio data to the desired sample rate
    audio_data = resampy.resample(audio_data, recorded_sample_rate, target_sample_rate, filter=filter)
    # Convert the resampled data back to int16 dtype
    return np.asarray(audio_data, dtype=audio_data_dtype)


def _resample(smp, scale=1.0):
    """Resample a sound to be a different length

    Sample must be mono.  May take some time for longer sounds
    sampled at 44100 Hz.

    Keyword arguments:
    scale - scale factor for length of sound (2.0 means double length)

    """
    # f*ing cool, numpy can do this with one command
    # calculate new length of sample
    n = round(len(smp) * scale)
    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    return numpy.interp(
        numpy.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        numpy.linspace(0.0, 1.0, len(smp), endpoint=False),  # known positions
        smp,  # known data points
    )


def _interleave(left, right):
    """Given two separate arrays, return a new interleaved array

    This function is useful for converting separate left/right audio
    streams into one stereo audio stream.  Input arrays and returned
    array are Numpy arrays.

    See also: uninterleave()

    """
    return numpy.ravel(numpy.vstack((left, right)), order='F')


def _uninterleave(data):
    """Given a stereo array, return separate left and right streams

    This function converts one array representing interleaved left and
    right audio streams into separate left and right arrays.  The return
    value is a list of length two.  Input array and output arrays are all
    Numpy arrays.

    See also: interleave()

    """
    return data.reshape(2, len(data) // 2, order='F')


def resample_audio(audio_chunk, recorded_sample_rate, target_sample_rate, target_channels=None, input_channels=None, dtype="int16"):
    """
    Resample audio data and optionally convert between different channel configurations.

    :param audio_chunk: The raw audio data chunk as bytes, NumPy array, or PyTorch Tensor.
    :param recorded_sample_rate: The sample rate of the input audio.
    :param target_sample_rate: The desired target sample rate for the output.
    :param target_channels: The desired number of channels in the output. If None, keep original number of channels.
                            If positive integer, resample to that many channels.
    :param input_channels: Number of channels in the input audio data. If None, auto-detect from data shape.
    :param dtype: The desired data type of the output audio, either "int16", "int32", "int8", or "float32".
    :return: A NumPy array containing the resampled and potentially re-channelled audio data.
    """
    dtype_map = {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32
    }
    audio_data_dtype = dtype_map.get(dtype, np.int16)

    if isinstance(audio_chunk, torch.Tensor):
        audio_chunk = audio_chunk.detach().cpu().numpy()
    elif isinstance(audio_chunk, bytes):
        audio_chunk = np.frombuffer(audio_chunk, dtype=audio_data_dtype)

    if audio_chunk.size % input_channels != 0:
        raise ValueError("The total size of audio_chunk is not a multiple of input_channels.")

    if input_channels is not None:
        # Ensure that the audio data is reshaped to (-1, input_channels) if possible
        audio_data = audio_chunk.reshape(-1, input_channels)
    else:
        audio_data = audio_chunk
        input_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1

    # Calculate resampling scale
    scale = target_sample_rate / recorded_sample_rate

    # Process the channels
    resampled_channels = []
    for i in range(input_channels):
        channel_data = audio_data[:, i]
        if recorded_sample_rate != target_sample_rate:
            channel_data = _resample(channel_data, scale)
        resampled_channels.append(channel_data)

    # Adjust the number of channels to match target_channels if specified
    if target_channels is not None:
        adjusted_channels = resampled_channels
        if target_channels > input_channels:
            # Extend by repeating the last channel to fill the new channels
            extended_channels = resampled_channels + [resampled_channels[-1]] * (target_channels - input_channels)
            adjusted_channels = extended_channels
        elif target_channels < input_channels:
            # Reduce channels by averaging them in groups
            group_size = len(resampled_channels) // target_channels
            adjusted_channels = [np.mean(resampled_channels[i:i+group_size], axis=0) for i in range(0, len(resampled_channels), group_size)]
        resampled_channels = adjusted_channels

    # Interleave channels back into a single array if more than one channel
    if len(resampled_channels) > 1:
        resampled_audio_data = _interleave(*resampled_channels)
    else:
        resampled_audio_data = resampled_channels[0]

    return np.asarray(resampled_audio_data, dtype=audio_data_dtype)


def get_closest_sample_rate_of_device(device_index, target_sample_rate, fallback_sample_rate=44100):
    p = pyaudio.PyAudio()
    device_info = p.get_device_info_by_index(
        device_index if device_index is not None else p.get_default_output_device_info()["index"])
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

    wav_file = AudioSegment.from_file(audio, format="raw", frame_rate=sample_rate, channels=channels,
                                      sample_width=sample_width)

    buff = io.BytesIO()
    wav_file.export(buff, format="wav")

    del audio
    return buff


stop_flags = {}  # Dictionary to manage stop flags for each tag
audio_threads = []  # List to manage all audio threads
audio_thread_lock = threading.Lock()
audio_list_lock = threading.Lock()  # Lock to protect the audio_threads list


def play_stream(p=None, device=None, audio_data=None, chunk=1024, audio_format=2, channels=2, sample_rate=44100,
                tag="", dtype="int16"):

    dev_info = p.get_device_info_by_index(device)
    max_channels = int(dev_info['maxOutputChannels'])
    print("playback with channels: {}".format(max_channels))

    try:
        audio_data = resample_audio(audio_data, sample_rate, sample_rate, target_channels=max_channels,
                                    input_channels=channels, dtype=dtype)

        stream = p.open(format=audio_format,
                        channels=max_channels,
                        rate=int(sample_rate),
                        output_device_index=device,
                        output=True)

        for i in range(0, len(audio_data), chunk * max_channels):
            if stop_flags[tag].is_set():
                break
            stream.write(audio_data[i:i + chunk * max_channels].tobytes())

        stream.close()
    except Exception as e:
        print("Error playing audio: {}".format(e))
        traceback.print_exc()


# play wav binary audio to device, converting audio sample_rate and channels if necessary
# audio can be bytes (in wav), torch.Tensor or numpy array - audio data might need to be in int16, as python does not support float32 by default. use `audio_data = np.int16(wav_numpy * 32767)` to convert. (needs more testing)
# tensor_sample_with is the sample width of the tensor (if audio is tensor and not bytes) [default is 4 bytes]
# tensor_channels is the number of channels of the tensor (if audio is tensor and not bytes) [default is 1 channel (mono)]
def play_audio(audio, device=None, source_sample_rate=44100, audio_device_channel_num=2, target_channels=2,
               input_channels=1, dtype="int16", tensor_sample_with=4, tensor_channels=1, secondary_device=None,
               stop_play=True, tag=""):
    global audio_threads

    if stop_play:
        stop_audio(tag=tag)

    if tag not in stop_flags:
        stop_flags[tag] = threading.Event()
    stop_flags[tag].clear()

    if isinstance(audio, bytes):
        buff = _generate_binary_buffer(audio)
    elif isinstance(audio, numpy.ndarray):
        buff = io.BytesIO()
        write_wav(buff, source_sample_rate, audio)
        buff.seek(0)
    elif isinstance(audio, torch.Tensor):
        buff = convert_tensor_to_wav_buffer(audio, sample_rate=source_sample_rate, channels=tensor_channels,
                                            sample_width=tensor_sample_with)
    else:
        raise ValueError("Unsupported audio format. Please provide bytes, numpy array, or torch tensor.")

    # Set chunk size of 1024 samples per data frame
    chunk = 1024

    # Open the sound file
    wf = wave.open(buff, 'rb')

    # Create an interface to PortAudio
    # p = pyaudio.PyAudio()
    p = pyaudio_pool.acquire()

    # Find the closest supported sample rate to the original sample rate
    closest_sample_rate = get_closest_sample_rate_of_device(device, wf.getframerate())

    # Read all audio data and resample if necessary
    frame_data = wf.readframes(wf.getnframes())

    sound_file_channels = wf.getnchannels()

    # get audio sample width
    audio_sample_width = wf.getsampwidth()

    wf.close()

    # resample audio data
    audio_data = resample_audio(frame_data, source_sample_rate, closest_sample_rate, target_channels=target_channels,
                                input_channels=input_channels, dtype=dtype)

    current_threads = []

    if secondary_device is not None:
        secondary_audio_thread = threading.Thread(target=play_stream, args=(
            p, secondary_device, audio_data, chunk,
            p.get_format_from_width(audio_sample_width),
            audio_device_channel_num,
            closest_sample_rate,
            tag,
            dtype
        ))
        secondary_audio_thread.start()
        current_threads.append((secondary_audio_thread, tag))

    # Open a .Stream object to write the WAV file to
    # 'output = True' indicates that the sound will be played rather than recorded
    main_audio_thread = threading.Thread(target=play_stream, args=(
        p, device, audio_data, chunk,
        p.get_format_from_width(audio_sample_width),
        audio_device_channel_num,
        closest_sample_rate,
        tag,
        dtype
    ))
    main_audio_thread.start()
    current_threads.append((main_audio_thread, tag))

    # Add the current threads to the global list
    with audio_list_lock:
        audio_threads.extend(current_threads)

    # Wait only for the threads that this invocation of play_audio has started
    for thread, _ in current_threads:
        thread.join()

    # Cleanup: Remove threads that have completed from the global list
    with audio_list_lock:
        for thread, _ in current_threads:
            if (thread, tag) in audio_threads:
                audio_threads.remove((thread, tag))

    # p.terminate()
    pyaudio_pool.release(p)
    pyaudio_pool.manage_unused()

    if tag in stop_flags:
        stop_flags[tag].clear()

    current_threads.clear()


def stop_audio(tag=None):
    """
    Stop the audio playback with a given tag.
    If no tag is provided, all audio threads will be stopped.
    """
    global audio_threads
    with audio_thread_lock:
        if tag:
            if tag in stop_flags:
                stop_flags[tag].set()
        else:
            for flag in stop_flags.values():
                flag.set()
        with audio_list_lock:
            for thread, t in audio_threads:
                if tag is None or tag == t:
                    thread.join()
            if tag is None:
                audio_threads.clear()
            else:
                audio_threads = [(thread, t) for thread, t in audio_threads if t != tag]


def is_audio_playing(tag=None):
    global audio_threads
    with audio_list_lock:
        if tag is None:
            return len(audio_threads) > 0
        else:
            return any(t == tag for _, t in audio_threads)

def calculate_chunk_size(recorded_sample_rate, target_sample_rate, chunk):
    # Calculate the resampling ratio
    resampling_ratio = recorded_sample_rate / target_sample_rate

    # Calculate the initial chunk size needed to achieve the target chunk size after resampling
    return int(chunk * resampling_ratio)

def start_recording_audio_stream(device_index=None, sample_format=pyaudio.paInt16, sample_rate=16000, channels=1,
                                 chunk=512, py_audio=None, audio_processor=None):
    if py_audio is None:
        py_audio = pyaudio.PyAudio()

    needs_sample_rate_conversion = False
    num_of_channels = 2
    recorded_sample_rate = sample_rate

    callback = None
    if audio_processor is not None and hasattr(audio_processor, "callback"):
        callback = audio_processor.callback

    initial_chunk_size = calculate_chunk_size(recorded_sample_rate, sample_rate, chunk)

    try:
        # First attempt with user-specified settings
        stream = py_audio.open(format=sample_format,
                               channels=channels,
                               rate=recorded_sample_rate,
                               input=True,
                               input_device_index=device_index,
                               frames_per_buffer=initial_chunk_size,
                               stream_callback=callback)
    except Exception as e:
        print(f"Failed to open stream with channels={channels} and rate={sample_rate}: {e}")
        print("Attempting to use default device settings...")

        dev_info = py_audio.get_device_info_by_index(device_index)
        recorded_sample_rate = int(dev_info['defaultSampleRate'])
        needs_sample_rate_conversion = (sample_rate != recorded_sample_rate)

        initial_chunk_size = calculate_chunk_size(recorded_sample_rate, sample_rate, chunk)

        print(f"Max channels supported by the device: {int(dev_info['maxInputChannels'])}")
        print(f"default SampleRate supported by the device: {int(dev_info['defaultSampleRate'])}")

        try:
            # First fallback with 2 channels
            stream = py_audio.open(format=sample_format,
                                   channels=2,
                                   rate=recorded_sample_rate,
                                   input=True,
                                   input_device_index=device_index,
                                   frames_per_buffer=initial_chunk_size,
                                   stream_callback=callback)
        except Exception as e:
            print(f"Failed with 2 channels at default rate {recorded_sample_rate}: {e}")
            initial_chunk_size = calculate_chunk_size(recorded_sample_rate, sample_rate, chunk)
            try:
                # Second fallback with 1 channel (mono)
                stream = py_audio.open(format=sample_format,
                                       channels=1,
                                       rate=recorded_sample_rate,
                                       input=True,
                                       input_device_index=device_index,
                                       frames_per_buffer=initial_chunk_size,
                                       stream_callback=callback)
                num_of_channels = 1
            except Exception as e:
                print(f"Failed with 1 channel at default rate {recorded_sample_rate}: {e}")
                # Third fallback with max channels supported by the device
                max_channels = int(dev_info['maxInputChannels'])
                initial_chunk_size = calculate_chunk_size(recorded_sample_rate, sample_rate, chunk)
                try:
                    stream = py_audio.open(format=sample_format,
                                           channels=max_channels,
                                           rate=recorded_sample_rate,
                                           input=True,
                                           input_device_index=device_index,
                                           frames_per_buffer=initial_chunk_size,
                                           stream_callback=callback)
                    num_of_channels = max_channels
                except Exception as e:
                    print(f"Failed with max channels ({max_channels}) at default rate {recorded_sample_rate}: {e}")
                    raise Exception("Unable to open any audio stream.")

    # Update the audio_processor with the final stream settings
    if callback is not None:
        audio_processor.needs_sample_rate_conversion = needs_sample_rate_conversion
        audio_processor.recorded_sample_rate = recorded_sample_rate
        audio_processor.input_channel_num = num_of_channels

    return stream, needs_sample_rate_conversion, recorded_sample_rate, num_of_channels


# Function to calculate LUFS
def calculate_lufs(audio, sample_rate):
    meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    return loudness


# Function to normalize the audio based on LUFS
def normalize_audio_lufs(audio, sample_rate, lower_threshold=-24.0, upper_threshold=-16.0, gain_factor=2.0,
                         verbose=False):
    block_size_samples = int(
        sample_rate * 0.400)  # calculate block size in samples. (0.400 is the default block size of pyloudnorm)
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
        audio = audio * np.power(10.0, gain / 20.0)

    # If LUFS is higher than the upper threshold, decrease volume
    elif lufs > upper_threshold:
        if verbose:
            print(f"audio is too loud, decreasing volume")
        gain = (upper_threshold - lufs) * gain_factor
        audio = audio * np.power(10.0, gain / 20.0)
    else:
        if verbose:
            print(f"audio is within the desired range")
        return audio, lufs

    # Limit audio values to [-1, 1] (this is important to avoid clipping when converting to 16-bit PCM)
    audio = np.clip(audio, -1, 1)

    return audio, lufs


def convert_audio_datatype_to_float(audio, dtype=np.float32):
    """
    Convert audio data to floating-point representation.

    The function checks if the audio data is an integer type. If it is, the function converts the data to a floating-point
    range between -1.0 and 1.0. If the data is already in floating-point format, it leaves the data unchanged.

    Parameters:
    audio (numpy array): The audio data to be converted.
    dtype (numpy dtype, optional): The desired float data type for the output. (Defaults to np.float32)

    Returns:
    audio (numpy array): The audio data in floating-point representation.
    """
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max + 1  # Use +1 to handle -32768 to 32767 symmetrically for int16
        audio = audio.astype(dtype) / max_val
        if np.any((audio < -1) | (audio > 1)):
            print("Warning: Clipping detected after normalization")
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
        # Clip audio to ensure it remains within the valid range
        np.clip(audio, -1, 1, out=audio)

        audio = (audio * np.iinfo(dtype).max).astype(dtype)
    return audio


# remove silence parts from audio. Make sure that keep_silence_length is less than or equal to half of
# max_silence_length, or else the entire silent section will be kept.
# fallback_silence_threshold is used if the audio is too short to calculate the LUFS
def remove_silence_parts(audio, sample_rate, silence_offset=-40.0, max_silence_length=30.0, keep_silence_length=0.20,
                         fallback_silence_threshold=0.15, trim_silence_end=True, verbose=False):
    # Store the original data type
    original_dtype = audio.dtype

    # Convert audio to floating-point if necessary
    if np.issubdtype(original_dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(original_dtype).max

    # Calculate LUFS and define silence threshold
    block_size_samples = int(
        sample_rate * 0.400)  # calculate block size in samples. (0.400 is the default block size of pyloudnorm)
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
        audio_sample_rate = wave_file.getframerate()
        audio_sample_width = wave_file.getsampwidth()
        format_type = wave_file.getcomptype()
        channels = wave_file.getnchannels()

        dtype = "int16"
        # Determine the dtype based on sample width and format type
        if audio_sample_width == 1:
            dtype = "int8"
        elif audio_sample_width == 2:
            dtype = "int16"
        elif audio_sample_width == 4:
            if format_type == 'NONE':
                dtype = "int32"
            elif format_type == 'FLOAT':
                dtype = "float32"
            else:
                print("Unsupported audio format for sample_with=4: " + str(format_type))
        else:
            print("Unsupported audio sample width: " + str(audio_sample_width))

    return resample_audio(audio_bytes, audio_sample_rate, target_sample_rate, target_channels=1,
                          input_channels=channels, dtype=dtype)


def numpy_array_to_wav_bytes(audio: np.ndarray, sample_rate: int = 22050) -> BytesIO:
    buff = io.BytesIO()
    write_wav(buff, sample_rate, audio)
    buff.seek(0)
    return buff


def audio_bytes_to_wav(audio_bytes, channels=1, sample_rate=16000, sample_width=2):
    final_wavfile = io.BytesIO()
    wavefile = wave.open(final_wavfile, 'wb')
    wavefile.setnchannels(channels)
    wavefile.setsampwidth(sample_width)
    wavefile.setframerate(sample_rate)
    wavefile.writeframes(audio_bytes)

    final_wavfile.seek(0)
    return_data = final_wavfile.read()
    wavefile.close()
    return return_data


def wav_bytes_to_numpy_array(wav_bytes):
    """
    Converts a WAV bytes object to a NumPy array.

    Args:
        wav_bytes (bytes): The bytes object containing WAV file data.

    Returns:
        np.ndarray: A NumPy array representing the audio data.
    """
    # Use an io.BytesIO object as the file for wave to read from.
    with io.BytesIO(wav_bytes) as wav_file:
        with wave.open(wav_file, 'rb') as wav_reader:
            # Extract audio data
            n_channels = wav_reader.getnchannels()
            sample_width = wav_reader.getsampwidth()
            frame_rate = wav_reader.getframerate()
            n_frames = wav_reader.getnframes()
            frames = wav_reader.readframes(n_frames)

            # Determine the correct data type for the numpy array
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            if sample_width in dtype_map:
                dtype = dtype_map[sample_width]
            else:
                print("Unsupported sample width")
                return None

            # Convert audio bytes to a NumPy array
            audio_array = np.frombuffer(frames, dtype=dtype)

            # If stereo (or more channels), reshape the array
            if n_channels > 1:
                audio_array = audio_array.reshape(-1, n_channels)

            return audio_array


def split_audio_with_padding(audio_bytes, chunk_size, bytes_per_sample = 2, merge_to_bytes=True):
    """
    Args:
        audio_bytes:
        chunk_size:
        bytes_per_sample: 1 byte for 8-bit audio, 2 bytes for 16-bit, 3 bytes for 24-bit
        merge_to_bytes:

    Returns:
        bytes or list of bytes if merge_to_bytes is true
    """

    bytes_per_frame = chunk_size * bytes_per_sample

    # Initialize the list to hold audio frames
    audio_frames = []

    # Iterate over the audio bytes to split into frames
    for i in range(0, len(audio_bytes), bytes_per_frame):
        frame = audio_bytes[i:i+bytes_per_frame]

        # If the frame is shorter than bytes_per_frame, pad it with zeros
        if len(frame) < bytes_per_frame:
            frame += b'\x00' * (bytes_per_frame - len(frame))

        audio_frames.append(frame)

    if merge_to_bytes:
        return b''.join(audio_frames)

    return audio_frames


def change_volume(audio_data, volume_factor=1, dtype=None):
    """
    Adjusts the volume of the audio data.

    Args:
        audio_data: The audio data, either as bytes, numpy array or torch.tensor.
        volume_factor: The factor by which to adjust the volume. Greater than 1 increases volume, less than 1 decreases it.
        dtype: The data type of the audio samples (e.g., np.int16, np.float32). Can be set to none if data is numpy array.

    Returns:
        The audio data with adjusted volume, in the same format as the input.
    """

    # no volume change, just return audio data
    if volume_factor == 1.0:
        return audio_data

    if isinstance(audio_data, bytes):
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=dtype)
    elif isinstance(audio_data, np.ndarray):
        audio_array = audio_data
        if dtype is None:
            dtype = audio_array.dtype
    elif isinstance(audio_data, torch.Tensor):
        # Convert torch.Tensor to numpy array
        audio_array = audio_data.detach().cpu().numpy()
        if dtype is None:
            dtype = audio_array.dtype
    else:
        raise ValueError("Unsupported audio format. Please provide bytes or numpy array.")

    # Adjust the volume
    audio_array = audio_array * volume_factor

    # Ensure the audio data stays within the valid range
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        min_val = np.iinfo(dtype).min
        audio_array = np.clip(audio_array, min_val, max_val)
    elif np.issubdtype(dtype, np.floating):
        audio_array = np.clip(audio_array, -1.0, 1.0)

    # Convert back to the original format if needed
    if isinstance(audio_data, bytes):
        return audio_array.astype(dtype).tobytes()
    elif isinstance(audio_data, torch.Tensor):
        return torch.from_numpy(audio_array.astype(dtype))
    else:
        return audio_array.astype(dtype)

# ======================================
# buffered audio streaming playback
# ======================================


class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = bytearray(capacity)
        self.head = 0  # Read pointer
        self.tail = 0  # Write pointer
        self.count = 0  # Number of items in the buffer

    def append(self, data):
        for byte in data:
            self.buffer[self.tail] = byte
            self.tail = (self.tail + 1) % self.capacity
            if self.count < self.capacity:
                self.count += 1
            else:
                self.head = (self.head + 1) % self.capacity  # Move head forward if overwriting

    def read(self, size):
        if size > self.count:
            size = self.count
        data = bytearray(size)
        for i in range(size):
            data[i] = self.buffer[self.head]
            self.head = (self.head + 1) % self.capacity
        self.count -= size
        return data

    def get_available_size(self):
        return self.capacity - self.count


class CircularByteBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = bytearray(size)
        self.head = 0
        self.tail = 0
        self.full = False

    def append(self, data):
        if not isinstance(data, bytes):
            raise TypeError("Data must be of type bytes")

        for byte in data:
            self.buffer[self.head] = byte
            if self.full:
                self.tail = (self.tail + 1) % self.size
            self.head = (self.head + 1) % self.size
            self.full = self.head == self.tail

    def get_full_buffer(self):
        if not self.full and self.head == self.tail:
            return bytes()
        elif self.full:
            return bytes(self.buffer[self.tail:] + self.buffer[:self.head])
        else:
            return bytes(self.buffer[self.tail:self.head])

    def get_ordered_buffer(self):
        if self.full or self.head < self.tail:
            return bytes(self.buffer[self.tail:] + self.buffer[:self.head])
        else:
            return bytes(self.buffer[self.tail:self.head])

    def is_full(self):
        return self.full

    def is_empty(self):
        return not self.full and self.head == self.tail

    def clear(self):
        self.head = 0
        self.tail = 0
        self.full = False
        self.buffer = bytearray(self.size)


class QueueBuffer:
    def __init__(self, element_size):
        self.buffer = bytearray()
        self.element_size = element_size

    def append(self, data):
        self.buffer += data

    def read(self, size):
        actual_read_size = min(size, len(self.buffer))
        data = self.buffer[:actual_read_size]
        self.buffer = self.buffer[actual_read_size:]
        return data

    def get_available_size(self):
        return len(self.buffer) - (len(self.buffer) % self.element_size)


class AudioStreamer:
    def __init__(self, device_index=0, source_sample_rate=44100, buffer_size=2048, input_channels=None, playback_channels=2, dtype=np.int16, tag=""):
        self.device_index = device_index
        self.source_sample_rate = source_sample_rate
        self.input_channels = input_channels
        self.playback_channels = playback_channels
        self.dtype = dtype
        self.np_dtype = np.int16 if self.dtype == "int16" else np.float32
        self.buffer_size = buffer_size * (np.dtype(self.dtype).itemsize * self.playback_channels)  # buffer size in bytes for resampled size
        self.buffer = QueueBuffer(self.buffer_size)
        self.tag = tag
        self.playback_thread = None
        self.lock = threading.Lock()
        self.stop_playing_timeout = 2.0
        # PyAudio instance and stream will be set in init_stream
        self.p = None
        self.stream = None
        self.init_stream(source_sample_rate)
        self.before_playback_hook_func = None
        self.verbose = False

    def set_before_playback_hook(self, hook_func):
        self.before_playback_hook_func = hook_func

    def init_stream(self, desired_sample_rate):
        if self.p is not None:
            pyaudio_pool.release(self.p)
        self.p = pyaudio_pool.acquire()
        print("dtype itemsize: ", np.dtype(self.dtype).itemsize)
        print("format width: ", self.p.get_format_from_width(np.dtype(self.dtype).itemsize))
        self.actual_sample_rate = get_closest_sample_rate_of_device(self.device_index, desired_sample_rate)
        print("Actual sample rate: ", int(self.actual_sample_rate))
        print("Playback channels: ", self.playback_channels)
        self.stream = self.p.open(format=self.p.get_format_from_width(np.dtype(self.dtype).itemsize),
                                  channels=self.playback_channels, rate=int(self.actual_sample_rate), output=True,
                                  output_device_index=self.device_index)

    def add_audio_chunk(self, chunk):
        with self.lock:
            if self.verbose:
                print("adding audio chunk of size: ", len(chunk))
            self.buffer.append(chunk)
            if self.buffer.get_available_size() >= self.buffer_size and (self.playback_thread is None or not self.playback_thread.is_alive()):
                if self.verbose:
                    print("Starting playback thread on add_chunk")
                self.start_playback()

    def start_playback(self):
        def playback_loop():
            data_accumulated = bytearray()
            last_data_time = time.time()
            has_data_been_written = False
            while True:
                with self.lock:
                    available_size = self.buffer.get_available_size()
                    if self.verbose:
                        print("Available size: ", available_size)
                    if available_size > 0:
                        # Read the maximum available data that fits into full elements
                        buffer_entry = self.buffer.read(available_size)
                        if self.verbose:
                            print("Buffer entry size: ", len(buffer_entry))
                        data_accumulated += buffer_entry
                        if self.verbose:
                            print("Data accumulated size: ", len(data_accumulated))

                while len(data_accumulated) >= self.buffer_size:
                    data_to_play = bytes(data_accumulated[:self.buffer_size])
                    data_accumulated = data_accumulated[self.buffer_size:]
                    if self.verbose:
                        print("Playing audio chunk of size 1: ", len(data_to_play))
                    last_data_time = time.time()

                    if isinstance(data_to_play, np.ndarray) or isinstance(data_to_play, bytes):
                        if self.before_playback_hook_func is not None:
                            data_to_play = self.before_playback_hook_func(data_to_play, self.source_sample_rate)

                        data_to_play = np.frombuffer(data_to_play, dtype=self.np_dtype) if isinstance(data_to_play, bytes) else data_to_play
                        data_to_play = resample_audio(data_to_play, self.source_sample_rate, self.actual_sample_rate, target_channels=self.playback_channels, input_channels=self.input_channels, dtype=self.dtype)
                        data_to_play = data_to_play.tobytes()
                        if self.verbose:
                            print("Resampled audio chunk size: ", len(data_to_play))
                    self.stream.write(data_to_play)
                    has_data_been_written = True

                with self.lock:
                    if available_size == 0 and len(data_accumulated) > 0:
                        # If this is the last chunk and it's not full, pad it
                        padding_length = self.buffer_size - len(data_accumulated)
                        padding = b'\x00' * padding_length
                        data_to_play = bytes(data_accumulated) + padding
                        if self.verbose:
                            print("Playing audio chunk of size 2: ", len(data_to_play))
                        last_data_time = time.time()
                        self.stream.write(data_to_play)
                        has_data_been_written = True
                        data_accumulated = bytearray()  # Clear after writing

                    if available_size == 0 and len(data_accumulated) == 0:
                        time.sleep(0.01)
                        # Check if playback should stop due to inactivity
                        if has_data_been_written and time.time() - last_data_time > self.stop_playing_timeout:
                            break  # Exiting the loop stops playback

            if self.verbose:
                print("Stopping playback due to inactivity.")
            with audio_list_lock:
                if (self.playback_thread, self.tag) in audio_threads:
                    audio_threads.remove((self.playback_thread, self.tag))
            self.playback_thread = None

        if self.verbose:
            print("Starting playback thread")
        self.playback_thread = threading.Thread(target=playback_loop, name=self.tag)
        with audio_list_lock:
            audio_threads.append((self.playback_thread, self.tag))
        self.playback_thread.start()

    def stop(self):
        with self.lock:
            with audio_list_lock:
                if (self.playback_thread, self.tag) in audio_threads:
                    audio_threads.remove((self.playback_thread, self.tag))
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join()
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            pyaudio_pool.release(self.p)
            self.p = None
            self.playback_thread = None
            print("Stopped audio streamer")

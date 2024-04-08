import io
import threading
import wave

import numpy
import pyloudnorm
# import resampy
import numpy as np
import pyaudio
import torch
from librosa.core.audio import resampy
from pydub import AudioSegment
from threading import Lock
import time
from scipy.io.wavfile import write as write_wav
from io import BytesIO


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


def resample_audio(audio_chunk, recorded_sample_rate, target_sample_rate, target_channels=-1, is_mono=None,
                   dtype="int16"):
    """
    Resample audio data and optionally convert between stereo and mono.

    :param audio_chunk: The raw audio data chunk as bytes, NumPy array or PyTorch Tensor.
    :param recorded_sample_rate: The sample rate of the input audio.
    :param target_sample_rate: The desired target sample rate for the output.
    :param target_channels: The desired number of channels in the output.
        - '-1': Average the left and right channels to create mono audio. (default)
        - '0': Extract the first channel (left channel) data.
        - '1': Extract the second channel (right channel) data.
        - '2': Keep stereo channels (or copy the mono channel to both channels if is_mono is True).
    :param is_mono: Specify whether the input audio is mono. If None, it will be determined from the shape of the audio data.
    :param dtype: The desired data type of the output audio, either "int16" or "float32".
    :return: A NumPy array containing the resampled audio data.
    """
    # Determine the data type for audio data
    audio_data_dtype = np.float32
    if dtype == "int8":
        audio_data_dtype = np.int8
    elif dtype == "int16":
        audio_data_dtype = np.int16
    elif dtype == "int32":
        audio_data_dtype = np.int32
    elif dtype == "float32":
        audio_data_dtype = np.float32

    # Convert the audio chunk to a numpy array
    if isinstance(audio_chunk, torch.Tensor):
        audio_chunk = audio_chunk.detach().cpu().numpy()

    audio_data = np.frombuffer(audio_chunk, dtype=audio_data_dtype)

    # Determine if the audio is mono or stereo; assume mono if the shape has one dimension
    if is_mono is None:
        is_mono = len(audio_data.shape) == 1

    # If stereo, reshape the data to have two columns (left and right channels)
    if not is_mono:
        audio_data = audio_data.reshape(-1, 2)

    # Handle channel conversion based on the target_channels parameter
    # -1 means converting stereo to mono by taking the mean of both channels
    # 0 or 1 means selecting one of the stereo channels
    # 2 means duplicating the mono channel to make it stereo
    if target_channels == -1 and not is_mono:
        audio_data = audio_data.mean(axis=1)
    elif target_channels in [0, 1] and not is_mono:
        audio_data = audio_data[:, target_channels]
    elif target_channels == 2 and is_mono:
        audio_data = _interleave(audio_data, audio_data)

    # Calculate the scaling factor for resampling
    scale = target_sample_rate / recorded_sample_rate

    # Perform resampling based on whether the audio is mono or stereo
    # If mono or selected one channel, use _resample directly
    # If stereo, split into left and right, resample separately, then interleave
    if is_mono or target_channels in [0, 1, -1]:
        audio_data = _resample(audio_data, scale)
    else:  # Stereo
        left, right = _uninterleave(audio_data)
        left_resampled = _resample(left, scale)
        right_resampled = _resample(right, scale)
        audio_data = _interleave(left_resampled, right_resampled)

    # Return the resampled audio data with the specified dtype
    return np.asarray(audio_data, dtype=audio_data_dtype)


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

    return buff


stop_flags = {}  # Dictionary to manage stop flags for each tag
audio_threads = []  # List to manage all audio threads
audio_thread_lock = threading.Lock()
audio_list_lock = threading.Lock()  # Lock to protect the audio_threads list


def play_stream(p=None, device=None, audio_data=None, chunk=1024, audio_format=2, channels=2, sample_rate=44100,
                tag=""):
    try:
        # frames_per_buffer = chunk * channels  # experiment with this value

        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=int(sample_rate),
                        output_device_index=device,
                        output=True)
        # frames_per_buffer=frames_per_buffer)

        for i in range(0, len(audio_data), chunk * channels):
            if stop_flags[tag].is_set():
                break
            stream.write(audio_data[i:i + chunk * channels].tobytes())

        stream.close()
    except Exception as e:
        print("Error playing audio: {}".format(e))


# play wav binary audio to device, converting audio sample_rate and channels if necessary
# audio can be bytes (in wav) or tensor
# tensor_sample_with is the sample width of the tensor (if audio is tensor and not bytes) [default is 4 bytes]
# tensor_channels is the number of channels of the tensor (if audio is tensor and not bytes) [default is 1 channel (mono)]
def play_audio(audio, device=None, source_sample_rate=44100, audio_device_channel_num=2, target_channels=2,
               is_mono=True, dtype="int16", tensor_sample_with=4, tensor_channels=1, secondary_device=None,
               stop_play=True, tag=""):
    global audio_threads

    if stop_play:
        stop_audio(tag=tag)

    if tag not in stop_flags:
        stop_flags[tag] = threading.Event()
    stop_flags[tag].clear()

    if isinstance(audio, bytes):
        buff = _generate_binary_buffer(audio)
    else:
        buff = convert_tensor_to_wav_buffer(audio, sample_rate=source_sample_rate, channels=tensor_channels,
                                            sample_width=tensor_sample_with)

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

    channels = wf.getnchannels()

    # get audio sample width
    audio_sample_width = wf.getsampwidth()

    wf.close()

    # resample audio data
    audio_data = resample_audio(frame_data, source_sample_rate, closest_sample_rate, target_channels=target_channels,
                                is_mono=is_mono, dtype=dtype)

    current_threads = []

    if secondary_device is not None:
        secondary_audio_thread = threading.Thread(target=play_stream, args=(
            p, secondary_device, audio_data, chunk,
            p.get_format_from_width(audio_sample_width),
            audio_device_channel_num,
            closest_sample_rate,
            tag
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
        tag
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


def start_recording_audio_stream(device_index=None, sample_format=pyaudio.paInt16, sample_rate=16000, channels=1,
                                 chunk=int(16000 / 10), py_audio=None, audio_processor=None):
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

        # channel_number = int(dev_info['maxInputChannels'])
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

    return resample_audio(audio_bytes, audio_sample_rate, target_sample_rate, target_channels=-1,
                          is_mono=channels == 1, dtype=dtype)


def numpy_array_to_wav_bytes(audio: np.ndarray, sample_rate: int = 22050) -> BytesIO:
    buff = io.BytesIO()
    write_wav(buff, sample_rate, audio)
    buff.seek(0)
    return buff


def audio_bytes_to_wav(audio_bytes, channels=1, sample_rate=16000):
    final_wavfile = io.BytesIO()
    wavefile = wave.open(final_wavfile, 'wb')
    wavefile.setnchannels(channels)
    wavefile.setsampwidth(2)
    wavefile.setframerate(sample_rate)
    wavefile.writeframes(audio_bytes)

    final_wavfile.seek(0)
    return_data = final_wavfile.read()
    wavefile.close()
    return return_data


def split_audio_with_padding(audio_bytes, chunk_size):
    # Assuming 16-bit (2 bytes) audio, calculate bytes per frame
    bytes_per_sample = 2
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

    return audio_frames

# ======================================
# buffered audio streaming playback
# ======================================


class CircularBuffer:
    def __init__(self, element_size):
        self.buffer = bytearray()
        self.element_size = element_size

    def append(self, data):
        self.buffer += data

    def read(self, size):
        data = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return data

    def get_available_size(self):
        return len(self.buffer) - (len(self.buffer) % self.element_size)


class AudioStreamer:
    def __init__(self, device_index=0, source_sample_rate=44100, buffer_size=2048, is_mono=None, playback_channels=2, dtype=np.int16, tag=""):
        self.device_index = device_index
        self.source_sample_rate = source_sample_rate
        self.is_mono = is_mono
        self.buffer_size = buffer_size
        self.playback_channels = playback_channels
        self.dtype = dtype
        self.buffer = CircularBuffer(self.buffer_size)
        self.tag = tag
        self.playback_thread = None
        self.lock = threading.Lock()
        # PyAudio instance and stream will be set in init_stream
        self.p = None
        self.stream = None
        self.init_stream(source_sample_rate)
        self.before_playback_hook_func = None

    def set_before_playback_hook(self, hook_func):
        self.before_playback_hook_func = hook_func

    def init_stream(self, desired_sample_rate):
        self.p = pyaudio_pool.acquire()
        self.actual_sample_rate = get_closest_sample_rate_of_device(self.device_index, desired_sample_rate)
        self.stream = self.p.open(format=self.p.get_format_from_width(np.dtype(self.dtype).itemsize),
                                  channels=self.playback_channels, rate=int(self.actual_sample_rate), output=True,
                                  output_device_index=self.device_index)

    def add_audio_chunk(self, chunk):
        with self.lock:
            if isinstance(chunk, np.ndarray) or isinstance(chunk, bytes):
                dtype = np.int16 if self.dtype == "int16" else np.float32

                if self.before_playback_hook_func is not None:
                    chunk = self.before_playback_hook_func(chunk, self.source_sample_rate)

                chunk = np.frombuffer(chunk, dtype=dtype) if isinstance(chunk, bytes) else chunk
                chunk = resample_audio(chunk, self.source_sample_rate, self.actual_sample_rate, target_channels=self.playback_channels, is_mono=self.is_mono, dtype=self.dtype)
                chunk = chunk.tobytes()
            self.buffer.append(chunk)
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.start_playback()

    def start_playback(self):
        def playback_loop():
            data_accumulated = bytearray()
            while True:
                with self.lock:  # Ensure thread-safe access to the buffer
                    available_size = self.buffer.get_available_size()
                    if available_size > 0:
                        data_accumulated += self.buffer.read(available_size)

                # Process and play back if there's enough data or if it's the final chunk
                while len(data_accumulated) >= self.buffer_size or (available_size == 0 and len(data_accumulated) > 0):
                    if len(data_accumulated) >= self.buffer_size:
                        data_to_play = bytes(data_accumulated[:self.buffer_size])
                        data_accumulated = data_accumulated[self.buffer_size:]
                    else:  # This is the last chunk, potentially smaller than buffer_size
                        # Pad the final chunk to ensure it's the correct size
                        padding_length = self.buffer_size - len(data_accumulated)
                        padding = b'\x00' * padding_length
                        data_to_play = bytes(data_accumulated) + padding
                        data_accumulated = bytearray()  # Clear the accumulator

                    self.stream.write(data_to_play)

                if available_size == 0 and len(data_accumulated) == 0:
                    break  # Exit loop if no data is left

            with audio_list_lock:
                if (self.playback_thread, self.tag) in audio_threads:
                    audio_threads.remove((self.playback_thread, self.tag))
            self.playback_thread = None

        self.playback_thread = threading.Thread(target=playback_loop, name=self.tag)
        with audio_list_lock:
            audio_threads.append((self.playback_thread, self.tag))
        self.playback_thread.start()

    def stop(self):
        with self.lock:
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join()
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            pyaudio_pool.release(self.p)
            self.p = None
            with audio_list_lock:
                if (self.playback_thread, self.tag) in audio_threads:
                    audio_threads.remove((self.playback_thread, self.tag))
            self.playback_thread = None

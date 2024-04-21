# -*- encoding: utf-8 -*-
import time
import pyaudiowpatch as pyaudio
import numpy as np
import torch
import audio_tools
import keyboard
import threading
from pathlib import Path
from whisper import audio as whisper_audio
import wave
import Utilities
import Models.STS.SpeakerDiarization as speaker_diarization

SAMPLE_RATE = whisper_audio.SAMPLE_RATE
CHUNK = int(SAMPLE_RATE / 10)
CHANNELS = 1


# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound


def save_to_wav(data, filename, sample_rate, channels=1):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # Assuming 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def call_plugin_sts(plugins, wavefiledata, sample_rate):
    for plugin_inst in plugins.plugins:
        if plugin_inst.is_enabled(False) and hasattr(plugin_inst, 'sts'):
            plugin_inst.sts(wavefiledata, sample_rate)


def process_audio_chunk(audio_chunk, sample_rate, vad_model=None):
    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)
    if vad_model is not None:
        new_confidence = vad_model(torch.from_numpy(audio_float32), sample_rate).item()
    else:
        new_confidence = 9.9
    peak_amplitude = np.max(np.abs(audio_int16))

    # clear the variables
    audio_int16 = None
    del audio_int16
    audio_float32 = None
    del audio_float32
    return new_confidence, peak_amplitude


class AudioProcessor:
    last_callback_time = time.time()

    def __init__(self,
                 default_sample_rate=SAMPLE_RATE,
                 previous_audio_chunk=None,
                 start_rec_on_volume_threshold=None,
                 push_to_talk_key=None,
                 keyboard_rec_force_stop=None,
                 vad_model=None,
                 needs_sample_rate_conversion=False,
                 recorded_sample_rate=None,
                 input_channel_num=2,

                 plugins=None,
                 audio_enhancer=None,

                 osc_ip=None,
                 osc_port=None,

                 chunk=None,
                 channels=None,
                 sample_format=None,

                 audio_queue=None,
                 settings=None,
                 typing_indicator_function=None,

                 verbose=False
                 ):
        if plugins is None:
            plugins = []
        self.frames = []
        self.default_sample_rate = default_sample_rate
        self.previous_audio_chunk = previous_audio_chunk
        self.start_rec_on_volume_threshold = start_rec_on_volume_threshold
        self.push_to_talk_key = push_to_talk_key
        self.keyboard_rec_force_stop = keyboard_rec_force_stop

        self.vad_model = vad_model

        self.needs_sample_rate_conversion = needs_sample_rate_conversion
        self.recorded_sample_rate = recorded_sample_rate
        self.input_channel_num = input_channel_num

        self.Plugins = plugins
        self.audio_enhancer = audio_enhancer

        self.osc_ip = osc_ip
        self.osc_port = osc_port

        self.verbose = verbose

        self.start_time = time.time()
        self.pause_time = time.time()
        self.intermediate_time_start = time.time()

        self.block_size_samples = int(
            self.default_sample_rate * 0.400)  # calculate block size in samples. (0.400 is the default block size of pyloudnorm)

        self.chunk = chunk
        self.channels = channels
        self.sample_format = sample_format

        self.audio_queue = audio_queue
        self.settings = settings

        self.diarization_model = None
        self.typing_indicator_function = None

        self._new_speaker = False
        self.new_speaker_audio = None
        # run callback after timeout even if no audio was detected (and such callback not called by pyAudio)
        # self.timer_reset_event = threading.Event()
        # self.timer_thread = threading.Thread(target=self.timer_expired)
        # self.timer_thread.start()
        # self.timer_reset_event.set()
        # self.last_callback_time = time.time()

    def should_start_recording(self, peak_amplitude, energy, new_confidence, confidence_threshold, keyboard_key=None):
        return ((keyboard_key is not None and keyboard.is_pressed(
            keyboard_key)) or (0 < energy <= peak_amplitude and new_confidence >= confidence_threshold))

    def should_stop_recording(self, new_confidence, confidence_threshold, peak_amplitude, energy, pause_time, pause,
                              keyboard_key=None):
        return (keyboard_key is not None and not keyboard.is_pressed(keyboard_key)) or (
                0 < energy > peak_amplitude and (
                    new_confidence < confidence_threshold or confidence_threshold == 0.0) and (
                        time.time() - pause_time) > pause > 0.0)

    # The function to call when the timer expires
    # def timer_expired(self):
    #    while True:
    #        current_time = time.time()
    #        time_since_last_callback = current_time - self.last_callback_time
    #        if self.recorded_sample_rate is not None:
    #            # wait double the chunk size to not accidentally call callback twice
    #            self.timer_reset_event.wait(timeout=(self.chunk / self.recorded_sample_rate)*2)
    #            if time_since_last_callback >= (self.chunk / self.recorded_sample_rate)*2 and len(self.frames) > 0:
    #                #print("Timer expired. Triggering callback.")
    #                try:
    #                    print("Timer expired. Triggering callback.")
    #                    self.callback(None, None, None, None)
    #                except Exception as e:
    #                    print(e)
    #        self.timer_reset_event.clear()

    def callback(self, in_data, frame_count, time_info, status):
        # Reset the timer each time the callback is triggered
        # self.last_callback_time = time.time()
        # self.timer_reset_event.set()

        if not self.settings.GetOption("stt_enabled"):
            return None, pyaudio.paContinue

        # disable gradient calculation
        with torch.no_grad():
            phrase_time_limit = self.settings.GetOption("phrase_time_limit")
            pause = self.settings.GetOption("pause")
            energy = self.settings.GetOption("energy")
            if phrase_time_limit == 0:
                phrase_time_limit = None

            silence_cutting_enabled = self.settings.GetOption("silence_cutting_enabled")
            silence_offset = self.settings.GetOption("silence_offset")
            max_silence_length = self.settings.GetOption("max_silence_length")
            keep_silence_length = self.settings.GetOption("keep_silence_length")

            normalize_enabled = self.settings.GetOption("normalize_enabled")
            normalize_lower_threshold = self.settings.GetOption("normalize_lower_threshold")
            normalize_upper_threshold = self.settings.GetOption("normalize_upper_threshold")
            normalize_gain_factor = self.settings.GetOption("normalize_gain_factor")

            use_speaker_diarization = self.settings.GetOption("speaker_diarization")
            if use_speaker_diarization and self.diarization_model is None:
                self.diarization_model = speaker_diarization.SpeakerDiarization()
            speaker_change_split = self.settings.GetOption("speaker_change_split")
            min_speakers = self.settings.GetOption("min_speakers")
            max_speakers = self.settings.GetOption("max_speakers")

            clip_duration = phrase_time_limit
            fps = 0
            if clip_duration is not None:
                fps = int(self.recorded_sample_rate / CHUNK * clip_duration)

            end_time = time.time()
            elapsed_time = end_time - self.start_time
            elapsed_intermediate_time = end_time - self.intermediate_time_start

            confidence_threshold = float(self.settings.GetOption("vad_confidence_threshold"))

            # if settings.GetOption("denoise_audio") and audio_enhancer is not None and num_samples < DeepFilterNet.ModelParams().hop_size:
            #    #print("increase num_samples for denoising")
            #    num_samples = DeepFilterNet.ModelParams().hop_size

            # audio_chunk = stream.read(num_samples, exception_on_overflow=False)

            # denoise audio chunk
            # if settings.GetOption("denoise_audio") and audio_enhancer is not None:
            # record more audio to denoise if it's too short
            # if len(audio_chunk) < DeepFilterNet.ModelParams().hop_size:
            # while len(audio_chunk) < DeepFilterNet.ModelParams().hop_size:
            #    audio_chunk += stream.read(num_samples, exception_on_overflow=False)
            # audio_chunk = audio_enhancer.enhance_audio(audio_chunk, recorded_sample_rate, default_sample_rate, is_mono=is_mono)
            # needs_sample_rate_conversion = False

            # denoise audio chunk
            # if settings.GetOption("denoise_audio") and audio_enhancer is not None:
            #    #if len(audio_chunk) < DeepFilterNet.ModelParams().hop_size:
            #    #    while len(audio_chunk) < DeepFilterNet.ModelParams().hop_size * 2:
            #    #        audio_chunk += stream.read(num_samples, exception_on_overflow=False)
            #    audio_chunk = audio_enhancer.enhance_audio(audio_chunk, recorded_sample_rate, default_sample_rate, is_mono=is_mono)
            #    needs_sample_rate_conversion = False
            #    #recorded_sample_rate = audio_enhancer.df_state.sr()

            test_audio_chunk = in_data
            audio_chunk = in_data
            # special case which seems to be needed for WASAPI
            if self.needs_sample_rate_conversion and test_audio_chunk is not None:
                test_audio_chunk = audio_tools.resample_audio(test_audio_chunk, self.recorded_sample_rate,
                                                              self.default_sample_rate, target_channels=1,
                                                              input_channels=self.input_channel_num).tobytes()

            new_confidence, peak_amplitude = 0, 0
            if test_audio_chunk is not None:
                new_confidence, peak_amplitude = process_audio_chunk(test_audio_chunk, self.default_sample_rate,
                                                                     self.vad_model)

            # put frames with recognized speech into a list and send to whisper
            if (clip_duration is not None and len(self.frames) > fps) or (
                    elapsed_time > pause > 0.0 and len(self.frames) > 0) or (
                    self.keyboard_rec_force_stop and self.push_to_talk_key is not None and not keyboard.is_pressed(
                self.push_to_talk_key) and len(self.frames) > 0) or (
                    use_speaker_diarization and self._new_speaker and self.diarization_model is not None and len(
                self.frames) > 0):

                clip = []
                # merge all frames to one audio clip
                for i in range(0, len(self.frames)):
                    if self.frames[i] is not None:
                        clip.append(self.frames[i])

                if len(clip) > 0:
                    wavefiledata = b''.join(clip)
                else:
                    return None, pyaudio.paContinue

                if self.needs_sample_rate_conversion:
                    wavefiledata = audio_tools.resample_audio(wavefiledata, self.recorded_sample_rate,
                                                              self.default_sample_rate, target_channels=1,
                                                              input_channels=self.input_channel_num).tobytes()

                # normalize audio (and make sure it's longer or equal the default block size by pyloudnorm)
                if normalize_enabled and len(wavefiledata) >= self.block_size_samples:
                    wavefiledata = audio_tools.convert_audio_datatype_to_float(np.frombuffer(wavefiledata, np.int16))
                    wavefiledata, lufs = audio_tools.normalize_audio_lufs(
                        wavefiledata, self.default_sample_rate, normalize_lower_threshold, normalize_upper_threshold,
                        normalize_gain_factor, verbose=self.verbose
                    )
                    wavefiledata = audio_tools.convert_audio_datatype_to_integer(wavefiledata, np.int16)
                    wavefiledata = wavefiledata.tobytes()

                # remove silence from audio
                if silence_cutting_enabled:
                    wavefiledata_np = np.frombuffer(wavefiledata, np.int16)
                    if len(wavefiledata_np) >= self.block_size_samples:
                        wavefiledata = audio_tools.remove_silence_parts(
                            wavefiledata_np, self.default_sample_rate,
                            silence_offset=silence_offset, max_silence_length=max_silence_length,
                            keep_silence_length=keep_silence_length,
                            verbose=self.verbose
                        )
                        wavefiledata = wavefiledata.tobytes()

                # debug save of audio clip
                # save_to_wav(wavefiledata, "resampled_audio_chunk.wav", self.default_sample_rate)

                # check if the full audio clip is above the confidence threshold
                vad_clip_test = self.settings.GetOption("vad_on_full_clip")
                full_audio_confidence = 0.
                if vad_clip_test:
                    audio_full_int16 = np.frombuffer(wavefiledata, np.int16)
                    audio_full_float32 = int2float(audio_full_int16)
                    full_audio_confidence = self.vad_model(torch.from_numpy(audio_full_float32),
                                                           self.default_sample_rate).item()
                    print(full_audio_confidence)

                if ((not vad_clip_test) or (vad_clip_test and full_audio_confidence >= confidence_threshold)) and len(
                        wavefiledata) > 0:
                    # denoise audio
                    if self.settings.GetOption("denoise_audio") and self.audio_enhancer is not None:
                        wavefiledata = self.audio_enhancer.enhance_audio(wavefiledata).tobytes()

                    # call sts plugin methods
                    call_plugin_sts(self.Plugins, wavefiledata, self.default_sample_rate)

                    wave_file_bytes = audio_tools.audio_bytes_to_wav(wavefiledata, channels=CHANNELS,
                                                                     sample_rate=SAMPLE_RATE)
                    if self.diarization_model is not None and use_speaker_diarization:
                        wave_file_bytes = self.diarization_model.diarize(wave_file_bytes, min_speakers=min_speakers,
                                                                         max_speakers=max_speakers)

                    if isinstance(wave_file_bytes, list):
                        for audio_segment in wave_file_bytes:
                            self.audio_queue.put(
                                {'time': time.time_ns(), 'data': audio_segment, 'final': True})
                    else:
                        self.audio_queue.put(
                            {'time': time.time_ns(), 'data': wave_file_bytes, 'final': True})
                    # vad_iterator.reset_states()  # reset model states after each audio

                    # write wav file if configured to do so
                    transcription_save_audio_dir = self.settings.GetOption("transcription_save_audio_dir")
                    if transcription_save_audio_dir is not None and transcription_save_audio_dir != "":
                        start_time_str = Utilities.ns_to_datetime(time.time_ns(), formatting='%Y-%m-%d %H_%M_%S-%f')
                        audio_file_name = f"audio_transcript_{start_time_str}.wav"

                        transcription_save_audio_dir = Path(transcription_save_audio_dir)
                        audio_file_path = transcription_save_audio_dir / audio_file_name

                        threading.Thread(
                            target=save_to_wav,
                            args=(wavefiledata, str(audio_file_path.resolve()), self.default_sample_rate,)
                        ).start()

                    # set typing indicator for VRChat and Websocket clients
                    if self.typing_indicator_function is not None:
                        typing_indicator_thread = threading.Thread(target=self.typing_indicator_function,
                                                                   args=(self.osc_ip, self.osc_port, True))
                        typing_indicator_thread.start()

                self.frames = []
                self.start_time = time.time()
                self.intermediate_time_start = time.time()
                self.keyboard_rec_force_stop = False
                if self.new_speaker_audio is not None:
                    print("Added new speaker audio of new speaker to new audio recording chunk")
                    self.frames.append(self.new_speaker_audio)
                    self.new_speaker_audio = None
                self._new_speaker = False

            if audio_chunk is None:
                return None, pyaudio.paContinue

            # set start recording variable to true if the volume and voice confidence is above the threshold
            if self.should_start_recording(peak_amplitude, energy, new_confidence, confidence_threshold,
                                           keyboard_key=self.push_to_talk_key):
                if self.verbose:
                    print("start recording - new_confidence: " + str(new_confidence) + " peak_amplitude: " + str(
                        peak_amplitude))
                if not self.start_rec_on_volume_threshold:
                    # start processing_start event
                    if self.typing_indicator_function is not None:
                        typing_indicator_thread = threading.Thread(target=self.typing_indicator_function,
                                                                   args=(self.osc_ip, self.osc_port, True))
                        typing_indicator_thread.start()
                if self.push_to_talk_key is not None and keyboard.is_pressed(self.push_to_talk_key):
                    self.keyboard_rec_force_stop = True
                self.start_rec_on_volume_threshold = True
                self.pause_time = time.time()

            # append audio frame to the list if the recording var is set and voice confidence is above the threshold (So it only adds the audio parts with speech)
            if self.start_rec_on_volume_threshold and new_confidence >= confidence_threshold:
                if self.verbose:
                    print("add chunk - new_confidence: " + str(new_confidence) + " peak_amplitude: " + str(
                        peak_amplitude))
                # append previous audio chunk to improve recognition on too late audio recording starts
                if self.previous_audio_chunk is not None:
                    self.frames.append(self.previous_audio_chunk)

                # TODO? send audio_chunk to plugins (RVC )
                # threading.Thread(target=call_plugin_sts_chunk, args=(self.Plugins, test_audio_chunk, self.default_sample_rate,)).start()

                self.frames.append(audio_chunk)
                self.start_time = time.time()
                if self.settings.GetOption("realtime"):
                    clip = []
                    frame_count = len(self.frames)
                    # send realtime intermediate results every x frames and every x seconds (making sure its at least x frame length)
                    if frame_count % self.settings.GetOption(
                            "realtime_frame_multiply") == 0 and elapsed_intermediate_time > self.settings.GetOption(
                        "realtime_frequency_time"):
                        # set typing indicator for VRChat but not websocket
                        if self.typing_indicator_function is not None:
                            typing_indicator_thread = threading.Thread(target=self.typing_indicator_function,
                                                                       args=(self.osc_ip, self.osc_port, False))
                            typing_indicator_thread.start()
                        # merge all frames to one audio clip
                        for i in range(0, len(self.frames)):
                            clip.append(self.frames[i])

                        if len(clip) > 0:
                            wavefiledata = b''.join(clip)
                        else:
                            return None, pyaudio.paContinue

                        if self.needs_sample_rate_conversion:
                            wavefiledata = audio_tools.resample_audio(wavefiledata, self.recorded_sample_rate,
                                                                      self.default_sample_rate, target_channels=1,
                                                                      input_channels=self.input_channel_num).tobytes()

                        # normalize audio (and make sure it's longer or equal the default block size by pyloudnorm)
                        if normalize_enabled and len(wavefiledata) >= self.block_size_samples:
                            wavefiledata = audio_tools.convert_audio_datatype_to_float(
                                np.frombuffer(wavefiledata, np.int16))
                            wavefiledata, lufs = audio_tools.normalize_audio_lufs(
                                wavefiledata, self.default_sample_rate, normalize_lower_threshold,
                                normalize_upper_threshold, normalize_gain_factor,
                                verbose=self.verbose
                            )
                            wavefiledata = audio_tools.convert_audio_datatype_to_integer(wavefiledata, np.int16)
                            wavefiledata = wavefiledata.tobytes()

                        # remove silence from audio
                        if silence_cutting_enabled:
                            wavefiledata_np = np.frombuffer(wavefiledata, np.int16)
                            if len(wavefiledata_np) >= self.block_size_samples:
                                wavefiledata = audio_tools.remove_silence_parts(
                                    wavefiledata_np, self.default_sample_rate,
                                    silence_offset=silence_offset, max_silence_length=max_silence_length,
                                    keep_silence_length=keep_silence_length,
                                    verbose=self.verbose
                                )
                                wavefiledata = wavefiledata.tobytes()

                        if wavefiledata is not None and len(wavefiledata) > 0:
                            # denoise audio
                            if self.settings.GetOption("denoise_audio") and self.audio_enhancer is not None:
                                wavefiledata = self.audio_enhancer.enhance_audio(wavefiledata).tobytes()

                            wave_file_bytes = audio_tools.audio_bytes_to_wav(wavefiledata, channels=CHANNELS,
                                                                             sample_rate=SAMPLE_RATE)
                            if self.diarization_model is not None and use_speaker_diarization:
                                wave_file_bytes = self.diarization_model.diarize(wave_file_bytes,
                                                                                 min_speakers=min_speakers,
                                                                                 max_speakers=max_speakers)

                            if isinstance(wave_file_bytes, list):
                                print("multiple speaker audio detected")
                                self.audio_queue.put(
                                    {'time': time.time_ns(), 'data': wave_file_bytes[-1], 'final': False})
                                if speaker_change_split and len(wave_file_bytes) > 1:
                                    self.new_speaker_audio = self.frames[-1]
                                    # self.new_speaker_audio = wave_file_bytes[-1]
                                    # remove last element from frames list
                                    self.frames.pop()
                                    # convert into bytes per audio frame
                                    # self.frames = audio_tools.split_audio_with_padding(wave_file_bytes[0], CHUNK)
                                    self._new_speaker = True
                            else:
                                self.audio_queue.put(
                                    {'time': time.time_ns(), 'data': wave_file_bytes, 'final': False})

                        else:
                            self.frames = []

                        self.intermediate_time_start = time.time()

            # stop recording if no speech is detected for pause seconds
            if self.should_stop_recording(new_confidence, confidence_threshold, peak_amplitude, energy, self.pause_time,
                                          pause,
                                          keyboard_key=self.push_to_talk_key):
                self.start_rec_on_volume_threshold = False
                self.intermediate_time_start = time.time()
                if self.push_to_talk_key is not None and not keyboard.is_pressed(
                        self.push_to_talk_key) and self.keyboard_rec_force_stop:
                    self.keyboard_rec_force_stop = True
                else:
                    self.keyboard_rec_force_stop = False

            # save chunk as previous audio chunk to reuse later
            if not self.start_rec_on_volume_threshold and (
                    new_confidence < confidence_threshold or confidence_threshold == 0.0):
                self.previous_audio_chunk = audio_chunk
            else:
                self.previous_audio_chunk = None

        # self.last_callback_time = time.time()
        return in_data, pyaudio.paContinue

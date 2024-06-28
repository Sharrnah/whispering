import io
import os

import numpy as np
import torch
import soundfile as sf
import wave

from Models.Singleton import SingletonMeta
from pathlib import Path
from pyannote.audio import Pipeline, Audio

import audio_tools

import downloader

cache_model_path = Path(Path.cwd() / ".cache" / "pyannote")


class SpeakerDiarization(metaclass=SingletonMeta):
    pipeline = None

    def __init__(self):
        if self.pipeline is None:
            print("loading pyannote pipeline...")
            pipeline_path = str(Path(cache_model_path / "diarization_config.yaml").resolve())

            self.pipeline = Pipeline.from_pretrained(pipeline_path)
            self.pipeline.to(torch.device("cuda"))
            print("pyannote pipeline loaded")
            # initialize audio reader
            self._io = Audio()

    def load_pipeline_from_pretrained(self, path_to_config: str | Path) -> Pipeline:
        path_to_config = Path(path_to_config)

        print(f"Loading pyannote pipeline from {path_to_config}...")
        # the paths in the config are relative to the current working directory
        # so we need to change the working directory to the model path
        # and then change it back

        cwd = Path.cwd().resolve()  # store current working directory

        # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
        cd_to = path_to_config.parent.parent.resolve()

        print(f"Changing working directory to {cd_to}")
        os.chdir(cd_to)

        pipeline = Pipeline.from_pretrained(path_to_config)

        print(f"Changing working directory back to {cwd}")
        os.chdir(cwd)

        return pipeline

    def tensor_to_wav_bytes(self, tensor_waveform, sample_rate, num_channels, dtype=np.int16):
        """
        Converts a tensor waveform to WAV bytes.

        Args:
            tensor_waveform (np.ndarray): The tensor representing audio waveform.
            sample_rate (int): The sampling rate of the audio.
            num_channels (int): The number of audio channels.
            dtype (data-type, optional): The target data type of the WAV file's audio samples. Default is np.int16.

        Returns:
            bytes: The WAV-formatted byte stream.
        """
        # Ensure the tensor is in the correct dtype
        tensor_waveform = tensor_waveform.astype(dtype)

        # Convert NumPy array to bytes
        audio_bytes = tensor_waveform.tobytes()

        # Create a BytesIO object to handle WAV file creation
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(np.dtype(dtype).itemsize)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)

        return buffer.getvalue()

    def _waveform_to_bytes(self, waveform, sample_rate, output_dtype=np.int16, audio_format='WAV',
                           audio_subtype='PCM_16', scale=True):
        """
        Convert a numpy waveform array to bytes using soundfile, allowing for flexible output formats and automatic scaling between data types.

        Parameters:
        - waveform: numpy array containing the audio samples.
        - sample_rate: integer, the sample rate of the audio in Hz.
        - output_dtype: data type for the output audio, default is np.int16.
        - audio_format: string, format of the output audio file (e.g., 'WAV', 'FLAC').
        - audio_subtype: string, subtype of the output audio file (e.g., 'PCM_16', 'PCM_24').
        - scale: boolean, whether to scale data between float and int ranges; default is True.

        Returns:
        - audio_bytes: bytes object containing the audio in the specified format and subtype.
        """
        with io.BytesIO() as audio_buffer:
            # Check if scaling is needed and perform conversion
            if scale:
                if np.issubdtype(waveform.dtype, np.integer) and np.issubdtype(output_dtype, np.floating):
                    # Scale integer to float range -1.0 to 1.0
                    max_int_value = np.iinfo(waveform.dtype).max
                    waveform = waveform.astype(np.float32) / max_int_value
                elif np.issubdtype(waveform.dtype, np.floating) and np.issubdtype(output_dtype, np.integer):
                    # Scale float to full range of the output integer type
                    max_int_value = np.iinfo(output_dtype).max
                    waveform = (waveform * max_int_value).astype(output_dtype)
                else:
                    # input + output types the same.
                    waveform = waveform.astype(output_dtype)
            elif waveform.dtype != output_dtype:
                # Convert types without scaling (direct conversion)
                waveform = waveform.astype(output_dtype)

            # Write the audio data to a buffer with the specified format and subtype
            sf.write(audio_buffer, waveform.T, sample_rate, format=audio_format, subtype=audio_subtype)
            audio_bytes = audio_buffer.getvalue()

        return audio_bytes

    def diarize(self, audio_bytes, sample_rate=16000, min_speakers=1, max_speakers=3, min_segment_length=1.0,
                bytes_sample_rate=16000, bytes_channel_num=1):
        audio_bytes = audio_tools.resample_audio(audio_bytes, sample_rate,
                                                 sample_rate,
                                                 input_channels=1,
                                                 target_channels=1,
                                                 )

        waveform = audio_bytes.flatten()

        # Ensure waveform is in (channel, time) format
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        elif waveform.ndim == 2:
            waveform = waveform.T

        # Convert the numpy array to a torch Tensor with shape (channel, time)
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

        # Pass the waveform tensor and sample rate to the diarization pipeline
        diarization = self.pipeline(
            {"waveform": waveform_tensor, "sample_rate": sample_rate},
            min_speakers=min_speakers, max_speakers=max_speakers,
        )

        # Process diarization to split audio at speaker changes
        segments = []
        last_speaker = None
        start_idx = 0
        min_samples = int(min_segment_length * sample_rate)

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            end_idx = int(turn.end * sample_rate)
            if last_speaker is None:
                start_idx = int(turn.start * sample_rate)  # First segment start
            elif speaker != last_speaker:
                if (end_idx - start_idx >= min_samples):
                    segments.append((start_idx, end_idx, last_speaker))
                start_idx = int(turn.start * sample_rate)
            last_speaker = speaker

        # Append the last segment if any speakers have been processed
        if last_speaker and (len(waveform) - start_idx >= min_samples):
            segments.append((start_idx, len(waveform), last_speaker))

        #print("detected Segments:")
        #print(segments)

        # Merge segments that belong to the same speaker
        merged_segments = []
        last_speaker = None
        segment_start, segment_end = None, None

        for start, end, speaker in segments:
            if last_speaker is None or last_speaker != speaker:
                if last_speaker is not None:
                    merged_segments.append((segment_start, segment_end, last_speaker))
                segment_start, segment_end = start, end
            else:
                segment_end = end
            last_speaker = speaker

        # Append the last processed segment
        if last_speaker:
            merged_segments.append((segment_start, segment_end, last_speaker))

        #print("Merged Segments:")
        #print(merged_segments)

        # Generate audio bytes for each valid segment
        audio_segments_bytes = []
        bytes_per_sample = 2  # Assuming 16-bit PCM
        channels = 1
        for start_idx, end_idx, speaker in segments:
            start_byte = start_idx * bytes_per_sample * channels
            end_byte = end_idx * bytes_per_sample * channels
            segment_bytes = audio_bytes[start_byte:end_byte]
            if not len(segment_bytes) > 0:
                #print("empty segment bytes")
                continue
            #print("segment_bytes")
            #print(segment_bytes)
            segment_bytes = audio_tools.resample_audio(segment_bytes, sample_rate, bytes_sample_rate,
                                                       input_channels=1,
                                                       target_channels=bytes_channel_num,
                                                       ).tobytes()
            audio_segments_bytes.append(segment_bytes)

            # start_time_str = Utilities.ns_to_datetime(time.time_ns(), formatting='%Y-%m-%d %H_%M_%S-%f')
            # audio_file_name = f"audio_transcript_{start_time_str}.wav"
            # transcription_save_audio_dir = Path("audio_debugging")
            # audio_file_path = transcription_save_audio_dir / audio_file_name
            # with wave.open(str(audio_file_path.resolve()), 'wb') as wf:
            #     wf.setnchannels(bytes_channel_num)
            #     wf.setsampwidth(2)  # Assuming 16-bit audio
            #     wf.setframerate(bytes_sample_rate)
            #     wf.writeframes(segment_bytes)

        #print("audio_segments_bytes")
        #print(audio_segments_bytes)

        return audio_segments_bytes

    def diarize2(self, audio_bytes, sample_rate=16000, min_speakers=1, max_speakers=3):
        # Convert bytes to a file-like object
        audio_file = io.BytesIO(audio_bytes)

        # Read audio data from the file-like object
        waveform, file_sample_rate = sf.read(audio_file, dtype='float32')

        # Ensure waveform is in (channel, time) format
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        elif waveform.ndim == 2:
            waveform = waveform.T

        # Convert the numpy array to a torch Tensor with shape (channel, time)
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

        # Pass the waveform tensor and sample rate to the diarization pipeline
        diarization = self.pipeline(
            {"waveform": waveform_tensor, "sample_rate": sample_rate},
            min_speakers=min_speakers, max_speakers=max_speakers,
        )
        # Process diarization to split audio at speaker changes
        last_speaker = None
        start_idx = 0  # Initialize start index
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if last_speaker is not None and speaker != last_speaker:
                return True
        return False

#    def diarize(self, audio_data, sample_rate=16000):
#
#
#
#        print("diarizing...")
#        # bytes to numpy array
#        audio_int16 = np.asarray(audio_data, dtype=np.int16)
#        print("diarizing...2")
#        waveform_tensor = torch.from_numpy(audio_int16).float().unsqueeze(0)
#        print("diarizing...3")
#
#        #audio = Audio(sample_rate=sample_rate, mono=True)
#        #waveform, sample_rate = audio({"waveform": waveform_tensor, "sample_rate": sample_rate})
#
#        print("waveform_tensor", waveform_tensor)
#        print("sample_rate", sample_rate)
#
#        diarization_result = self.pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})
#        print("diarizing...4")
#
#        if isinstance(diarization_result, Timeline):
#            diarization_result = diarization_result.to_annotation(generator="string")
#            print("diarizing...5")
#
#        if isinstance(diarization_result, Annotation):
#            #for s, t, l in diarization_result.itertracks(yield_label=True):
#            #    line = (
#            #        f"SPEAKER {diarization_result.uri} 1 {s.start:.3f} {s.duration:.3f} "
#            #        f"<NA> <NA> {l} <NA> <NA>\n"
#            #    )
#            #    print(line)
#            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}, turn={turn}")
#            print("diarizing...6")
#
#        print("diarizing...7")
#
#        #print(diarization_result)
#        #for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#        #    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}, turn={turn}")
#

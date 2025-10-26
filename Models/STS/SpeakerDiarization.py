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

# --- Begin: PyTorch 2.6+ safe allowlist for checkpoint loading ---
# PyTorch 2.6 defaults torch.load to weights_only=True which can fail to unpickle
# certain safe classes in legacy checkpoints. We add an allowlist for these.
_PATCH_APPLIED = False


def _apply_torch_load_patch_if_needed():
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    try:
        # Allowlist TorchVersion used in some checkpoints metadata and
        # pyannote classes commonly referenced in checkpoints.
        from torch.serialization import add_safe_globals  # type: ignore
        import importlib
        import inspect

        # Helper to add a class/function by dotted path if available
        def _add_safe_global_by_name(dotted_name: str):
            try:
                module_path, attr_name = dotted_name.rsplit('.', 1)
                mod = importlib.import_module(module_path)
                obj = getattr(mod, attr_name, None)
                if obj is not None:
                    add_safe_globals([obj])
            except Exception:
                pass

        # Helper to add all classes from a module by dotted path
        def _add_all_classes_from_module(module_path: str):
            try:
                mod = importlib.import_module(module_path)
                classes = [obj for _, obj in inspect.getmembers(mod, inspect.isclass)]
                if classes:
                    add_safe_globals(classes)
            except Exception:
                pass

        # TorchVersion
        try:
            import torch.torch_version as _tv  # type: ignore
            try:
                add_safe_globals([_tv.TorchVersion])
            except Exception:
                pass
        except Exception:
            pass

        # Known pyannote class required by weights-only safe loader
        _add_safe_global_by_name('pyannote.audio.core.task.Specifications')

        # Proactively allow all classes from relevant pyannote core modules
        _add_all_classes_from_module('pyannote.audio.core.task')
        _add_all_classes_from_module('pyannote.audio.core.model')

    except Exception:
        # If torch doesn't expose these (older versions), just continue.
        pass
    _PATCH_APPLIED = True
# --- End: PyTorch 2.6+ safe allowlist for checkpoint loading ---

model_cache_path = Path(Path.cwd() / ".cache" / "pyannote")

MODEL_LINKS = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/pyannote/pyannote-speakerdiarization.zip",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/pyannote/pyannote-speakerdiarization.zip",
        "https://s3.libs.space:9000/ai-models/pyannote/pyannote-speakerdiarization.zip"
    ],
    "checksum": "3491d2025492f1519990fa01be4e6cde444a50b4cc681815431033e432848ba1",
    "file_checksums": {
        "diarization_config.yaml": "25ce75f7c26f6ca5232602afb9b95bf1c4f4b1764970566b3edcce448dd25c99",
        "models\\pyannote_model_segmentation-3.0.bin": "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298",
        "models\\pyannote_model_wespeaker-voxceleb-resnet34-LM.bin": "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"
    }
}


class SpeakerDiarization(metaclass=SingletonMeta):
    pipeline = None

    def __init__(self):
        if self.pipeline is None:
            print("loading pyannote pipeline...")
            if self._needs_download():
                self.download_model()
            pipeline_path = str(Path(model_cache_path / "diarization_config.yaml").resolve())

            # Ensure torch.load compatibility before loading pyannote pipeline
            _apply_torch_load_patch_if_needed()

            self.pipeline = Pipeline.from_pretrained(pipeline_path)
            self.pipeline.to(torch.device("cuda"))
            print("pyannote pipeline loaded")
            # initialize audio reader
            self._io = Audio()

    def _needs_download(self):
        if not model_cache_path.exists():
            return True

        expected_hashes = MODEL_LINKS["file_checksums"]
        actual_hashes = downloader.load_hashes(model_cache_path)

        if not actual_hashes:
            if downloader.check_file_hashes(model_cache_path, expected_hashes):
                return False
            else:
                return True

        for file_name, expected_hash in expected_hashes.items():
            actual_hash = actual_hashes.get(file_name)
            if actual_hash.lower() != expected_hash.lower():
                if downloader.sha256_checksum(model_cache_path / file_name).lower() == expected_hash.lower():
                    actual_hashes[file_name] = expected_hash.lower()
                else:
                    return True
        return False

    def download_model(self):
        os.makedirs(model_cache_path, exist_ok=True)

        file_checksums_check_need_dl = False
        hash_checked_file = model_cache_path / "hash_checked"

        if "file_checksums" in MODEL_LINKS:
            if not hash_checked_file.is_file():
                file_checksums_check_need_dl = True

        if not model_cache_path.exists() or file_checksums_check_need_dl:
            print("downloading Speaker Diarization...")
            if not downloader.download_extract(
                    MODEL_LINKS["urls"],
                    str(model_cache_path.resolve()),
                    MODEL_LINKS["checksum"],
                    title="Speaker Diarization"
            ):
                print("Model download failed")
            if file_checksums_check_need_dl:
                downloader.save_hashes(model_cache_path, MODEL_LINKS["file_checksums"])


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

        # Ensure torch.load compatibility before loading pyannote pipeline
        _apply_torch_load_patch_if_needed()

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

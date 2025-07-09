import io
import os
import queue
import re
from pathlib import Path

import numpy as np
import transformers

import Plugins
import audio_tools
import downloader
import settings
from Models.Singleton import SingletonMeta

cache_path = Path(Path.cwd() / ".cache" / "orpheus-tts-cache")
os.makedirs(cache_path, exist_ok=True)

#from Models.TTS.orpheus import OrpheusModel

from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import BaseStreamer

from scipy.io.wavfile import write as write_wav
import torch

failed = False

#https://github.com/blak-code-tech/orpheus-tts
#https://github.com/isaiahbjork/orpheus-tts-local
# https://canopylabs.ai/releases/orpheus_can_speak_any_language#info
TTS_MODEL_LINKS = {
    # Models
    "orpheus-3b-0.1-ft": {
        "urls": [
        ],
        "checksum": "",
        "file_checksums": {
        },
        "voices": [
            "tara",
            "leah",
            "jess",
            "leo",
            "dan",
            "mia",
            "zac",
            "zoe",
        ],
        "path": "orpheus-3b-0.1-ft",
    },
    "snac_24khz": {
        "urls": [
            "https://canopylabs.ai/releases/orpheus_can_speak_any_language/snac_24khz.zip",
        ],
        "checksum": "",
        "file_checksums": {
        },
        "path": "snac_24khz",
    }
}

speed_mapping = {
    "": 1.0,        # Default speed when empty
    "x-slow": 0.1,  # Extra slow speed
    "slow": 0.5,   # Slow speed
    "medium": 1.0,  # Medium speed (default)
    "fast": 2.0,   # Fast speed
    "x-fast": 3.0   # Extra fast speed
}

model_list = {
    "English": ["orpheus-3b-0.1-ft"],
}


# patching EspeakWrapper of phonemizer library.
# See https://github.com/open-mmlab/Amphion/issues/323#issuecomment-2646709006



class OrpheusTTS(metaclass=SingletonMeta):
    model = None
    snac_model = None
    tokenizer = None

    snac_device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_stream = None

    sample_rate = 24000
    last_generation = {"audio": None, "sample_rate": None}
    voice_list = []
    audio_streamer = None

    pipeline = None

    last_model = ""

    special_settings = {
        "language": 'a',
    }

    compute_device = "cpu"

    download_state = {"is_downloading": False}

    def __init__(self):
        self.compute_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        if not self.pipeline:
            #self.download_model("orpheus-3b-0.1-ft")

            self.load()

        # Prepare CUDA streams for parallel processing if available
        if self.snac_device == "cuda":
            self.cuda_stream = torch.cuda.Stream()

        if not self.voice_list:
            self.update_voices()
        pass

    def flatten_tensors(self, lst):
        for item in lst:
            if isinstance(item, list):
                yield from self.flatten_tensors(item)
            else:
                yield item

    def prompt_builder(self, text, voice=""):
        all_input_ids = []

        prompt = f"{voice}: " + text.strip()

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        all_input_ids.append(input_ids)

        start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human

        all_modified_input_ids = []
        for input_ids in all_input_ids:
            modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
            all_modified_input_ids.append(modified_input_ids)

        all_padded_tensors = []
        all_attention_masks = []
        max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
        for modified_input_ids in all_modified_input_ids:
            padding = max_length - modified_input_ids.shape[1]
            padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
            attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)

        all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        input_ids = all_padded_tensors.to("cuda")
        attention_mask = all_attention_masks.to("cuda")

        return input_ids, attention_mask

    def set_compute_device(self, device):
        self.compute_device_str = device
        if device is None or device == "cuda" or device == "auto" or device == "":
            self.compute_device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            #device = torch.device(self.compute_device_str)
            device = self.compute_device_str
        self.compute_device = device

    def list_models(self):
        return model_list

    def list_models_indexed(self):
        return tuple([{"language": language, "models": models} for language, models in self.list_models().items()])

    def download_model(self, model_name):
        downloader.download_model({
            "model_path": cache_path,
            "model_link_dict": TTS_MODEL_LINKS,
            "model_name": model_name,
            "title": "Text 2 Speech (Orpheus TTS)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)

    def set_special_setting(self, special_settings):
        self.special_settings = special_settings

    def stop(self):
        print("TTS Stop requested")
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    def _get_model_name(self):
        model = "orpheus-3b-0.1-ft"
        if len(settings.GetOption('tts_model')) == 2:
            #language = settings.GetOption('tts_model')[0]
            model = settings.GetOption('tts_model')[1]
            # remove language part from string example: " (en & zh)"
            model = re.sub(r'\(.*?\)', '', model).strip()

        if model == "" or model not in TTS_MODEL_LINKS:
            model = "orpheus-3b-0.1-ft"
        return model

    def load(self, model='orpheus-3b-0.1-ft'):
        self.load_snac_model("snac_24khz")
        self.load_model(model)

    def load_snac_model(self, model_name):
        if self.snac_model is None:
            self.snac_model = SNAC.from_pretrained(str(Path(cache_path / model_name).resolve())).eval()
            self.snac_model = self.snac_model.to(self.snac_device)
            print(f"SNAC model {model_name} loaded")

    # def _parse_output(self, generated_ids):
    #     token_to_find = 128257
    #     token_to_remove = 128258
    #
    #     token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    #
    #     if len(token_indices[1]) > 0:
    #         last_occurrence_idx = token_indices[1][-1].item()
    #         cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    #     else:
    #         cropped_tensor = generated_ids
    #
    #     mask = cropped_tensor != token_to_remove
    #
    #     processed_rows = []
    #
    #     for row in cropped_tensor:
    #         masked_row = row[row != token_to_remove]
    #         processed_rows.append(masked_row)
    #
    #     code_lists = []
    #
    #     for row in processed_rows:
    #         row_length = row.size(0)
    #         new_length = (row_length // 7) * 7
    #         trimmed_row = row[:new_length]
    #         trimmed_row = [t - 128266 for t in trimmed_row]
    #         code_lists.append(trimmed_row)
    #
    #
    #     def redistribute_codes(code_list):
    #         device = next(self.snac_model.parameters()).device
    #         layer_1 = []
    #         layer_2 = []
    #         layer_3 = []
    #         for i in range((len(code_list)+1)//7):
    #             layer_1.append(code_list[7*i])
    #             layer_2.append(code_list[7*i+1]-4096)
    #             layer_3.append(code_list[7*i+2]-(2*4096))
    #             layer_3.append(code_list[7*i+3]-(3*4096))
    #             layer_2.append(code_list[7*i+4]-(4*4096))
    #             layer_3.append(code_list[7*i+5]-(5*4096))
    #             layer_3.append(code_list[7*i+6]-(6*4096))
    #         codes = [torch.tensor(layer_1).unsqueeze(0).to(device),
    #                  torch.tensor(layer_2).unsqueeze(0).to(device),
    #                  torch.tensor(layer_3).unsqueeze(0).to(device)]
    #         audio_hat = self.snac_model.decode(codes)
    #         return audio_hat
    #
    #     my_samples = []
    #     for code_list in code_lists:
    #         samples = redistribute_codes(code_list)
    #         my_samples.append(samples)
    #
    #     return my_samples


    def _parse_output(self, generated_ids):
        token_to_find = 128257
        token_to_remove = 128258

        # Step 1: Find last occurrence in all rows (just as in original)
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_ids

        processed_rows = []

        # Step 2: Remove token_to_remove from each row (tensor ops, but per row)
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []

        # Step 3: Cut to length divisible by 7 and subtract 128266
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            if new_length == 0:
                code_lists.append([])  # This will match original (empty row)
                continue
            trimmed_row = row[:new_length] - 128266
            code_lists.append(trimmed_row.tolist())  # .tolist() for compatibility

        def redistribute_codes(code_list):
            # If code_list is empty, return empty tensors
            if len(code_list) == 0:
                empty = torch.empty((1, 0), dtype=torch.int64, device=self.snac_device)
                return [empty, empty, empty]
            layer_1 = []
            layer_2 = []
            layer_3 = []
            for i in range((len(code_list)+1)//7):
                layer_1.append(code_list[7*i])
                layer_2.append(code_list[7*i+1]-4096)
                layer_3.append(code_list[7*i+2]-(2*4096))
                layer_3.append(code_list[7*i+3]-(3*4096))
                layer_2.append(code_list[7*i+4]-(4*4096))
                layer_3.append(code_list[7*i+5]-(5*4096))
                layer_3.append(code_list[7*i+6]-(6*4096))
            codes = [torch.tensor(layer_1).unsqueeze(0).to(self.snac_device),
                     torch.tensor(layer_2).unsqueeze(0).to(self.snac_device),
                     torch.tensor(layer_3).unsqueeze(0).to(self.snac_device)]
            audio_hat = self.snac_model.decode(codes)
            return audio_hat

        my_samples = []
        for code_list in code_lists:
            samples = redistribute_codes(code_list)
            my_samples.append(samples)

        return my_samples

    def load_model(self, model='orpheus-3b-0.1-ft'):
        if self.model is None:
            # self.model = OrpheusModel(
            #     model_name=str(Path(cache_path / model).resolve()),
            #     tokenizer=str(Path(cache_path / model).resolve()),
            #     max_model_len=2048
            # )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            attention_implementation = 'sdpa'
            if transformers.utils.is_flash_attn_2_available():
                attention_implementation = 'flash_attention_2'

            self.model = AutoModelForCausalLM.from_pretrained(str(Path(cache_path / model).resolve()),
                                                              torch_dtype=torch.bfloat16,
                                                              quantization_config=quantization_config,
                                                              _attn_implementation=attention_implementation
                                                              )
            self.model.cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(str(Path(cache_path / model).resolve()))
            self.last_model = model
        pass

    def _get_voices(self):
        return self.voice_list

    def update_voices(self):
        self.voice_list = TTS_MODEL_LINKS[self._get_model_name()]["voices"]

    def list_voices(self):
        self.update_voices()
        return [voice for voice in self._get_voices()]

    def get_voice_by_name(self, voice_name):
        for voice in self._get_voices():
            if voice == voice_name:
                return voice
        return None

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    def tts(self, text, remove_silence=True, silence_after_segments=0.2, normalize=True):
        print("TTS requested Orpheus TTS")
        self.set_compute_device(settings.GetOption('tts_ai_device'))

        tts_volume = settings.GetOption("tts_volume")

        voice_name = settings.GetOption('tts_voice')
        selected_voice = self.get_voice_by_name(voice_name)
        if selected_voice is None:
            print("No voice selected or does not exist. Using default voice 'tara'.")
            voice_name = "tara"
            selected_voice = self.get_voice_by_name(voice_name)

        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)

        # generator = self.model.generate_speech(
        #     prompt=text,
        #     voice=selected_voice,
        # )


        # https://blog.gopenai.com/how-to-stream-output-in-llm-based-applications-to-boost-user-experience-e9fcf582777a
        input_ids, attention_mask = self.prompt_builder(text, voice=selected_voice)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                #max_new_tokens=1200,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                num_return_sequences=1,
                eos_token_id=128258,
            )

        audio_chunks = []

        tmp_audio_chunks = self._parse_output(generated_ids)

        for audio in self.flatten_tensors(tmp_audio_chunks):
            print("type(audio):", type(audio))
            print("audio:", audio)
            #with self.stop_flag_lock:
            #    if self.stop_flag:
            #        break
            # change volume
            audio = audio.detach().squeeze().to("cpu").numpy()
            if tts_volume != 1.0:
                audio = audio_tools.change_volume(audio, tts_volume)
            audio_chunks.append(audio)

        full_audio = np.concatenate(audio_chunks, axis=-1)
        # numpy array to torch.Tensor
        full_audio = torch.from_numpy(full_audio).float()

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': full_audio, 'sample_rate': self.sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            full_audio = plugin_audio['audio']

        # save last generation in memory
        self.last_generation = {"audio": full_audio, "sample_rate": self.sample_rate}

        print("TTS generation finished")

        return full_audio, self.sample_rate


    def tts_streaming(self, text, ref_audio=None):
        """
        Low-latency Orpheus-TTS streaming.
        Streams chunks to self.audio_streamer.add_audio_chunk(bytes) and,
        when finished, returns the full utterance as (audio_tensor, sample_rate).
        """
        # ---- house-keeping ---------------------------------------------------
        #self.set_compute_device(settings.GetOption("tts_ai_device"))
        tts_volume   = settings.GetOption("tts_volume")
        voice_name   = settings.GetOption("tts_voice")
        selected_voice = self.get_voice_by_name(voice_name) or "tara"

        # make sure the output device is ready
        self.init_audio_stream_playback()

        # build LLaMA-style prompt → input_ids / attention_mask
        input_ids, attention_mask = self.prompt_builder(text, voice=selected_voice)


        # ---- custom streamer -------------------------------------------------
        class _OrpheusAudioStreamer(BaseStreamer):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
                self.token_buf   : list[int]  = []   # all tokens received so far
                self.processed   = 0                 # index of last decoded token
                self.audio_parts : list[np.ndarray] = []

            def put(self, value):
                """
                Receives freshly generated token IDs from HF `generate()`.
                Only IDs that come *after* the 128257 start-marker are treated as
                Orpheus acoustic codes.
                """
                ids = value.view(-1).tolist()          # flatten in case of batch

                for tok in ids:
                    # wait for Start-of-Audio (128257)
                    if not hasattr(self, "in_audio"):
                        self.in_audio = False
                    if not self.in_audio:
                        if tok == 128257:
                            self.in_audio = True        # flip the switch
                        continue                        # ignore everything else
                    # after 128257: collect codes
                    self.token_buf.append(tok)

                # decode while we have ≥1 complete 7-token frame
                while len(self.token_buf) - self.processed >= 7:
                    seg = self.token_buf[self.processed:]
                    usable_len = (len(seg) // 7) * 7
                    if usable_len == 0:
                        break

                    new_ids = seg[:usable_len].copy()   # ALWAYS a real list
                    self.processed += usable_len

                    ids_t = torch.tensor([new_ids],
                                         dtype=torch.int64,
                                         device=self.outer.snac_device)

                    with torch.no_grad():
                        nested = self.outer._parse_output(ids_t)

                    for t in self.outer.flatten_tensors(nested):
                        chunk = t.squeeze().cpu().numpy()

                        if tts_volume != 1.0:
                            chunk = audio_tools.change_volume(chunk, tts_volume)

                        # call custom plugin event method
                        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': chunk, 'sample_rate': self.outer.sample_rate})
                        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
                            chunk = plugin_audio['audio']

                        self.audio_parts.append(chunk)

                        chunk_bytes = chunk.astype(np.float32, copy=True).tobytes()
                        if self.outer.audio_streamer is not None:
                            self.outer.audio_streamer.add_audio_chunk(
                                #chunk.astype(np.float32, copy=False).tobytes()
                                chunk_bytes
                            )

            # called once after EOS
            def end(self):
                if self.audio_parts:
                    full = np.concatenate(self.audio_parts, axis=-1)
                    self.outer.last_generation = {
                        "audio": torch.from_numpy(full).float(),
                        "sample_rate": self.outer.sample_rate
                    }

        streamer = _OrpheusAudioStreamer(self)

        # ---- run generation --------------------------------------------------
        self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=128258,   # Orpheus EOT
        )

        # return the whole utterance once generation stopped
        return (self.last_generation["audio"],
                self.last_generation["sample_rate"])

    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")

        chunk_size = settings.GetOption("tts_streamed_chunk_size")
        #if self.audio_streamer is not None:
        #    self.audio_streamer.stop()
        #    self.audio_streamer = None
        #else:
        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(audio_device,
                                                            source_sample_rate=int(self.sample_rate),
                                                            playback_channels=2,
                                                            buffer_size=chunk_size,
                                                            input_channels=1,
                                                            dtype="float32",
                                                            tag="tts",
                                                            )

    def play_audio(self, audio, device=None):
        source_channels = 1

        if device is None:
            device = settings.GetOption("device_default_out_index")

        secondary_audio_device = None
        if settings.GetOption("tts_use_secondary_playback") and (
                (settings.GetOption("tts_secondary_playback_device") == -1 and device != settings.GetOption("device_default_out_index")) or
                (settings.GetOption("tts_secondary_playback_device") > -1 and device != settings.GetOption("tts_secondary_playback_device"))):
            secondary_audio_device = settings.GetOption("tts_secondary_playback_device")
            if secondary_audio_device == -1:
                secondary_audio_device = settings.GetOption("device_default_out_index")

        allow_overlapping_audio = settings.GetOption("tts_allow_overlapping_audio")
        #audio = np.int16(audio * 32767)  # Convert to 16-bit PCM
        #audio = audio_tools.convert_audio_datatype_to_integer(audio)

        # play audio tensor
        audio_tools.play_audio(audio, device,
                               source_sample_rate=int(self.sample_rate),
                               audio_device_channel_num=1,
                               target_channels=1,
                               input_channels=source_channels,
                               dtype="float32",
                               tensor_sample_with=4,
                               tensor_channels=1,
                               secondary_device=secondary_audio_device,
                               stop_play=not allow_overlapping_audio,
                               tag="tts"
                               )

    def return_wav_file_binary(self, audio, sample_rate=sample_rate):
        # convert pytorch tensor to numpy array
        np_arr = audio.detach().cpu().numpy()

        # convert numpy array to wav file
        buff = io.BytesIO()
        write_wav(buff, sample_rate, np_arr)

        return buff.read()

    def return_pcm_audio(self, audio):
        # convert pytorch tensor to numpy array
        np_arr = audio.detach().cpu().numpy()

        # convert numpy array to raw PCM bytes
        pcm_bytes = np_arr.tobytes()

        return pcm_bytes

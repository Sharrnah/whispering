import os
from pathlib import Path

from ..Singleton import SingletonMeta

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC

import Plugins
import audio_tools
import downloader
import settings

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009


TTS_MODEL_LINKS = {
    # Models
    "maya1": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/maya1-tts/maya1.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/maya1-tts/maya1.zip",
            "https://s3.libs.space:9000/ai-models/maya1-tts/maya1.zip",
        ],
        "checksum": "cc70921dc618c230eb78b4476bc9292b038687185dfc0182dc02675f7627561f",
        "file_checksums": {
            "chat_template.jinja": "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
            "config.json": "2cfdde4a84cf7ac3654ea5cca50d782b987dfd2a942dc3a3101814e93228c92f",
            "generation_config.json": "1373bcfcb512cf3230c6ffbce471d325a5ab4e28f168beff6179e449d3faece5",
            "model-00001-of-00002.safetensors": "f1dae409f70c5beb92916662c6bc389b9b235ac8aa5edd19a4dcb87e37a73074",
            "model-00002-of-00002.safetensors": "df22e9a90c1bea262250982640b119e6020474736991da482cb6ed56dd23d045",
            "special_tokens_map.json": "bebc5d683318466acb710f7ea1956bc1b88a3fa0c9fa515a4b35b912d8826935",
            "tokenizer.json": "6c5e5b1d89b7e3738e5a5a4f93c326d8f3292ea83f9c560b8dbb6d66fb851973",
            "tokenizer\\chat_template.jinja": "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
            "tokenizer\\special_tokens_map.json": "bebc5d683318466acb710f7ea1956bc1b88a3fa0c9fa515a4b35b912d8826935",
            "tokenizer\\tokenizer.json": "6c5e5b1d89b7e3738e5a5a4f93c326d8f3292ea83f9c560b8dbb6d66fb851973",
            "tokenizer\\tokenizer_config.json": "e188e557f86c2d1893e0bb973cd868a7420472ac4c6b7de1eed398612e3c7518",
            "tokenizer_config.json": "e188e557f86c2d1893e0bb973cd868a7420472ac4c6b7de1eed398612e3c7518"
        },
        "path": "maya1",
    },
    "snac_24khz": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/orpheus-tts/snac_24khz.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/orpheus-tts/snac_24khz.zip",
            "https://s3.libs.space:9000/ai-models/orpheus-tts/snac_24khz.zip",
        ],
        "checksum": "8ab9b76ea04fc4579c78a7707bb6504d3fb7ac72a0781d2747931f3a22de0b28",
        "file_checksums": {
            "config.json": "e119b9366d4f5e73c6ca5f31137c4ff361578bbb132953a5203afe037c4012be",
            "pytorch_model.bin": "4b8164cc6606bfa627f1a784734c1e539891518f1191ed9194fe1e3b9b4bff40"
        },
        "path": "snac_24khz",
    }
}

model_list = {
    "English": ["maya1"],
}

class MayaOne(metaclass=SingletonMeta):
    model = None
    snac_model = None
    tokenizer = None
    sample_rate = 24000
    last_generation = {"audio": None, "sample_rate": None}
    compute_device = "cpu"
    compute_device_str = "cpu"
    snac_device = "cuda" if torch.cuda.is_available() else "cpu"

    last_model = ""
    download_state = {"is_downloading": False}
    cache_path = Path(Path.cwd() / ".cache" / "maya1-tts")

    special_settings = {
        "voice_description": "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
    }

    def __init__(self):
        os.makedirs(self.cache_path, exist_ok=True)
        self.set_compute_device(settings.GetOption("tts_ai_device"))

        if not self.snac_model:
            self.download_model("snac_24khz")
        if not self.model:
            self.download_model("maya1")
        if not self.snac_model or not self.model:
            self.load()

    def list_models(self):
        return model_list

    def list_models_indexed(self):
        return tuple([{"language": language, "models": models} for language, models in self.list_models().items()])

    def set_special_setting(self, special_settings):
        self.special_settings = special_settings

    def _ensure_special_settings(self):
        # ensure special settings are in global settings
        special_settings = settings.GetOption("special_settings")
        if not isinstance(special_settings, dict):
            special_settings = {}

        tts_cfg = special_settings.get("tts_maya1")
        if isinstance(tts_cfg, dict):
            # Merge defaults to ensure new keys exist
            merged = {**self.special_settings, **tts_cfg}
            self.special_settings = merged
        else:
            # add without dropping other keys
            special_settings["tts_maya1"] = self.special_settings
            settings.SetOption("special_settings", special_settings)

    def load_snac_model(self, model_name):
        if self.snac_model is None:
            self.snac_model = SNAC.from_pretrained(str(Path(self.cache_path / model_name).resolve())).eval()
            self.snac_model = self.snac_model.to(self.snac_device)
            print(f"SNAC model {model_name} loaded")

    def load_model(self, model='maya1'):
        if self.model is None:
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=False,
            #     load_in_8bit=True,
            #     bnb_4bit_use_double_quant=False,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16
            # )

            # attention_implementation = 'sdpa'
            # if transformers.utils.is_flash_attn_2_available():
            #     attention_implementation = 'flash_attention_2'

            self.model = AutoModelForCausalLM.from_pretrained(
                str(Path(self.cache_path / model).resolve()),
                torch_dtype=torch.bfloat16,
                # quantization_config=quantization_config,

                #attn_implementation=attention_implementation,
                #_attn_implementation=attention_implementation,
            )

            self.model.to(self.compute_device)
            self.tokenizer = AutoTokenizer.from_pretrained(str(Path(self.cache_path / model).resolve()))
            self.last_model = model
        pass

    def load(self, model='maya1'):
        self.load_snac_model("snac_24khz")
        self.load_model(model)

    def download_model(self, model_name):
        downloader.download_model({
            "model_path": self.cache_path,
            "model_link_dict": TTS_MODEL_LINKS,
            "model_name": model_name,
            "title": "Text 2 Speech (Maya1 TTS)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]


    def set_compute_device(self, device):
        self.compute_device_str = device
        if device is None or device == "cuda" or device == "auto" or device == "":
            self.compute_device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            #device = torch.device(self.compute_device_str)
            device = self.compute_device_str
        self.compute_device = device

    def tts(self, text, remove_silence=True, silence_after_segments=0.2, normalize=True):
        print("TTS requested Maya1 TTS")
        self._ensure_special_settings()

        self.set_compute_device(settings.GetOption('tts_ai_device'))

        tts_volume = settings.GetOption("tts_volume")

        description = self.special_settings["voice_description"]

        prompt = self.build_prompt(self.tokenizer, description, text)


        # Generate emotional speech
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}


        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # Increase to let model finish naturally
                min_new_tokens=28,  # At least 4 SNAC frames
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,  # Prevent loops
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Extract generated tokens (everything after the input prompt)
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()

        print(f"Generated {len(generated_ids)} tokens")

        # Debug: Check what tokens we got
        print(f"   First 20 tokens: {generated_ids[:20]}")
        print(f"   Last 20 tokens: {generated_ids[-20:]}")

        # Check if EOS was generated
        if CODE_END_TOKEN_ID in generated_ids:
            eos_position = generated_ids.index(CODE_END_TOKEN_ID)
            print(f" EOS token found at position {eos_position}/{len(generated_ids)}")

        # Extract SNAC audio tokens
        snac_tokens = self.extract_snac_codes(generated_ids)

        print(f"Extracted {len(snac_tokens)} SNAC tokens")


        # Check for SOS token
        if CODE_START_TOKEN_ID in generated_ids:
            sos_pos = generated_ids.index(CODE_START_TOKEN_ID)
            print(f"   SOS token at position: {sos_pos}")
        else:
            print(f"   No SOS token found in generated output!")

        if len(snac_tokens) < 7:
            print("Error: Not enough SNAC tokens generated")
            return

        # Unpack SNAC tokens to 3 hierarchical levels
        levels = self.unpack_snac_from_7(snac_tokens)
        frames = len(levels[0])

        print(f"Unpacked to {frames} frames")
        print(f"   L1: {len(levels[0])} codes")
        print(f"   L2: {len(levels[1])} codes")
        print(f"   L3: {len(levels[2])} codes")

        # Convert to tensors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes_tensor = [
            torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
            for level in levels
        ]

        # Generate final audio with SNAC decoder
        print("\n[4/4] Decoding to audio...")
        with torch.inference_mode():
            z_q = self.snac_model.quantizer.from_codes(codes_tensor)
            audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()

        # Trim warmup samples (first 2048 samples)
        if len(audio) > 2048:
            audio = audio[2048:]

        duration_sec = len(audio) / 24000
        print(f"Audio generated: {len(audio)} samples ({duration_sec:.2f}s)")

        # print type of audio
        print(f"Audio type: {type(audio)}")

        #audio = audio.detach().squeeze().to("cpu").numpy()
        if tts_volume != 1.0:
            audio = audio_tools.change_volume(audio, tts_volume)

        # numpy array to torch.Tensor
        audio = torch.from_numpy(audio).float()

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': audio, 'sample_rate': self.sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            audio = plugin_audio['audio']

        # save last generation in memory
        self.last_generation = {"audio": audio, "sample_rate": self.sample_rate}

        print("TTS generation finished")

        return audio, self.sample_rate

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

    def build_prompt(self, tokenizer, description: str, text: str) -> str:
        """Build formatted prompt for Maya1."""
        soh_token = tokenizer.decode([SOH_ID])
        eoh_token = tokenizer.decode([EOH_ID])
        soa_token = tokenizer.decode([SOA_ID])
        sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
        eot_token = tokenizer.decode([TEXT_EOT_ID])
        bos_token = tokenizer.bos_token

        formatted_text = f'<description="{description}"> {text}'

        prompt = (
                soh_token + bos_token + formatted_text + eot_token +
                eoh_token + soa_token + sos_token
        )

        return prompt


    def extract_snac_codes(self, token_ids: list) -> list:
        """Extract SNAC codes from generated tokens."""
        try:
            eos_idx = token_ids.index(CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx = len(token_ids)

        snac_codes = [
            token_id for token_id in token_ids[:eos_idx]
            if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
        ]

        return snac_codes


    def unpack_snac_from_7(self, snac_tokens: list) -> list:
        """Unpack 7-token SNAC frames to 3 hierarchical levels."""
        if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
            snac_tokens = snac_tokens[:-1]

        frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
        snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]

        if frames == 0:
            return [[], [], []]

        l1, l2, l3 = [], [], []

        for i in range(frames):
            slots = snac_tokens[i*7:(i+1)*7]
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,
                ])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
                ])

        return [l1, l2, l3]

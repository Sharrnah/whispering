import io
import os
import re
import time
import traceback
from pathlib import Path

#import numpy
import numpy as np
#import soundfile as sf
import torch
import torchaudio
#import torchaudio
from scipy.io.wavfile import write as write_wav

import Plugins
import audio_tools
import downloader
#from cached_path import cached_path

import settings
from Models.Singleton import SingletonMeta

from Models.TTS.F5TTS.model.backbones.unett import UNetT
from Models.TTS.F5TTS.model.backbones.dit import DiT

from Models.TTS.F5TTS.infer.utils_infer import (
    load_vocoder,
    load_model,
    #preprocess_ref_audio_text,
    infer_process,
    infer_batch_process,
    chunk_text,
    #remove_silence_for_generated_wav,
)


def estimate_remaining_time(total_segments, segment_times, segments_for_estimate=3, last_x_segments=None):
    """
    Estimates the remaining time based on the average time of specified segment times.

    Parameters:
    total_segments (int): Total number of segments.
    segment_times (list): List of times taken for each segment.
    segments_for_estimate (int): Minimum number of segments needed to start estimating.
    last_x_segments (int, optional): Number of recent segments to use for estimating the average time.
                                     If None, all available segments are used.

    Returns:
    str: Formatted string of estimated remaining time or "[estimating...]" if not enough data.
    """
    if len(segment_times) < segments_for_estimate:
        return " [estimating...]"

    if last_x_segments is not None:
        # Use only the last x segments for estimation
        relevant_segment_times = segment_times[-last_x_segments:]
    else:
        # Use all available segment times for estimation
        relevant_segment_times = segment_times

    # Calculate the average time per segment
    avg_time_per_segment = sum(relevant_segment_times) / len(relevant_segment_times)

    # Calculate the remaining segments
    remaining_segments = total_segments - len(segment_times)

    # Estimate the remaining time
    estimated_remaining_time = avg_time_per_segment * remaining_segments

    # Convert estimated time to hours, minutes, and seconds
    hours, rem = divmod(estimated_remaining_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format the estimated time
    estimated_time_str = ""
    if hours > 0:
        estimated_time_str += f"{int(hours)} hrs. "
    if minutes > 0:
        estimated_time_str += f"{int(minutes)} min. "
    estimated_time_str += f"{int(seconds)} sec."

    if estimated_time_str:
        return f" [~ {estimated_time_str} remaining]"
    else:
        return ""


cache_path = Path(Path.cwd() / ".cache" / "f5tts-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)

vocoder_paths = {
    "vocos": Path(cache_path / "vocos"),
    "bigvgan": Path(cache_path / "bigvgan"),
}

failed = False

TTS_MODEL_LINKS = {
    # Models
    "F5-TTS": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Base.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Base.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Base.zip",
        ],
        "checksum": "06fa9afce84910a1b6ec6126a001b9f4e716a1122ad8cd5c8914ab9da5351494",
        "file_checksums": {
            "model.safetensors": "4180310f91d592cee4bc14998cd37c781f779cf105e8ca8744d9bd48ca7046ae",
            "vocab.txt": "4e173934be56219eb38759fa8d4c48132d5a34454f0c44abce409bcf6a07ec46",
        },
        "path": "F5-TTS_Base",
    },
    "E2-TTS": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/E2-TTS_Base.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/E2-TTS_Base.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/E2-TTS_Base.zip",
        ],
        "checksum": "175cde51b2496498a09d9eae52ecb00a1ac5abcf2109f1f61b17c48504607997",
        "file_checksums": {
            "model.safetensors": "8a813cd26fd21b298734eece9474bfea842a585adff98d2bb89fd384b1c00ac7",
            "vocab.txt": "4e173934be56219eb38759fa8d4c48132d5a34454f0c44abce409bcf6a07ec46",
        },
        "path": "E2-TTS_Base",
    },
    "F5-TTS-bigvgan": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Base-bigvgan.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Base-bigvgan.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Base-bigvgan.zip",
        ],
        "checksum": "109c7fd25af309d2e87b622c0af2a26e02aa2e923ede35cfaaf6e5cb7478536a",
        "file_checksums": {
            "model.pt": "bdab3e92fc2b77447aa8c46aac77531d970822b191ca198e5ab94aef99265df9",
            "vocab.txt": "4e173934be56219eb38759fa8d4c48132d5a34454f0c44abce409bcf6a07ec46"
        },
        "mel_spec_type": "bigvgan",
        "path": "F5-TTS_Base-bigvgan",
    },
    "F5-TTS_French": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_French.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_French.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_French.zip",
        ],
        "checksum": "29b5196e17e49aa79303389a3e73fff5b2f6112a0f4be637a636f91105669b96",
        "file_checksums": {
            "model.pt": "812347d5dcd375e5d84fd3cb156468ca4ee99bd6fbe29d86f487a3c964b69e34",
            "vocab.txt": "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
        },
        "path": "F5-TTS_French",
    },
    "F5-TTS_German": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_German.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_German.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_German.zip",
        ],
        "checksum": "d5a37a8a1cf803723e6a887a966b8ae3b0cbf90ed345eeb853a2472ec61256a7",
        "file_checksums": {
            "model.safetensors": "3f01d5e49e63a6811a22600ddfba8a5229fc73e8441371d184a5e05029da6ae7",
            "vocab.txt": "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
        },
        "path": "F5-TTS_German",
    },
    "F5-TTS_German-bigvgan": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_German-bigvgan.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_German-bigvgan.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_German-bigvgan.zip",
        ],
        "checksum": "97c02cce5b460a0531a8cad66c641869040c4b7214b6501543f3d67ed6d57fdd",
        "file_checksums": {
            "model.safetensors": "fb4740a09a2c2513cb337437724fd8a1213ddaf9d60fb18d29812d6b5d161be7",
            "vocab.txt": "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
        },
        "mel_spec_type": "bigvgan",
        "path": "F5-TTS_German-bigvgan",
    },
    "F5-TTS_Italian": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Italian.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Italian.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Italian.zip",
        ],
        "checksum": "047d44af4d67358c6d7866bb93f362bbd1fb9f53dde97628b248d83a4fe3095a",
        "file_checksums": {
            "model.safetensors": "c92b19a07843bda8bf55c8b525e67051ef4f95d3bd28cec2d330a45a48a1ac91",
            "vocab.txt": "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
        },
        "path": "F5-TTS_Italian",
    },
    "F5-TTS_Japanese": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Japanese.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Japanese.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Japanese.zip",
        ],
        "checksum": "09e778610e9061204c8168eeb6c1dd6348628e123545ae6edd250faf89c9ac4b",
        "file_checksums": {
            "model.pt": "6e7fec0716a09401b92c39c6869325e331b20bcf6cc16a414e7e3ac87ff6c854",
            "vocab.txt": "f405cceeeaf2461b8ee2247118f93140db617ddc25d1447790d7e93a761f89e3"
        },
        "mel_spec_type": "bigvgan",
        "path": "F5-TTS_Japanese",
    },
    "F5-TTS_Spanish": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Spanish.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Spanish.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Spanish.zip",
        ],
        "checksum": "a47edb335aa82ec9aa0737ddd8d6b7d8a1795a2a1216f4d8253a24d24d193dc6",
        "file_checksums": {
            "model.safetensors": "9cd5757a92c0b979e9769558fea922243c4916090ad93dd2c595b73e6f05a3b2",
            "vocab.txt": "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
        },
        "path": "F5-TTS_Spanish",
    },
    "F5-TTS_Russian": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Russian.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Russian.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Russian.zip",
        ],
        "checksum": "ddd63d60f12e1ab62df67294f5105528157a9cc9fbc7db7e1b2785057003187a",
        "file_checksums": {
            "model.safetensors": "2f1b8f2c4df0f3d79569e1a3df7dcb5ea55e535198e30deff030a62d96c5e463",
            "vocab.txt": "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
        },
        "path": "F5-TTS_Russian",
    },
    "F5-TTS_Vietnamese": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Vietnamese.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Vietnamese.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Vietnamese.zip",
        ],
        "checksum": "06d7b24e9c419142d3ea1e1e09e9b26e61d7951bfb072373fe3b8582a1dd2649",
        "file_checksums": {
            "model.safetensors": "482032c87417d421afdc821e866f0b9eea0300ecf22062d6adef4cb24e9f5488",
            "vocab.txt": "cf30aa5d265aeddbb4804cccb3f46c45dae945d40ecd5838e66f69a94a9090c8"
        },
        "path": "F5-TTS_Vietnamese",
    },
    "F5-TTS_Malaysian": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Malaysian.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Malaysian.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Malaysian.zip",
        ],
        "checksum": "ab465605eabb9713a83374d223b3cf7f0ece984cbb081bc36780b1fdb001387f",
        "file_checksums": {
            "model.pt": "d2f61c8ad573e9b95f9da3bcf49dddebb9dbc3d0549682fb774d2c041d6af034",
            "vocab.txt": "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
        },
        "path": "F5-TTS_Malaysian",
    },
    "F5-TTS_Greek": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5-TTS_Greek.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5-TTS_Greek.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5-TTS_Greek.zip",
        ],
        "checksum": "0a453c321de7a1be2b5041180f30566e3392cbb404227560593522a69a6a83b1",
        "file_checksums": {
            "model.safetensors": "3051d025dc0fbb70733098fc9fd0d0a516bb404921002c4c1b9bf7283a00effa",
            "vocab.txt": "7eaef18c7b89ae0c6ac52a1d6ff6e121e396cc87827f88eaf032f44150842a78"
        },
        "path": "F5-TTS_Greek",
    },
    # Vocoders
    "vocos": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/vocos.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/vocos.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/vocos.zip",
        ],
        "checksum": "e9af55494b4729ca10b10e7c3d1e1769fe12dc1fcca75cdbd0219699dce55677",
        "file_checksums": {
            "README.md": "5858497d76d58914b29958a0f448e7b4fd3bb54940100baaa7520d4a56d8b3df",
            "config.yaml": "da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc",
            "pytorch_model.bin": "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
        },
        "path": "vocos",
    },
    "bigvgan": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/bigvgan.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/bigvgan.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/bigvgan.zip",
        ],
        "checksum": "16a9be4fcff4cbfdc3dd644a3374a66069b08123eb6556e2af893f06475d63c8",
        "file_checksums": {
            "LICENSE": "90459cd52fc41bd723df7c0c76fac1e4dd60e6bfd644a7e2a93f325bed4f6d95",
            "README.md": "1ab4fe827dc65913b6745cb0fa7a96fb0a4a89ff32ec143a0eecfac079e03d1e",
            "activations.py": "3ba94028aebabfc994bcd746bf9cbe92ecace528434c922c480be6ada182cad6",
            "alias_free_activation\\cuda\\__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "alias_free_activation\\cuda\\activation1d.py": "54778a4308d359cac8348bc28f5407652982b22732c3c768a6721183c26f5b0e",
            "alias_free_activation\\cuda\\anti_alias_activation.cpp": "222ce6cd687fdc1d541a4bedbf5f3578f321bc2649996b2747ebc60ddd2e1d8e",
            "alias_free_activation\\cuda\\anti_alias_activation_cuda.cu": "e3fa60fbf80fd95cad6f05dd605c85393b6539d6684502a56b64c348c655de7d",
            "alias_free_activation\\cuda\\compat.h": "39e530d6d9cf5eda60c25b899cf0ba87c70cdca3424de7cb1716adad8f212388",
            "alias_free_activation\\cuda\\load.py": "6ee5cbfaedc6b73cf2bff9677f18bcc5a46c91a088523e853bca12805a909e8a",
            "alias_free_activation\\cuda\\type_shim.h": "3e32f2fec72b2b6749389acd64c336b2932a3764b686634c696af8151df0e38c",
            "alias_free_activation\\torch\\__init__.py": "2e3138e1052e377ba2e51ea59c7d5d255a519559001757af21dbec4cb9c22471",
            "alias_free_activation\\torch\\act.py": "651448005dd5ae0c193da60d1a50c0aa53550a4a3050e29ed7d35cf86410e212",
            "alias_free_activation\\torch\\filter.py": "acf2257276e617dd3161e53abc0a1582e586a3d2d618a43235c7f83818b4e179",
            "alias_free_activation\\torch\\resample.py": "4d7e1bf4169d03f59360f34f33d029c04b99364150f6737563ebfbee0ab49d79",
            "bigvgan.py": "2b2c5d7bdc818b90ffc2e1a65776823e46030f51a4e0d7a2d42e9b6d05cba101",
            "bigvgan_discriminator_optimizer.pt": "41f7718f34fbbd85975316e89b1fbb10e3fb36e7a0dc9e03aefa9b7aed65f8f5",
            "bigvgan_discriminator_optimizer_3msteps.pt": "efcc619cd2f6e7bd3f9827efeb63212a882e86eddd6e90da120c49d8b279ff90",
            "bigvgan_generator.pt": "6f9c5715550c9d0f11159ceb8935638da5aeb19e27d1e63677632df095e376f5",
            "bigvgan_generator_3msteps.pt": "f6e3ad0dc7efe9a60b13be041a8bc5c20f2080a0faa54ed8d625fa52ae0846c7",
            "config.json": "d77e2c96583ca2296ac112a56ec7cc6bd5da4bf7681ceff18448bedc4fcf6512",
            "env.py": "54ae665797fbb20ed3fc9b856be688f6b8a903b0bed8ff7edabff81dd9cdcd33",
            "meldataset.py": "691e8413fb4c65ee3f603a43b19bdd69652a06a150ce82e72c43363cd268dfc6",
            "nv-modelcard++\\.gitkeep": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "nv-modelcard++\\bias.md": "0b374821eb4a1a1a0c4f7e3d5d0474620eea573ec3cc4bd238cce78b6595b375",
            "nv-modelcard++\\explainability.md": "58d21b235cca347db09b7e120199351840b456d7475d90006df0ff7c49239e35",
            "nv-modelcard++\\overview.md": "0c9016e379a90a4cbb8476e09845d3e197fa4bb12bd1912ccb01b96436849721",
            "nv-modelcard++\\privacy.md": "87f3de69d52bf16ea8bc5fe8671eee7fd2a4ba0ce9fd7ae1fe9fbe9c7771df0d",
            "nv-modelcard++\\safety.md": "f244890262cd9074fad924fcadfc8bfa66e99ddeb2bf33b379ad04f325861386",
            "utils.py": "04eee590ca04ca33a6b2b79802a6514bbe7d6867164c72d61f9404ea9b0101c4"
        },
        "path": "bigvgan",
    },
    # Default Voices
    "voices": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/voices.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/voices.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/voices.zip",
        ],
        "checksum": "3997e6e1c7a7c0255bac3fd6ad9493098cfda410091755399e873e848459ac96",
        "file_checksums": {
            #"Info.txt": "19455228a1ef69f6abe1dfa7f8477a1da600ed371ca237032263786d8a7a51ec",
            "Announcer_Ahri.txt": "65cdbe885b89037dc651bea9bb7c41077471a2d7168e25c905c7034da7de285d",
            "Announcer_Ahri.wav": "2a3fd17d45b3c5633dd64e2d80a6e3fc924fa829da405e3b591a5bacdc88f9fc",
            "Attenborough.txt": "4c617f7adc60b992de93abd18bd447da5c882db7d04d9d51b241cdf79cbda6a1",
            "Attenborough.wav": "358540c89932baf1103960d99949b78ea7466f95b2225fdcd8f8bb8b976f09ee",
            "Jane.txt": "58e939100b6422f76e631d445a957047fa915ba6727f984ebdcecfa3418f5d08",
            "Jane.wav": "d1d2235af1a4408c641a765427978486f5cca9b369fc6456d8086449f1f92fe3",
            "Justin.txt": "6ce2802c88bd83ef12ecb3338f1bf6f8bc5bc12212b3cd1d2863d0d3ab93632b",
            "Justin.wav": "a83c37f408b53efaeb9189f166c6669d1a0dc6cf779e85913fa9cbbbbe0d5aaf",
            "Xiaochen.txt": "1316b1e27871565b1d7cd4f64b0521a37632cc15d1ea0944d18394bdaf76d8e2",
            "Xiaochen.wav": "7f0b735e188a06dc9f104eeb3fd71a3ef580d1f2133c95630c92a244dd253732",
            "en_0.txt": "3bb999d455ca88b8eca589bd16d3b99db8b324b5f8c57e3283e4bb4db8593243",
            "en_0.wav": "f006e2e9c76523bde4f5bbe67a7be9a600786d7432cbcc9486bc9501053298b7",
            "en_1.txt": "79cccada817b316fa855dc8ca04823f59a11c956b5780fbb3267ddf684c8e145",
            "en_1.wav": "b0e22048e72414fcc1e6b6342e47a774d748a195ed34e4a5b3fcf416707f2b71",
            "test_zh_1_ref_short.wav": "96724a113240d1f82c6ded1334122f0176b96c9226ccd3c919e625bcfd2a3ede"
        },
        "path": "voices",
    },
}

model_list = {
    "English & Chinese": ["F5-TTS", "E2-TTS", "F5-TTS-bigvgan"],
    "French": ["F5-TTS_French"],
    "German": ["F5-TTS_German", "F5-TTS_German-bigvgan"],
    "Greek": ["F5-TTS_Greek"],
    "Italian": ["F5-TTS_Italian"],
    "Japanese": ["F5-TTS_Japanese"],
    "Spanish": ["F5-TTS_Spanish"],
    "Russian": ["F5-TTS_Russian"],
    "Vietnamese": ["F5-TTS_Vietnamese"],
    "Malaysian": ["F5-TTS_Malaysian"],
    "Custom": ["F5-TTS_custom", "F5-TTS_custom-bigvgan", "E2-TTS_custom"],
}

speed_mapping = {
    "": 1.0,        # Default speed when empty
    "x-slow": 0.5,  # Extra slow speed
    "slow": 0.75,   # Slow speed
    "medium": 1.0,  # Medium speed (default)
    "fast": 1.25,   # Fast speed
    "x-fast": 1.5   # Extra fast speed
}

class F5TTS(metaclass=SingletonMeta):
    lang = 'en'
    model_id = 'F5-TTS'
    model = None
    compute_device = "cpu"

    audio_streamer = None

    target_sample_rate = 24000
    target_rms = 0.1
    n_mel_channels = 100
    hop_length = 256
    ode_method = "euler"
    speed = 1.0
    #nfe_step = 32
    nfe_step = 64

    device = None
    ema_model = None
    vocoder = None
    vocoder_name = "vocos"

    #currently_downloading = False
    download_state = {"is_downloading": False}

    last_generation = {"audio": None, "sample_rate": None}

    voice_list = []

    config = {
        "model": "F5-TTS",
        #"model": "E2-TTS",
        "ref_audio": str(Path(cache_path / "voices" / "en_1.wav").resolve()),
        # If an empty "", transcribes the reference audio automatically.
        "ref_text": "Some call me nature, others call me mother nature.",
        #"gen_text": "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.",
        "gen_text": "This is a test.",
        # File with text to generate. Ignores the text above.
        "gen_file": "",
        "remove_silence": True,
        "output_dir": "tests",
        "voices": {},
    }

    def __init__(self):
        self.compute_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if not self.voice_list:
            self.update_voices()
        pass

    def download_model(self, model_name):
        downloader.download_model({
            "model_path": cache_path,
            "model_link_dict": TTS_MODEL_LINKS,
            "model_name": model_name,
            "title": "Text 2 Speech (F5/E2 TTS)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)

    def _get_model_name(self):
        model = "F5-TTS"
        if len(settings.GetOption('tts_model')) == 2:
            #language = settings.GetOption('tts_model')[0]
            model = settings.GetOption('tts_model')[1]
            # remove language part from string example: " (en & zh)"
            model = re.sub(r'\(.*?\)', '', model).strip()

        if "custom" in model:
            return model

        if model == "" or model not in TTS_MODEL_LINKS:
            model = "F5-TTS"

        return model

    # def apply_vocos_on_audio(self, audio_data, sample_rate=24000):
    #     # check if audio_data is bytes
    #     wav_file = audio_data
    #     if isinstance(audio_data, bytes):
    #         wav_file = io.BytesIO(audio_data)
    #
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     y, sr = torchaudio.load(wav_file)
    #     if y.size(0) > 1:  # mix to mono
    #         y = y.mean(dim=0, keepdim=True)
    #     y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=sample_rate)
    #     y = y.to(device)
    #     bandwidth_id = torch.tensor([2]).to(device)  # 6 kbps
    #     y_hat = self.vocoder.decode(y, bandwidth_id=bandwidth_id)
    #
    #     #audio_data_np_array = audio_tools.resample_audio(y_hat, 24000, sample_rate, target_channels=1,
    #     #                                                 input_channels=1, dtype="float32")
    #
    #     #audio_data_16bit = np.int16(audio_data_np_array * 32767)  # Convert to 16-bit PCM
    #
    #     #buff = io.BytesIO()
    #     #write_wav(buff, sample_rate, audio_data_16bit)
    #
    #     #buff.seek(0)
    #     return y_hat

    def set_compute_device(self, device):
        self.compute_device_str = device
        if device is None or device == "cuda" or device == "auto" or device == "":
            self.compute_device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            #device = torch.device(self.compute_device_str)
            device = self.compute_device_str
        #elif device == "mps":
            #device = torch.device("mps")
        #    device = "mps"
        #elif device == "cpu":
            #device = torch.device("cpu")
        #    device = "cpu"
        #elif device.startswith("direct-ml"):
            #device_id = 0
            #device_id_split = device.split(":")
            #if len(device_id_split) > 1:
            #    device_id = int(device_id_split[1])
            #import torch_directml
            #device = torch_directml.device(device_id)
        self.compute_device = device

    def load(self):
        model = self._get_model_name()

        if "custom" not in model:
            model_directory = Path(cache_path / TTS_MODEL_LINKS[model]["path"])
        else:
            model_directory = Path(cache_path / model)
            os.makedirs(model_directory, exist_ok=True)

        self.download_model("vocos")
        self.download_model("bigvgan")
        self.download_model("voices")
        if "custom" not in model:
            self.download_model(model)

        self.model_id = model

        self.set_compute_device(settings.GetOption('tts_ai_device'))

        # load models
        if model.startswith("F5-TTS"):
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            if Path(model_directory / f"model.safetensors").exists():
                ckpt_file = str(Path(model_directory / f"model.safetensors").resolve())
            elif Path(model_directory / f"model.pt").exists():
                ckpt_file = str(Path(model_directory / f"model.pt").resolve())
            vocab_file = str(Path(model_directory / "vocab.txt").resolve())

        # check if model starts with the string "E2-TTS"
        elif model.startswith("E2-TTS"):
            model_cls = UNetT
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            if Path(model_directory / f"model.safetensors").exists():
                ckpt_file = str(Path(model_directory / f"model.safetensors").resolve())
            elif Path(model_directory / f"model.pt").exists():
                ckpt_file = str(Path(model_directory / f"model.pt").resolve())
            vocab_file = str(Path(model_directory / "vocab.txt").resolve())

        if "custom" not in model:
            # set model specific cfg if needed.
            if "model_cfg" in TTS_MODEL_LINKS[model]:
                model_cfg = TTS_MODEL_LINKS[model]["model_cfg"]

            mel_spec_type = "vocos"
            if "mel_spec_type" in TTS_MODEL_LINKS[model]:
                mel_spec_type = TTS_MODEL_LINKS[model]["mel_spec_type"]
        else:
            # set custom model vocoder / mel_spec_type.
            mel_spec_type = "vocos"
            if "bigvgan" in model:
                mel_spec_type = "bigvgan"

        try:
            self.ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=mel_spec_type, vocab_file=vocab_file, device=self.compute_device)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return

        # load vocoder model
        vocoder_name = "vocos"
        try:
            if "custom" not in model:
                if self.vocoder is None or ("mel_spec_type" not in TTS_MODEL_LINKS[self.model_id] and self.vocoder_name != "vocos") or ("mel_spec_type" in TTS_MODEL_LINKS[self.model_id] and self.vocoder_name != TTS_MODEL_LINKS[self.model_id]["mel_spec_type"]):
                    if "mel_spec_type" in TTS_MODEL_LINKS[self.model_id]:
                        vocoder_name = TTS_MODEL_LINKS[self.model_id]["mel_spec_type"]
                    print(f"loading vocoder model: {vocoder_name}")
                    try:
                        self.load_vocoder(vocoder_name, device=self.compute_device)
                    except Exception as e:
                        print(e)
                        print(f"Failed to load vocoder model: {vocoder_name}, falling back to default vocoder.")
                        self.load_vocoder("vocos", device=self.compute_device)
            else:
                vocoder_name = mel_spec_type
                print(f"loading vocoder model: {vocoder_name}")
                try:
                    self.load_vocoder(vocoder_name, device=self.compute_device)
                except Exception as e:
                    print(e)
                    print(f"Failed to load vocoder model: {vocoder_name}, falling back to default vocoder.")
                    self.load_vocoder("vocos", device=self.compute_device)

        except Exception as e:
            print(e)
            traceback.print_exc()
            return

        print(f"Model {model} with vocoder {vocoder_name} loaded successfully.")

    def list_models(self):
        return model_list

    def list_models_indexed(self):
        model_list = self.list_models()
        return tuple([{"language": language, "models": models} for language, models in model_list.items()])

    def _get_voices(self):
        return self.voice_list

    def update_voices(self):
        # find all voices that have both a .wav and .txt file
        voice_files = [f.stem for f in voices_path.iterdir() if f.is_file() and f.suffix == ".wav"]
        voice_list = []
        for voice_id in voice_files:
            wav_file = voices_path / f"{voice_id}.wav"
            txt_file = voices_path / f"{voice_id}.txt"
            if wav_file.exists() and txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as file:
                    text_content = file.read().strip()
                voice_list.append({"name": voice_id, "wav_filename": str(wav_file.resolve()), "text_content": text_content})
        self.voice_list = voice_list

    def list_voices(self):
        self.update_voices()
        voice_list = [{"name": voice["name"], "value": voice["name"]} for voice in self._get_voices()]
        voice_list.append({"name": "open_voice_dir", "value": "open_dir:"+str(voices_path.resolve())})
        return voice_list

    def get_voice_by_name(self, voice_name):
        for voice in self._get_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    def remove_silence_parts(self, audio_data: bytes|np.ndarray, sample_rate, dtype=None, silence_threshold=0.03, max_silence_length=0.8,
                             keep_silence_length=0.20):
        if isinstance(audio_data, bytes):
            # Convert bytes to numpy array
            audio = np.frombuffer(audio_data, dtype=dtype)
        elif isinstance(audio_data, np.ndarray):
            audio = audio_data
            if dtype is None:
                dtype = audio_data.dtype
        else:
            raise ValueError("Unsupported audio format. Please provide bytes or numpy array.")

        audio_abs = np.abs(audio)
        above_threshold = audio_abs > silence_threshold

        # Convert length parameters to number of samples
        max_silence_samples = int(max_silence_length * sample_rate)
        keep_silence_samples = int(keep_silence_length * sample_rate)

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
                        # Subtract keep_silence_samples from the start and add it to the end
                        start = max(last_silence_end - keep_silence_samples, 0)
                        end = min(silence_start + keep_silence_samples, len(audio))
                        chunks.append(audio[start:end])
                        last_silence_end = i
                    silence_start = None

        # Append the final chunk of audio after the last silence
        if last_silence_end < len(audio):
            start = max(last_silence_end - keep_silence_samples, 0)
            end = len(audio)
            chunks.append(audio[start:end])

        if len(chunks) == 0:
            print("No non-silent sections found in audio.")
            return_np_array = np.array([]).astype(dtype)
        else:
            print(f"found {len(chunks)} non-silent sections in audio")
            return_np_array = np.concatenate(chunks).astype(dtype)

        if isinstance(audio_data, bytes):
            return return_np_array.tobytes()
        else:
            return return_np_array

    def load_vocoder(self, vocoder_name="vocos", device=None):
        self.vocoder_name = vocoder_name
        vocoder_local_path = vocoder_paths[vocoder_name]
        self.vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=True, local_path=vocoder_local_path, device=device)

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    def stop(self):
        print("TTS Stop requested")
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    def tts(self, text, ref_audio=None, ref_text=None, remove_silence=True, silence_after_segments=0.2, normalize=True, streaming=False):
        is_streaming_text = ""
        if streaming:
            is_streaming_text = " (streaming)"
        print("TTS requested F5/E2 TTS", is_streaming_text)
        tts_volume = settings.GetOption("tts_volume")
        tts_normalize = settings.GetOption("tts_normalize")
        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)
        return_sample_rate = self.target_sample_rate
        if ref_audio is None and ref_text is None:
            voice_name = settings.GetOption('tts_voice')
            selected_voice = self.get_voice_by_name(voice_name)
            if selected_voice is None:
                print("No voice selected or does not exist. Using default voice 'en_1'.")
                voice_name = "en_1"
                selected_voice = self.get_voice_by_name(voice_name)
            ref_audio = selected_voice["wav_filename"]
            ref_text = selected_voice["text_content"]

        if ref_audio is None:
            ref_audio = self.config["ref_audio"]
        if ref_text is None:
            ref_text = self.config["ref_text"]

        main_voice = {"ref_audio":ref_audio, "ref_text":ref_text}

        # get voice list from _get_voices function
        voices_list = self._get_voices()
        for voice_entry in voices_list:
            self.config["voices"][voice_entry["name"]] = {'ref_audio': voice_entry['wav_filename'], 'ref_text': voice_entry['text_content']}

        if "voices" not in self.config or self.config["voices"] == {}:
            voices = {"main": main_voice}
        else:
            voices = self.config["voices"]
            voices["main"] = main_voice
        #for voice in voices:
            #voices[voice]['ref_audio'], voices[voice]['ref_text'] = preprocess_ref_audio_text(voices[voice]['ref_audio'], voices[voice]['ref_text'])
            #print("Voice:", voice)
            #print("Ref_audio:", voices[voice]['ref_audio'])
            #print("Ref_text:", voices[voice]['ref_text'])

        segment_times = []

        generated_audio_segments = []
        reg1 = r'(?=\[\w+\])'
        chunks = re.split(reg1, text)
        reg2 = r'\[(\w+)\]'

        last_ref_audio_data = None
        streamed_last_ref_audio_filepath = None
        streamed_last_ref_audio_sample_rate = None
        streamed_audio_reference = None

        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        for i, text_chunk in enumerate(chunks):
            text_chunk = text_chunk.strip()  # Remove leading/trailing whitespace
            if not text_chunk:
                continue  # Skip empty chunks

            match = re.match(reg2, text_chunk)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text_chunk = re.sub(reg2, "", text_chunk)
            gen_text = text_chunk.strip()

            if not gen_text:
                continue  # Skip processing if there's no text to generate

            ref_audio = voices[voice]['ref_audio']
            ref_text = voices[voice]['ref_text']

            start_time = time.time()
            estimate_time_full_str = estimate_remaining_time(len(chunks), segment_times, 3)
            print(f"\nTTS progress: {int((i) / len(chunks) * 100)}% ({i} of {len(chunks)} segments){estimate_time_full_str}")

            if streaming:
                chunk_size = settings.GetOption("tts_streamed_chunk_size")

                # cached audio reference loading
                if streamed_last_ref_audio_filepath != ref_audio or streamed_last_ref_audio_filepath is None or streamed_audio_reference is None:
                    streamed_audio_reference, sr = torchaudio.load(ref_audio)
                    # noinspection PyUnusedLocal
                    last_ref_audio_data = streamed_audio_reference
                    # noinspection PyUnusedLocal
                    streamed_last_ref_audio_filepath = ref_audio
                    # noinspection PyUnusedLocal
                    streamed_last_ref_audio_sample_rate = sr
                else:
                    sr = streamed_last_ref_audio_sample_rate
                    streamed_audio_reference = last_ref_audio_data

                max_chars = int(len(ref_text.encode("utf-8")) / (streamed_audio_reference.shape[-1] / sr) * (25 - streamed_audio_reference.shape[-1] / sr))
                gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
                return infer_batch_process(
                    (streamed_audio_reference, sr),
                    ref_text,
                    gen_text_batches,
                    self.ema_model,
                    self.vocoder,
                    mel_spec_type=self.vocoder_name,
                    speed=tts_speed,
                    progress=None,
                    device=self.compute_device,
                    nfe_step=self.nfe_step,
                    streaming=True,
                    chunk_size=chunk_size,
                )
            else:
                audio, final_sample_rate, spectragram = infer_process(ref_audio, ref_text, gen_text, self.ema_model, self.vocoder, mel_spec_type=self.vocoder_name, speed=tts_speed, device=self.compute_device, nfe_step=self.nfe_step, show_info=None)
                return_sample_rate = final_sample_rate

                # Add silence when silence_after_segments > 0 and not last segment
                if silence_after_segments > 0 and i < len(chunks) - 1:
                    silence_samples = int(silence_after_segments * return_sample_rate)
                    audio = np.concatenate([audio, np.zeros(silence_samples, dtype=np.float32)])

                generated_audio_segments.append(audio)

                end_time = time.time()
                segment_times.append(end_time - start_time)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)

            #final_wave = np.frombuffer(final_wave, dtype=np.float32)
            #final_wave = self.apply_vocos_on_audio(final_wave, sample_rate=return_sample_rate)

            if remove_silence:
                final_wave = self.remove_silence_parts(final_wave, sample_rate=return_sample_rate)

            if tts_normalize:
                final_wave, _ = audio_tools.normalize_audio_lufs(
                    final_wave, return_sample_rate, -24.0, -16.0,
                    1.3, verbose=True
                )

            # change volume
            if tts_volume != 1.0:
                final_wave = audio_tools.change_volume(final_wave, tts_volume)

            # save last generation in memory
            self.last_generation = {"audio": final_wave, "sample_rate": return_sample_rate}

            return final_wave, return_sample_rate
            # with open(wave_path, "wb") as f:
            #     sf.write(f.name, final_wave, final_sample_rate)
            #     # Remove silence
            #     if remove_silence:
            #         remove_silence_for_generated_wav(f.name)
            #     print(f.name)

    def tts_streaming(self, text, ref_audio=None, ref_text=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        tts_volume = settings.GetOption("tts_volume")
        tts_normalize = settings.GetOption("tts_normalize")
        self.init_audio_stream_playback()
        audio_stream = self.tts(text, ref_audio, ref_text, remove_silence, silence_after_segments, normalize, streaming=True)

        audio_chunks = []
        for codes_chunk, _ in audio_stream:
            if self.audio_streamer is None:
                break

            if tts_normalize:
                codes_chunk, _ = audio_tools.normalize_audio_lufs(
                    codes_chunk, self.target_sample_rate, -24.0, -16.0,
                    1.3, verbose=False
                )

            # change volume
            if tts_volume != 1.0:
                codes_chunk = audio_tools.change_volume(codes_chunk, tts_volume)

            audio_chunks.append(codes_chunk)
            if self.audio_streamer is not None:
                self.audio_streamer.add_audio_chunk(codes_chunk.tobytes())

        full_audio = np.concatenate(audio_chunks, axis=-1)
        self.last_generation = {"audio": full_audio, "sample_rate": self.target_sample_rate}

        print("TTS generation finished")

        return full_audio, self.target_sample_rate

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
                                                            source_sample_rate=int(self.target_sample_rate),
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
        audio = audio_tools.convert_audio_datatype_to_integer(audio)

        audio_bytes = self.return_wav_file_binary(audio, int(self.target_sample_rate))

        # play audio tensor
        audio_tools.play_audio(audio_bytes, device,
                               source_sample_rate=int(self.target_sample_rate),
                               audio_device_channel_num=1,
                               target_channels=1,
                               input_channels=source_channels,
                               dtype="int16",
                               secondary_device=secondary_audio_device,
                               stop_play=not allow_overlapping_audio,
                               tag="tts"
                               )

    def return_wav_file_binary(self, audio, sample_rate=24000):
        # convert numpy array to wav file
        buff = io.BytesIO()
        write_wav(buff, sample_rate, audio)

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio',
                                                        {'audio': buff, 'sample_rate': sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            buff = plugin_audio['audio']

        return buff.read()

    def return_pcm_audio(self, audio):
        # convert numpy array to raw PCM bytes
        pcm_bytes = audio.tobytes()

        return pcm_bytes

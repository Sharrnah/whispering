import settings
from Models.cuda_inference import cuda_inference_guard, guard_cuda_model_methods
from Models.TTS.tts_config import get_tts_device
from Models.TTS.orpheus_tts import OrpheusTTS
from Models.TTS.silero import Silero
from Models.TTS.f5_tts import F5TTS
from Models.TTS.zonos_tts import ZonosTTS
from Models.TTS.zonos2_tts import Zonos2TTS
from Models.TTS.kokoro_tts import KokoroTTS
from Models.TTS.chatterbox_tts import Chatterbox
from Models.TTS.maya1_tts import MayaOne
from Models.TTS.index_tts import IndexTTS
from Models.TTS.qwen3_tts import Qwen3TTS
from Models.TTS.audio8_tts import Audio8TTS

tts = None
failed = None


def _activate_tts(adapter_class):
    global tts

    selected_type = settings.GetOption("tts_type")
    device = get_tts_device
    with cuda_inference_guard(device, lambda: f"TTS/{selected_type}.load"):
        adapter = adapter_class()

    tts = guard_cuda_model_methods(
        adapter,
        ("load", "load_model", "release_model", "tts", "tts_streaming"),
        device=device,
        runtime_label=lambda: f"TTS/{settings.GetOption('tts_type')}",
        parallel_same_runtime=False,
    )
    return not failed


def init():
    global tts, failed
    match settings.GetOption("tts_type"):
        case "silero":
            return _activate_tts(Silero)
        case "f5_e2":
            return _activate_tts(F5TTS)
        case "zonos":
            return _activate_tts(ZonosTTS)
        case "zonos2":
            return _activate_tts(Zonos2TTS)
        case "kokoro":
            return _activate_tts(KokoroTTS)
        case "orpheus":
            return _activate_tts(OrpheusTTS)
        case "chatterbox":
            return _activate_tts(Chatterbox)
        case "maya1":
            return _activate_tts(MayaOne)
        case "index_tts":
            return _activate_tts(IndexTTS)
        case "qwen3_tts":
            return _activate_tts(Qwen3TTS)
        case "audio8_tts":
            return _activate_tts(Audio8TTS)
        case _:
            if tts is not None and not failed:
                return True
            else:
                return False

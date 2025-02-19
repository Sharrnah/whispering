import settings
from Models.TTS.silero import Silero
from Models.TTS.f5_tts import F5TTS
from Models.TTS.zonos_tts import ZonosTTS

tts = None
failed = None

def init():
    global tts, failed
    match settings.GetOption("tts_type"):
        case "silero":
            tts = Silero()
            if not failed:
                return True
            else:
                return False
        case "f5_e2":
            tts = F5TTS()
            if not failed:
                return True
            else:
                return False
        case "zonos":
            tts = ZonosTTS()
            if not failed:
                return True
            else:
                return False
        case _:
            if tts is not None and not failed:
                return True
            else:
                return False

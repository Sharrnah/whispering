import settings
from Models.TTS.silero import Silero
from Models.TTS.f5_tts import F5TTS

tts = None
failed = None

def init():
    global tts, failed
    if settings.GetOption("tts_type") == "silero" and tts is None:
        tts = Silero()
        if not failed:
            return True
        else:
            return False
    elif settings.GetOption("tts_type") == "f5_e2" and tts is None:
        tts = F5TTS()
        if not failed:
            return True
        else:
            return False
    else:
        if tts is not None and not failed:
            return True
        else:
            return False

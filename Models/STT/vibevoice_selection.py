"""Lightweight model routing, shared by startup, capture and inference."""

STREAMING_MODELS = ("VibeVoice-ASR-Streaming-1.5B", "VibeVoice-ASR-Streaming-7B")


def is_streaming_selection(stt_type, model):
    # Keep profiles from the initial separate-type integration loadable.
    return stt_type == "vibevoice_asr_streaming" or (
        stt_type == "vibevoice_asr" and model in (*STREAMING_MODELS, "custom-streaming")
    )


def uses_vibevoice_streaming(settings):
    return is_streaming_selection(settings.GetOption("stt_type"), settings.GetOption("model"))

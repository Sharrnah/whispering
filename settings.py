# noinspection PyPackageRequirements
import yaml
import os
from pathlib import Path
from click import core
from whisper import available_models
import threading

import Utilities

DEBOUNCE_TIME = 1.5  # 1.5 second, adjust as necessary
_save_timer = None

SETTINGS_PATH = Path(Path.cwd() / "Profiles" / 'settings.yaml')

TRANSLATE_SETTINGS = {
    "process_id": 0,  # the process id of the running instance

    # text translate settings
    "txt_translate": False,  # if enabled, pipes whisper A.I. results through text translator
    "txt_translator_device": "cpu",  # auto, cuda, cpu
    "src_lang": "auto",  # source language for text translator (Whisper A.I. in translation mode always translates to "en")
    "trg_lang": "fra_Latn",  # target language for text translator
    "txt_romaji": False,  # if enabled, text translator will convert text to romaji.
    "txt_translator": "NLLB200_CT2",  # can be "NLLB200", "NLLB200_CT2" or "M2M100"
    "txt_translator_size": "small",  # for M2M100 model size: Can be "small" or "large", for NLLB200 model size: Can be "small", "medium", "large".
    "txt_translator_precision": "float32",  # for ctranwslate based: can be "default", "auto", "int8", "int8_float16", "int16", "float16", "float32".
    "txt_translate_realtime": False,  # use text translator in realtime mode

    # ocr settings
    "ocr_lang": "en",  # language for OCR image to text recognition.
    "ocr_window_name": "VRChat",  # window name for OCR image to text recognition.
    "ocr_txt_src_lang": "auto",  # (internal setting for ocr text translation)
    "ocr_txt_trg_lang": "eng_Latn",  # (internal setting for ocr text translation)

    # audio settings
    "audio_api": "MME",  # The name of the audio API. (MME, DirectSound, WASAPI)
    "audio_input_device": "",  # audio input device name - used by whispering tiger UI to select audio input device by name
    "audio_output_device": "",  # audio output device name - used by whispering tiger UI to select audio output device by name
    "device_index": None,  # input device index for STT
    "device_out_index": None,  # output device index for TTS

    # whisper settings
    "stt_enabled": True,  # enable STT (if disabled, stops sending audio to whisper)
    "ai_device": None,  # can be None (auto), "cuda" or "cpu".
    "whisper_task": "transcribe",  # Whisper A.I. Can do "transcribe" or "translate"
    "current_language": None,  # can be None (auto) or any Whisper supported language.
    "target_language": "eng",  # can be any M4T supported language.
    "model": "small",  # Whisper model size. Can be "tiny", "base", "small", "medium" or "large"
    "condition_on_previous_text": False,  # if enabled, Whisper will condition on previous text. (more prone to loops or getting stuck)
    "prompt_reset_on_temperature": 0.5,  # after which temperature fallback step the prompt with the previous text should be reset (default value is 0.5)
    "energy": 300,  # energy of audio volume to start whisper processing. Can be 0-1000
    "phrase_time_limit": 0,  # time limit for Whisper to generate a phrase. (0 = no limit)
    "pause": 1.0,  # pause between phrases.
    "initial_prompt": "",  # initial prompt for Whisper. for example "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking." will give more filler words.
    "logprob_threshold": "-1.0",
    "no_speech_threshold": "0.6",
    "length_penalty": 1.0,
    "beam_search_patience": 1.0,
    "repetition_penalty": 1.0,  # penalize the score of previously generated tokens (set > 1 to penalize)
    "no_repeat_ngram_size": 0,  # prevent repetitions of ngrams with this size
    "whisper_precision": "float32",  # for original Whisper can be "float16" or "float32", for faster-whisper "default", "auto", "int8", "int8_float16", "int16", "float16", "float32".
    "stt_type": "faster_whisper",  # can be "faster_whisper", "original_whisper", "transformer_whisper", "speech_t5", "seamless_m4t" etc.
    "temperature_fallback": True,  # Set to False to disable temperature fallback which is the reason for some slowdowns, but decreases quality.
    "beam_size": 5,  # Beam size for beam search. (higher = more accurate, but slower)
    "whisper_cpu_threads": 0,  # Number of threads to use when running on CPU (4 by default)
    "whisper_num_workers": 1,  # When transcribe() is called from multiple Python threads
    "vad_enabled": True,  # Enable Voice activity detection (VAD)
    "vad_on_full_clip": False,  # Make an additional VAD check on the full clip (Not only on each frame).
    "vad_confidence_threshold": 0.4,  # Voice activity detection (VAD) confidence threshold. Can be 0-1
    "vad_frames_per_buffer": 1536,  # Voice activity detection (VAD) sample size (how many audio samples should be tested). Values other than 512, 1024, 1536 may affect model perfomance!!! (default: 1536)
    "vad_thread_num": 1,  # number of threads to use for VAD.
    "push_to_talk_key": "",  # Push to talk key. (empty or None to disable)
    "word_timestamps": False,  # if enabled, Whisper will add timestamps to the transcribed text.
    "faster_without_timestamps": False,  # if enabled, faster whisper will only sample text tokens. (only when using stt_type=faster_whisper)
    "whisper_apply_voice_markers": False,  # if enabled, Whisper will apply voice markers.
    "max_sentence_repetition": -1,  # set max sentence repetition in result (-1 = disabled)

    "transcription_auto_save_file": "",  # set to filepath to save transcriptions. (empty or None to disable)
    "transcription_auto_save_continous_text": False,  # set to save continous text line instead of CSV
    "transcription_save_audio_dir": "",  # set to filepath to save transcriptions wav files. (empty or None to disable)

    "silence_cutting_enabled": True,
    "silence_offset": -40.0,
    "max_silence_length": 30.0,
    "keep_silence_length": 0.20,
    "normalize_enabled": True,
    "normalize_lower_threshold": -24.0,
    "normalize_upper_threshold": -16.0,
    "normalize_gain_factor": 2.0,
    "denoise_audio": False,  # if enabled, audio will be de-noised before processing.
    "denoise_audio_post_filter": False,  # Enable post filter for some minor, extra noise reduction.
    "thread_per_transcription": False,  # Use a separate thread for each transcription.

    "realtime": False,  # if enabled, Whisper will process audio in realtime.
    "realtime_whisper_model": "",  # model used for realtime transcription. (empty for using same model as model setting)
    "realtime_whisper_precision": "float16",  # precision used for realtime transcription model. (only used when realtime_whisper_model is set)
    "realtime_whisper_beam_size": 1,  # beam size used for realtime transcription model.
    "realtime_temperature_fallback": False,  # Set to False to disable temperature fallback for realtime transcription. (see temperature_fallback setting)
    "realtime_frame_multiply": 15,  # Only sends the audio clip to Whisper every X frames (and if its minimum this length, to prevent partial frames). (higher = less whisper updates and less processing time)
    "realtime_frequency_time": 1.0,  # Only sends the audio clip to Whisper every X seconds. (higher = less whisper updates and less processing time)

    # OSC settings
    "osc_ip": "127.0.0.1",  # OSC IP address. set to "0" to disable.
    "osc_port": 9000,
    "osc_address": "/chatbox/input",
    "osc_min_time_between_messages": 1.5,  # defines the minimum time between OSC messages in seconds.
    "osc_typing_indicator": True,  # Display typing indicator while processing audio
    "osc_convert_ascii": False,
    "osc_chat_prefix": "",  # Prefix for OSC messages.
    "osc_chat_limit": 144,  # defines the maximum length of a chat message.
    "osc_time_limit": 15.0,  # defines the time between OSC messages in seconds.
    "osc_scroll_time_limit": 1.5,  # defines the scroll time limit for scrolling OSC messages. (only used when osc_send_type is set to "scroll")
    "osc_initial_time_limit": 15.0,  # defines the initial time after the first message is send.
    "osc_scroll_size": 3,  # defines the scroll size for scrolling OSC messages. (only used when osc_send_type is set to "scroll")
    "osc_max_scroll_size": 30,  # defines the maximum scroll size for scrolling OSC messages. ~30 to scroll on only a single line (only used when osc_send_type is set to "scroll")
    "osc_send_type": "chunks",  # defines the type of OSC messages are send. Can be "scroll", "full_or_scroll", "chunks" or "full". Where "scroll" sends the text scrollung until all is send, "full_or_scroll" to only scroll if it is too long, "chunks" sends the text in chunks and "full" sends the whole text at once.
    "osc_auto_processing_enabled": True,  # Toggle auto sending of OSC messages on WhisperAI results. (not saved)
    "osc_type_transfer": "translation_result",  # defines which type of data to send. Can be "source", "translation_result" or "both".
    "osc_type_transfer_split": " 🌐 ",  # defines how source and translation results are split. (only used when osc_type_transfer is set to "both")
    "osc_delay_until_audio_playback": False,  # if enabled, OSC messages will be delayed until audio playback starts. (if no TTS is used, this will prevent messages from being send.)
    "osc_delay_until_audio_playback_tag": "tts",  # defines the tag used for detecting audio playback. (only used when osc_delay_until_audio_playback is enabled. Set empty to detect any audio playback)
    "osc_delay_timeout": 10,  # defines the timeout for delayed OSC messages. (only used when osc_delay_until_audio_playback is enabled)

    # websocket settings
    "websocket_ip": "127.0.0.1",
    "websocket_port": 5000,
    "websocket_final_messages": True,  # if enabled, websocket will send final messages. (internal use)

    # TTS settings
    "tts_enabled": True,  # enable TTS
    "tts_ai_device": "cpu",  # can be "auto", "cuda" or "cpu".
    "tts_answer": False,  # send whisper results to TTS engine
    "tts_model": ["en", "v3_en"],  # TTS language and model to use
    "tts_voice": "en_0",  # TTS voice (one of silero tts voices, or "last" to use last used voice)
    "tts_prosody_rate": "",  # TTS voice speed. Can be "x-slow", "slow", "medium", "fast", "x-fast" or "" for default.
    "tts_prosody_pitch": "",  # TTS voice pitch. Can be "x-low", "low", "medium", "high", "x-high" or "" for default.
    "tts_use_secondary_playback": False,  # Play TTS audio to a secondary audio device at the same time.
    "tts_secondary_playback_device": -1,  # Play TTS audio to this specified audio device at the same time. (set to -1 to use default audio device)
    "tts_allow_overlapping_audio": False,  # Allow overlapping audio (if disabled, TTS will stop previous audio before playing new audio)

    # Plugins
    "plugins": {},  # active plugins
    "plugin_settings": {},  # plugin settings
    "plugin_timer_timeout": 15.0,  # Timer timeout for plugins
    "plugin_timer": 2.0,  # Timer for plugins
    "plugin_timer_stopped": False,
    "plugin_current_timer": 0.0
}

NON_PERSISTENT_SETTINGS = ["stt_enabled",
                           # "txt_translate", "tts_answer", "osc_auto_processing_enabled",
                           "whisper_languages", "lang_swap", "verbose",
                           "transl_result_textarea_savetts_voice", "transl_result_textarea_sendtts_download",
                           "plugin_timer_stopped", "plugin_current_timer", "websocket_final_messages",
                           "device_default_in_index", "device_default_out_index", "ui_download"]


def SetOption(setting, value):
    if setting in TRANSLATE_SETTINGS:
        if TRANSLATE_SETTINGS[setting] != value:
            TRANSLATE_SETTINGS[setting] = value
            # Save settings
            if TRANSLATE_SETTINGS[setting] not in NON_PERSISTENT_SETTINGS:
                debounced_save_yaml(SETTINGS_PATH)
    else:
        TRANSLATE_SETTINGS[setting] = value
        # Save settings
        if TRANSLATE_SETTINGS[setting] not in NON_PERSISTENT_SETTINGS:
            debounced_save_yaml(SETTINGS_PATH)
    return value


def GetOption(setting):
    return TRANSLATE_SETTINGS[setting]


def LoadYaml(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
            sanitized_data = Utilities.handle_bytes(loaded_data)
            TRANSLATE_SETTINGS.update(sanitized_data)


def debounced_save_yaml(path):
    global _save_timer

    # Cancel the existing timer if it exists
    if _save_timer is not None:
        _save_timer.cancel()

    # Start a new timer
    _save_timer = threading.Timer(DEBOUNCE_TIME, SaveYaml, [path])
    _save_timer.start()


def SaveYaml(path):
    global _save_timer

    # If this function was called directly, cancel the timer
    if _save_timer is not None:
        _save_timer.cancel()
        _save_timer = None

    to_save_settings = TRANSLATE_SETTINGS.copy()

    # Remove settings that are in NON_PERSISTENT_SETTINGS
    for setting in NON_PERSISTENT_SETTINGS:
        if setting in to_save_settings:
            del to_save_settings[setting]

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(to_save_settings, f)


def IsArgumentSetting(ctx, argument_name):
    return ctx.get_parameter_source(argument_name) == core.ParameterSource.COMMANDLINE


# Get Setting from argument if it is set, otherwise get setting from settings file
def GetArgumentSettingFallback(ctx, argument_name, fallback_setting_name):
    if IsArgumentSetting(ctx, argument_name):
        return ctx.params[argument_name]
    else:
        return GetOption(fallback_setting_name)


def get_available_models():
    available_models_list = available_models()

    # add custom models to list
    if GetOption("stt_type") == "faster_whisper":
        available_models_list.insert(0, "medium-distilled.en")
        available_models_list.insert(0, "large-distilled-v2.en")
        available_models_list.insert(0, "small.eu")
        available_models_list.insert(0, "medium.eu")
        available_models_list.insert(0, "small.de")
        available_models_list.insert(0, "medium.de")
        available_models_list.insert(0, "large-v2.de2")
        available_models_list.insert(0, "small.de-swiss")
        available_models_list.insert(0, "medium.mix-jpv2")
        available_models_list.insert(0, "large-v2.mix-jp")
        available_models_list.insert(0, "small.jp")
        available_models_list.insert(0, "medium.jp")
        available_models_list.insert(0, "large-v2.jp")
        available_models_list.insert(0, "medium.ko")
        available_models_list.insert(0, "large-v2.ko")
        available_models_list.insert(0, "small.zh")
        available_models_list.insert(0, "medium.zh")
        available_models_list.insert(0, "large-v2.zh")
    return available_models_list


def GetAvailableSettingValues():
    possible_settings = {
        "ai_device": ["None", "cuda", "cpu"],
        "model": get_available_models(),
        "whisper_task": ["transcribe", "translate"],
        "stt_type": ["faster_whisper", "original_whisper", "transformer_whisper", "seamless_m4t", "speech_t5", "wav2vec_bert", "nemo_canary", ""],
        "tts_ai_device": ["cuda", "cpu"],
        "txt_translator_device": ["cuda", "cpu"],
        "txt_translator": ["", "NLLB200_CT2", "NLLB200", "M2M100", "Seamless_M4T"],
        "txt_translator_size": ["small", "medium", "large"],
        "txt_translator_precision": ["float32", "float16", "int16", "int8_float16", "int8", "bfloat16", "int8_bfloat16", "4bit", "8bit"],
        "tts_prosody_rate": ["", "x-slow", "slow", "medium", "fast", "x-fast"],
        "tts_prosody_pitch": ["", "x-low", "low", "medium", "high", "x-high"],
        "whisper_precision": ["float32", "float16", "int16", "int8_float16", "int8", "bfloat16", "int8_bfloat16", "4bit", "8bit"],
        #"whisper_precision": ["float32", "float16", "int16", "int8_float16", "int8"],
        "realtime_whisper_model": [""] + get_available_models(),
        "realtime_whisper_precision": ["float32", "float16", "int16", "int8_float16", "int8", "bfloat16", "int8_bfloat16", "4bit", "8bit"],
        #"realtime_whisper_precision": ["float32", "float16", "int16", "int8_float16", "int8"],
        "osc_type_transfer": ["source", "translation_result", "both", "both_inverted"],
        "osc_send_type": ["full", "full_or_scroll", "scroll", "chunks"],
    }

    return possible_settings

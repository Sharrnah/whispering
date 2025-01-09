# Configurations

There are two ways to configure the application. Either by [command-line flags](#command-line-flags) or by a [settings file](#settings-file).
The settings file allows you to configure more options, but the command-line flags take precedence over the settings file.

On the first run, the application will create a `settings.yaml` file in the same folder as the executable with the default settings and the command-line flags that were used to start the application.

Edit the `settings.yaml` file with any text editor to change the settings.

Used settings file can be changed by using the `--config` command-line flag.

## Command-line flags
_These take precedence to the [settings file](#settings-file). But not all options are available as command-line flags._

|            --flags             | Default Value  |                                                                  Description                                                                  |
|:------------------------------:|:--------------:|:---------------------------------------------------------------------------------------------------------------------------------------------:|
|          `--devices`           |     False      |                                                         Print all available devices.                                                          |
|        `--device_index`        |       -1       |              Choose the input device to listen to and transcribe the audio from this device. '-1' = auto-select mic by default.               |
|      `--device_out_index`      |       -1       |                                           the id of the output device (-1 = default active Speaker)                                           |
|        `--sample_rate`         |     16000      |                                                      Sample rate of the audio recording.                                                      |
|         `--ai_device`          |      None      |                           defines on which device the AI is loaded. can be `cuda` or `cpu`. auto-select by default                            |
|            `--task`            |   transcribe   |                                    Choose between to `transcribe` or to `translate` the audio to English.                                     |
|           `--model`            |     small      |             Select model list. can be `tiny, base, small, medium, large`. where large models are not available for english only.              |
|          `--language`          |      None      |                                   language spoken in the audio, specify None to perform language detection                                    |
| `--condition_on_previous_text` |     False      |   Feed it the previous result to keep it consistent across recognition windows, but makes it more prone to getting stuck in a failure loop    |
|           `--energy`           |      300       |                                                        Energy level for mic to detect.                                                        |
|       `--dynamic_energy`       |     False      |                                                            Enable dynamic energy.                                                             |
|           `--pause`            |      0.8       |                                                         Pause time before entry ends.                                                         |
|     `--phrase_time_limit`      |      None      |                             Phrase time limit (in seconds) before entry ends to break up long recognition tasks.                              |
|           `--osc_ip`           |   127.0.0.1    |                        IP to send OSC messages to. Set to '0' to disable. (For VRChat this should mostly be 127.0.0.1)                        |
|          `--osc_port`          |      9000      |                                          Port to send OSC message to. ('9000' as default for VRChat)                                          |
|        `--osc_address`         | /chatbox/input |                              The Address the OSC messages are send to. ('/chatbox/input' as default for VRChat)                               |
|     `--osc_convert_ascii`      |     False      |                                            Convert Text to ASCII compatible when sending over OSC.                                            |
|        `--websocket_ip`        |       0        |                                         IP where Websocket Server listens on. Set to '0' to disable.                                          |
|       `--websocket_port`       |      5000      |                                                    Port where Websocket Server listens on.                                                    |
|       `--txt_translator`       |    NLLB200     |                           The Model the AI is loading for text translations. can be 'M2M100', 'NLLB200' or 'None'.                            |
|    `--txt_translator_size`     |     small      | The Model size of M2M100 or NLLB200 text translator is used. can be 'small', 'medium', 'large' for NLLB200, or 'small' or 'large' for M2M100. |
|   `--txt_translator_device`    |      auto      |                                    The device used for M2M100 translation. can be 'auto', 'cuda' or 'cpu'.                                    |
|      `--ocr_window_name`       |     VRChat     |                                             Window name of the application for OCR translations.                                              |
|        `--open_browser`        |     False      |                       Open default Browser with websocket-remote on start. (requires --websocket_ip to be set as well)                        |
|           `--config`           |      None      |      Use the specified config file instead of the default 'settings.yaml' (relative to the current path) [overwrites without asking!!!]       |
|          `--verbose`           |     False      |                                                       Whether to print verbose output.                                                        |


## Settings File
All possible options of the settings file.

_Default name is `settings.yaml`, but can be customized with the `--config` option._

```yaml
# audio settings
audio_api: "MME",  # The name of the audio API. (MME, DirectSound, WASAPI)
audio_input_device: "",  # audio input device name - used by whispering tiger UI to select audio input device by name
audio_output_device: "",  # audio output device name - used by whispering tiger UI to select audio output device by name
device_index: None,  # input device index for STT

# whisper settings
ai_device: null  # can be null (auto), "cuda" or "cpu".
whisper_task: translate  # Whisper A.I. Can do "transcribe" or "translate".
current_language: null  # can be null (auto) or any Whisper supported language (improves accuracy if whisper does not have to detect the language).
model: small  # Whisper model size. Can be "tiny", "base", "small", "medium" or "large".
condition_on_previous_text: true  # if enabled, Whisper will condition on previous text (more prone to loops or getting stuck).
prompt_reset_on_temperature: 0.5,  # after which temperature fallback step the prompt with the previous text should be reset (default value is 0.5)
energy: 300,  # energy of audio volume to start whisper processing. Can be 0-?????
phrase_time_limit: 0,  # time limit for Whisper to generate a phrase. (0 = no limit)
pause: 1.0,  # pause between phrases.
initial_prompt: ""  # initial prompt for Whisper to try to follow its style. for example "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking." will give more filler words.
logprob_threshold: "-1.0",  # log probability threshold for Whisper to treat as failed. (can be negative or positive).
no_speech_threshold: "0.6",  # If the no_speech probability is higher than this value AND the average log probability over sampled tokens is below `logprob_threshold`, consider the segment as silent
length_penalty: 1.0,
beam_search_patience: 1.0,
repetition_penalty: 1.0,  # penalize the score of previously generated tokens (set > 1 to penalize)
no_repeat_ngram_size: 0,  # prevent repetitions of ngrams with this size
whisper_precision: "float32"  # for original Whisper can be "float16" or "float32", for faster-whisper "default", "auto", "int8", "int8_float16", "int16", "float16", "float32".
stt_type: "faster_whisper",  # can be "faster_whisper", "original_whisper", "speech_t5" or "seamless_m4t".
temperature_fallback: true  # Set to False to disable temperature fallback which is the reason for some slowdowns, but decreases quality.
beam_size: 5  # Beam size for beam search. (higher = more accurate, but slower)
whisper_cpu_threads: 0  # Number of threads to use when running on CPU (4 by default)
whisper_num_workers: 1  # When transcribe() is called from multiple Python threads
vad_enabled: True,  # Enable Voice activity detection (VAD)
vad_on_full_clip: False,  # If enabled,  an additional VAD check will be applied to the full clip, not just the frames.
vad_confidence_threshold: "0.4",  # Voice activity detection (VAD) confidence threshold. Can be 0-1
vad_frames_per_buffer: 2000,  # Voice activity detection (VAD) sample size (how many audio samples should be tested).
vad_thread_num: 1,  # number of threads to use for VAD.
push_to_talk_key: "",  # Push to talk key or key combination. (empty or None to disable)
word_timestamps: False,  # if enabled, Whisper will add timestamps to the transcribed text.
faster_without_timestamps: False,  # if enabled, faster whisper will only sample text tokens. (only when using stt_type=faster_whisper)
whisper_apply_voice_markers: False,  # if enabled, Whisper will apply voice markers.
max_sentence_repetition: -1,  # set max sentence repetition in result (-1 = disabled)

transcription_auto_save_file: "",  # set to filepath to save transcriptions. (empty or None to disable)
transcription_auto_save_continuous_text: False,  # set to save continuous text line instead of CSV
transcription_save_audio_dir: "",  # set to filepath to save transcriptions wav files. (empty or None to disable)

silence_cutting_enabled: True,
silence_offset: -40.0,
max_silence_length: 30.0,
keep_silence_length: 0.20,
normalize_enabled: True,
normalize_lower_threshold: -24.0,
normalize_upper_threshold: -16.0,
normalize_gain_factor: 2.0,
denoise_audio: False,  # if enabled, audio will be de-noised before processing.
denoise_audio_post_filter: False,  # Enable post filter for some minor, extra noise reduction.

realtime: false  # if enabled, Whisper will process audio in realtime.
realtime_whisper_model: ''  # model used for realtime transcription. (empty for using same model as model setting)
realtime_whisper_precision: "float16"  # precision used for realtime transcription model.
realtime_whisper_beam_size: 1  # beam size used for realtime transcription model.
realtime_temperature_fallback: false  # Set to False to disable temperature fallback for realtime transcription. (see temperature_fallback setting)
realtime_frame_multiply: 15  # Only sends the audio clip to Whisper every X frames. (higher = less whisper updates and less processing time)
realtime_frequency_time: 1.0  # Only sends the audio clip to Whisper every X seconds. (higher = less whisper updates and less processing time)

# text translate settings
txt_translate: false  # if enabled, pipes whisper A.I. results through text translator.
txt_translator_device: 'cuda'  # can be "auto", "cuda" or "cpu".
src_lang: auto  # source language for text translator.
trg_lang: fra_Latn  # target language for text translator.
txt_romaji: false  # if enabled, text translator will convert text to romaji.
txt_translator: NLLB200  # can be "NLLB200" or "M2M100".
txt_translator_size: small  # for M2M100 model size: Can be "small" or "large", for NLLB200 model size: Can be "small", "medium", "large".
txt_translator_precision: float32  # for ctranwslate based text translators: can be "default", "auto", "int8", "int8_float16", "int16", "float16", "float32".
txt_translate_realtime: false  # use text translator in realtime mode

# websocket settings
websocket_ip: 127.0.0.1
websocket_port: 5000

# OSC settings
osc_ip: '127.0.0.1'
osc_port: 9000
osc_address: /chatbox/input
osc_typing_indicator: true  # Display typing indicator while processing audio
osc_convert_ascii: false
osc_chat_prefix: ''  # Prefix for OSC messages. (is prepended in front of the OSC message)
osc_chat_limit: 144,  # defines the maximum length of a chat message.
osc_time_limit: 15.0,  # defines the time between OSC messages in seconds.
osc_scroll_time_limit: 1.5,  # defines the scroll time limit for scrolling OSC messages. (only used when osc_send_type is set to "scroll")
osc_initial_time_limit: 15.0,  # defines the initial time after the first message is send.
osc_scroll_size: 3,  # defines the scroll size for scrolling OSC messages. (only used when osc_send_type is set to "scroll")
osc_max_scroll_size: 30,  # defines the maximum scroll size for scrolling OSC messages. ~30 to scroll on only a single line (only used when osc_send_type is set to "scroll")
osc_send_type: "chunks",  # defines the type of OSC messages are send. Can be "scroll", "full_or_scroll", "chunks" or "full". Where "scroll" sends the text scrollung until all is send, "full_or_scroll" to only scroll if it is too long, "chunks" sends the text in chunks and "full" sends the whole text at once.
osc_auto_processing_enabled: True,  # Toggle auto sending of OSC messages on WhisperAI results. (not saved)
osc_type_transfer: "translation_result",  # defines which type of data to send. Can be "source", "translation_result" or "both".
osc_type_transfer_split: " 🌐 ",  # defines how source and translation results are split. (only used when osc_type_transfer is set to "both")
osc_delay_until_audio_playback: False,  # if enabled, OSC messages will be delayed until audio playback starts. (if no TTS is used, this will prevent messages from being send.)
osc_delay_until_audio_playback_tag: "tts",  # defines the tag used for detecting audio playback. (only used when osc_delay_until_audio_playback is enabled. Set empty to detect any audio playback)
osc_delay_timeout: 10,  # defines the timeout for delayed OSC messages. (only used when osc_delay_until_audio_playback is enabled)

# OCR settings
ocr_lang: en  # language for OCR image to text recognition.
ocr_window_name: VRChat  # window name for OCR image to text recognition.

# TTS settings
tts_enabled: True,  # enable TTS
tts_ai_device: "cuda",  # can be "auto", "cuda" or "cpu".
tts_answer: True,  # answer to whisper results
device_out_index: None,  # output device index for TTS
tts_model: ["en", "v3_en"],  # TTS language and model to use
tts_voice: "en_0",  # TTS voice (one of silero tts voices, or "last" to use last used voice)
tts_prosody_rate: ""  # TTS voice speed. Can be "x-slow", "slow", "medium", "fast", "x-fast" or "" for default.
tts_prosody_pitch: "" # TTS voice pitch. Can be "x-low", "low", "medium", "high", "x-high" or "" for default.

# plugin settings
plugins: {}  # list of plugins to load.
plugin_settings: {}  # settings for plugins.
plugin_timer_timeout: 15.0  # timeout for plugin timer in seconds. (Timer pause time after whisper event)
plugin_timer: 2.0  # time between plugin timer events in seconds.
```

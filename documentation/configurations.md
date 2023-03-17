# Configurations

There are two ways to configure the application. Either by [command-line flags](#command-line-flags) or by a [settings file](#settings-file).
The settings file allows you to configure more options, but the command-line flags take precedence over the settings file.

On the first run, the application will create a `settings.yaml` file in the same folder as the executable with the default settings and the command-line flags that were used to start the application.

Edit the `settings.yaml` file with any text editor to change the settings.

Used settings file can be changed by using the `--config` command-line flag.

## Command-line flags
_These take precedence to the [settings file](#settings-file). But not all options are available as command-line flags._

|            --flags             | Default Value  |                                                                                        Description                                                                                        |
|:------------------------------:|:--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          `--devices`           |     False      |                                                                               Print all available devices.                                                                                |
|        `--device_index`        |       -1       |                                    Choose the input device to listen to and transcribe the audio from this device. '-1' = auto-select mic by default.                                     |
|      `--device_out_index`      |       -1       |                                                                 the id of the output device (-1 = default active Speaker)                                                                 |
|        `--sample_rate`         |     16000      |                                                                            Sample rate of the audio recording.                                                                            |
|         `--ai_device`          |      None      |                                                 defines on which device the AI is loaded. can be `cuda` or `cpu`. auto-select by default                                                  |
|            `--task`            |   transcribe   |                                                          Choose between to `transcribe` or to `translate` the audio to English.                                                           |
|           `--model`            |     small      |                                   Select model list. can be `tiny, base, small, medium, large`. where large models are not available for english only.                                    |
|          `--language`          |      None      |                                                         language spoken in the audio, specify None to perform language detection                                                          |
| `--condition_on_previous_text` |     False      |                         Feed it the previous result to keep it consistent across recognition windows, but makes it more prone to getting stuck in a failure loop                          |
|           `--energy`           |      300       |                                                                              Energy level for mic to detect.                                                                              |
|       `--dynamic_energy`       |     False      |                                                                                  Enable dynamic energy.                                                                                   |
|           `--pause`            |      0.8       |                                                                               Pause time before entry ends.                                                                               |
|     `--phrase_time_limit`      |      None      |                                                   Phrase time limit (in seconds) before entry ends to break up long recognition tasks.                                                    |
|           `--osc_ip`           |   127.0.0.1    |                                              IP to send OSC messages to. Set to '0' to disable. (For VRChat this should mostly be 127.0.0.1)                                              |
|          `--osc_port`          |      9000      |                                                                Port to send OSC message to. ('9000' as default for VRChat)                                                                |
|        `--osc_address`         | /chatbox/input |                                                    The Address the OSC messages are send to. ('/chatbox/input' as default for VRChat)                                                     |
|     `--osc_convert_ascii`      |     False      |                                                                  Convert Text to ASCII compatible when sending over OSC.                                                                  |
|        `--websocket_ip`        |       0        |                                                               IP where Websocket Server listens on. Set to '0' to disable.                                                                |
|       `--websocket_port`       |      5000      |                                                                          Port where Websocket Server listens on.                                                                          |
|       `--txt_translator`       |     M2M100     |                                                  The Model the AI is loading for text translations. can be 'M2M100', 'ARGOS' or 'None'.                                                   |
|    `--txt_translator_size`     |     small      | The Model size of M2M100 or NLLB200 text translator is used. can be 'small', 'medium', 'large' for NLLB200, or 'small' or 'large' for M2M100. (has no effect with --txt_translator ARGOS) |
|   `--txt_translator_device`    |      auto      |                                    The device used for M2M100 translation. can be 'auto', 'cuda' or 'cpu' (has no effect with --txt_translator ARGOS)                                     |
|      `--ocr_window_name`       |     VRChat     |                                                                   Window name of the application for OCR translations.                                                                    |
|        `--open_browser`        |     False      |                                             Open default Browser with websocket-remote on start. (requires --websocket_ip to be set as well)                                              |
|           `--config`           |      None      |                            Use the specified config file instead of the default 'settings.yaml' (relative to the current path) [overwrites without asking!!!]                             |
|          `--verbose`           |     False      |                                                                             Whether to print verbose output.                                                                              |


## Settings File
All possible options of the settings file.

_Default name is `settings.yaml`, but can be customized with the `--config` option._

```yaml
# whisper settings
ai_device: null  # can be null (auto), "cuda" or "cpu".
whisper_task: translate  # Whisper A.I. Can do "transcribe" or "translate".
current_language: null  # can be null (auto) or any Whisper supported language (improves accuracy if whisper does not have to detect the language).
model: small  # Whisper model size. Can be "tiny", "base", "small", "medium" or "large".
condition_on_previous_text: true  # if enabled, Whisper will condition on previous text (more prone to loops or getting stuck).
initial_prompt: ""  # initial prompt for Whisper to try to follow its style. for example "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking." will give more filler words.
logprob_threshold: "-1.0",  # log probability threshold for Whisper to treat as failed. (can be negative or positive).
no_speech_threshold: "0.6",  # If the no_speech probability is higher than this value AND the average log probability over sampled tokens is below `logprob_threshold`, consider the segment as silent
vad_enabled: True,  # Enable Voice activity detection (VAD)
vad_on_full_clip: False,  # If enabled,  an additional VAD check will be applied to the full clip, not just the frames.
vad_confidence_threshold: "0.4",  # Voice activity detection (VAD) confidence threshold. Can be 0-1
vad_num_samples: 3000,  # Voice activity detection (VAD) sample size (how many audio samples should be tested).
vad_thread_num: 1,  # number of threads to use for VAD.
fp16: false  # Set to true to use FP16 instead of FP32.

# text translate settings
txt_translate: false  # if enabled, pipes whisper A.I. results through text translator.
txt_translator_device: 'cuda'  # can be "cuda" or "cpu".
src_lang: auto  # source language for text translator.
trg_lang: fra_Latn  # target language for text translator.
txt_ascii: false  # if enabled, text translator will convert text to romaji.
txt_translator: NLLB200  # can be "NLLB200", "M2M100" or "ARGOS".
txt_translator_size: small  # for M2M100 model size: Can be "small" or "large", for NLLB200 model size: Can be "small", "medium", "large".

# websocket settings
websocket_ip: 127.0.0.1
websocket_port: 5000

# OSC settings
osc_ip: '127.0.0.1'
osc_port: 9000
osc_address: /chatbox/input
osc_typing_indicator: true
osc_convert_ascii: false

# OCR settings
ocr_lang: en  # language for OCR image to text recognition.
ocr_window_name: VRChat  # window name for OCR image to text recognition.

# TTS settings
tts_enabled: True,  # enable TTS
tts_ai_device: "cuda",  # can be "cuda" or "cpu".
tts_answer: True,  # answer to whisper results
device_out_index: None,  # output device index for TTS
tts_model: ["en", "v3_en"],  # TTS language and model to use
tts_voice: "en_0",  # TTS voice (one of silero tts voices, or "last" to use last used voice)

# plugin settings
plugins: {}  # list of plugins to load.
plugin_settings: {}  # settings for plugins.
plugin_timer_timeout: 15.0  # timeout for plugin timer in seconds. (Timer pause time after whisper event)
plugin_timer: 2.0  # time between plugin timer events in seconds.
```

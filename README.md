# <img src=images/app-icon.png width=90> Whispering Tiger (Live Translate/Transcribe)

This application listens to any audio stream on your machine and prints out the transcription or translation of the audio.

Based on OpenAI's [Whisper](https://github.com/openai/whisper) project.

It allows connecting to OSC (for VRChat for example) and Websockets (For Streaming Overlays and Remote Controlling some settings using the websocket_remote)

<img src=images/vrchat.png width=400><img src=images/streaming-overlay.png width=400>

## Content:
- [Release Downloads](#release-downloads)
- [Usage](#usage)
- [Websocket Clients](documentation/websocket-clients.md)
- [Command-line flags](#command-line-flags)
- [Usage with 3rd Party Applications](#usage-with-3rd-party-applications)
  - [VRChat](#vrchat)
  - [Live Streaming Applications (OBS, vMix, XSplit ...)](#live-streaming-applications-obs-vmix-xsplit-)
  - [Desktop+](#desktop-currently-only-new-ui-beta-with-embedded-browser)
- [Working with the Code](documentation/working-with-code.md)
- [FAQ](documentation/faq.md)
- [Sources](#sources)

## Release Downloads
Standalone Releases with all dependencies included are now provided.

Go to the [GitHub Releases Page](https://github.com/Sharrnah/whispering/releases) and Download from the download Link in the description or find the [Latest Release here.](https://github.com/Sharrnah/whispering/releases/latest)

_(because of the 2 GB Limit, no direct release files on GitHub)_

- [Install CUDA for GPU Acceleration](https://developer.nvidia.com/cuda-downloads) (recommended)
- Extract the Files on a Drive with enough free Space.
  - _(After download of medium Whisper Model + Argos Translations, it can take up to 20 GB)_
- Run only using the *.bat files. Edit or copy an existing `start-*.bat` file and edit the parameters in any text editor for your own command-line flags.

## Usage
1. run `audioWhisper\audioWhisper.exe --devices true` (or `get-device-list.bat`) and get the Index of the audio device. (the number in `[*]` at the end)
2. run `audioWhisper\audioWhisper.exe`. By default, it tries to find your default Microphone. Otherwise, you need to add `--device_index *` to the run command where the `*` is the device index found at step 3. Find more command-line flags in the following table.
3. If websocket option is enabled, you can control the whisper task (translate or transcript) as well as textual translation options while the AI is running.
   
   <img src=images/remote_control.png width=600>
   
   For this: open the `websocket_clients/websocket-remote/` folder and start the index.html there.
   
   _If you have the AI running on a secondary PC, open the HTML file with the IP as parameter like this: `index.html?ws_server=ws://127.0.0.1:5000`_

## Command-line flags
|            --flags             | Default Value  |                                                               Description                                                                |
|:------------------------------:|:--------------:|:----------------------------------------------------------------------------------------------------------------------------------------:|
|          `--devices`           |     False      |                                                       Print all available devices.                                                       |
|        `--device_index`        |       -1       |             Choose the output device to listen to and transcribe the audio from this device. '-1' = auto-select by default.              |
|        `--sample_rate`         |     16000      |                                                   Sample rate of the audio recording.                                                    |
|         `--ai_device`          |      None      |                         defines on which device the AI is loaded. can be `cuda` or `cpu`. auto-select by default                         |
|            `--task`            |   transcribe   |                                  Choose between to `transcribe` or to `translate` the audio to English.                                  |
|           `--model`            |     small      |           Select model list. can be `tiny, base, small, medium, large`. where large models are not available for english only.           |
|          `--language`          |      None      |                                 language spoken in the audio, specify None to perform language detection                                 |
| `--condition_on_previous_text` |     False      | Feed it the previous result to keep it consistent across recognition windows, but makes it more prone to getting stuck in a failure loop |
|           `--energy`           |      300       |                                                     Energy level for mic to detect.                                                      |
|       `--dynamic_energy`       |     False      |                                                          Enable dynamic energy.                                                          |
|           `--pause`            |      0.8       |                                                      Pause time before entry ends.                                                       |
|     `--phrase_time_limit`      |      None      |                           Phrase time limit (in seconds) before entry ends to break up long recognition tasks.                           |
|           `--osc_ip`           |       0        |                     IP to send OSC messages to. Set to '0' to disable. (For VRChat this should mostly be 127.0.0.1)                      |
|          `--osc_port`          |      9000      |                                       Port to send OSC message to. ('9000' as default for VRChat)                                        |
|        `--osc_address`         | /chatbox/input |                            The Address the OSC messages are send to. ('/chatbox/input' as default for VRChat)                            |
|     `--osc_convert_ascii`      |     False      |                                         Convert Text to ASCII compatible when sending over OSC.                                          |
|        `--websocket_ip`        |       0        |                                       IP where Websocket Server listens on. Set to '0' to disable.                                       |
|       `--websocket_port`       |      5000      |                                                 Port where Websocket Server listens on.                                                  |
|       `--txt_translator`       |     M2M100     |                          The Model the AI is loading for text translations. can be 'M2M100', 'ARGOS' or 'None'.                          |
|        `--m2m100_size`         |     small      |         The Model size if M2M100 text translator is used. can be 'small' or 'large'. (has no effect with --txt_translator ARGOS)         |
|       `--m2m100_device`        |      auto      |            The device used for M2M100 translation. can be 'auto', 'cuda' or 'cpu' (has no effect with --txt_translator ARGOS)            |
|      `--ocr_window_name`       |     VRChat     |                                           Window name of the application for OCR translations.                                           |
|        `--flan_enabled`        |     False      |                               Enable FLAN-T5 A.I. (General A.I. which can be used for Question Answering.)                               |
|        `--open_browser`        |     False      |                     Open default Browser with websocket-remote on start. (requires --websocket_ip to be set as well)                     |
|           `--config`           |      None      |    Use the specified config file instead of the default 'settings.yaml' (relative to the current path) [overwrites without asking!!!]    |
|          `--verbose`           |     False      |                                                     Whether to print verbose output.                                                     |

## Usage with 3rd Party Applications
### VRChat
- run the script with `--osc_ip 127.0.0.1` parameter. This way it automatically writes the recognized text into the in-game chat-box.
  
  example:

  > `audioWhisper\audioWhisper.exe --model medium --task transcribe --energy 300 --osc_ip 127.0.0.1 --phrase_time_limit 9`

### Live Streaming Applications (OBS, vMix, XSplit ...)
1. run the script with `--websocket_ip 127.0.0.1` parameter (127.0.0.1 if you are running everything on the same machine), and set a `--phrase_time_limit` if you expect not many pauses that could be recognized by the configured `--energy` and `--pause` values.

   example:

   > `audioWhisper\audioWhisper.exe --model medium --task translate --device_index 4 --energy 300 --phrase_time_limit 15 --websocket_ip 127.0.0.1`
2. Find a streaming overlay website in the `websocket_clients` folder. (So far only `streaming-overlay-01` is optimized as overlay with transparent background.)
3. Add the HTML file to your streaming application.

### Desktop+ (Currently only new-ui Beta with embedded Browser)
1. Run the Application listening on your Audio-Device with the VRChat Sound.
2. Add the Overlay in the [Desktop+ Beta](https://github.com/elvissteinjr/DesktopPlus/tree/new-ui) with the embedded Browser with (`index.html?no_scroll=1&auto_hide_message=25`)
3. Set the Browser to allow Transparency.
4. Attach the Browser to your VR-Headset.

Voilà, you have live translated subtitles in VR of other people speaking (or videos playing) which automatically disappear after 25 seconds.

<img src=images/vr_subtitles.gif width=410> <img src=images/vrchat_live_subtitles.gif width=410>

### _Sources_
A thanks goes to
- OpenAI https://github.com/openai/whisper
- Awexander https://github.com/Awexander/audioWhisper
- Blake https://github.com/mallorbc/whisper_mic
- Argos Translate https://github.com/argosopentech/argos-translate

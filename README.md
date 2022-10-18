# Whispering to OSC and Websocket
This application listens to any audio stream on your machine and prints out the transcription or translation of the audio.
Based on OpenAI's [Whisper](https://github.com/openai/whisper) project.

It allows connecting to OSC (for VRChat for example) and Websockets (For Streaming Overlays and Remote Controlling some settings using the websocket_remote)

<img src=screenshots/vrchat.png width=400><img src=screenshots/streaming-overlay.png width=400>

## Content:
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Command-line flags](#command-line-flags)
- [Usage with 3rd Party Applications](#usage-with-3rd-party-applications)
  - [VRChat](#vrchat)
  - [Live Streaming Applications (OBS, vMix, XSplit ...)](#live-streaming-applications-obs-vmix-xsplit-)
- [FAQ](#faq)
- [Sources](#sources)


## Prerequisites
1. [**Turn on stereo mix settings on windows if you want to fetch the PCs audio**](https://www.howtogeek.com/howto/39532/how-to-enable-stereo-mix-in-windows-7-to-record-audio/)
2. [**Install and add ffmpeg to your PATH**](https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10#:~:text=Click%20New%20and%20type%20the,Click%20OK%20to%20apply%20changes.)
3. [**Install CUDA to your system**](https://developer.nvidia.com/cuda-downloads)
4. [**Install Python (Version 3.10) to your system**](https://www.python.org/downloads/windows/)
5. [**Install Git to your system**](https://git-scm.com/download/win)

## Setup
1. Git clone or download this repository
2. `pip install -r requirements.txt -U` (or `install.bat`)
3. run `python audioWhisper.py --devices true` (or `get-device-list.bat`) and get the Index of the audio device. (the number in `[*]` at the end)

## Usage
1. run `python audioWhisper.py`. By default it tries to find your default Mic. Otherwise you need to add `--device_index *` to the run command where the `*` is the device index found at step 3. Find more command-line flags in the following table.

2. If websocket option is enabled, you can control the whisper task (translate or transcript) as well as textual translation options while the AI is running.
   
   <img src=screenshots/remote_control.png width=600>
   
   For this: open the `websocket_remote/` folder and start the index.html there.
   
   (If you have the AI running on a secondary PC, change the IP in the line `var websocketServer = "ws://127.0.0.1:5000"` inside the HTML.)

## Command-line flags
|      --flags                   |  Default Value  |      Description                                                                                                                          |
|:------------------------------:|:---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|
|`--devices`                     | False           | Print all available devices.                                                                                                              |
|`--device_index`                | -1              | Choose the output device to listen to and transcribe the audio from this device. '-1' = autoselect by default.                            |
|`--sample_rate`                 | 44100           | Sample rate of the audio recording.                                                                                                       |
|`--ai_device`                   | None            | defines on which device the AI is loaded. can be `cuda` or `cpu`. autoselect by default                                                   |
|`--task`                        | transcribe      | Choose between to `transcribe` or to `translate` the audio to English.                                                                    |
|`--model`                       | small           | Select model list. can be `tiny, base, small, medium, large`. where large models are not for english only.                                |
|`--english`                     | False           | Use english only model.                                                                                                                   |
|`--condition_on_previous_text`  | False           | Feed it the previous result to keep it consistent across recognition windows, but makes it more prone to getting stuck in a failure loop  |
|`--energy`                      | 300             | Energy level for mic to detect.                                                                                                           |
|`--dynamic_energy`              | False           | Enable dynamic engergy.                                                                                                                   |
|`--pause`                       | 0.8             | Pause time before entry ends.                                                                                                             |
|`--phrase_time_limit`           | None            | Phrase time limit (in seconds) before entry ends to break up long recognition tasks.                                                      |
|`--osc_ip`                      | 0               | IP to send OSC messages to. Set to '0' to disable. (For VRChat this should mostly be 127.0.0.1)                                           |
|`--osc_port`                    | 9000            | Port to send OSC message to. ('9000' as default for VRChat)                                                                               |
|`--osc_address`                 | /chatbox/input  | The Address the OSC messages are send to. ('/chatbox/input' as default for VRChat)                                                        |
|`--websocket_ip`                | 0               | IP where Websocket Server listens on. Set to '0' to disable.                                                                              |
|`--verbose`                     | False           | Whether to print verbose output.                                                                                                          |

## Usage with 3rd Party Applications
### VRChat
- run the script with `--osc_ip 127.0.0.1` parameter. This way it automatically writes the recognized text into the ingame chatbox.
  
  example:

  > `python audioWhisper.py --model medium --task transcribe --energy 300 --osc_ip 127.0.0.1 --phrase_time_limit 8`

### Live Streaming Applications (OBS, vMix, XSplit ...)
1. run the script with `--websocket_ip 127.0.0.1` parameter (127.0.0.1 if you are running everything on the same machine), and preferably also set a `--phrase_time_limit` if you expect not many pauses that could be recognized by the configured `--energy` and `--pause` values.

   example:

   > `python audioWhisper.py --model medium --task translate --device_index 4 --energy 300 --phrase_time_limit 8 --websocket_ip 127.0.0.1`
2. Find a streaming overlay website in the `websocket_clients` folder. (So far only `streaming-overlay-01` is optimized.)
3. Add the HTML file to your streaming application.

### FAQ
- **Problem**: _The translation/transcript is too slow and it shows the warning:_

  _`UserWarning: FP16 is not supported on CPU; using FP32 instead.`_

  **Answers**: This means that the AI is not running on your GPU but on your CPU instead.
  - The whisper AI is running best on CUDA enabled GPUs.
    
    NVIDIA starting from GTX 1080 should do.
  - It is possible that CUDA is not installed on your System. (see [Prerequisites](#prerequisites))

- **Problem**: _The translation/transcript is still too slow. But no warning appears._

  **Answers**:
  - Your GPU might be busy with another Task or you are using a too big model for your GPU.
    
    Look in Task-Manager how much the GPU is used without running the AI or change to a smaller model like `small` or `tiny` using `--model small` or `--model tiny`. (see [Command-line flags](#command-line-flags))
  
  - If you happen to have a secondary PC with an GPU, you can outsorce the workload to it:
    
    Run the AI on the secondary PC, start the Websocket-Server with `--websocket_ip 0.0.0.0` to have it listen on all its IP's.
    
    Change the IP in the websocket-client to the one from the **Secondary** PC. (change the line: `var websocketServer = "ws://127.0.0.1:5000"` in the index.html of the example websocket_clients directory.)

    For OSC, give it the IP of your **Primary** PC using the `--websocket_ip 127.0.0.1` argument and change the `127.0.0.1` to its IP.

    Stream the Audio to the **Secondary PC** using for example https://vb-audio.com/Voicemeeter/vban.htm


### _Sources_
A thanks goes to
- OpenAI https://github.com/openai/whisper
- Awexander https://github.com/Awexander/audioWhisper
- Blake https://github.com/mallorbc/whisper_mic

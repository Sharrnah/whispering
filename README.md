# <img src=images/app-icon.png width=90> Whispering Tiger (Live Translate/Transcribe)

Whispering Tiger is a free and Open-Source tool that can listen/watch to any **audio stream** or **in-game image** on your machine and prints out the transcription or translation
to a web browser using Websockets or over OSC (examples are **Streaming-overlays** or **VRChat**).

<img src=images/vrchat.png width=400><img src=images/streaming-overlay.png width=400>


## Content:
- [Features](#features)
- [Quickstart](#quickstart)
- [Release Downloads](#release-downloads)
- [Usage](documentation/usage.md)
  - [Usage with 3rd Party Applications](documentation/usage.md#usage-with-3rd-party-applications)
    - [VRChat](documentation/usage.md#vrchat)
    - [Live Streaming Applications (OBS, vMix, XSplit ...)](documentation/usage.md#live-streaming-applications--obs-vmix-xsplit--)
    - [Desktop+](documentation/usage.md#desktop--currently-only-new-ui-beta-with-embedded-browser-) 
- [Websocket Clients](documentation/websocket-clients.md)
- [Configurations](documentation/configurations.md)
  - [Command-line flags](documentation/configurations.md#command-line-flags)
  - [Settings file](documentation/configurations.md#settings-file)
- [Working with the Code](documentation/working-with-code.md)
- [FAQ](documentation/faq.md)
- [Sources](#sources)


### Features
- Runs 100% locally on your machine. (Once A.I. Models are downloaded, no further internet connection is required)
- **Speech recognition, translation and transcription**
  - OpenAI's [Whisper](https://github.com/openai/whisper) project, Supports ~98 languages)
- **Text translation**
  - LID _[Language Identification]_ (Supports 200 languages)
  - NLLB-200 (single model, Supporting 200 languages, high accuracy)
  - M2M-100 (single model, Supporting 100 languages, high accuracy)
  - ARGOS (multiple models, Supporting 56 languages, depending on the language, low accuracy)
- **OCR** _[Optical Character Recognition]_ (to capture game images and translate in-game text)
  - EasyOCR (Supports 80+ languages)
- **TTS** _[Text-to-Speech]_ (Read out transcriptions/translations)
  - Silero
- **LLM** _[Large language model]_ (Continuation of text. automatic answer generation etc.)
  - FLAN-T5

## Quickstart
For a quick and easy start, download the latest Whispering Tiger UI from here: [https://github.com/Sharrnah/whispering-ui](https://github.com/Sharrnah/whispering-ui#readme)

This is a native UI application that allows keeping your Whispering Tiger version up-to-date and manage the settings more easily.

<img src=https://github.com/Sharrnah/whispering-ui/raw/main/doc/images/speech2text.png width=825>


## Release Downloads
Standalone Releases with all dependencies included.

Go to the [GitHub Releases Page](https://github.com/Sharrnah/whispering/releases) and Download from the download Link in the description or find the [Latest Release here.](https://github.com/Sharrnah/whispering/releases/latest)

_(because of the 2 GB Limit, no direct release files on GitHub)_

- [Install CUDA for GPU Acceleration](https://developer.nvidia.com/cuda-downloads) (recommended)
- Extract the Files on a Drive with enough free Space.
  - _(After download of medium Whisper Model + medium NLLB-200 Translation model, it can take up to 20 GB)_
- Run only using the *.bat files. Edit or copy an existing `start-*.bat` file and edit the parameters in any text editor for your own command-line flags.
  - `start-transcribe-mic.bat` tries to use your default microphone and is a good starting point.


### _Sources_
A thanks goes to
- OpenAI https://github.com/openai/whisper
- Awexander https://github.com/Awexander/audioWhisper
- Blake https://github.com/mallorbc/whisper_mic
- Meta AI (LID, NLLB-200, M2M-100) https://ai.facebook.com/blog/nllb-200-high-quality-machine-translation/
- Argos Translate https://github.com/argosopentech/argos-translate
- EasyOCR https://github.com/jaidedai/easyocr
- Google Research (FLAN-T5) https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html
- Silero (TTS) https://github.com/snakers4/silero-models

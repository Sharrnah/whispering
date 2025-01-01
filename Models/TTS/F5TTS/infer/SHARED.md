<!-- omit in toc -->
# Shared Model Cards

<!-- omit in toc -->
### **Prerequisites of using**
- This document is serving as a quick lookup table for the community training/finetuning result, with various language support.
- The models in this repository are open source and are based on voluntary contributions from contributors.
- The use of models must be conditioned on respect for the respective creators. The convenience brought comes from their efforts.

<!-- omit in toc -->
### **Welcome to share here**
- Have a pretrained/finetuned result: model checkpoint (pruned best to facilitate inference, i.e. leave only `ema_model_state_dict`) and corresponding vocab file (for tokenization).
- Host a public [huggingface model repository](https://huggingface.co/new) and upload the model related files.
- Make a pull request adding a model card to the current page, i.e. `src\f5_tts\infer\SHARED.md`.

<!-- omit in toc -->
### Supported Languages
- [Multilingual](#multilingual)
    - [F5-TTS Base @ zh \& en @ F5-TTS](#f5-tts-base--zh--en--f5-tts)
- [English](#english)
- [Finnish](#finnish)
    - [F5-TTS Base @ fi @ AsmoKoskinen](#f5-tts-base--fi--asmokoskinen)
- [French](#french)
    - [F5-TTS Base @ fr @ RASPIAUDIO](#f5-tts-base--fr--raspiaudio)
- [Hindi](#hindi)
    - [F5-TTS Small @ hi @ SPRINGLab](#f5-tts-small--hi--springlab)
- [Italian](#italian)
    - [F5-TTS Base @ it @ alien79](#f5-tts-base--it--alien79)
- [Japanese](#japanese)
    - [F5-TTS Base @ ja @ Jmica](#f5-tts-base--ja--jmica)
- [Mandarin](#mandarin)
- [Russian](#russian)
    - [F5-TTS Base @ ru @ HotDro4illa](#f5-tts-base--ru--hotdro4illa)
- [Spanish](#spanish)
    - [F5-TTS Base @ es @ jpgallegoar](#f5-tts-base--es--jpgallegoar)


## Multilingual

#### F5-TTS Base @ zh & en @ F5-TTS
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Base|[ckpt & vocab](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_Base)|[Emilia 95K zh&en](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07)|cc-by-nc-4.0|

```bash
Model: hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors
Vocab: hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```

*Other infos, e.g. Author info, Github repo, Link to some sampled results, Usage instruction, Tutorial (Blog, Video, etc.) ...*


## English


## Finnish

#### F5-TTS Base @ fi @ AsmoKoskinen
|Model|🤗Hugging Face|Data|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Base|[ckpt & vocab](https://huggingface.co/AsmoKoskinen/F5-TTS_Finnish_Model)|[Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0), [Vox Populi](https://huggingface.co/datasets/facebook/voxpopuli)|cc-by-nc-4.0|

```bash
Model: hf://AsmoKoskinen/F5-TTS_Finnish_Model/model_common_voice_fi_vox_populi_fi_20241206.safetensors
Vocab: hf://AsmoKoskinen/F5-TTS_Finnish_Model/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```


## French

#### F5-TTS Base @ fr @ RASPIAUDIO
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Base|[ckpt & vocab](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced)|[LibriVox](https://librivox.org/)|cc-by-nc-4.0|

```bash
Model: hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/model_last_reduced.pt
Vocab: hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```

- [Online Inference with Hugging Face Space](https://huggingface.co/spaces/RASPIAUDIO/f5-tts_french).
- [Tutorial video to train a new language model](https://www.youtube.com/watch?v=UO4usaOojys).
- [Discussion about this training can be found here](https://github.com/SWivid/F5-TTS/issues/434).


## Hindi

#### F5-TTS Small @ hi @ SPRINGLab
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Small|[ckpt & vocab](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz)|[IndicTTS Hi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) & [IndicVoices-R Hi](https://huggingface.co/datasets/SPRINGLab/IndicVoices-R_Hindi) |cc-by-4.0|

```bash
Model: hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors
Vocab: hf://SPRINGLab/F5-Hindi-24KHz/vocab.txt
Config: {"dim": 768, "depth": 18, "heads": 12, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```

- Authors: SPRING Lab, Indian Institute of Technology, Madras
- Website: https://asr.iitm.ac.in/


## Italian

#### F5-TTS Base @ it @ alien79
|Model|🤗Hugging Face|Data|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Base|[ckpt & vocab](https://huggingface.co/alien79/F5-TTS-italian)|[ylacombe/cml-tts](https://huggingface.co/datasets/ylacombe/cml-tts) |cc-by-nc-4.0|

```bash
Model: hf://alien79/F5-TTS-italian/model_159600.safetensors
Vocab: hf://alien79/F5-TTS-italian/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```

- Trained by [Mithril Man](https://github.com/MithrilMan)
- Model details on [hf project home](https://huggingface.co/alien79/F5-TTS-italian)
- Open to collaborations to further improve the model


## Japanese

#### F5-TTS Base @ ja @ Jmica
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Base|[ckpt & vocab](https://huggingface.co/Jmica/F5TTS/tree/main/JA_25498980)|[Emilia 1.7k JA](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07) & [Galgame Dataset 5.4k](https://huggingface.co/datasets/OOPPEENN/Galgame_Dataset)|cc-by-nc-4.0|

```bash
Model: hf://Jmica/F5TTS/JA_25498980/model_25498980.pt
Vocab: hf://Jmica/F5TTS/JA_25498980/vocab_updated.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```


## Mandarin


## Russian

#### F5-TTS Base @ ru @ HotDro4illa
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Base|[ckpt & vocab](https://huggingface.co/hotstone228/F5-TTS-Russian)|[Common voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)|cc-by-nc-4.0|

```bash
Model: hf://hotstone228/F5-TTS-Russian/model_last.safetensors
Vocab: hf://hotstone228/F5-TTS-Russian/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```
- Finetuned by [HotDro4illa](https://github.com/HotDro4illa)
- Any improvements are welcome


## Spanish

#### F5-TTS Base @ es @ jpgallegoar
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Base|[ckpt & vocab](https://huggingface.co/jpgallegoar/F5-Spanish)|[Voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) & Crowdsourced & TEDx, 218 hours|cc0-1.0|

- @jpgallegoar [GitHub repo](https://github.com/jpgallegoar/Spanish-F5), Jupyter Notebook and Gradio usage for Spanish model.

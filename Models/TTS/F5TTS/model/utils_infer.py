# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format

import re
import tempfile

import numpy as np
import torch
import torchaudio
import tqdm
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from . import CFM
from .utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device,
)

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")


# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces

def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[；：，。！？])', text)

    for sentence in sentences:
        if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder

def load_vocoder(is_local=False, local_path=""):
    if is_local:
        print(f"Load vocos from local path {local_path}")
        vocos = Vocos.from_hparams(f"{local_path}/config.yaml")
        state_dict = torch.load(f"{local_path}/pytorch_model.bin", map_location=device)
        vocos.load_state_dict(state_dict)
        vocos.eval()
    else:
        print("Download Vocos from huggingface charactr/vocos-mel-24khz")
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocos


# load model for inference

def load_model(model_cls, model_cfg, ckpt_path, vocab_file=""):
    
    if vocab_file == "":
        vocab_file = "Emilia_ZH_EN"
        tokenizer = "pinyin"
    else:
        tokenizer = "custom"

    print("\nvocab : ", vocab_file, tokenizer) 
    print("tokenizer : ", tokenizer) 
    print("model : ", ckpt_path,"\n")    

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema = True)

    return model


# preprocess reference audio and text

def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave

        audio_duration = len(aseg)
        if audio_duration > 15000:
            show_info("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.strip():
        show_info("No reference text provided, transcribing reference audio...")
        ref_text = asr_pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        show_info("Finished transcription")
    else:
        show_info("Using custom reference text...")

    # Add the functionality to ensure it ends with ". "
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]

def infer_process(ref_audio, ref_text, gen_text, model_obj, cross_fade_duration=0.15, speed=speed, show_info=print, progress=tqdm):

    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    #for i, gen_text in enumerate(gen_text_batches):
    #    print(f'gen_text {i}', gen_text)

    if show_info is not None:
        show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return infer_batch_process((audio, sr), ref_text, gen_text_batches, model_obj, cross_fade_duration, speed, progress)


# infer batches

def infer_batch_process(ref_audio, ref_text, gen_text_batches, model_obj, cross_fade_duration=0.15, speed=1, progress=tqdm):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode('utf-8')) == 1:
        ref_text = ref_text + " "
    #for i, gen_text in enumerate(progress.tqdm(gen_text_batches)):
    for i, gen_text in enumerate(gen_text_batches):
        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        ref_text_len = len(ref_text.encode('utf-8'))
        gen_text_len = len(gen_text.encode('utf-8'))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = generated.permute(0, 2, 1)
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate([
                prev_wave[:-cross_fade_samples],
                cross_faded_overlap,
                next_wave[cross_fade_samples:]
            ])

            final_wave = new_wave

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    return final_wave, target_sample_rate, combined_spectrogram


# remove silence from generated wav

def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")

# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from pathlib import Path
import os
import settings
import random
import downloader

# MODEL_LINKS = {
#     "small": "google/flan-t5-small",
#     "base": "google/flan-t5-base",
#     "large": "google/flan-t5-large",
#     "xl": "google/flan-t5-xl",
#     "xxl": "google/flan-t5-xxl"
# }

MODEL_LINKS = {
    "small": "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5%2Fsmall.zip",
    "base": "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5%2Fbase.zip",
    "large": "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5%2Flarge.zip",
    "xl": "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5%2Fxl.zip",
    "xxl": "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5%2Fxxl.zip"
}

cache_path = Path(Path.cwd() / ".cache" / "flan-t5-cache")
os.makedirs(cache_path, exist_ok=True)
weight_offload_folder = Path(cache_path / "weight_offload")
os.makedirs(weight_offload_folder, exist_ok=True)

flan = None

PROMPT_FORMATTING = {
    "question": ["about ", "across ", "after ", "against ", "along ", "am ", "amn't ", "among ", "are ", "aren't ", "around ", "at ", "before ", "behind ", "between ",
                 "beyond ", "but ", "by ", "can ", "can't ", "concerning ", "could ", "couldn't ", "despite ", "did ", "didn't ", "do ", "does ", "doesn't ", "don't ",
                 "down ", "during ", "except ", "following ", "for ", "from ", "had ", "hadn't ", "has ", "hasn't ", "have ", "haven't ", "how ", "how's ", "in ",
                 "including ", "into ", "is ", "isn't ", "like ", "may ", "mayn't ", "might ", "mightn't ", "must ", "mustn't ", "near ", "of ", "off ", "on ", "out ",
                 "over ", "plus ", "shall ", "shan't ", "should ", "shouldn't ", "since ", "through ", "throughout ", "to ", "towards ", "under ", "until ", "up ", "upon ",
                 "was ", "wasn't ", "were ", "weren't ", "what ", "what's ", "when ", "when's ", "where ", "where's ", "which ", "which's ", "who ", "who's ", "why ",
                 "why's ", "will ", "with ", "within ", "without ", "won't ", "would ", "wouldn't "]
}


class FlanLanguageModel:
    tokenizer = None
    model = None
    model_size = "large"
    max_length = 50  # max result token length. default is 20
    bit_length = 32  # can be 32 = 32 float, 16 = 16 float or 8 = 8 int
    device_map = "auto"  # can be "auto" or None
    low_cpu_mem_usage = True

    # Set the device. "cuda" for GPU or None for CPU
    def __init__(self, model_size, device="auto", bit_length=32):
        self.model_size = model_size
        self.device_map = device
        self.bit_length = bit_length

        model_path = Path(cache_path / model_size)

        if not model_path.exists():
            print(f"Downloading {model_size} FLAN-T5 model...")
            downloader.download_extract(MODEL_LINKS[model_size], str(cache_path.resolve()))

        model_path_string = str(model_path.resolve())

        self.tokenizer = T5Tokenizer.from_pretrained(model_path_string, cache_dir=str(cache_path.resolve()))

        match self.bit_length:
            case 16:  # 16 bit float
                self.model = T5ForConditionalGeneration.from_pretrained(model_path_string, device_map=self.device_map, torch_dtype=torch.float16,
                                                                        offload_folder=str(weight_offload_folder.resolve()))
            case 8:  # 8 bit int
                self.model = T5ForConditionalGeneration.from_pretrained(model_path_string, device_map=self.device_map, load_in_8bit=True,
                                                                        offload_folder=str(weight_offload_folder.resolve()))
            case _:  # 32 bit float
                self.model = T5ForConditionalGeneration.from_pretrained(model_path_string, device_map=self.device_map,
                                                                        offload_folder=str(weight_offload_folder.resolve()))

    # Try to modify prompts to get better results
    @staticmethod
    def whisper_result_prompter(whisper_result: str):
        prompt_change = False
        question = whisper_result.strip().lower()
        question_prompt = whisper_result.strip()

        possible_prompt_prefixes = []
        # looks like a question
        if "?" in question and any(ele in question for ele in PROMPT_FORMATTING['question']):
            possible_prompt_prefixes.append("Answer the following question by reasoning step-by-step. ")
            possible_prompt_prefixes.append("Answer the following question. ")
            possible_prompt_prefixes.append("Question: ")
            possible_prompt_prefixes.append("Q: ")
            prompt_change = True

        if prompt_change:
            question_prompt = random.choice(possible_prompt_prefixes) + question_prompt

        return question_prompt, prompt_change

    def encode(self, input_text, token_length=max_length):
        if self.device_map == "auto":
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        else:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        outputs = self.model.generate(input_ids, max_new_tokens=token_length)
        result = self.tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "").replace("<unk>", "").strip()

        return result


def init():
    global flan
    if settings.GetOption("flan_enabled") and flan is None:
        model_size = settings.GetOption("flan_size")
        flan_bits = settings.GetOption("flan_bits")
        flan_device = "auto" if settings.GetOption("flan_device") == "cuda" or settings.GetOption("flan_device") == "auto" else None
        print(f"Flan {model_size} is Loading to {('GPU' if flan_device == 'auto' else 'CPU')} using {flan_bits} bit {('INT' if flan_bits == 8 else 'float')} precision...")

        flan = FlanLanguageModel(model_size, bit_length=flan_bits, device=flan_device)
        print("Flan loaded.")
        return True
    else:
        if flan is not None:
            return True
        else:
            return False

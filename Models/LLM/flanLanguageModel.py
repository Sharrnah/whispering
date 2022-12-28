# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from pathlib import Path
import os

import loading_state
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
    "small": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/FLAN-T5/small.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5/small.zip",
            "https://s3.libs.space:9000/ai-models/FLAN-T5/small.zip",
        ],
        "checksum": "34415fe1e13813e5e3037b950794197870deb5573b0de899d785a1094c1a5e0e"
    },
    "base": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/FLAN-T5/base.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5/base.zip",
            "https://s3.libs.space:9000/ai-models/FLAN-T5/base.zip",
         ],
        "checksum": "577e84801ac0a0dac18a8c66962a688b828f3a567546e988879ae5279d51fcbe"
     },
    "large": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/FLAN-T5/large.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5/large.zip",
            "https://s3.libs.space:9000/ai-models/FLAN-T5/large.zip",
        ],
        "checksum": "8a713559ffc9ba6ac1af3d8d24478261a085cac3d5448a934596379cb0420518"
    },
    "xl": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/FLAN-T5/xl.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5/xl.zip",
            "https://s3.libs.space:9000/ai-models/FLAN-T5/xl.zip",
        ],
        "checksum": "15c85799f083b284c73e724068a7ff95c901cd7b55a206a49661940ae5bd4778"
    },
    "xxl": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/FLAN-T5/xxl.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/FLAN-T5/xxl.zip",
            "https://s3.libs.space:9000/ai-models/FLAN-T5/xxl.zip",
        ],
        "checksum": "66c221e46e230714675ae2f1af86852d886ddb2b26ba9ca7f73acd6ed27160dc"
    }
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
                 "why's ", "will ", "with ", "within ", "without ", "won't ", "would ", "wouldn't "],
    "statement": ["i ", "i am ", "i am not ", "i was ", "i was not ", "i will ", "i will not ", "i would ", "i would not ", "i have ", "i have not ", "i had ", "i had not ",
                  "i can ", "i want ", "i need ", "i like ", "i love ", "i hate ", "i don't like ", "i don't love ", "i don't hate ", "i don't want ", "i don't need ", "i don't ",
                  "i do ", "i do not ", "i did ", "i did not ", "i will ", "i will not ", "i would ", "i would not ", "i have ", "i have not ", "i had ", "i had not ", "i can ",
                  "i can not ", "i cannot ", "i am not ", "i am ", "i am not ", "i was ", "i was not ", "i will ", "i will not ", "i would ", "i would not ", "i have ",
                  "i have not ", "i had ", "i had not ", "i can ", "i can not ", "i cannot ", "i am not ", "i am ", "i am not ", "i was ", "i was not ", "i will ", "i will not ",
                  "i would ", "i would not ", "i have ", "i have not ", "i had ", "i had not ", "i can ", "i can not ", "i cannot ", "i am not ", "i am ", "i am not ", "i was ",
                  "i was not ", "i will ", "i will not ", "i would ", "i would not ", "i have ", "i have not ", "i had ", "i had not ", "i can ", "i can not ", "i cannot ",
                  "i am not ", "i am ", "i am not ", "i was ", "i was not ", "i will ", "i will not ", "i would ", "i would not ", "i have ", "i have not ", "i had ", "i had not ",
                  "i can ", "i can not ", "i cannot ", "i am not ", "i am ", "i am not ", "i was ", "i was not ", "i will ", "i will not ", "i would ", "i would not ", "i have ",
                  "i have not ", "i had ", "i had not ", "i can ", "i can not ", "i cannot ", "i am not ", "i am ", "i am not ", "i was "],
    "command": ["you have ", "you should ", "you do ", "ai ", "artificial intelligence"],
}


class FlanLanguageModel:
    tokenizer = None
    model = None
    model_size = "large"
    max_length = 80  # max result token length. default is 20
    bit_length = 32  # can be 32 = 32 float, 16 = 16 float or 8 = 8 int
    device_map = "auto"  # can be "auto" or None
    low_cpu_mem_usage = True

    conditioning_lines = []

    # Set the device. "cuda" for GPU or None for CPU
    def __init__(self, model_size=model_size, device=device_map, bit_length=bit_length):
        self.model_size = model_size
        self.device_map = device
        self.bit_length = bit_length

        model_path = Path(cache_path / model_size)

        if not model_path.exists():
            print(f"Downloading {model_size} FLAN-T5 model...")
            downloader.download_extract(MODEL_LINKS[model_size]["urls"], str(cache_path.resolve()), MODEL_LINKS[model_size]["checksum"])

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
            # possible_prompt_prefixes.append("Answer the following question by reasoning step-by-step. ")
            possible_prompt_prefixes.append("Answer the following question. ")
            possible_prompt_prefixes.append("Question: ")
            possible_prompt_prefixes.append("Q: ")
            prompt_change = True

        if prompt_change:
            question_prompt = random.choice(possible_prompt_prefixes) + question_prompt

        return question_prompt, prompt_change

    def encode(self, input_text, token_length=max_length):
        # Add flan prompt prefix
        if settings.GetOption("flan_prompt") != "":
            flan_prompt = settings.GetOption("flan_prompt")
            if flan_prompt.count("??") > 0:
                input_text = flan_prompt.replace("??", input_text)
            else:
                input_text = flan_prompt + input_text
        conditioning_input_text = input_text

        # Add conditioning lines
        if settings.GetOption("flan_conditioning_history") > 0 and len(self.conditioning_lines) > 0:
            input_text = "\n".join(self.conditioning_lines) + "\n" + input_text

        # Add flan long-term memory
        if settings.GetOption("flan_memory") != "":
            input_text = settings.GetOption("flan_memory") + "\n" + input_text

        if self.device_map == "auto":
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        else:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        outputs = self.model.generate(input_ids, max_new_tokens=token_length)
        result = self.tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "").replace("<unk>", "").strip()

        # Add the result to the conditioning history and remove the oldest lines if needed
        if settings.GetOption("flan_conditioning_history") > 0:
            if len(self.conditioning_lines) >= settings.GetOption("flan_conditioning_history"):
                difference = len(self.conditioning_lines) - settings.GetOption("flan_conditioning_history")
                del self.conditioning_lines[0:difference - 1]

            self.conditioning_lines.append(conditioning_input_text + result)
        else:
            self.conditioning_lines.clear()

        # remove some common prefixes from the start of the result (@todo: make this configurable)
        result = result.removeprefix("A: ")
        result = result.removeprefix("AI: ")
        result = result.removeprefix("Human: ")

        return result


def init():
    global flan
    if settings.GetOption("flan_enabled") and flan is None:
        loading_state.set_loading_state("flan_loading", False)
        model_size = settings.GetOption("flan_size")
        flan_bits = settings.GetOption("flan_bits")
        flan_device = "auto" if settings.GetOption("flan_device") == "cuda" or settings.GetOption("flan_device") == "auto" else None
        print(f"Flan {model_size} is Loading to {('GPU' if flan_device == 'auto' else 'CPU')} using {flan_bits} bit {('INT' if flan_bits == 8 else 'float')} precision...")

        flan = FlanLanguageModel(model_size, bit_length=flan_bits, device=flan_device)
        print("Flan loaded.")
        loading_state.set_loading_state("flan_loading", False)
        return True
    else:
        if flan is not None:
            return True
        else:
            return False

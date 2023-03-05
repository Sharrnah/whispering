# pip install accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import os
from time import strftime

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
}
TMP_CHECKPOINT = "EleutherAI/gpt-j-6B"

cache_path = Path(Path.cwd() / ".cache" / "gptj-cache")
os.makedirs(cache_path, exist_ok=True)
weight_offload_folder = Path(cache_path / "weight_offload")
os.makedirs(weight_offload_folder, exist_ok=True)

model = None

PROMPT_FORMATTING = {
    "question": ["about ", "across ", "after ", "against ", "along ", "am ", "amn't ", "among ", "are ", "aren't ",
                 "around ", "at ", "before ", "behind ", "between ",
                 "beyond ", "but ", "by ", "can ", "can't ", "concerning ", "could ", "couldn't ", "despite ", "did ",
                 "didn't ", "do ", "does ", "doesn't ", "don't ",
                 "down ", "during ", "except ", "following ", "for ", "from ", "had ", "hadn't ", "has ", "hasn't ",
                 "have ", "haven't ", "how ", "how's ", "in ",
                 "including ", "into ", "is ", "isn't ", "like ", "may ", "mayn't ", "might ", "mightn't ", "must ",
                 "mustn't ", "near ", "of ", "off ", "on ", "out ",
                 "over ", "plus ", "shall ", "shan't ", "should ", "shouldn't ", "since ", "through ", "throughout ",
                 "to ", "towards ", "under ", "until ", "up ", "upon ",
                 "was ", "wasn't ", "were ", "weren't ", "what ", "what's ", "when ", "when's ", "where ", "where's ",
                 "which ", "which's ", "who ", "who's ", "why ",
                 "why's ", "will ", "with ", "within ", "without ", "won't ", "would ", "wouldn't "],
    "statement": ["i ", "i am ", "i am not ", "i was ", "i was not ", "i will ", "i will not ", "i would ",
                  "i would not ", "i have ", "i have not ", "i had ", "i had not ",
                  "i can ", "i want ", "i need ", "i like ", "i love ", "i hate ", "i don't like ", "i don't love ",
                  "i don't hate ", "i don't want ", "i don't need ", "i don't ",
                  "i do ", "i do not ", "i did ", "i did not ", "i will ", "i will not ", "i would ", "i would not ",
                  "i have ", "i have not ", "i had ", "i had not ", "i can ",
                  "i can not ", "i cannot ", "i am not ", "i am ", "i am not ", "i was ", "i was not ", "i will ",
                  "i will not ", "i would ", "i would not ", "i have ",
                  "i have not ", "i had ", "i had not ", "i can ", "i can not ", "i cannot ", "i am not ", "i am ",
                  "i am not ", "i was ", "i was not ", "i will ", "i will not ",
                  "i would ", "i would not ", "i have ", "i have not ", "i had ", "i had not ", "i can ", "i can not ",
                  "i cannot ", "i am not ", "i am ", "i am not ", "i was ",
                  "i was not ", "i will ", "i will not ", "i would ", "i would not ", "i have ", "i have not ",
                  "i had ", "i had not ", "i can ", "i can not ", "i cannot ",
                  "i am not ", "i am ", "i am not ", "i was ", "i was not ", "i will ", "i will not ", "i would ",
                  "i would not ", "i have ", "i have not ", "i had ", "i had not ",
                  "i can ", "i can not ", "i cannot ", "i am not ", "i am ", "i am not ", "i was ", "i was not ",
                  "i will ", "i will not ", "i would ", "i would not ", "i have ",
                  "i have not ", "i had ", "i had not ", "i can ", "i can not ", "i cannot ", "i am not ", "i am ",
                  "i am not ", "i was "],
    "command": ["ai? ", "ai. ", "ai ", "a.i. ", "ai, ", "ai! ", "artificial intelligence", "tiger"],
}


class GPTJLanguageModel:
    tokenizer = None
    model = None
    model_size = "large"
    max_length = 512  # max result token length. default is 20
    bit_length = 16  # can be 32 = 32 float, 16 = 16 float or 8 = 8 int
    device_map = "auto"  # can be "auto" or None
    low_cpu_mem_usage = True
    stop_token = "[human]"

    conditioning_lines = []

    # Set the device. "cuda" for GPU or None for CPU
    def __init__(self, model_size=model_size, device=device_map, bit_length=bit_length):
        self.model_size = model_size
        self.device_map = device
        self.bit_length = bit_length

        model_path = Path(cache_path / model_size)

        self.tokenizer = AutoTokenizer.from_pretrained(TMP_CHECKPOINT)

        match self.bit_length:
            case 16:  # 16 bit float
                self.model = AutoModelForCausalLM.from_pretrained(TMP_CHECKPOINT, cache_dir=str(cache_path.resolve()),
                                                                  revision="float16", device_map=self.device_map,
                                                                  torch_dtype=torch.float16,
                                                                  low_cpu_mem_usage=self.low_cpu_mem_usage)
            case 8:  # 8 bit int
                self.model = AutoModelForCausalLM.from_pretrained(TMP_CHECKPOINT, cache_dir=str(cache_path.resolve()),
                                                                  device_map=self.device_map, load_in_8bit=True,
                                                                  low_cpu_mem_usage=self.low_cpu_mem_usage)
            case _:  # 32 bit float
                self.model = AutoModelForCausalLM.from_pretrained(TMP_CHECKPOINT, cache_dir=str(cache_path.resolve()),
                                                                  device_map=self.device_map,
                                                                  low_cpu_mem_usage=self.low_cpu_mem_usage)

    # Try to modify prompts to get better results
    @staticmethod
    def whisper_result_prompter(whisper_result: str):
        prompt_change = False
        question = whisper_result.strip().lower()
        question_prompt = whisper_result.strip()

        possible_prompt_prefixes = []
        # looks like a question
        #if "?" in question and any(ele in question for ele in PROMPT_FORMATTING['question']):
        if ("?" in question and any(ele in question for ele in PROMPT_FORMATTING['question'])) or any(ele in question for ele in PROMPT_FORMATTING['command']):
        #if any(ele in question for ele in PROMPT_FORMATTING['command']):
            # possible_prompt_prefixes.append("Answer the following question by reasoning step-by-step. ")
            #possible_prompt_prefixes.append("Answer the following question. ")
            #possible_prompt_prefixes.append("Question: ")
            #possible_prompt_prefixes.append("Q: ")
            prompt_change = True

        if prompt_change:
            question_prompt = random.choice(possible_prompt_prefixes) + question_prompt

        return question_prompt, prompt_change

    def encode(self, input_text, token_length=max_length, retry=0):
        original_input_text = input_text

        # make sure input has an end token
        if not input_text.endswith(".") and not input_text.endswith("!") and not input_text.endswith(
                "?") and not input_text.endswith(",") and not input_text.endswith(";") and not input_text.endswith(":"):
            input_text += "."

        # Add flan prompt prefix
        if settings.GetOption("flan_prompt") != "":
            flan_prompt = settings.GetOption("flan_prompt")
            if flan_prompt.count("??") > 0:
                input_text = flan_prompt.replace("??", input_text)
            else:
                input_text = flan_prompt + input_text
        conditioning_input_text = input_text

        # add current time infos
        input_text = strftime("It is %A the %d %B %Y and the time is %H:%M.") + "\n" + input_text

        # Add conditioning lines
        if settings.GetOption("flan_conditioning_history") > 0 and len(self.conditioning_lines) > 0:
            input_text = "\n".join(self.conditioning_lines) + "\n" + input_text

        # Add flan long-term memory
        if settings.GetOption("flan_memory") != "":
            input_text = settings.GetOption("flan_memory") + "\n" + input_text

        if self.device_map == "auto":
            input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids'].cuda(0)
            # input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=self.max_length).to("cuda")
            # input_ids = self.model.tokenizer.encode(input_text, return_tensors="pt", max_length=512).input_ids.to("cuda")
        else:
            input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids']
            # input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=self.max_length)
            # input_ids = self.model.tokenizer.encode(input_text, return_tensors="pt", max_length=512)

        stop_token = self.tokenizer.encode(self.stop_token, return_tensors="pt")[0]

        with torch.no_grad():
            # outputs = self.model.generate(input_ids, max_length=512, top_p=0.9, pad_token_id=stop_token, num_return_sequences=1)
            output_tokens = self.model.generate(
                input_ids,
                do_sample=True,
                use_cache=True,
                temperature=0.8,
                min_length=len(input_ids[0]) + 10,
                max_length=len(input_ids[0]) + 40,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )

        # result = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = self.tokenizer.batch_decode(output_tokens)[0]

        result = result.replace("<pad>", "").replace("</s>", "").replace("<unk>", "").strip()

        # remove the input text and human hallucination from the result
        ol = len(input_text)
        n = 0
        for i in range(ol, len(result)):
            if result[i] == '\n\n' or result[i] == '[':
                n = i
                break
        result = result[ol:n]

        # remove some common prefixes from the start of the result (@todo: make this configurable)
        result = result.strip().removeprefix(settings.GetOption("flan_memory"))
        result = result.strip().removeprefix("\n".join(self.conditioning_lines) + "\n")
        result = result.strip().removeprefix(conditioning_input_text)

        result = result.removeprefix("A: ")
        result = result.removeprefix("AI: ")
        result = result.removeprefix("Human: ")
        result = result.removeprefix("[human]")
        result = result.removeprefix(":")

        if result.strip() == "":
            if retry < 3:
                return self.encode(original_input_text, token_length, retry + 1)
            else:
                result = "hmm..."

        # Add the result to the conditioning history and remove the oldest lines if needed
        if settings.GetOption("flan_conditioning_history") > 0:
            if len(self.conditioning_lines) >= settings.GetOption("flan_conditioning_history"):
                difference = len(self.conditioning_lines) - settings.GetOption("flan_conditioning_history")
                del self.conditioning_lines[0:difference - 1]

            self.conditioning_lines.append(conditioning_input_text + result)
        else:
            self.conditioning_lines.clear()

        return result.strip()


def init():
    global model
    if settings.GetOption("flan_enabled") and model is None:
        loading_state.set_loading_state("gpt-j_loading", True)
        model_size = settings.GetOption("flan_size")
        flan_bits = settings.GetOption("flan_bits")
        flan_device = "auto" if settings.GetOption("flan_device") == "cuda" or settings.GetOption(
            "flan_device") == "auto" else None
        print(
            f"GPT-J {model_size} is Loading to {('GPU' if flan_device == 'auto' else 'CPU')} using {flan_bits} bit {('INT' if flan_bits == 8 else 'float')} precision...")

        model = GPTJLanguageModel(model_size, bit_length=flan_bits, device=flan_device)
        print("GPT-J loaded.")
        loading_state.set_loading_state("gpt-j_loading", False)
        return True
    else:
        if model is not None:
            return True
        else:
            return False

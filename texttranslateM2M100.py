from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# requires protobuf==3.20.1

LANGUAGES = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Arabic": "ar",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Breton": "br",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greeek": "el",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian": "fa",
    "Fulah": "ff",
    "Finnish": "fi",
    "French": "fr",
    "Western Frisian": "fy",
    "Irish": "ga",
    "Gaelic": "gd",
    "Galician": "gl",
    "Gujarati": "gu",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Croatian": "hr",
    "Haitian": "ht",
    "Hungarian": "hu",
    "Armenian": "hy",
    "Indonesian": "id",
    "Igbo": "ig",
    "Iloko": "ilo",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Central Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Luxembourgish": "lb",
    "Ganda": "lg",
    "Lingala": "ln",
    "Lao": "lo",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Malagasy": "mg",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Burmese": "my",
    "Nepali": "ne",
    "Dutch": "nl",
    "Norwegian": "no",
    "Northern Sotho": "ns",
    "Occitan (post 1500)": "oc",
    "Oriya": "or",
    "Panjabi": "pa",
    "Polish": "pl",
    "Pushto": "ps",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Albanian": "sq",
    "Serbian": "sr",
    "Swati": "ss",
    "Sundanese": "su",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Thai": "th",
    "Tagalog": "tl",
    "Tswana": "tn",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Wolof": "wo",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Chinese": "zh",
    "Zulu": "zu"
}

model = M2M100ForConditionalGeneration
tokenizer = M2M100Tokenizer


def get_installed_language_names():
    return tuple([{"code": code, "name": language} for language, code in LANGUAGES.items()])


def load_model(size="small"):
    global model
    global tokenizer
    match size:
        case "small":
            model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

        case "large":
            model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
            tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

        case _:
            model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


def translate_language(text, from_code, to_code):
    tokenizer.src_lang = from_code
    encoded_hi = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(to_code))
    translation_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return ' '.join(translation_text)

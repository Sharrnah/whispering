import torch
import ctranslate2
import sentencepiece as spm
import os
import downloader
from pathlib import Path
from Models import languageClassification

nltk_path = Path(Path.cwd() / ".cache" / "nltk")
os.makedirs(nltk_path, exist_ok=True)
os.environ["NLTK_DATA"] = str(nltk_path.resolve())
from nltk.tokenize import sent_tokenize
import nltk

LANGUAGES = {
    "Achinese (Arab)": "ace_Arab",
    "Achinese (Latn)": "ace_Latn",
    "Mesopotamian": "acm_Arab",
    "Ta’izzi-Adeni": "acq_Arab",
    "Tunisian": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Aymara": "ayr_Latn",
    "Azerbaijani (South)": "azb_Arab",
    "Azerbaijani (North)": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjarese (Arab)": "bjn_Arab",
    "Banjarese (Latn)": "bjn_Latn",
    "Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Sorani": "ckb_Arab",
    "Crimean": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek (Modern)": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Persian": "pes_Arab",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Fula": "fuv_Latn",
    "Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Iloko": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Kachin": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri (Arab)": "kas_Arab",
    "Kashmiri (Deva)": "kas_Deva",
    "Georgian": "kat_Geor",
    "Kanuri (Arab)": "knc_Arab",
    "Kanuri (Latn)": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiye": "kbp_Latn",
    "Kriolu": "kea_Latn",
    "Central Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kirghiz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Kongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Kurmanji": "kmr_Latn",
    "Lao": "lao_Laoo",
    "Latvian": "lvs_Latn",
    "Ligurian": "lij_Latn",
    "Limburgan": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": "ltg_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luba-Lulua": "lua_Latn",
    "Ganda": "lug_Latn",
    "Luo": "luo_Latn",
    "Lushai": "lus_Latn",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Minangkabau": "min_Latn",
    "Macedonian": "mkd_Cyrl",
    "Malagasy": "plt_Latn",
    "Maltese": "mlt_Latn",
    "Manipuri": "mni_Beng",
    "Mongolian": "khk_Cyrl",
    "Mossi": "mos_Latn",
    "Māori": "mri_Latn",
    "Malaysian Malay": "zsm_Latn",
    "Burmese": "mya_Mymr",
    "Dutch": "nld_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Nepali": "npi_Deva",
    "Northern Sotho": "nso_Latn",
    "Nuer": "nus_Latn",
    "Chichewa, Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "Oromo": "gaz_Latn",
    "Odia": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Panjabi": "pan_Guru",
    "Papiamento": "pap_Latn",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Dari": "prs_Arab",
    "Southern Pashto": "pbt_Arab",
    "Ayacucho Quechua": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Beng",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Samoan": "smo_Latn",
    "Shona": "sna_Latn",
    "Sindhi": "snd_Arab",
    "Somali": "som_Latn",
    "Sotho": "sot_Latn",
    "Spanish": "spa_Latn",
    "Tosk Albanian": "als_Latn",
    "Sardinian": "srd_Latn",
    "Serbian": "srp_Cyrl",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swedish": "swe_Latn",
    "Swahili": "swh_Latn",
    "Silesian": "szl_Latn",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Tajik": "tgk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Thai": "tha_Thai",
    "Tigrinya": "tir_Ethi",
    "Tamasheq (Latn)": "taq_Latn",
    "Tamasheq": "taq_Tfng",
    "Tok Pisin, Pidgin": "tpi_Latn",
    "Tswana": "tsn_Latn",
    "Tsonga": "tso_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Turkish": "tur_Latn",
    "Twi": "twi_Latn",
    "Atlasic": "tzm_Tfng",
    "Uighur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Uzbek": "uzn_Latn",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Yiddish": "ydd_Hebr",
    "Yoruba": "yor_Latn",
    "Yue Chinese": "yue_Hant",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Zulu": "zul_Latn"
}

# mapping from ISO 639-1 to ISO 639-3.
LANGUAGES_ISO1_TO_ISO3 = {
    "ace": ["ace_Arab", "ace_Latn"],
    "acm": ["acm_Arab"],
    "acq": ["acq_Arab"],
    "aeb": ["aeb_Arab"],
    "af": ["afr_Latn"],
    "ajp": ["ajp_Arab"],
    "ak": ["aka_Latn"],
    "am": ["amh_Ethi"],
    "apc": ["apc_Arab"],
    "arb": ["arb_Arab"],
    "ars": ["ars_Arab"],
    "ary": ["ary_Arab"],
    "arz": ["arz_Arab"],
    "as": ["asm_Beng"],
    "ast": ["ast_Latn"],
    "awa": ["awa_Deva"],
    "ayr": ["ayr_Latn"],
    "azb": ["azb_Arab"],
    "azj": ["azj_Latn"],
    "ba": ["bak_Cyrl"],
    "bm": ["bam_Latn"],
    "ban": ["ban_Latn"],
    "be": ["bel_Cyrl"],
    "bem": ["bem_Latn"],
    "bn": ["ben_Beng"],
    "bho": ["bho_Deva"],
    "bjn": ["bjn_Arab", "bjn_Latn"],
    "bo": ["bod_Tibt"],
    "bs": ["bos_Latn"],
    "bug": ["bug_Latn"],
    "bg": ["bul_Cyrl"],
    "ca": ["cat_Latn"],
    "ceb": ["ceb_Latn"],
    "cs": ["ces_Latn"],
    "cjk": ["cjk_Latn"],
    "ckb": ["ckb_Arab"],
    "crh": ["crh_Latn"],
    "cy": ["cym_Latn"],
    "da": ["dan_Latn"],
    "de": ["deu_Latn"],
    "dik": ["dik_Latn"],
    "dyu": ["dyu_Latn"],
    "dz": ["dzo_Tibt"],
    "el": ["ell_Grek"],
    "en": ["eng_Latn"],
    "eo": ["epo_Latn"],
    "et": ["est_Latn"],
    "eu": ["eus_Latn"],
    "ee": ["ewe_Latn"],
    "fo": ["fao_Latn"],
    "pes": ["pes_Arab"],
    "fj": ["fij_Latn"],
    "fi": ["fin_Latn"],
    "fon": ["fon_Latn"],
    "fr": ["fra_Latn"],
    "fur": ["fur_Latn"],
    "fuv": ["fuv_Latn"],
    "gd": ["gla_Latn"],
    "ga": ["gle_Latn"],
    "gl": ["glg_Latn"],
    "gn": ["grn_Latn"],
    "gu": ["guj_Gujr"],
    "ht": ["hat_Latn"],
    "ha": ["hau_Latn"],
    "he": ["heb_Hebr"],
    "hi": ["hin_Deva"],
    "hne": ["hne_Deva"],
    "hr": ["hrv_Latn"],
    "hu": ["hun_Latn"],
    "hy": ["hye_Armn"],
    "ig": ["ibo_Latn"],
    "ilo": ["ilo_Latn"],
    "id": ["ind_Latn"],
    "is": ["isl_Latn"],
    "it": ["ita_Latn"],
    "jv": ["jav_Latn"],
    "ja": ["jpn_Jpan"],
    "kab": ["kab_Latn"],
    "kac": ["kac_Latn"],
    "kam": ["kam_Latn"],
    "kn": ["kan_Knda"],
    "ks": ["kas_Arab", "kas_Deva"],
    "ka": ["kat_Geor"],
    "knc": ["knc_Arab", "knc_Latn"],
    "kk": ["kaz_Cyrl"],
    "kbp": ["kbp_Latn"],
    "kea": ["kea_Latn"],
    "km": ["khm_Khmr"],
    "ki": ["kik_Latn"],
    "rw": ["kin_Latn"],
    "ky": ["kir_Cyrl"],
    "kmb": ["kmb_Latn"],
    "kg": ["kon_Latn"],
    "ko": ["kor_Hang"],
    "kmr": ["kmr_Latn"],
    "lo": ["lao_Laoo"],
    "lvs": ["lvs_Latn"],
    "lij": ["lij_Latn"],
    "li": ["lim_Latn"],
    "ln": ["lin_Latn"],
    "lt": ["lit_Latn"],
    "lmo": ["lmo_Latn"],
    "ltg": ["ltg_Latn"],
    "lb": ["ltz_Latn"],
    "lua": ["lua_Latn"],
    "lg": ["lug_Latn"],
    "luo": ["luo_Latn"],
    "lus": ["lus_Latn"],
    "mag": ["mag_Deva"],
    "mai": ["mai_Deva"],
    "ml": ["mal_Mlym"],
    "mr": ["mar_Deva"],
    "min": ["min_Latn"],
    "mk": ["mkd_Cyrl"],
    "plt": ["plt_Latn"],
    "mt": ["mlt_Latn"],
    "mni": ["mni_Beng"],
    "khk": ["khk_Cyrl"],
    "mos": ["mos_Latn"],
    "mi": ["mri_Latn"],
    "zsm": ["zsm_Latn"],
    "my": ["mya_Mymr"],
    "nl": ["nld_Latn"],
    "nn": ["nno_Latn"],
    "nb": ["nob_Latn"],
    "npi": ["npi_Deva"],
    "nso": ["nso_Latn"],
    "nus": ["nus_Latn"],
    "ny": ["nya_Latn"],
    "oc": ["oci_Latn"],
    "gaz": ["gaz_Latn"],
    "ory": ["ory_Orya"],
    "pag": ["pag_Latn"],
    "pa": ["pan_Guru"],
    "pap": ["pap_Latn"],
    "pl": ["pol_Latn"],
    "pt": ["por_Latn"],
    "prs": ["prs_Arab"],
    "pbt": ["pbt_Arab"],
    "quy": ["quy_Latn"],
    "ro": ["ron_Latn"],
    "rn": ["run_Latn"],
    "ru": ["rus_Cyrl"],
    "sg": ["sag_Latn"],
    "sa": ["san_Deva"],
    "sat": ["sat_Beng"],
    "scn": ["scn_Latn"],
    "shn": ["shn_Mymr"],
    "si": ["sin_Sinh"],
    "sk": ["slk_Latn"],
    "sl": ["slv_Latn"],
    "sm": ["smo_Latn"],
    "sn": ["sna_Latn"],
    "sd": ["snd_Arab"],
    "so": ["som_Latn"],
    "st": ["sot_Latn"],
    "es": ["spa_Latn"],
    "als": ["als_Latn"],
    "sc": ["srd_Latn"],
    "sr": ["srp_Cyrl"],
    "ss": ["ssw_Latn"],
    "su": ["sun_Latn"],
    "sv": ["swe_Latn"],
    "swh": ["swh_Latn"],
    "szl": ["szl_Latn"],
    "ta": ["tam_Taml"],
    "tt": ["tat_Cyrl"],
    "te": ["tel_Telu"],
    "tg": ["tgk_Cyrl"],
    "tl": ["tgl_Latn"],
    "th": ["tha_Thai"],
    "ti": ["tir_Ethi"],
    "taq": ["taq_Tfng", "taq_Latn"],
    "tpi": ["tpi_Latn"],
    "tn": ["tsn_Latn"],
    "ts": ["tso_Latn"],
    "tk": ["tuk_Latn"],
    "tum": ["tum_Latn"],
    "tr": ["tur_Latn"],
    "tw": ["twi_Latn"],
    "tzm": ["tzm_Tfng"],
    "ug": ["uig_Arab"],
    "uk": ["ukr_Cyrl"],
    "umb": ["umb_Latn"],
    "ur": ["urd_Arab"],
    "uzn": ["uzn_Latn"],
    "vec": ["vec_Latn"],
    "vi": ["vie_Latn"],
    "war": ["war_Latn"],
    "wo": ["wol_Latn"],
    "xh": ["xho_Latn"],
    "ydd": ["ydd_Hebr"],
    "yo": ["yor_Latn"],
    "yue": ["yue_Hant"],
    "zh": ["zho_Hans", "zho_Hant"],
    "zu": ["zul_Latn"]
}

# List from https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.xml
NLTK_LANGUAGE_CODES = {
    "ces_Latn": "Czech",
    "dan_Latn": "Danish",
    "nld_Latn": "Dutch",
    "eng_Latn": "English",
    "est_Latn": "Estonian",
    "fin_Latn": "Finnish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ell_Grek": "Greek",
    "ita_Latn": "Italian",
    "mal_Mlym": "Malayalam",
    "nno_Latn": "Norwegian",
    "nob_Latn": "Norwegian",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "slv_Latn": "Slovene",
    "spa_Latn": "Spanish",
    "swe_Latn": "Swedish",
    "tur_Latn": "Turkish",
}

SUPPORTED_LANGUAGES = set()
for lang_codes in LANGUAGES_ISO1_TO_ISO3.values():
    SUPPORTED_LANGUAGES.update(lang_codes)

MODEL_LINKS = {
    "small": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/NLLB-200/CT2/small.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/NLLB-200/CT2/small.zip",
            "https://s3.libs.space:9000/ai-models/NLLB-200/CT2/small.zip",
        ],
        "checksum": "54188e59e5267329996f93a559befc0c14c09ef6a4f5f4e9645b0da94e380d47"
    },
    "medium": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/NLLB-200/CT2/medium.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/NLLB-200/CT2/medium.zip",
            "https://s3.libs.space:9000/ai-models/NLLB-200/CT2/medium.zip",
        ],
        "checksum": "88efd459f37d098bc44262721add08c57d22e482aab986edb4c7cbde5bd17cf9"
    },
    "large": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/NLLB-200/CT2/large.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/NLLB-200/CT2/large.zip",
            "https://s3.libs.space:9000/ai-models/NLLB-200/CT2/large.zip",
        ],
        "checksum": "c1f5618552cdfad2a5daf74e8218e5c583a6ee10acd3b8dc139ae2d94067af85"
    },
    "sentencepiece": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/NLLB-200/CT2/sentencepiece.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/NLLB-200/CT2/sentencepiece.zip",
            "https://s3.libs.space:9000/ai-models/NLLB-200/CT2/sentencepiece.zip",
        ],
        "checksum": "7e7fe41261d253ebba549de48b280021b1ae9d7915aa583689b34aa1f8604d13"
    }
}

# [Modify] Set paths to the models
ct_model_path = Path(Path.cwd() / ".cache" / "nllb200_ct2")
os.makedirs(ct_model_path, exist_ok=True)

sp_model_path = Path(ct_model_path / ct_model_path / "flores200_sacrebleu_tokenizer_spm.model")

model = None

sentencepiece = None

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_installed_language_names():
    return tuple([{"code": code, "name": language} for language, code in LANGUAGES.items()])


def set_device(device: str):
    global torch_device
    if device == "cuda" or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = device


def load_model(size="small", compute_type="float32"):
    global model
    global sentencepiece

    model_path = Path(ct_model_path / size)

    print(f"NLLB-200_CTranslate2 {size} is Loading to {torch_device} using {compute_type} precision...")

    pretrained_lang_model_file = Path(model_path / "model.bin")

    if not model_path.exists() or not pretrained_lang_model_file.is_file():
        print(f"Downloading {size} NLLB-200 model...")
        downloader.download_extract(MODEL_LINKS[size]["urls"], str(ct_model_path.resolve()), MODEL_LINKS[size]["checksum"], title="Text Translation (NLLB200CT2)")

    if not sp_model_path.is_file():
        print(f"Downloading sentencepiece model...")
        downloader.download_extract(MODEL_LINKS["sentencepiece"]["urls"], str(ct_model_path.resolve()), MODEL_LINKS["sentencepiece"]["checksum"], title="Text Translation (Sentencepiece)")

    sentencepiece = spm.SentencePieceProcessor()
    sentencepiece.load(str(sp_model_path.resolve()))

    # load nltk sentence splitting dependency
    nltk.download('punkt')

    # init NLLB 200 model
    model_path_string = str(model_path.resolve())
    model = ctranslate2.Translator(model_path_string, device=torch_device, compute_type=compute_type)

    print(f"NLLB-200_CTranslate2 model loaded.")


def translate_language(text, from_code, to_code, as_iso1=False):
    global model
    global sentencepiece

    if as_iso1 and from_code in LANGUAGES_ISO1_TO_ISO3:
        from_code = LANGUAGES_ISO1_TO_ISO3[from_code][0]
    if as_iso1 and to_code in LANGUAGES_ISO1_TO_ISO3:
        to_code = LANGUAGES_ISO1_TO_ISO3[to_code][0]

    if from_code == "auto":
        from_code = languageClassification.classify(text)

    language_unsupported = False
    if from_code not in SUPPORTED_LANGUAGES:
        print(f"error translating. {from_code} not supported.")
        language_unsupported = True
    if to_code not in SUPPORTED_LANGUAGES:
        print(f"error translating. {to_code} not supported.")
        language_unsupported = True
    if language_unsupported:
        return text, from_code, to_code

    if from_code == to_code:
        return text, from_code, to_code

    # Split the source text into sentences
    nltk_sentence_split_lang = "english"
    if from_code in NLTK_LANGUAGE_CODES:
        nltk_sentence_split_lang = NLTK_LANGUAGE_CODES[from_code]
    sentences = sent_tokenize(text, language=nltk_sentence_split_lang)
    translated_sentences = []

    for sentence in sentences:
        # Tokenize the source text
        source_text_subworded = sentencepiece.encode([sentence], out_type=str)[0] + ["</s>", from_code]

        # Add target language code as the target prefix
        source_sents_target_prefix = [[to_code]]

        translated_tokens = model.translate_batch([source_text_subworded], batch_type="tokens", max_batch_size=2024, target_prefix=source_sents_target_prefix, beam_size=4)
        translated_tokens = [translation[0]['tokens'] for translation in translated_tokens]
        for translation in translated_tokens:
            if to_code in translation:
                translation.remove(to_code)

        translated_sentence = sentencepiece.decode(translated_tokens[0])
        translated_sentences.append(translated_sentence)

    # Join the translated sentences into a single text
    translation = ' '.join(translated_sentences)

    return translation, from_code, to_code

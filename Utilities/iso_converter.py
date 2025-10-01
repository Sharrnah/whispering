from typing import Dict, List, Tuple, Union

class LanguageCodeConverter:
    """
    LanguageCodeConverter converts between ISO-639 (iso1/iso2/iso3) and NLLB language codes.
    - Case-insensitive for lookups.
    - Backed by a single data structure LANGUAGES (a list of dicts).

    Public methods:
      - any_to_code_type(code: str) -> Union[str, List[str]]
      - any_to_language(code: str) -> Dict[str, Union[str, List[str]]]
      - convert(code: str, target_type: str, *, return_all: bool = False) -> Union[str, List[str]]
      - has_language(code_or_name: str) -> bool
    """

    # --- Single authoritative data structure (extend as needed) ---
    LANGUAGES: List[Dict[str, Union[str, List[str], None]]] = [
        { "name": "Achinese (Arab)", "iso1": None, "iso2": "ace", "iso3": "ace", "nllb": ["ace_Arab"]},
        { "name": "Achinese (Latn)", "iso1": None, "iso2": "ace", "iso3": "ace", "nllb": ["ace_Latn"]},
        { "name": "Mesopotamian", "iso1": None, "iso2": None, "iso3": "acm", "nllb": ["acm_Arab"]},
        { "name": "Ta’izzi-Adeni", "iso1": None, "iso2": None, "iso3": "acq", "nllb": ["acq_Arab"]},
        { "name": "Tunisian", "iso1": None, "iso2": None, "iso3": "aeb", "nllb": ["aeb_Arab"]},
        { "name": "Afrikaans", "iso1": "af", "iso2": "afr", "iso3": "afr", "nllb": ["afr_Latn"]},
        { "name": "South Levantine", "iso1": None, "iso2": None, "iso3": "ajp", "nllb": ["ajp_Arab"]},
        { "name": "Akan", "iso1": "ak", "iso2": "aka", "iso3": "aka", "nllb": ["aka_Latn"]},
        { "name": "Amharic", "iso1": "am", "iso2": "amh", "iso3": "amh", "nllb": ["amh_Ethi"]},
        { "name": "North Levantine Arabic", "iso1": None, "iso2": None, "iso3": "apc", "nllb": ["apc_Arab"]},
        { "name": "Modern Standard Arabic", "iso1": "ar", "iso2": "ara", "iso3": "arb", "nllb": ["arb_Arab"]},
        { "name": "Najdi Arabic", "iso1": None, "iso2": None, "iso3": "ars", "nllb": ["ars_Arab"]},
        { "name": "Moroccan Arabic", "iso1": None, "iso2": None, "iso3": "ary", "nllb": ["ary_Arab"]},
        { "name": "Egyptian Arabic", "iso1": None, "iso2": None, "iso3": "arz", "nllb": ["arz_Arab"]},
        { "name": "Assamese", "iso1": "as", "iso2": "asm", "iso3": "asm", "nllb": ["asm_Beng"]},
        { "name": "Asturian", "iso1": None, "iso2": "ast", "iso3": "ast", "nllb": ["ast_Latn"]},
        { "name": "Awadhi", "iso1": None, "iso2": None, "iso3": "awa", "nllb": ["awa_Deva"]},
        { "name": "Aymara", "iso1": "ay", "iso2": "aym", "iso3": "aym", "nllb": ["ayr_Latn"]},
        { "name": "Azerbaijani (South)", "iso1": None, "iso2": None, "iso3": "azb", "nllb": ["azb_Arab"]},
        { "name": "Azerbaijani (North)", "iso1": "az", "iso2": "aze", "iso3": "azj", "nllb": ["azj_Latn"]},
        { "name": "Bashkir", "iso1": "ba", "iso2": "bak", "iso3": "bak", "nllb": ["bak_Cyrl"]},
        { "name": "Bambara", "iso1": "bm", "iso2": "bam", "iso3": "bam", "nllb": ["bam_Latn"]},
        { "name": "Balinese", "iso1": None, "iso2": None, "iso3": "ban", "nllb": ["ban_Latn"]},
        { "name": "Belarusian", "iso1": "be", "iso2": "bel", "iso3": "bel", "nllb": ["bel_Cyrl"]},
        { "name": "Bemba", "iso1": None, "iso2": None, "iso3": "bem", "nllb": ["bem_Latn"]},
        { "name": "Bengali", "iso1": "bn", "iso2": "ben", "iso3": "ben", "nllb": ["ben_Beng"]},
        { "name": "Bhojpuri", "iso1": None, "iso2": None, "iso3": "bho", "nllb": ["bho_Deva"]},
        { "name": "Banjarese (Arab)", "iso1": None, "iso2": None, "iso3": "bjn", "nllb": ["bjn_Arab"]},
        { "name": "Banjarese (Latn)", "iso1": None, "iso2": None, "iso3": "bjn", "nllb": ["bjn_Latn"]},
        { "name": "Tibetan", "iso1": "bo", "iso2": "bod", "iso3": "bod", "nllb": ["bod_Tibt"]},
        { "name": "Bosnian", "iso1": "bs", "iso2": "bos", "iso3": "bos", "nllb": ["bos_Latn"]},
        { "name": "Buginese", "iso1": None, "iso2": None, "iso3": "bug", "nllb": ["bug_Latn"]},
        { "name": "Bulgarian", "iso1": "bg", "iso2": "bul", "iso3": "bul", "nllb": ["bul_Cyrl"]},
        { "name": "Catalan", "iso1": "ca", "iso2": "cat", "iso3": "cat", "nllb": ["cat_Latn"]},
        { "name": "Cebuano", "iso1": None, "iso2": None, "iso3": "ceb", "nllb": ["ceb_Latn"]},
        { "name": "Czech", "iso1": "cs", "iso2": "ces", "iso3": "ces", "nllb": ["ces_Latn"]},
        { "name": "Chokwe", "iso1": None, "iso2": None, "iso3": "cjk", "nllb": ["cjk_Latn"]},
        { "name": "Sorani", "iso1": None, "iso2": None, "iso3": "ckb", "nllb": ["ckb_Arab"]},
        { "name": "Crimean", "iso1": None, "iso2": None, "iso3": "crh", "nllb": ["crh_Latn"]},
        { "name": "Welsh", "iso1": "cy", "iso2": "cym", "iso3": "cym", "nllb": ["cym_Latn"]},
        { "name": "Danish", "iso1": "da", "iso2": "dan", "iso3": "dan", "nllb": ["dan_Latn"]},
        { "name": "German", "iso1": "de", "iso2": "deu", "iso3": "deu", "nllb": ["deu_Latn"]},
        { "name": "Dinka", "iso1": None, "iso2": None, "iso3": "dik", "nllb": ["dik_Latn"]},
        { "name": "Dyula", "iso1": None, "iso2": "dyu", "iso3": "dyu", "nllb": ["dyu_Latn"]},
        { "name": "Dzongkha", "iso1": "dz", "iso2": "dzo", "iso3": "dzo", "nllb": ["dzo_Tibt"]},
        { "name": "Greek (Modern)", "iso1": "el", "iso2": "ell", "iso3": "ell", "nllb": ["ell_Grek"]},
        { "name": "English", "iso1": "en", "iso2": "eng", "iso3": "eng", "nllb": ["eng_Latn"]},
        { "name": "Esperanto", "iso1": "eo", "iso2": "epo", "iso3": "epo", "nllb": ["epo_Latn"]},
        { "name": "Estonian", "iso1": "et", "iso2": "est", "iso3": "est", "nllb": ["est_Latn"]},
        { "name": "Basque", "iso1": "eu", "iso2": "eus", "iso3": "eus", "nllb": ["eus_Latn"]},
        { "name": "Ewe", "iso1": "ee", "iso2": "ewe", "iso3": "ewe", "nllb": ["ewe_Latn"]},
        { "name": "Faroese", "iso1": "fo", "iso2": "fao", "iso3": "fao", "nllb": ["fao_Latn"]},
        { "name": "Persian", "iso1": "fa", "iso2": "fas", "iso3": "fas", "nllb": ["pes_Arab"]},
        { "name": "Fijian", "iso1": "fj", "iso2": "fij", "iso3": "fij", "nllb": ["fij_Latn"]},
        { "name": "Finnish", "iso1": "fi", "iso2": "fin", "iso3": "fin", "nllb": ["fin_Latn"]},
        { "name": "Fon", "iso1": None, "iso2": None, "iso3": "fon", "nllb": ["fon_Latn"]},
        { "name": "French", "iso1": "fr", "iso2": "fra", "iso3": "fra", "nllb": ["fra_Latn"]},
        { "name": "Friulian", "iso1": None, "iso2": None, "iso3": "fur", "nllb": ["fur_Latn"]},
        { "name": "Fula", "iso1": None, "iso2": None, "iso3": "fuv", "nllb": ["fuv_Latn"]},
        { "name": "Gaelic", "iso1": "gd", "iso2": "gla", "iso3": "gla", "nllb": ["gla_Latn"]},
        { "name": "Irish", "iso1": "ga", "iso2": "gle", "iso3": "gle", "nllb": ["gle_Latn"]},
        { "name": "Galician", "iso1": "gl", "iso2": "glg", "iso3": "glg", "nllb": ["glg_Latn"]},
        { "name": "Guarani", "iso1": "gn", "iso2": "grn", "iso3": "grn", "nllb": ["grn_Latn"]},
        { "name": "Gujarati", "iso1": "gu", "iso2": "guj", "iso3": "guj", "nllb": ["guj_Gujr"]},
        { "name": "Haitian", "iso1": "ht", "iso2": "hat", "iso3": "hat", "nllb": ["hat_Latn"]},
        { "name": "Hausa", "iso1": "ha", "iso2": "hau", "iso3": "hau", "nllb": ["hau_Latn"]},
        { "name": "Hebrew", "iso1": "he", "iso2": "heb", "iso3": "heb", "nllb": ["heb_Hebr"]},
        { "name": "Hindi", "iso1": "hi", "iso2": "hin", "iso3": "hin", "nllb": ["hin_Deva"]},
        { "name": "Chhattisgarhi", "iso1": None, "iso2": None, "iso3": "hne", "nllb": ["hne_Deva"]},
        { "name": "Croatian", "iso1": "hr", "iso2": "hrv", "iso3": "hrv", "nllb": ["hrv_Latn"]},
        { "name": "Hungarian", "iso1": "hu", "iso2": "hun", "iso3": "hun", "nllb": ["hun_Latn"]},
        { "name": "Armenian", "iso1": "hy", "iso2": "hye", "iso3": "hye", "nllb": ["hye_Armn"]},
        { "name": "Igbo", "iso1": "ig", "iso2": "ibo", "iso3": "ibo", "nllb": ["ibo_Latn"]},
        { "name": "Iloko", "iso1": None, "iso2": None, "iso3": "ilo", "nllb": ["ilo_Latn"]},
        { "name": "Indonesian", "iso1": "id", "iso2": "ind", "iso3": "ind", "nllb": ["ind_Latn"]},
        { "name": "Icelandic", "iso1": "is", "iso2": "isl", "iso3": "isl", "nllb": ["isl_Latn"]},
        { "name": "Italian", "iso1": "it", "iso2": "ita", "iso3": "ita", "nllb": ["ita_Latn"]},
        { "name": "Javanese", "iso1": "jv", "iso2": "jav", "iso3": "jav", "nllb": ["jav_Latn"]},
        { "name": "Japanese", "iso1": "ja", "iso2": "jpn", "iso3": "jpn", "nllb": ["jpn_Jpan"]},
        { "name": "Kabyle", "iso1": None, "iso2": None, "iso3": "kab", "nllb": ["kab_Latn"]},
        { "name": "Kachin", "iso1": None, "iso2": None, "iso3": "kac", "nllb": ["kac_Latn"]},
        { "name": "Kamba", "iso1": None, "iso2": None, "iso3": "kam", "nllb": ["kam_Latn"]},
        { "name": "Kannada", "iso1": "kn", "iso2": "kan", "iso3": "kan", "nllb": ["kan_Knda"]},
        { "name": "Kashmiri (Arab)", "iso1": "ks", "iso2": "kas", "iso3": "kas", "nllb": ["kas_Arab"]},
        { "name": "Kashmiri (Deva)", "iso1": "ks", "iso2": "kas", "iso3": "kas", "nllb": ["kas_Deva"]},
        { "name": "Georgian", "iso1": "ka", "iso2": "kat", "iso3": "kat", "nllb": ["kat_Geor"]},
        { "name": "Kanuri (Arab)", "iso1": None, "iso2": None, "iso3": "knc", "nllb": ["knc_Arab"]},
        { "name": "Kanuri (Latn)", "iso1": None, "iso2": None, "iso3": "knc", "nllb": ["knc_Latn"]},
        { "name": "Kazakh", "iso1": "kk", "iso2": "kaz", "iso3": "kaz", "nllb": ["kaz_Cyrl"]},
        { "name": "Kabiye", "iso1": None, "iso2": None, "iso3": "kbp", "nllb": ["kbp_Latn"]},
        { "name": "Kriolu", "iso1": None, "iso2": None, "iso3": "kea", "nllb": ["kea_Latn"]},
        { "name": "Central Khmer", "iso1": "km", "iso2": "khm", "iso3": "khm", "nllb": ["khm_Khmr"]},
        { "name": "Kikuyu", "iso1": "ki", "iso2": "kik", "iso3": "kik", "nllb": ["kik_Latn"]},
        { "name": "Kinyarwanda", "iso1": "rw", "iso2": "kin", "iso3": "kin", "nllb": ["kin_Latn"]},
        { "name": "Kirghiz", "iso1": "ky", "iso2": "kir", "iso3": "kir", "nllb": ["kir_Cyrl"]},
        { "name": "Kimbundu", "iso1": None, "iso2": None, "iso3": "kmb", "nllb": ["kmb_Latn"]},
        { "name": "Kongo", "iso1": "kg", "iso2": "kon", "iso3": "kon", "nllb": ["kon_Latn"]},
        { "name": "Korean", "iso1": "ko", "iso2": "kor", "iso3": "kor", "nllb": ["kor_Hang"]},
        { "name": "Kurmanji", "iso1": None, "iso2": None, "iso3": "kmr", "nllb": ["kmr_Latn"]},
        { "name": "Lao", "iso1": "lo", "iso2": "lao", "iso3": "lao", "nllb": ["lao_Laoo"]},
        { "name": "Latvian", "iso1": "lv", "iso2": "lav", "iso3": "lav", "nllb": ["lvs_Latn"]},
        { "name": "Ligurian", "iso1": None, "iso2": None, "iso3": "lij", "nllb": ["lij_Latn"]},
        { "name": "Limburgan", "iso1": "li", "iso2": "lim", "iso3": "lim", "nllb": ["lim_Latn"]},
        { "name": "Lingala", "iso1": "ln", "iso2": "lin", "iso3": "lin", "nllb": ["lin_Latn"]},
        { "name": "Lithuanian", "iso1": "lt", "iso2": "lit", "iso3": "lit", "nllb": ["lit_Latn"]},
        { "name": "Lombard", "iso1": None, "iso2": None, "iso3": "lmo", "nllb": ["lmo_Latn"]},
        { "name": "Latgalian", "iso1": None, "iso2": None, "iso3": "ltg", "nllb": ["ltg_Latn"]},
        { "name": "Luxembourgish", "iso1": "lb", "iso2": "ltz", "iso3": "ltz", "nllb": ["ltz_Latn"]},
        { "name": "Luba-Lulua", "iso1": None, "iso2": None, "iso3": "lua", "nllb": ["lua_Latn"]},
        { "name": "Ganda", "iso1": "lg", "iso2": "lug", "iso3": "lug", "nllb": ["lug_Latn"]},
        { "name": "Luo", "iso1": None, "iso2": None, "iso3": "luo", "nllb": ["luo_Latn"]},
        { "name": "Lushai", "iso1": None, "iso2": None, "iso3": "lus", "nllb": ["lus_Latn"]},
        { "name": "Magahi", "iso1": None, "iso2": None, "iso3": "mag", "nllb": ["mag_Deva"]},
        { "name": "Maithili", "iso1": None, "iso2": None, "iso3": "mai", "nllb": ["mai_Deva"]},
        { "name": "Malayalam", "iso1": "ml", "iso2": "mal", "iso3": "mal", "nllb": ["mal_Mlym"]},
        { "name": "Marathi", "iso1": "mr", "iso2": "mar", "iso3": "mar", "nllb": ["mar_Deva"]},
        { "name": "Minangkabau", "iso1": None, "iso2": None, "iso3": "min", "nllb": ["min_Latn"]},
        { "name": "Macedonian", "iso1": "mk", "iso2": "mkd", "iso3": "mkd", "nllb": ["mkd_Cyrl"]},
        { "name": "Malagasy", "iso1": "mg", "iso2": "mlg", "iso3": "mlg", "nllb": ["plt_Latn"]},
        { "name": "Maltese", "iso1": "mt", "iso2": "mlt", "iso3": "mlt", "nllb": ["mlt_Latn"]},
        { "name": "Manipuri", "iso1": None, "iso2": None, "iso3": "mni", "nllb": ["mni_Beng"]},
        { "name": "Mongolian", "iso1": "mn", "iso2": "mon", "iso3": "khk", "nllb": ["khk_Cyrl"]},
        { "name": "Mossi", "iso1": None, "iso2": None, "iso3": "mos", "nllb": ["mos_Latn"]},
        { "name": "Māori", "iso1": "mi", "iso2": "mri", "iso3": "mri", "nllb": ["mri_Latn"]},
        { "name": "Malaysian Malay", "iso1": "ms", "iso2": "msa", "iso3": "msa", "nllb": ["zsm_Latn"]},
        { "name": "Burmese", "iso1": "my", "iso2": "mya", "iso3": "mya", "nllb": ["mya_Mymr"]},
        { "name": "Dutch", "iso1": "nl", "iso2": "nld", "iso3": "nld", "nllb": ["nld_Latn"]},
        { "name": "Norwegian Nynorsk", "iso1": "nn", "iso2": "nno", "iso3": "nno", "nllb": ["nno_Latn"]},
        { "name": "Norwegian Bokmål", "iso1": "nb", "iso2": "nob", "iso3": "nob", "nllb": ["nob_Latn"]},
        { "name": "Nepali", "iso1": "ne", "iso2": "nep", "iso3": "nep", "nllb": ["npi_Deva"]},
        { "name": "Northern Sotho", "iso1": None, "iso2": None, "iso3": "nso", "nllb": ["nso_Latn"]},
        { "name": "Nuer", "iso1": None, "iso2": None, "iso3": "nus", "nllb": ["nus_Latn"]},
        { "name": "Chichewa, Nyanja", "iso1": "ny", "iso2": "nya", "iso3": "nya", "nllb": ["nya_Latn"]},
        { "name": "Occitan", "iso1": "oc", "iso2": "oci", "iso3": "oci", "nllb": ["oci_Latn"]},
        { "name": "Oromo", "iso1": "om", "iso2": "orm", "iso3": "orm", "nllb": ["gaz_Latn"]},
        { "name": "Odia", "iso1": "or", "iso2": "ori", "iso3": "ory", "nllb": ["ory_Orya"]},
        { "name": "Pangasinan", "iso1": None, "iso2": None, "iso3": "pag", "nllb": ["pag_Latn"]},
        { "name": "Panjabi", "iso1": "pa", "iso2": "pan", "iso3": "pan", "nllb": ["pan_Guru"]},
        { "name": "Papiamento", "iso1": None, "iso2": None, "iso3": "pap", "nllb": ["pap_Latn"]},
        { "name": "Polish", "iso1": "pl", "iso2": "pol", "iso3": "pol", "nllb": ["pol_Latn"]},
        { "name": "Portuguese", "iso1": "pt", "iso2": "por", "iso3": "por", "nllb": ["por_Latn"]},
        { "name": "Dari", "iso1": None, "iso2": None, "iso3": "prs", "nllb": ["prs_Arab"]},
        { "name": "Southern Pashto", "iso1": None, "iso2": None, "iso3": "pbt", "nllb": ["pbt_Arab"]},
        { "name": "Ayacucho Quechua", "iso1": None, "iso2": None, "iso3": "quy", "nllb": ["quy_Latn"]},
        { "name": "Romanian", "iso1": "ro", "iso2": "ron", "iso3": "ron", "nllb": ["ron_Latn"]},
        { "name": "Rundi", "iso1": "rn", "iso2": "run", "iso3": "run", "nllb": ["run_Latn"]},
        { "name": "Russian", "iso1": "ru", "iso2": "rus", "iso3": "rus", "nllb": ["rus_Cyrl"]},
        { "name": "Sango", "iso1": "sg", "iso2": "sag", "iso3": "sag", "nllb": ["sag_Latn"]},
        { "name": "Sanskrit", "iso1": "sa", "iso2": "san", "iso3": "san", "nllb": ["san_Deva"]},
        { "name": "Santali", "iso1": None, "iso2": None, "iso3": "sat", "nllb": ["sat_Beng"]},
        { "name": "Sicilian", "iso1": None, "iso2": None, "iso3": "scn", "nllb": ["scn_Latn"]},
        { "name": "Shan", "iso1": None, "iso2": None, "iso3": "shn", "nllb": ["shn_Mymr"]},
        { "name": "Sinhala", "iso1": "si", "iso2": "sin", "iso3": "sin", "nllb": ["sin_Sinh"]},
        { "name": "Slovak", "iso1": "sk", "iso2": "slk", "iso3": "slk", "nllb": ["slk_Latn"]},
        { "name": "Slovenian", "iso1": "sl", "iso2": "slv", "iso3": "slv", "nllb": ["slv_Latn"]},
        { "name": "Samoan", "iso1": "sm", "iso2": "smo", "iso3": "smo", "nllb": ["smo_Latn"]},
        { "name": "Shona", "iso1": "sn", "iso2": "sna", "iso3": "sna", "nllb": ["sna_Latn"]},
        { "name": "Sindhi", "iso1": "sd", "iso2": "snd", "iso3": "snd", "nllb": ["snd_Arab"]},
        { "name": "Somali", "iso1": "so", "iso2": "som", "iso3": "som", "nllb": ["som_Latn"]},
        { "name": "Sotho", "iso1": "st", "iso2": "sot", "iso3": "sot", "nllb": ["sot_Latn"]},
        { "name": "Spanish", "iso1": "es", "iso2": "spa", "iso3": "spa", "nllb": ["spa_Latn"]},
        { "name": "Tosk Albanian", "iso1": None, "iso2": None, "iso3": "als", "nllb": ["als_Latn"]},
        { "name": "Sardinian", "iso1": "sc", "iso2": "srd", "iso3": "srd", "nllb": ["srd_Latn"]},
        { "name": "Serbian", "iso1": "sr", "iso2": "srp", "iso3": "srp", "nllb": ["srp_Cyrl"]},
        { "name": "Swati", "iso1": "ss", "iso2": "ssw", "iso3": "ssw", "nllb": ["ssw_Latn"]},
        { "name": "Sundanese", "iso1": "su", "iso2": "sun", "iso3": "sun", "nllb": ["sun_Latn"]},
        { "name": "Swedish", "iso1": "sv", "iso2": "swe", "iso3": "swe", "nllb": ["swe_Latn"]},
        { "name": "Swahili", "iso1": "sw", "iso2": "swa", "iso3": "swa", "nllb": ["swh_Latn"]},
        { "name": "Silesian", "iso1": None, "iso2": None, "iso3": "szl", "nllb": ["szl_Latn"]},
        { "name": "Tamil", "iso1": "ta", "iso2": "tam", "iso3": "tam", "nllb": ["tam_Taml"]},
        { "name": "Tatar", "iso1": "tt", "iso2": "tat", "iso3": "tat", "nllb": ["tat_Cyrl"]},
        { "name": "Telugu", "iso1": "te", "iso2": "tel", "iso3": "tel", "nllb": ["tel_Telu"]},
        { "name": "Tajik", "iso1": "tg", "iso2": "tgk", "iso3": "tgk", "nllb": ["tgk_Cyrl"]},
        { "name": "Tagalog", "iso1": "tl", "iso2": "tgl", "iso3": "tgl", "nllb": ["tgl_Latn"]},
        { "name": "Thai", "iso1": "th", "iso2": "tha", "iso3": "tha", "nllb": ["tha_Thai"]},
        { "name": "Tigrinya", "iso1": "ti", "iso2": "tir", "iso3": "tir", "nllb": ["tir_Ethi"]},
        { "name": "Tamasheq (Latn)", "iso1": None, "iso2": None, "iso3": "taq", "nllb": ["taq_Latn"]},
        { "name": "Tamasheq", "iso1": None, "iso2": None, "iso3": "taq", "nllb": ["taq_Tfng"]},
        { "name": "Tok Pisin, Pidgin", "iso1": None, "iso2": None, "iso3": "tpi", "nllb": ["tpi_Latn"]},
        { "name": "Tswana", "iso1": "tn", "iso2": "tsn", "iso3": "tsn", "nllb": ["tsn_Latn"]},
        { "name": "Tsonga", "iso1": "ts", "iso2": "tso", "iso3": "tso", "nllb": ["tso_Latn"]},
        { "name": "Turkmen", "iso1": "tk", "iso2": "tuk", "iso3": "tuk", "nllb": ["tuk_Latn"]},
        { "name": "Tumbuka", "iso1": None, "iso2": None, "iso3": "tum", "nllb": ["tum_Latn"]},
        { "name": "Turkish", "iso1": "tr", "iso2": "tur", "iso3": "tur", "nllb": ["tur_Latn"]},
        { "name": "Twi", "iso1": None, "iso2": None, "iso3": "twi", "nllb": ["twi_Latn"]},
        { "name": "Atlasic", "iso1": None, "iso2": None, "iso3": "tzm", "nllb": ["tzm_Tfng"]},
        { "name": "Uighur", "iso1": "ug", "iso2": "uig", "iso3": "uig", "nllb": ["uig_Arab"]},
        { "name": "Ukrainian", "iso1": "uk", "iso2": "ukr", "iso3": "ukr", "nllb": ["ukr_Cyrl"]},
        { "name": "Umbundu", "iso1": None, "iso2": None, "iso3": "umb", "nllb": ["umb_Latn"]},
        { "name": "Urdu", "iso1": "ur", "iso2": "urd", "iso3": "urd", "nllb": ["urd_Arab"]},
        { "name": "Uzbek", "iso1": "uz", "iso2": "uzb", "iso3": "uzn", "nllb": ["uzn_Latn"]},
        { "name": "Venetian", "iso1": None, "iso2": None, "iso3": "vec", "nllb": ["vec_Latn"]},
        { "name": "Vietnamese", "iso1": "vi", "iso2": "vie", "iso3": "vie", "nllb": ["vie_Latn"]},
        { "name": "Waray", "iso1": None, "iso2": None, "iso3": "war", "nllb": ["war_Latn"]},
        { "name": "Wolof", "iso1": "wo", "iso2": "wol", "iso3": "wol", "nllb": ["wol_Latn"]},
        { "name": "Xhosa", "iso1": "xh", "iso2": "xho", "iso3": "xho", "nllb": ["xho_Latn"]},
        { "name": "Yiddish", "iso1": "yi", "iso2": "yid", "iso3": "yid", "nllb": ["ydd_Hebr"]},
        { "name": "Yoruba", "iso1": "yo", "iso2": "yor", "iso3": "yor", "nllb": ["yor_Latn"]},
        { "name": "Yue Chinese", "iso1": None, "iso2": None, "iso3": "yue", "nllb": ["yue_Hant"]},
        { "name": "Chinese (Simplified)", "iso1": "zh", "iso2": "zho", "iso3": "zho", "nllb": ["zho_Hans"]},
        { "name": "Chinese (Traditional)", "iso1": "zh", "iso2": "zho", "iso3": "zho", "nllb": ["zho_Hant"]},
        { "name": "Zulu", "iso1": "zu", "iso2": "zul", "iso3": "zul", "nllb": ["zul_Latn"]},

        # ===== A few extras to keep dual-script variants discoverable (consistent with earlier behavior) =====
        { "name": "Serbian (Latin)", "iso1": "sr", "iso2": "srp", "iso3": "srp", "nllb": ["srp_Latn"]},
    ]

    CODE_TYPES = ("iso1", "iso2", "iso3", "nllb")

    def __init__(self) -> None:
        # Build a reverse index: lowercased value -> list of (idx, type)
        self._index: Dict[str, List[Tuple[int, str]]] = {}
        for i, entry in enumerate(self.LANGUAGES):
            # index by name
            name_val = str(entry["name"])
            self._index.setdefault(name_val.lower(), []).append((i, "name"))
            # index by iso codes
            for t in ("iso1", "iso2", "iso3"):
                v = entry.get(t)
                if isinstance(v, str) and v:
                    self._index.setdefault(v.lower(), []).append((i, t))
            # index by nllb (list or str)
            nllb_values = entry.get("nllb", [])
            if isinstance(nllb_values, list):
                for code in nllb_values:
                    self._index.setdefault(str(code).lower(), []).append((i, "nllb"))
            elif isinstance(nllb_values, str):
                self._index.setdefault(nllb_values.lower(), []).append((i, "nllb"))

    def _lookup(self, code_or_name: str) -> List[Tuple[int, str]]:
        return self._index.get(code_or_name.strip().lower(), [])

    def any_to_code_type(self, code: str) -> Union[List[str]]:
        """
        Return the type(s) of the given code: 'iso1', 'iso2', 'iso3', 'nllb', or 'name'.
        return a list of unique types.
        """
        matches = self._lookup(code)
        if not matches:
            raise ValueError(f"Unknown code or language name: {code!r}")
        types: List[str] = []
        seen = set()
        for _, t in matches:
            if t not in seen:
                seen.add(t)
                types.append(t)
        return types

    def any_to_language(self, code_or_name: str) -> Dict[str, Union[str, List[str], None]]:
        """
        Return the canonical language dictionary for the given code or name.
        If the code maps to multiple languages, the first one is returned.
        """
        matches = self._lookup(code_or_name)
        if not matches:
            raise ValueError(f"Unknown code or language name: {code_or_name!r}")
        idx, _ = matches[0]
        return self.LANGUAGES[idx]

    def convert(self, code: str, target_type: str, *, return_all: bool = False) -> Union[str, List[str]]:
        """
        Convert any code or name to the specified target_type: 'iso1', 'iso2', 'iso3', 'nllb', or 'name'.
        - For 'nllb', a language may have multiple codes. By default returns the first; set return_all=True to get all.
        """
        target_type = target_type.strip().lower()
        if target_type not in ("iso1", "iso2", "iso3", "nllb", "name"):
            raise ValueError(f"Invalid target_type {target_type!r}. Expected one of 'iso1','iso2','iso3','nllb','name'.")

        lang = self.any_to_language(code)
        if target_type == "name":
            return str(lang["name"])
        if target_type == "nllb":
            values = lang.get("nllb", [])
            if not values:
                raise ValueError(f"No NLLB code available for {lang['name']}")
            return list(values) if return_all else values[0]
        value = lang.get(target_type)
        if not value:
            raise ValueError(f"No {target_type.upper()} code available for {lang['name']}")
        return str(value)

    def has_language(self, code_or_name: str) -> bool:
        return bool(self._lookup(code_or_name))

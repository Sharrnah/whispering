"""German text normalization and G2P for Kokoro.

Adapted from ``semidark/misaki`` revision
``6d252a2e02f3b030f22f56686f1a73786c16ffc8`` under Apache-2.0.
Changes: the pronunciation lexicon is embedded for standalone packaging and
Misaki's version-dependent EspeakG2P return shape is normalized internally.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional, Tuple
import re
import unicodedata


_ONES = [
    "", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht",
    "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn",
    "sechzehn", "siebzehn", "achtzehn", "neunzehn",
]
_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
_ORD_IRREG = {1: "erst", 2: "zweit", 3: "dritt", 7: "siebt", 8: "acht"}
_MONTHS = [
    "", "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
    "August", "September", "Oktober", "November", "Dezember",
]
_CURRENCY = {"€": "Euro", "$": "Dollar", "£": "Pfund", "¥": "Yen"}


def _int_to_de(number: int, standalone: bool = True) -> str:
    if number < 0:
        return "minus " + _int_to_de(-number)
    if number == 0:
        return "null"
    if number == 1:
        return "eins" if standalone else "ein"
    if number < 20:
        return _ONES[number]
    if number < 100:
        ones, tens = number % 10, number // 10
        return (_ONES[ones] + "und" if ones else "") + _TENS[tens]
    if number < 1_000:
        hundreds, remainder = divmod(number, 100)
        return _ONES[hundreds] + "hundert" + (
            _int_to_de(remainder, standalone=False) if remainder else ""
        )
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1_000)
        prefix = _int_to_de(thousands, standalone=False) if thousands != 1 else "ein"
        return prefix + "tausend" + (
            _int_to_de(remainder, standalone=False) if remainder else ""
        )
    if number < 1_000_000_000:
        millions, remainder = divmod(number, 1_000_000)
        word = "eine Million" if millions == 1 else _int_to_de(millions, standalone=False) + " Millionen"
        return word + (" " + _int_to_de(remainder, standalone=False) if remainder else "")
    billions, remainder = divmod(number, 1_000_000_000)
    word = "eine Milliarde" if billions == 1 else _int_to_de(billions, standalone=False) + " Milliarden"
    return word + (" " + _int_to_de(remainder, standalone=False) if remainder else "")


def _ordinal_stem_de(number: int) -> str:
    if number in _ORD_IRREG:
        return _ORD_IRREG[number]
    return _int_to_de(number, standalone=False) + ("t" if number < 20 else "st")


def _year_de(number: int) -> str:
    if 1100 <= number <= 1999:
        century, remainder = divmod(number, 100)
        return _int_to_de(century, standalone=False) + "hundert" + (
            _int_to_de(remainder, standalone=False) if remainder else ""
        )
    return _int_to_de(number)


def _currency_replacement(symbol: str, number: str) -> str:
    cleaned = number.replace(".", "").replace(",", ".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return symbol + number
    cents_total = int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    units, cents = divmod(cents_total, 100)
    rendered = _int_to_de(units) + " " + _CURRENCY.get(symbol, symbol)
    if cents:
        rendered += " und " + _int_to_de(cents) + " Cent"
    return rendered


def normalize_text_de(text: str) -> str:
    """Expand German numbers, dates, times, currency, and abbreviations."""
    if not text:
        return text

    text = text.replace("„", '"').replace("“", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("‹", '"').replace("›", '"')
    text = re.sub(r"[^\S \n]", " ", text)

    replacements = (
        (r"\bDr\.(?=\s)", "Doktor", 0),
        (r"\bProf\.(?=\s)", "Professor", 0),
        (r"\bHr\.(?=\s)", "Herr ", 0),
        (r"\bFr\.(?=\s[A-ZÄÖÜ])", "Frau", 0),
        (r"\bDipl\.\s*-?\s*Ing\.", "Diplom-Ingenieur", 0),
        (r"\bStr\.(?=\s)", "Straße", 0),
        (r"\bNr\.(?=\s*\d)", "Nummer", 0),
        (r"\bTel\.(?=\s)", "Telefon", 0),
        (r"\bAbt\.(?=\s)", "Abteilung", 0),
        (r"\bGmbH\b", "Gesellschaft mit beschränkter Haftung", 0),
        (r"\bAG\b(?=[\s,.]|$)", "Aktiengesellschaft", 0),
        (r"\bz\.\s*B\.", "zum Beispiel", re.IGNORECASE),
        (r"\bd\.\s*h\.", "das heißt", re.IGNORECASE),
        (r"\bu\.\s*a\.", "unter anderem", re.IGNORECASE),
        (r"\bbzw\.", "beziehungsweise", re.IGNORECASE),
        (r"\busw\.", "und so weiter", re.IGNORECASE),
        (r"\betc\.", "et cetera", re.IGNORECASE),
        (r"\bca\.", "circa", re.IGNORECASE),
        (r"\bvgl\.", "vergleiche", re.IGNORECASE),
        (r"\binkl\.", "inklusive", re.IGNORECASE),
        (r"\bexkl\.", "exklusive", re.IGNORECASE),
        (r"\bggf\.", "gegebenenfalls", re.IGNORECASE),
        (r"\bi\.\s*d\.\s*R\.", "in der Regel", re.IGNORECASE),
        (r"\bo\.\s*ä\.", "oder ähnliches", re.IGNORECASE),
        (r"\bu\.\s*U\.", "unter Umständen", re.IGNORECASE),
    )
    for pattern, replacement, flags in replacements:
        text = re.sub(pattern, replacement, text, flags=flags)

    for abbreviation, full_name in (
        ("Jan", "Januar"), ("Feb", "Februar"), ("Mär", "März"),
        ("Apr", "April"), ("Jun", "Juni"), ("Jul", "Juli"),
        ("Aug", "August"), ("Sep", "September"), ("Okt", "Oktober"),
        ("Nov", "November"), ("Dez", "Dezember"),
    ):
        text = re.sub(rf"\b{abbreviation}\.(?=\s)", full_name, text)

    currency_symbol = r"[€$£¥]"
    text = re.sub(
        rf"({currency_symbol})\s*(\d[\d.,]*)",
        lambda match: _currency_replacement(match.group(1), match.group(2)),
        text,
    )
    text = re.sub(
        rf"(\d[\d.,]*)\s*({currency_symbol})",
        lambda match: _currency_replacement(match.group(2), match.group(1)),
        text,
    )

    def replace_time(match):
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            return match.group(0)
        return _int_to_de(hour) + " Uhr" + (" " + _int_to_de(minute) if minute else "")

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b(?:\s*Uhr\b)?", replace_time, text)

    def replace_date(match):
        day, month, year = map(int, match.groups())
        if not 1 <= day <= 31 or not 1 <= month <= 12:
            return match.group(0)
        return _ordinal_stem_de(day) + "e " + _MONTHS[month] + " " + _year_de(year)

    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", replace_date, text)
    text = re.sub(
        r"(?<!\d)(\d{1,2})\.\s",
        lambda match: _ordinal_stem_de(int(match.group(1))) + "e ",
        text,
    )
    text = re.sub(
        r"\b(\d{4})\b",
        lambda match: _year_de(int(match.group(1)))
        if 1100 <= int(match.group(1)) <= 2099
        else _int_to_de(int(match.group(1))),
        text,
    )

    def replace_grouped_number(match):
        cleaned = match.group(0).replace(".", "").replace(",", ".")
        try:
            value = float(cleaned)
        except ValueError:
            return match.group(0)
        if value == int(value):
            return _int_to_de(int(value))
        integer, fraction = cleaned.split(".")
        return _int_to_de(int(integer)) + " Komma " + " ".join(
            _int_to_de(int(digit)) for digit in fraction
        )

    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b", replace_grouped_number, text)
    text = re.sub(
        r"\b(\d+),(\d+)\b",
        lambda match: _int_to_de(int(match.group(1))) + " Komma " + " ".join(
            _int_to_de(int(digit)) for digit in match.group(2)
        ),
        text,
    )

    remaining_time = re.compile(r"\b\d{1,2}:\d{2}\b(?:\s*Uhr\b)?")

    def replace_integer(match):
        start = max(0, match.start() - 3)
        end = min(len(text), match.end() + len(":00 Uhr"))
        for time_match in remaining_time.finditer(text, start, end):
            if time_match.start() <= match.start() and match.end() <= time_match.end():
                return match.group(0)
        return _int_to_de(int(match.group(1)))

    text = re.sub(r"\b(\d+)\b", replace_integer, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_OVERRIDE_DATA = {
    "brand": {
        "bark": "bˈaːɐk", "claude": "klˈoːt", "coqui": "kˈoːki",
        "cuda": "kˈuːda", "deepseek": "dˈiːpsiːk", "espeak-ng": "ˈiːspiːk ɛndʒiː",
        "fastpitch": "fˈaːstpɪtʃ", "geforce": "dʒiːfˈɔɐs", "gemini": "dʒˈɛmɪnaɪ",
        "hifigan": "haɪfˈaɪɡæn", "intellij": "ˈɪntɛlaɪdʒ", "kikiri": "kɪkˈiːʁi",
        "kokoro": "kˈoːkoːʁoː", "llama": "lˈaːma", "mistral": "mˈɪstʁal",
        "mixtral": "mˈɪkstʁal", "neovim": "nˈiːovɪm", "phonemizer": "foːnəmˈaɪzɐ",
        "piper": "pˈaɪpɐ", "pycharm": "pˈaɪtʃaːɐm", "qwen": "kwˈɛn",
        "radeon": "ɹˈeɪdɪɔn", "ryzen": "ɹˈaɪzən", "tacotron": "tˈækotʁɔn",
        "tacotron2": "tˈækotʁɔn tuː", "triton": "tʁˈaɪtɔn", "typescript": "tˈaɪpskʁɪpt",
        "unsloth": "ˈʌnslɔːθ", "vits": "vˈɪts", "vscode": "viːˈɛs koːt",
    },
    "en": {
        "accelerate": "ɐksˈɛlɚɹˌAt", "amd": "ˌAˌɛmdˈiː", "apache": "ɐpˈæʧi",
        "api": "eɪpiːˈaɪ", "bert": "bˈɜːt", "checkpoint": "tʃˈɛkpɔɪnt",
        "cli": "tseːɛlˈaɪ", "debian": "dˈɛbiən", "disneyplus": "dˈɪzni plˈʌs",
        "dropout": "dɹˈɑːpWt", "fallback": "fˈɔːlbæk", "finetuning": "fˈIn tˈuːnɪŋ",
        "gan": "ɡˈæn", "github": "ɡˈɪthab", "githubactions": "ɡˈɪt hˈʌb ˈɛkʃəns",
        "gpu": "dʒiːpiːjˈuː", "https": "ˌAʧtˌiːtˈiːpˌiːˈɛs", "huggingface": "hˈaɡɪŋfeɪs",
        "ipad": "ˈI pˈæd", "jameswebb": "ʤˈAmz wˈɛb", "json": "dʒˈeɪsən",
        "kde": "kˌAdˌiːˈiː", "louisvuitton": "lwˈi vyitˈɔ̃", "macos": "mˈɛk oː ˈɛs",
        "moetchandon": "mɔˈɛ ʃɑ̃dˈɔ̃", "nvidia": "ɛnˈviːdiːa", "ollama": "olˈaːma",
        "pipeline": "pˈaɪplaɪn", "primevideo": "pɹˈIm vˈɪdɪO", "protocol": "pʁotokˈɔl",
        "pytorch": "pˈaɪtɔːɹtʃ", "rag": "ɹˈæɡ", "repository": "ɹᵻpˈɑːzɪtˌɔːɹi",
        "review": "ɹᵻvjˈuː", "rnn": "ˌɑːɹɹˌɛnˈɛn", "runtime": "ˈɹantaɪm",
        "styletts": "stˈaɪl tiːtiːˈɛs", "styletts2": "stˈaɪl tiːtiːˈɛs tsvai",
        "surface": "sˈɜːfɪs", "tcp": "tˌiːsˌiːpˈiː", "thread": "θɹˈɛd",
        "tpu": "tˌiːpˌiːjˈuː", "transformers": "tɹænsfˈɔːɹmɚz", "ubuntu": "uːbˈuːntuː",
        "ui": "juːˈaɪ", "wavlm": "wˈɛɪv ɛlˈɛm", "wsl": "dˌʌbəljˌuːˌɛsˈɛl",
        "zero-shot": "zˈiːɹo ʃˈɔt",
    },
    "de_foreign": {
        "diathese": "diaˈteːzə", "ekstase": "ɛkstˈaːzə", "epiklese": "epiˈkleːzə",
        "epithese": "epiˈteːzə", "glucose": "ɡlukˈoːzə", "hypnose": "hˈyːpnoːzə",
        "metamorphose": "metamɔʁfˈoːzə", "oase": "oˈaːzə", "prosthese": "pʁɔstˈeːzə",
        "prothese": "pʁotˈeːzə", "symbiose": "zymbɪˈoːzə", "synthese": "zyntˈeːzə",
    },
    "aliases": {"moetandchandon": "moetchandon"},
}

_LOOKUP_REPLACEMENTS = {"+": "plus", "&": "and", "@": "at"}
_TRAILING_PUNCTUATION = frozenset(".,!?;:%)]}»”")
_OVERRIDE_WORD = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿß]+(?:['\-][0-9A-Za-zÀ-ÖØ-öø-ÿß]+)*\+?")


def normalize_for_lookup(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    parts = []
    for character in text:
        if unicodedata.category(character) == "Mn":
            continue
        replacement = _LOOKUP_REPLACEMENTS.get(character)
        if replacement is not None:
            parts.append(replacement)
        elif character.isalnum():
            parts.append(character)
    return "".join(parts)


_OVERRIDES = {}
for _section in ("brand", "en", "de_foreign"):
    for _word, _phonemes in _OVERRIDE_DATA[_section].items():
        _OVERRIDES.setdefault(normalize_for_lookup(_word), _phonemes)
_OVERRIDE_ALIASES = {
    normalize_for_lookup(key): normalize_for_lookup(value)
    for key, value in _OVERRIDE_DATA["aliases"].items()
}


def override_for(word: str) -> Optional[str]:
    key = normalize_for_lookup(word)
    if not key:
        return None
    return _OVERRIDES.get(_OVERRIDE_ALIASES.get(key, key))


class GermanG2P:
    """Normalize German text, then phonemize it with the bundled eSpeak path."""

    def __init__(self):
        from misaki.espeak import EspeakG2P

        self.espeak = EspeakG2P(language="de")

    def _espeak_phonemes(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        result = self.espeak(text)
        phonemes = result[0] if isinstance(result, tuple) else result
        return phonemes or ""

    @staticmethod
    def _render(parts) -> str:
        rendered = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if not rendered:
                rendered = part
            elif part[0] in _TRAILING_PUNCTUATION:
                rendered += part
            else:
                rendered += " " + part
        return rendered

    def __call__(self, text: str) -> Tuple[str, None]:
        text = normalize_text_de(text)
        parts = []
        cursor = 0
        for match in _OVERRIDE_WORD.finditer(text):
            phonemes = override_for(match.group(0))
            if phonemes is None:
                continue
            preceding = text[cursor:match.start()]
            if preceding.strip():
                parts.append(self._espeak_phonemes(preceding))
            parts.append(phonemes)
            cursor = match.end()

        if cursor == 0:
            result = self.espeak(text)
            if isinstance(result, tuple):
                return result
            return result, None

        trailing = text[cursor:]
        if trailing.strip():
            parts.append(self._espeak_phonemes(trailing))
        return self._render(parts), None

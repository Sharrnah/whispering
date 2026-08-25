"""German text normalization for Whispering Tiger's IndexTTS fine-tune.

The exported checkpoint was trained with the training project's ``german_v2``
frontend. Keep this implementation in sync with that frontend so synthesis
receives the same token distribution as training.
"""

import re
import unicodedata


_DE_SMALL = {
    0: "null", 1: "eins", 2: "zwei", 3: "drei", 4: "vier",
    5: "fünf", 6: "sechs", 7: "sieben", 8: "acht", 9: "neun",
    10: "zehn", 11: "elf", 12: "zwölf", 13: "dreizehn",
    14: "vierzehn", 15: "fünfzehn", 16: "sechzehn",
    17: "siebzehn", 18: "achtzehn", 19: "neunzehn",
}

_DE_TENS = {
    20: "zwanzig", 30: "dreißig", 40: "vierzig", 50: "fünfzig",
    60: "sechzig", 70: "siebzig", 80: "achtzig", 90: "neunzig",
}

_DE_MONTHS = {
    1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai",
    6: "Juni", 7: "Juli", 8: "August", 9: "September",
    10: "Oktober", 11: "November", 12: "Dezember",
}

_DE_ABBREV = {
    "z. B.": "zum Beispiel",
    "z.B.": "zum Beispiel",
    "bzw.": "beziehungsweise",
    "usw.": "und so weiter",
    "u. a.": "unter anderem",
    "u.a.": "unter anderem",
    "d. h.": "das heißt",
    "d.h.": "das heißt",
    "ggf.": "gegebenenfalls",
    "ca.": "circa",
    "inkl.": "einschließlich",
    "Nr.": "Nummer",
    "Dr.": "Doktor",
    "Prof.": "Professor",
}

_DE_PRONUNCIATION_HELPERS = {
    "VRAM": "vie RAM",
    "C#": "Cie-Sharp",
    "C++": "C plus plus",
    "AMD": "A-Em-D",
    "NVIDIA": "En-Vidia",
    "ARD": "A-Er-D",
    "ZDF": "Zet-de ef",
    "WDR": "We-De-Er",
}

_DE_UNITS = {
    "km/h": "Kilometer pro Stunde",
    "km²": "Quadratkilometer", "km2": "Quadratkilometer",
    "m²": "Quadratmeter", "m2": "Quadratmeter",
    "m³": "Kubikmeter", "m3": "Kubikmeter",
    "km": "Kilometer", "cm": "Zentimeter", "mm": "Millimeter",
    "kg": "Kilogramm", "mg": "Milligramm", "g": "Gramm",
    "ml": "Milliliter", "l": "Liter", "Hz": "Hertz",
    "kHz": "Kilohertz", "MHz": "Megahertz", "GHz": "Gigahertz",
    "KB": "Kilobyte", "MB": "Megabyte", "GB": "Gigabyte", "TB": "Terabyte",
    "°C": "Grad Celsius", "°F": "Grad Fahrenheit",
}

_DE_CURRENCY = {
    "€": "Euro", "EUR": "Euro", "Euro": "Euro",
    "$": "Dollar", "USD": "Dollar",
    "£": "Pfund", "GBP": "Pfund",
}

_DE_FILLERS = {
    "呃": "äh", "嗯": "hm", "噢": "oh", "哦": "oh",
    "诶": "äh", "欸": "äh", "啊": "ah",
}


def _de_under_hundred(n: int) -> str:
    if n < 20:
        return _DE_SMALL[n]
    tens, ones = divmod(n, 10)
    tens_word = _DE_TENS[tens * 10]
    if not ones:
        return tens_word
    ones_word = "ein" if ones == 1 else _DE_SMALL[ones]
    return ones_word + "und" + tens_word


def _de_under_thousand(n: int) -> str:
    if n < 100:
        return _de_under_hundred(n)
    hundreds, rest = divmod(n, 100)
    prefix = "einhundert" if hundreds == 1 else _DE_SMALL[hundreds] + "hundert"
    return prefix + (_de_under_hundred(rest) if rest else "")


def de_number_to_words(n: int) -> str:
    """Convert an integer to standard written German without dependencies."""
    if n < 0:
        return "minus " + de_number_to_words(-n)
    if n < 1000:
        return _de_under_thousand(n)
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        prefix = "eintausend" if thousands == 1 else de_number_to_words(thousands) + "tausend"
        return prefix + (de_number_to_words(rest) if rest else "")

    scales = (
        (1_000_000_000_000_000, "Billiarde", "Billiarden"),
        (1_000_000_000_000, "Billion", "Billionen"),
        (1_000_000_000, "Milliarde", "Milliarden"),
        (1_000_000, "Million", "Millionen"),
    )
    for value, singular, plural in scales:
        if n >= value:
            count, rest = divmod(n, value)
            prefix = "eine " + singular if count == 1 else de_number_to_words(count) + " " + plural
            return prefix + ((" " + de_number_to_words(rest)) if rest else "")

    return " ".join(_DE_SMALL[int(d)] for d in str(n))


def de_ordinal_to_words(n: int) -> str:
    special = {1: "erste", 3: "dritte", 7: "siebte", 8: "achte"}
    if n in special:
        return special[n]
    suffix = "te" if n < 20 else "ste"
    return de_number_to_words(n) + suffix


def _de_digit_string(digits: str) -> str:
    return " ".join(_DE_SMALL[int(d)] for d in digits)


def _de_num_token(raw: str) -> str:
    raw = raw.replace(".", "")
    if "," in raw:
        whole, frac = raw.split(",", 1)
        return de_number_to_words(int(whole)) + " Komma " + _de_digit_string(frac)
    return de_number_to_words(int(raw))


def _de_year_to_words(raw: str) -> str:
    year = int(raw)
    if 1100 <= year <= 1999:
        century, rest = divmod(year, 100)
        return de_number_to_words(century) + "hundert" + (de_number_to_words(rest) if rest else "")
    return de_number_to_words(year)


def _normalize_german(
    text: str,
    *,
    named_dates: bool = False,
    consume_colon_uhr: bool = False,
) -> str:
    """Expand common German numeric forms and remove non-German ASR artefacts."""
    text = unicodedata.normalize("NFC", text)

    for source, replacement in _DE_FILLERS.items():
        text = text.replace(source, " " + replacement + " ")
    text = re.sub(r"[\u3400-\u9fff]+", " ", text)

    for abbr, full in sorted(_DE_ABBREV.items(), key=lambda item: -len(item[0])):
        text = re.sub(r"(?<!\w)" + re.escape(abbr), full, text, flags=re.IGNORECASE)

    for word, pronunciation in sorted(
        _DE_PRONUNCIATION_HELPERS.items(), key=lambda item: -len(item[0])
    ):
        text = re.sub(
            r"(?<!\w)" + re.escape(word) + r"(?!\w)",
            pronunciation,
            text,
            flags=re.IGNORECASE,
        )

    def replace_date(match: re.Match[str]) -> str:
        day, month, year = (int(match.group(i)) for i in range(1, 4))
        if not (1 <= day <= 31 and month in _DE_MONTHS):
            return match.group(0)
        return de_ordinal_to_words(day) + "n " + _DE_MONTHS[month] + " " + _de_year_to_words(str(year))

    text = re.sub(r"\b([0-3]?\d)\.([01]?\d)\.(\d{4})\b", replace_date, text)

    if named_dates:
        # 24. August 2026 -> vierundzwanzigsten August ...
        month_names = "|".join(re.escape(name) for name in _DE_MONTHS.values())
        text = re.sub(
            r"\b([0-3]?\d)\.\s+(" + month_names + r")\s+(\d{4})\b",
            lambda m: de_ordinal_to_words(int(m.group(1))) + "n " + m.group(2)
            + " " + _de_year_to_words(m.group(3)),
            text,
            flags=re.IGNORECASE,
        )

    colon_time = (
        r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*Uhr)?\b"
        if consume_colon_uhr
        else r"\b([01]?\d|2[0-3]):([0-5]\d)\b"
    )
    text = re.sub(
        colon_time,
        lambda m: de_number_to_words(int(m.group(1))) + " Uhr " + de_number_to_words(int(m.group(2))),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([01]?\d|2[0-3])\.([0-5]\d)\s*Uhr\b",
        lambda m: de_number_to_words(int(m.group(1))) + " Uhr " + de_number_to_words(int(m.group(2))),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"%\s*(\d[\d.,]*)", lambda m: _de_num_token(m.group(1)) + " Prozent", text)
    text = re.sub(r"(\d[\d.,]*)\s*%", lambda m: _de_num_token(m.group(1)) + " Prozent", text)

    for symbol, word in sorted(_DE_CURRENCY.items(), key=lambda item: -len(item[0])):
        escaped = re.escape(symbol)
        text = re.sub(
            r"(\d[\d.]*)[,]([0-9]{2})\s*" + escaped + r"(?!\w)",
            lambda m, w=word: de_number_to_words(int(m.group(1).replace(".", "")))
            + " " + w + " " + de_number_to_words(int(m.group(2))) + " Cent",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            escaped + r"\s*(\d[\d.,]*)",
            lambda m, w=word: _de_num_token(m.group(1)) + " " + w,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(\d[\d.,]*)\s*" + escaped + r"(?!\w)",
            lambda m, w=word: _de_num_token(m.group(1)) + " " + w,
            text,
            flags=re.IGNORECASE,
        )

    for unit, word in sorted(_DE_UNITS.items(), key=lambda item: -len(item[0])):
        text = re.sub(
            r"\b(\d[\d.,]*)\s*" + re.escape(unit) + r"(?!\w)",
            lambda m, w=word: _de_num_token(m.group(1)) + " " + w,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", lambda m: m.group(0).replace(".", ""), text)
    text = re.sub(
        r"\b(\d+),(\d+)\b",
        lambda m: de_number_to_words(int(m.group(1))) + " Komma " + _de_digit_string(m.group(2)),
        text,
    )
    text = re.sub(r"\b(?:1[1-9]\d{2}|20\d{2})\b", lambda m: _de_year_to_words(m.group(0)), text)
    text = re.sub(r"\b\d+\b", lambda m: de_number_to_words(int(m.group(0))), text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_german(text: str) -> str:
    """Legacy German frontend retained for compatibility and comparisons."""
    return _normalize_german(text)


def normalize_german_v2(text: str) -> str:
    """Frontend used to train the current German checkpoint."""
    return _normalize_german(text, named_dates=True, consume_colon_uhr=True)

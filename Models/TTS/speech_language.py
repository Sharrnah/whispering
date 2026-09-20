"""Language metadata for one utterance, independent of mutable TTS settings."""

from Utilities.iso_converter import LanguageCodeConverter

_CONVERTER = LanguageCodeConverter()
_AUTO = {"", "auto", "none", "null", "und"}


def language_code(value):
    value = str(value or "").strip().lower()
    if value in _AUTO:
        return ""
    # Convert NLLB/ISO-3/names before treating underscores as locale separators.
    try:
        converted = _CONVERTER.convert(value, "iso1")
    except ValueError:
        converted = ""
    return str(converted or value).lower().replace("_", "-")


def spoken_language(result, settings):
    """Return the language of the outgoing text, including translated output."""
    if result.get("txt_translation"):
        return language_code(str(result.get("txt_translation_target") or
                                 settings.GetOption("trg_lang") or "").split("|", 1)[0])
    if settings.GetOption("whisper_task") == "translate":
        return "en"
    return (language_code(result.get("language")) or
            language_code(settings.GetOption("current_language")))


def supported_language(value, supported):
    code = language_code(value)
    if not code:
        return ""
    if supported is None:
        return code
    if code in supported:
        return code
    base = code.split("-", 1)[0]
    if base in supported:
        return base
    for alias in {"en": ("en-us",), "pt": ("pt-br",), "ar": ("ar-msa",),
                  "fr": ("fr-fr",)}.get(base, ()):
        if alias in supported:
            return alias
    return ""

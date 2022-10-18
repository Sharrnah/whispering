from pathlib import Path
from typing import List
import argostranslate.package
import argostranslate.translate
import argostranslate.settings
import pykakasi
#import torch

argostranslate.settings.home_dir = Path(Path.cwd() / ".cache" / "argos-translate")
argostranslate.settings.cache_dir = argostranslate.settings.home_dir
argostranslate.settings.downloads_dir = Path(argostranslate.settings.home_dir / "downloads")
argostranslate.settings.data_dir = Path(argostranslate.settings.home_dir / "data")
argostranslate.settings.package_data_dir = Path(argostranslate.settings.home_dir / "packages")
argostranslate.settings.package_dirs = [argostranslate.settings.package_data_dir]
argostranslate.settings.local_package_index = argostranslate.settings.cache_dir / "index.json"
#argostranslate.settings.device = "cuda" if torch.cuda.is_available() else "cpu"

TRANSLATE_SETTINGS = {
    "txt_translate": False,
    "src_lang": "en",
    "trg_lang": "fr",
    "txt_ascii": False
}

LANGUAGES = list()

argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
# Download and install Argos Translate package
def InstallLanguages():
    for translationPackage in available_packages:
        print("Downloading translation: " + translationPackage.get_description())
        argostranslate.package.install_from_path(translationPackage.download())
    
    print("Download of " + str(list.count(available_packages)) + " translations finished.")

def LoadLanguages():
    LANGUAGES = argostranslate.translate.get_installed_languages()
    return LANGUAGES

def GetInstalledLanguageNames():
    return tuple([{"code": language.code, "name": language.name} for language in LoadLanguages()])

def GetLanguageCodeFromName(language_name):
    for language in LANGUAGES:
        if language.name == language_name:
            return language.code

def GetLanguageNameFromCode(language_code):
    for language in LANGUAGES:
        print(language)
        if language.code == language_code:
            return language.name

def TranslateLanguage(text, from_code, to_code, to_romaji = False):
    installed_languages = LoadLanguages()
    from_lang = list(filter(
        lambda x: x.code == from_code,
        installed_languages))[0]
    to_lang = list(filter(
        lambda x: x.code == to_code,
        installed_languages))[0]
    translation = from_lang.get_translation(to_lang)
    translationText = translation.translate(text)

    # Convert Hiragana, Katakana, Japanese to romaji (ascii compatible)
    if to_romaji:
        kks = pykakasi.kakasi()
        convertedText = kks.convert(translationText)
        fullConvertedText = []
        for convertedTextItem in convertedText:
            fullConvertedText.append(convertedTextItem['hepburn'])

        translationText = ' '.join(fullConvertedText)
    
    return translationText

def SetOption(setting, value):
    TRANSLATE_SETTINGS[setting] = value

def GetOption(setting):
    return TRANSLATE_SETTINGS[setting]

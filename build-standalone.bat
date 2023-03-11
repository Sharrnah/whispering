pip install -U pyinstaller

rem install plugin dependencies (also added in pyinstaller)
pip install winsdk

pyinstaller audioWhisper.py -y ^
            --python-option=u ^
            --hidden-import=pytorch --collect-data torch --copy-metadata torch ^
            --hidden-import=torchaudio.lib.libtorchaudio ^
            --copy-metadata tqdm ^
            --copy-metadata regex ^
            --copy-metadata requests ^
            --copy-metadata packaging ^
            --copy-metadata filelock ^
            --copy-metadata numpy ^
            --copy-metadata tokenizers ^
            --collect-data whisper ^
            --collect-data pykakasi ^
            --collect-all easyocr ^
            --collect-all winsdk ^
            --collect-all ctranslate2 ^
            -i app-icon.ico

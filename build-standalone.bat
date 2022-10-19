pip install -U pyinstaller
pyinstaller audioWhisper.py -y ^
            --hidden-import=pytorch --collect-data torch --copy-metadata torch ^
            --copy-metadata tqdm ^
            --copy-metadata regex ^
            --copy-metadata requests ^
            --copy-metadata packaging ^
            --copy-metadata filelock ^
            --copy-metadata numpy ^
            --copy-metadata tokenizers ^
            --collect-data whisper ^
            --collect-data pykakasi ^
            -i app-icon.ico

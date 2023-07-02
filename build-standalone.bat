pip install -U pyinstaller

rem install plugin dependencies (also added in pyinstaller)
pip install winsdk

rem pyinstaller audioWhisper.py -y ^
rem             --python-option=u ^
rem             --hidden-import=pytorch --collect-data torch --copy-metadata torch ^
rem             --hidden-import=torchaudio.lib.libtorchaudio ^
rem             --hidden-import=scipy.signal ^
rem             --hidden-import=transformers.models.nllb ^
rem             --hidden-import=sentencepiece ^
rem             --copy-metadata tqdm ^
rem             --copy-metadata regex ^
rem             --copy-metadata requests ^
rem             --copy-metadata packaging ^
rem             --copy-metadata filelock ^
rem             --copy-metadata numpy ^
rem             --copy-metadata tokenizers ^
rem             --copy-metadata rich ^
rem             --copy-metadata sentencepiece ^
rem             --copy-metadata transformers ^
rem             --collect-data whisper ^
rem             --collect-data pykakasi ^
rem             --collect-all easyocr ^
rem             --collect-all winsdk ^
rem             --collect-all ctranslate2 ^
rem             --collect-all torchaudio ^
rem             --collect-all scipy ^
rem             --collect-all lazy_loader ^
rem             --collect-all decorator ^
rem             --collect-all librosa ^
rem             --collect-all torchlibrosa ^
rem             --collect-all sentencepiece ^
rem             --collect-all transformers ^
rem             -i app-icon.ico

rem set pyrecursivelimit=import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
rem (echo %pyrecursivelimit%) > temp.txt & type audioWhisper.spec >> temp.txt & move /y temp.txt audioWhisper.spec >nul

rem 1. In your program's .spec file add this line near the top::
rem      import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

pyinstaller audioWhisper.spec -y

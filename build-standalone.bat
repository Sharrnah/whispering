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
rem             --hidden-import=df.deepfilternet3 ^
rem             --hidden-import=bitsandbytes ^
rem             --hidden-import=faiss-cpu ^
rem             --hidden-import=praat-parselmouth ^
rem             --hidden-import=parselmouth ^
rem             --hidden-import=pyworld ^
rem             --hidden-import=torchcrepe ^
rem             --hidden-import=grpcio ^
rem             --hidden-import=grpc ^
rem             --hidden-import=annotated_types ^
rem             --hidden-import=Cython ^
rem             --hidden-import=nemo_toolkit ^
rem             --hidden-import=nemo ^
rem             --copy-metadata rich ^
rem             --copy-metadata tqdm ^
rem             --copy-metadata regex ^
rem             --copy-metadata requests ^
rem             --copy-metadata packaging ^
rem             --copy-metadata filelock ^
rem             --copy-metadata numpy ^
rem             --copy-metadata tokenizers ^
rem             --copy-metadata sentencepiece ^
rem             --copy-metadata transformers ^
rem             --copy-metadata huggingface-hub ^
rem             --copy-metadata safetensors ^
rem             --copy-metadata pyyaml ^
rem             --copy-metadata pyloudnorm ^
rem             --copy-metadata nltk ^
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
rem             --collect-all df ^
rem             --collect-all nltk ^
rem             --collect-all fairseq ^
rem             --collect-all bitsandbytes ^
rem             --collect-all faiss-cpu ^
rem             --collect-all praat-parselmouth ^
rem             --collect-all pyworld ^
rem             --collect-all torchcrepe ^
rem             --collect-all grpcio ^
rem             --collect-all grpc ^
rem             --collect-all annotated_types ^
rem             --collect-all Cython ^
rem             --collect-all nemo_toolkit ^
rem             --collect-all nemo ^
rem             --collect-submodules fairseq ^
rem             --add-data ".cache/nltk/tokenizers/punkt;./nltk_data/tokenizers/punkt" ^
rem             -i app-icon.ico

rem set pyrecursivelimit=import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
rem (echo %pyrecursivelimit%) > temp.txt & type audioWhisper.spec >> temp.txt & move /y temp.txt audioWhisper.spec >nul

rem 1. In your program's .spec file add this line near the top::
rem      import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

pyinstaller audioWhisper.spec -y

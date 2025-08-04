# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

datas = []
binaries = []
# Collect dynamic libraries from onnxruntime
binaries= collect_dynamic_libs('onnxruntime', destdir='onnxruntime/capi')

hiddenimports = [
    'torch', 'pytorch', 'torchaudio.lib.libtorchaudio', 'scipy.signal', 'transformers', 'transformers.models.nllb', 'sentencepiece',
    'df.deepfilternet3', 'bitsandbytes', 'faiss', 'faiss-cpu', 'praat-parselmouth', 'parselmouth', 'pyworld', 'torchcrepe',
    'grpcio', 'grpc', 'annotated_types', 'Cython', 'nemo_toolkit', 'nemo', 'speechbrain', 'pyannote', 'pyannote.audio',
    'pyannote.pipeline', 'pyloudnorm', 'future', 'noisereduce', 'frozendict', 'torch_directml', 'x_transformers', 'inflect', 'backoff',
    'language_tags', 'spacy', 'en-core-web-sm', 'en_core_web_sm', 'misaki', 'fugashi', 'mojimoji', 'ordered_set', 'phonemizer', 'triton', 'mistral_common', 'snac'
]
datas += collect_data_files('torch', include_py_files=True)
datas += collect_data_files('whisper')
datas += collect_data_files('pykakasi')
datas += collect_data_files('lightning_fabric')
datas += collect_data_files('transformers', include_py_files=True)
datas += collect_data_files('x_transformers', include_py_files=True)
datas += collect_data_files('inflect', include_py_files=True)
datas += collect_data_files('language_tags', include_py_files=True)
datas += collect_data_files('spacy', include_py_files=True)
datas += collect_data_files('en-core-web-sm', include_py_files=True)
datas += collect_data_files('en_core_web_sm', include_py_files=True)
datas += collect_data_files('misaki', include_py_files=True)
datas += collect_data_files('phonemizer')
datas += collect_data_files('backoff')
datas += collect_data_files('triton')
datas += collect_data_files('mistral_common')
datas += collect_data_files('snac', include_py_files=True)
datas += copy_metadata('rich')
datas += copy_metadata('torch')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('sentencepiece')
datas += copy_metadata('transformers')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('safetensors')
datas += copy_metadata('pyyaml')
datas += copy_metadata('pyloudnorm')
datas += copy_metadata('future')
datas += copy_metadata('nltk')
datas += copy_metadata('noisereduce')
datas += copy_metadata('spacy')
datas += copy_metadata('en-core-web-sm')
datas += copy_metadata('en_core_web_sm')
datas += copy_metadata('misaki')
datas += copy_metadata('backoff')
hiddenimports += collect_submodules('fairseq')
tmp_ret = collect_all('easyocr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('winsdk')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ctranslate2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchaudio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lazy_loader')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('decorator')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('librosa')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchlibrosa')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sentencepiece')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('df')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nltk')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fairseq')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('bitsandbytes')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faiss')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faiss-cpu')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faiss_cpu')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('praat-parselmouth')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('praat_parselmouth')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('parselmouth')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyworld')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchcrepe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('grpcio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('grpc')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('annotated_types')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('Cython')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nemo_toolkit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nemo-toolkit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lightning_fabric')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lightning')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nemo')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('speechbrain')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyannote')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyannote.audio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyannote.pipeline')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('noisereduce')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('frozendict')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch_directml')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('inflect')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('language_tags')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('spacy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('en-core-web-sm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('en_core_web_sm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('misaki')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('backoff')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('triton')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mistral_common')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

workdir = os.environ.get('WORKDIR_WIN', r'\drone\src')
workdir = "C:" + workdir  # Now workdir = "C:\drone\src"

# Check for the existence of various possible locations for the punkt tokenizer
punkt_path_options = [
    r'.cache/nltk/tokenizers/punkt',
    r'C:\src\.cache\nltk\tokenizers\punkt',
    workdir + r'\.cache\nltk\tokenizers\punkt',
]
for path_option in punkt_path_options:
    if os.path.exists(path_option):
        datas.append((path_option, r'./nltk_data/tokenizers/punkt'))
        break  # Exit the loop once we find the existing path


punkt_tab_path_options = [
    r'.cache/nltk/tokenizers/punkt_tab',
    r'C:\src\.cache\nltk\tokenizers\punkt_tab',
    workdir + r'\.cache\nltk\tokenizers\punkt_tab',
]
for path_option in punkt_tab_path_options:
    if os.path.exists(path_option):
        datas.append((path_option, r'./nltk_data/tokenizers/punkt_tab'))
        break  # Exit the loop once we find the existing path

corpora_path_options = [
    r'.cache/nltk/corpora',
    r'C:\src\.cache\nltk\corpora',
    workdir + r'\.cache\nltk\corpora',
]
for path_option in corpora_path_options:
    if os.path.exists(path_option):
        datas.append((path_option, r'./nltk_data/corpora'))
        break  # Exit the loop once we find the existing path


# add local module src
#datas.append((r'./Models/TTS/F5TTS', r'Models.TTS.F5TTS'))
#datas.append((r'./Models/TTS/zonos', r'Models.TTS.zonos'))

block_cipher = None


a = Analysis(
    ['audioWhisper.py'],
    pathex=['C:\\src\\', workdir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

options = [
    ('u', None, 'OPTION'),
    ('X utf8', None, 'OPTION'),
]

exe = EXE(
    pyz,
    a.scripts,
    options,
    exclude_binaries=True,
    name='audioWhisper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app-icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='audioWhisper',
)

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_dynamic_libs
import os
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

datas = []
binaries = []
# Collect dynamic libraries from onnxruntime
binaries= collect_dynamic_libs('onnxruntime', destdir='onnxruntime/capi')
try:
    binaries += collect_dynamic_libs('torch')
except Exception:
    pass

hiddenimports = [
    'torch', 'pytorch', 'torchaudio.lib.libtorchaudio', 'scipy.signal', 'transformers', 'transformers.models.nllb', 'sentencepiece',
    'df.deepfilternet3', 'bitsandbytes', 'faiss', 'faiss-cpu', 'praat-parselmouth', 'parselmouth', 'pyworld', 'torchcrepe',
    'grpcio', 'grpc', 'annotated_types', 'Cython', 'nemo_toolkit', 'nemo', 'speechbrain', 'pyannote', 'pyannote.audio',
    'pyannote.pipeline', 'pyloudnorm', 'future', 'noisereduce', 'frozendict', 'torch_directml', 'x_transformers', 'inflect', 'backoff',
    'language_tags', 'spacy', 'en_core_web_sm', 'misaki', 'fugashi', 'mojimoji', 'unidic', 'unidic-lite', 'ordered_set', 'phonemizer',
    'flash_attn', 'mistral_common', 'snac',
    'espeakng_loader', 'unidic_lite', 'mamba_ssm', 'audiotools'
]
datas += collect_data_files('whisper')
datas += collect_data_files('pykakasi')
datas += collect_data_files('lightning_fabric')
datas += collect_data_files('phonemizer')
datas += collect_data_files('backoff')
datas += collect_data_files('mistral_common')
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
datas += copy_metadata('en_core_web_sm')
datas += copy_metadata('misaki')
datas += copy_metadata('backoff')

# ---- Bundle these as real modules (code + extensions) ----
for pkg in [
    'easyocr', 'winsdk', 'ctranslate2', 'torchaudio', 'scipy', 'lazy_loader',
    'decorator', 'librosa', 'torchlibrosa', 'sentencepiece', 'transformers',
    'df', 'nltk', 'fairseq', 'bitsandbytes', 'faiss', 'faiss-cpu', 'faiss_cpu',
    'praat-parselmouth', 'praat_parselmouth', 'parselmouth', 'pyworld',
    'torchcrepe', 'grpcio', 'grpc', 'annotated_types', 'Cython', 'nemo_toolkit',
    'nemo-toolkit', 'lightning_fabric', 'lightning', 'nemo', 'speechbrain',
    'pyannote', 'pyannote.audio', 'pyannote.pipeline', 'noisereduce',
    'frozendict', 'torch_directml', 'inflect', 'language_tags', 'spacy',
    'en_core_web_sm', 'misaki', 'unidic', 'unidic-lite', 'backoff',
    'flash_attn', 'mistral_common', 'espeakng_loader', 'unidic_lite',
    'mamba_ssm', 'audiotools', 'x_transformers', 'snac'
]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

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

# add warmed triton cache
datas.append((r'C:\src\triton_cache_warm', 'triton_cache'))

block_cipher = None

# ---- Runtime hook to set TRITON_CACHE_DIR (very important for first-run JIT) ----
runtime_hooks = [
    'rthooks/rt_mamba_triton_shim.py',
    'rthooks/rt_disable_triton_backend.py',
    'rthooks/rt_fix_flash_attn_spec.py',
    'rthooks/rt_inspect_fallback.py',
    'rthooks/rt_triton_env.py'
]

a = Analysis(
    ['audioWhisper.py'],
    pathex=['C:\\src\\', workdir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
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

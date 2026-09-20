#!/bin/bash
set -euo pipefail
if (( $# )); then
    exec "$@"
fi
cd "${SRCDIR:-/src}"
python -m unittest discover -s tests -p 'test_linux_build.py' -v
flavor=${TORCH_FLAVOR:-cu128}
case "$flavor" in
    cpu|cu128) ;;
    *) echo "Unsupported build flavor: $flavor (use cpu or cu128)" >&2; exit 2 ;;
esac
python -m pip install "torch==2.7.1+$flavor" "torchvision==0.22.1+$flavor" "torchaudio==2.7.1+$flavor" \
    --index-url "https://download.pytorch.org/whl/$flavor"
python -m pip install numpy==1.26.4 flit-core==3.12.0
python builder/linux-compat-wheels.py /tmp/linux-wheels
if [[ "$flavor" == cu128 ]]; then
    python builder/linux-onnx-wheels.py requirements.txt /tmp/linux-wheels
    # Both distributions own the same onnxruntime/ files. Keep only GPU.
    python -m pip uninstall -y onnxruntime
fi
python builder/linux-requirements.py requirements.txt /tmp/requirements-linux.txt "$flavor"
python -m pip install --no-build-isolation -r /tmp/requirements-linux.txt
python -m pip check
# Exercise the application's TTS playback path while recording from a virtual
# PulseAudio sink. No host sound devices or GPU are used by these checks.
pulseaudio --start --exit-idle-time=-1
python -m unittest discover -s tests -p 'test_audio_playback.py' -v
python -m unittest discover -s tests -p 'test_audio_input_switching.py' -v
timeout 90s python builder/linux-audio-smoke.py
python -m unittest discover -s tests -p 'test_linux_packaging.py' -v
python -m unittest discover -s tests -p 'test_audio_cpp_runtime_linux.py' -v
mkdir -p "${DIST_DIR:-/out}"
python builder/linux-cuda-check.py --flavor "$flavor" --output "${DIST_DIR:-/out}/linux-runtime-info.json"
python builder/linux-prepare.py
mkdir -p "${DIST_DIR:-/out}"
python -m pip freeze --all > "${DIST_DIR:-/out}/linux-python-packages.txt"
pyinstaller --clean -y --distpath "${DIST_DIR:-/out}" --workpath /tmp/pyinstaller audioWhisper.spec
# Keep the container-local PulseAudio server alive for frozen imports, too.
python builder/linux-startup-check.py "${DIST_DIR:-/out}/audioWhisper/audioWhisper" \
    --log "${WT_BUILD_LOG_DIR:-${DIST_DIR:-/out}}/linux-startup.log"

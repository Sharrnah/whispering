#!/bin/sh
# Run as the unprivileged tester in Dockerfile-linux-smoke.
set -eu
trap 'status=$?; if [ "$status" -ne 0 ]; then
    echo "Runtime check failed (status $status). Recent application logs:" >&2
    for log in /home/tester/app/*-*.log; do
        [ -f "$log" ] || continue
        echo "$log" >&2
        tail -n 40 "$log" >&2
    done
fi; exit "$status"' 0
flavor=${1:-cpu}
case "$flavor" in cpu|cu128) ;; *) exit 2 ;; esac
cd /artifacts
sha256sum -c SHA256SUMS
# Accept versioned releases and the earlier preview names. An optional second
# argument selects a specific ZIP when testing a directory with several builds.
archive=${2:-}
if [ -z "$archive" ]; then
    for candidate in whispering-tiger*_linux_amd64_${flavor}.zip "audioWhisper_linux_amd64_${flavor}.zip"; do
        [ -f "$candidate" ] || continue
        if [ -n "$archive" ]; then
            echo 'Multiple backend ZIPs found; pass the filename as the second argument.' >&2
            exit 2
        fi
        archive=$candidate
    done
fi
test -n "$archive" && test -f "$archive"
echo "Extracting $archive..."
mkdir -p /home/tester/app
unzip -oq "$archive" -d /home/tester/app
cp "whispering-tiger-linux-amd64-${flavor}" /home/tester/app/
cd /home/tester/app
echo 'Checking release files and bundled media tools...'
test -x audioWhisper/audioWhisper
test -s LICENSE
test -s ignorelist.txt
test -s .current_platform.yaml
test -s Plugins/place_plugins_here.txt
test -s markers/OKW-MRK.wav
test -s websocket_clients/simple/index.html
for build_report in PACKAGE-CONTENTS.txt README-LINUX.txt toolchain/README-LINUX.txt linux-runtime-info.json linux-python-packages.txt; do
    test ! -e "$build_report"
done
test -x help.sh
test -x get-device-list.sh
./toolchain/ffmpeg/ffmpeg -version > ffmpeg-version.log
./toolchain/ffmpeg/ffprobe -version > ffprobe-version.log
./toolchain/ffmpeg/ffmpeg -v error -f lavfi -i 'sine=frequency=440:duration=0.2' \
    -ar 16000 -ac 1 -y media-smoke.wav
./toolchain/ffmpeg/ffprobe -v error -show_entries stream=sample_rate,channels \
    -of default=noprint_wrappers=1 media-smoke.wav > ffprobe-audio.log
grep 'sample_rate=16000' ffprobe-audio.log
grep 'channels=1' ffprobe-audio.log
echo 'Checking bundled audio.cpp without a GPU...'
set -- toolchain/audio.cpp/v*-linux-x86_64/audiocpp_server
test "$#" -eq 1 && test -x "$1"
"$1" --list-devices > audio-cpp-devices.log 2>&1
grep 'CPU:0' audio-cpp-devices.log
chmod +x "whispering-tiger-linux-amd64-${flavor}"
test -x "whispering-tiger-linux-amd64-${flavor}"
test ! -e /usr/local/bin/python
test ! -e /usr/bin/python3
echo 'Checking frozen startup and PulseAudio recording...'
pulseaudio --start --exit-idle-time=-1
pactl load-module module-null-sink sink_name=wt_runtime sink_properties=device.description=WT_Runtime_Test
./help.sh > frozen-help.log 2>&1
./get-device-list.sh > frozen-devices.log 2>&1
grep 'API=PulseAudio' frozen-devices.log
./audioWhisper/audioWhisper --detect_energy --detect_energy_time 2 --audio_api PulseAudio \
    --audio_input_device 'Monitor of WT_Runtime_Test' > frozen-recording.log 2>&1
grep 'detected_energy:' frozen-recording.log
echo 'Checking UI startup...'
set +e
timeout 15s xvfb-run -a "./whispering-tiger-linux-amd64-${flavor}" > ui-startup.log 2>&1
ui_status=$?
set -e
test "$ui_status" -eq 124
echo 'PASS: archive hashes, executable permissions, unprivileged frozen startup and recording, UI stays open; no system Python.'

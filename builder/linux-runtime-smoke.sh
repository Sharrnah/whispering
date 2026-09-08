#!/bin/sh
# Run as the unprivileged tester in Dockerfile-linux-smoke.
set -eu
flavor=${1:-cpu}
case "$flavor" in cpu|cu128) ;; *) exit 2 ;; esac
cd /artifacts
sha256sum -c SHA256SUMS
mkdir -p /home/tester/app
unzip -oq "audioWhisper_linux_amd64_${flavor}.zip" -d /home/tester/app
cp "whispering-tiger-linux-amd64-${flavor}" /home/tester/app/
cd /home/tester/app
test -x audioWhisper/audioWhisper
chmod +x "whispering-tiger-linux-amd64-${flavor}"
test -x "whispering-tiger-linux-amd64-${flavor}"
test ! -e /usr/local/bin/python
test ! -e /usr/bin/python3
pulseaudio --start --exit-idle-time=-1
pactl load-module module-null-sink sink_name=wt_runtime sink_properties=device.description=WT_Runtime_Test
./audioWhisper/audioWhisper --help > frozen-help.log 2>&1
./audioWhisper/audioWhisper --devices true > frozen-devices.log 2>&1
grep 'API=PulseAudio' frozen-devices.log
./audioWhisper/audioWhisper --detect_energy --detect_energy_time 2 --audio_api PulseAudio \
    --audio_input_device 'Monitor of WT_Runtime_Test' > frozen-recording.log 2>&1
grep 'detected_energy:' frozen-recording.log
set +e
timeout 15s xvfb-run -a "./whispering-tiger-linux-amd64-${flavor}" > ui-startup.log 2>&1
ui_status=$?
set -e
test "$ui_status" -eq 124
echo 'PASS: archive hashes, executable permissions, unprivileged frozen startup and recording, UI stays open; no system Python.'

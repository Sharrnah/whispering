IF EXIST %cd%\ffmpeg\bin SET PATH=%cd%\ffmpeg\bin;%PATH%

audioWhisper\audioWhisper.exe --model medium --task translate --device_index 4 --energy 300 --phrase_time_limit 12 --websocket_ip 0.0.0.0 --open_browser

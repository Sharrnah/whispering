IF EXIST %cd%\ffmpeg\bin SET PATH=%cd%\ffmpeg\bin;%PATH%

audioWhisper\audioWhisper.exe --model medium --task transcribe --energy 300 --pause 1.0 --osc_ip 127.0.0.1 --websocket_ip 127.0.0.1 --phrase_time_limit 9 --condition_on_previous_text --open_browser

# Usage
1. run `audioWhisper\audioWhisper.exe --devices true` (or `get-device-list.bat`) and get the Index of the audio device. (the number in `[*]` at the end)
2. run `audioWhisper\audioWhisper.exe`. By default, it tries to find your default Microphone. Otherwise, you need to add `--device_index *` to the run command where the `*` is the device index found at step 3. Find more command-line flags [here](documentation/configurations.md#command-line-flags).
3. If websocket option is enabled, you can control the whisper task (translate or transcript) as well as textual translation options while the AI is running.
   
   <img src=images/remote_control.png width=750>
   
   For this: open the `websocket_clients/websocket-remote/` folder and start the index.html there.
   
   _If you have the AI running on a secondary PC, open the HTML file with the IP as parameter like this: `index.html?ws_server=ws://127.0.0.1:5000`_

## Usage with 3rd Party Applications
### VRChat
- run the script with `--osc_ip 127.0.0.1` parameter. This way it automatically writes the recognized text into the in-game chat-box.
  
  example:

  > `audioWhisper\audioWhisper.exe --model medium --task transcribe --energy 300 --osc_ip 127.0.0.1 --phrase_time_limit 9`

### Live Streaming Applications (OBS, vMix, XSplit ...)
1. run the script with `--websocket_ip 127.0.0.1` parameter (127.0.0.1 if you are running everything on the same machine), and set a `--phrase_time_limit` if you expect not many pauses that could be recognized by the configured `--energy` and `--pause` values.

   example:

   > `audioWhisper\audioWhisper.exe --model medium --task translate --device_index 4 --energy 300 --phrase_time_limit 15 --websocket_ip 127.0.0.1`
2. Find a streaming overlay website in the `websocket_clients` folder. (So far only `streaming-overlay-01` is optimized as overlay with transparent background.)
3. Add the HTML file to your streaming application. (With some additional arguments if needed. See [[Websocket Clients]](documentation/websocket-clients.md#all-possible-configuration-url-arguments) for all possible arguments.)
   
   _For example:_ `websocket_clients/streaming-overlay-01/index.html?no_scroll=1&no_loader=1&bottom_align=1&auto_rm_message=15`

### Desktop+ (Currently only new-ui Beta with embedded Browser)
1. Run the Application listening on your Audio-Device with the VRChat Sound.
2. Add the Overlay in the [Desktop+ Beta](https://github.com/elvissteinjr/DesktopPlus/tree/new-ui) with the embedded Browser with (`index.html?no_scroll=1&auto_hide_message=25`)
3. Set the Browser to allow Transparency.
4. Attach the Browser to your VR-Headset.

Voilà, you have live translated subtitles in VR of other people speaking (or videos playing) which automatically disappear after 25 seconds.

<img src=../images/vr_subtitles.gif width=410> <img src=../images/vrchat_live_subtitles.gif width=410>

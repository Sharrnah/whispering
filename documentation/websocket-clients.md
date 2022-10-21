# Websocket Clients
All Interactions with the Application besides the Speech recognition and OCR is done using Websockets.

Currently, all Websocket Clients are provided as HTML files. They should run on every relatively modern Browser that supports Websockets. (see [support table on caniuse.com](https://caniuse.com/websockets))

You have build a nice Client and would like to share?, let me know so i can add it to this list.

## Using the Websocket Clients

- **Simple** [_websocket_clients/simple_](../websocket_clients/simple/)

  Is the simplest implementation.
  
  Mostly intended to show how to use it without much around it, to make it easier to build your own Overlay.

  _Supports only the `ws_server` [URL argument](#all-possible-configuration-url-arguments)._

- **Streaming Overlay 01** [_websocket_clients/streaming-overlay-01_](../websocket_clients/streaming-overlay-01/)

  Has a lot more around the actual websocket implementation. Including a Styling that should work well for Streaming Overlays.

- **Websocket Remote** [_websocket_clients/websocket-remote_](../websocket_clients/websocket-remote/)

  Is the most complete Websocket implementation with additional Configuration Options that can be changed without restarting the application.

  Clicking on any Transcription Box feeds the Transcription and/or Translation into the textual translation Fields to Translate it into another language, edit it or send again over OSC.

  <img src=../images/remote_control.png width=600>


All Clients Support some options that can be set without changing the HTML

using Parameters added to the URL in the form:

> `index.html?ws_server=ws://127.0.0.1:5000&no_scroll=1&auto_hide_message=30`

_(first option is added with a `?`, all following with a `&`)_


## All possible Configuration URL arguments

|      Arguments      |    Default Value    |                                                                           Description                                                                           |
|:-------------------:|:-------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     `ws_server`     | ws://127.0.0.1:5000 |                                                      Sets the Websocket Server IP and Port to connect to.                                                       |
|     `no_scroll`     |        None         |                                                                  Removes the Scrollbar if set.                                                                  |
| `auto_hide_message` |          0          | Sets the Seconds after which Transcriptions are hidden, but not removed. (Can be shown again by clicking in the lower part where the Boxes appear) 0 = Disabled |
|  `auto_rm_message`  |          0          |                        Sets the Seconds after which Transcriptions are removed. (makes it impossible to access them again) 0 = Disabled                         |


## Other Use-Cases
Since it is possible to run multiple AI's concurrently, it is possible to combine multiple Websocket Clients.

For Example, you can create a HTML Page with I-Frames that show the different Websocket Clients.

> _Following example shows on the left the local remote control webpage and on the right the streaming overlay page connecting to a secondary PC._
> 
> _(but could also be the same PC using a different websocket port)._

<img src=../images/parallel-live-translation.png width=700>

```html
<!DOCTYPE html>
<html>
   
   <head>
      <title>Websocket Client Frames</title>
   </head>
   
   <frameset cols = "60%,40%">
      <frame name = "left" src = "audioWhisper/websocket_clients/websocket-remote/index.html" />
      <frame name = "right" src = "audioWhisper/websocket_clients/streaming-overlay-01/index.html?ws_server=ws://192.168.2.136:5000" />
      
      <noframes>
         <body>Your browser does not support frames.</body>
      </noframes>
   </frameset>
   
</html>
```

# FAQ
- **Problem**: _The translation/transcript is too slow, and it shows the warning:_

  _`UserWarning: FP16 is not supported on CPU; using FP32 instead.`_

  **Answers**: This means that the AI is not running on your GPU but on your CPU instead.
  - The whisper AI is running best on CUDA enabled GPUs.
    
    NVIDIA starting from GTX 1080 should do.
  - It is possible that CUDA is not installed on your System. (see [Prerequisites](#prerequisites))

- **Problem**: _The translation/transcript is still too slow. But no warning appears._

  **Answers**:
  - Your GPU might be busy with another Task or you are using a too big model for your GPU.
    
    Look in Task-Manager how much the GPU is used without running the AI or change to a smaller model like `small` or `tiny` using `--model small` or `--model tiny`. (see [Command-line flags](#command-line-flags))
  
  - If you happen to have a secondary PC with an GPU, you can outsource the workload to it:
    
    Run the AI on the secondary PC, start the Websocket-Server with `--websocket_ip 0.0.0.0` to have it listen on all its IP-Addresses.
    
    Change the IP of the websocket-client to use to the one from the **Secondary** PC. (Open the html file with parameter: `index.html?ws_server=ws://127.0.0.1:5000` to use secondary's PC IP-Address.)

    For OSC, give it the IP of your **Primary** PC using the `--websocket_ip 127.0.0.1` argument and change the `127.0.0.1` to its IP.

    Stream the Audio to the **Secondary PC** using for example https://vb-audio.com/Voicemeeter/vban.htm

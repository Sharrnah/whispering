import io
import json
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import tempfile
import os
import click
import VRC_OSCLib
import websocket


temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "temp.wav")

# some regular mistakenly recognized words/sentences on mostly silence audio, which are ignored in processing
blacklist = [
    "",
    "Thanks for watching!",
    "Thank you for watching!",
    "Thanks for watching.",
    "Thank you for watching.",
    "you"
]
# make all list entries lowercase for later comparison
blacklist = list((map(lambda x: x.lower(), blacklist)))

@click.command()
@click.option('--devices', default='False', help='print all available devices id', type=str)
@click.option('--device_index', default=-1, help='the id of the device (-1 = default active Mic)', type=int)
@click.option('--sample_rate', default=44100, help='sample rate of recording', type=int)
@click.option("--task", default="transcribe", help="task for the model whether to only transcribe the audio or translate the audio to english", type=click.Choice(["transcribe", "translate"]))
@click.option("--model", default="small", help="Model to use", type=click.Choice(whisper.available_models()))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--condition_on_previous_text", default=False, help="Feed it the previous result to keep it consistent across recognition windows, but makes it more prone to getting stuck in a failure loop",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--phrase_time_limit", default=None, help="phrase time limit before entry ends to break up long recognitions.", type=float)
@click.option("--osc_ip", default="0", help="IP to send OSC message to. Set to '0' to disable", type=str)
@click.option("--websocket_ip", default="0", help="IP where Websocket Server listens on. Set to '0' to disable", type=str)
@click.option("--ai_device", default=None, help="The Device the AI is loaded on. can be 'cuda' or 'cpu'. default does autodetect", type=click.Choice(["cuda", "cpu"]))
def main(devices, device_index, sample_rate, task, model, english, condition_on_previous_text, verbose, energy, pause,dynamic_energy, phrase_time_limit, osc_ip, websocket_ip, ai_device):

    if str2bool(devices) == True:
        index = 0
        for device in sr.Microphone.list_microphone_names():
            print(device, end = ' [' + str(index) + '] ' + "\n")
            index = index + 1
        return

    if websocket_ip != "0":
        websocket.StartWebsocketServer(websocket_ip, 5000)

    #there are no english models for large
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model, download_root=".cache/whisper", device=ai_device)
    
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=sample_rate, device_index=(device_index if device_index > -1 else None)) as source:
        print("Say something!")
        
        while True:
            #get and save audio to wav file
            audio = r.listen(source, phrase_time_limit=phrase_time_limit)
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            audio_clip.export(save_path, format="wav")

            if english:
                result = audio_model.transcribe(save_path, task=task, language='english', condition_on_previous_text=condition_on_previous_text)
            else:
                result = audio_model.transcribe(save_path, task=task, condition_on_previous_text=condition_on_previous_text)

            predicted_text = result.get('text').strip()

            if not predicted_text.lower() in blacklist:
                if not verbose:
                    print("Transcribe" + (" (OSC)" if osc_ip != "0" else "") + ": " + predicted_text)
                else:
                    print(result)
                # Send to VRChat
                if osc_ip != "0":
                    VRC_OSCLib.Chat(predicted_text, True, "/chatbox/input", IP = osc_ip, PORT = 9000)
                # Send to Websocket
                if websocket_ip != "0":
                    websocket.BroadcastMessage(json.dumps(result))

def str2bool(string):
    str2val = {"true": True, "false": False}
    if string.lower() in str2val:
        return str2val[string.lower()]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

main()

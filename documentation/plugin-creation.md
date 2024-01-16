# Plugin Creation

Plugins are a way to extend the functionality of Whispering Tiger.

There mignt still be changes to the plugin system in the future.

## How to use

Plugins are loaded from the `Plugins` directory in the root of the project. The directory is scanned for `.py` files and each file is loaded as a plugin.

## How to write

Plugins are written as Python classes. The class must inherit from `Plugins.Base` and implement the `init`, `timer`, `stt` and `tts` methods.

At the very top of the file, you should add a comment with a short description and most importantly, a version line, so the version can be compared inside the UI application.
example:
```python
# ============================================================
# This is the plugin xyz for Whispering Tiger
# Version: 1.0.0
# some more information about the plugin
# ============================================================
```
The format of the version line can start with `Version: `, `Version `, `V`, `V: ` followed by `<major>.<minor>.<patch>`

| Function                                                                            | Optionality | Description                                                                                                                                                                                                                                                                                                      |
|-------------------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `init(self)`                                                                        | Required    | is called at the initialization of whispering tiger, right after the settings file is loaded.                                                                                                                                                                                                                    |
| `on_enable(self)`                                                                   | Optional    | is called when the plugin is enabled.                                                                                                                                                                                                                                                                            |
| `on_disable(self)`                                                                  | Optional    | is called when the plugin is disabled.                                                                                                                                                                                                                                                                           |
| `timer(self)`                                                                       | Optional    | is called every x seconds (defined in `plugin_timer`) and is paused for x seconds (defined in `plugin_timer_timeout`) when the Speech-to-Text engine returned a result.<br>_This can be used for a regular output that is stopped occasionally when a more important transcription is supposed to be displayed._ |
| `stt(self, text, result_obj)`                                                       | Optional    | is called when the Speech-to-Text engine returns a result.                                                                                                                                                                                                                                                       |
| `stt_intermediate(self, text, result_obj)`                                          | Optional    | is called when a live transcription result is available. Make sure to use the `stt` function for final results.                                                                                                                                                                                                  |
| `tts(self, text, device_index, websocket_connection=None, download=False)`          | Optional    | is called when the TTS engine is about to play a result, except when called by the sst engine.<br>_if you want to play a sound when the Speech-to-Text engine returns a result, you should do it in the `stt` method as well._                                                                                   |
| `sts(self, wavefiledata, sample_rate)`                                              | Optional    | is called when a recording is finished (which is sent to the Speech-to-Text model). This function gets the audio recording to be processed by the plugin.                                                                                                                                                        |
| `text_translate(self, text, from_code, to_code) -> tuple (txt, from_lang, to_lang)` | Optional    | is called when a translation is requested and no included translator is available.<br>_Must return a tuple of translation_text, from_lang_code, to_lang_code._                                                                                                                                                   |
| `on_{event_name}_call(self, data_obj) -> dict (data_obj)`                           | Optional    | is called when a custom plugin event is called via `Plugins.plugin_custom_event_call(event_name, data_obj)`. See [Custom Plugin events](#Custom-Plugin-events) for more info.                                                                                                                                    |

## Helper methods

The `Base` class provides some helper methods to make it easier to write plugins.

`init_plugin_settings(self, settings, settings_groups=None)` - Prepare all possible plugin settings and their default values. This method must be called in the `init` method of the plugin. The `settings` parameter is a dictionary with the settings and their default values. The `settings_groups` parameter is an optional dictionary with the settings groups and the settings that belong to that group. The settings groups are used in the settings window to group the settings. If the `settings_groups` parameter is not provided no groups are displayed in the UI.

**IMPORTANT: Settings that are not initialized with `init_plugin_settings` are deleted when calling `init_plugin_settings`, so make sure to define every setting your Plugin needs.**

`get_plugin_setting(self, setting, default=None)` - Get a plugin setting from the settings file. If the setting is not yet in the settings file, the default value is used. (if default is not set, the default from _init_plugin_settings()_ is used)

`set_plugin_setting(self, setting, value)` - Set a plugin setting in the settings file.

__Note:__ _When using the *_plugin_setting methods, the settings are saved with the class name as the section name. So if you have a plugin called `ExamplePlugin`, the settings will be saved in the `ExamplePlugin` section._

`is_enabled(self, default=True)` - Check if the plugin is enabled. If the plugin is not yet in the settings file, the default value is used. So by default, plugins will be enabled. Use this around your main functionality to allow enabling/disabling of your plugin even at runtime.


## use specific Widgets in plugin settings

To use specific widgets in plugin settings, you can add specific structs to the init_plugin_settings method.

The following structs are available:
- `{"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.7}` - A slider
- `{"type": "button", "label": "Batch Generate", "style": "primary"}` - A button (style can be "primary" or "default")
- `{"type": "select", "label": "Label", "value": "default value", "options": ["default value", "option2", "option3"]}` - A select box 
- `{"type": "textarea", "rows": 5, "value": ""}` - A textarea
- `{"type": "hyperlink", "label": "hyperlink", "value": "https://github.com/Sharrnah/whispering-ui"}`
- `{"type": "label", "label": "Some infotext in a label.", "style": "center"}` - A label (style can be "left", "right" or "center")
- `{"type": "file_open", "accept": ".wav,.mp3", "value": "bark_clone_voice/clone_voice.wav"}` - A file open dialog (accept can be any file extension or a comma separated list of file extensions)
- `{"type": "file_save", "accept": ".npz", "value": "last_prompt.npz"}` - A file save dialog (accept can be any file extension or a comma separated list of file extensions)
- `{"type": "folder_open", "accept": "", "value": ""}` - A folder open dialog
- `{"type": "dir_open", "accept": "", "value": ""}` - Alias for a folder open dialog

## Custom Plugin events
You can use event calls in plugins using `Plugins.plugin_custom_call(event_name, data_obj)`.
The function names have the form of `on_{event_name}_call`.

`event_name` should be unique and self explaining.

The function needs to return `None` if something failed or should be skipped,
or the `data_obj` again with your necessary changes to the object.

As an example the call from the Silero TTS:
```py
plugin_audio = Plugins.plugin_custom_event_call('silero_tts_after_audio', {'audio': audio})
if plugin_audio is not None:
    audio = plugin_audio['audio']
```

The called function in the Plugin looks like this:
```py
def on_silero_tts_after_audio_call(self, data_obj):
    if self.is_enabled(False) and self.get_plugin_setting("voice_change_source") == CONSTANTS["SILERO_TTS"]:
        audio = data_obj['audio']
        # doing stuff
        data_obj['audio'] = audio
        return data_obj
    return None
```

Before calling Events from other Plugins, make sure all Plugins are already loaded. (Should not be called in `__init__`).

Make sure to check if the Event should be callable. (Is Plugin enabled? Are the plugin settings configured properly? ...). Otherwise return `None`.

### List of core plugin events

| Function Name                            | Description                                                                                                                     |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| on_silero_tts_after_audio_call(data_obj) | Called after the Included Silero TTS generated audio. Expects the `audio` key in the `data_obj` <br> (audio as Pytorch Tensor). |


## Example plugin
```python
# ============================================================
# This is the example plugin for Whispering Tiger
# Version: 1.0.0
# some more information about the plugin
# ============================================================
import Plugins
import settings
import VRC_OSCLib

class ExamplePlugin(Plugins.Base):
    def init(self):
        # prepare all possible plugin settings and their default values
        self.init_plugin_settings(
            {
                "hello_world": "default value",
                "hello_world2": "foo bar",
                "osc_auto_processing_enabled": False,
                "tts_answer": False,
                "homepage_link": {"label": "Whispering Tiger Link", "value": "https://github.com/Sharrnah/whispering-ui", "type": "hyperlink"},

                "more_settings_a": "default value",
                "more_settings_b": "default value\nmultiline",
                "more_settings_c": 0.15,
                "more_settings_d": 60,
            },
            settings_groups={
                "General": ["osc_auto_processing_enabled", "tts_answer", "hello_world", "hello_world2", "homepage_link"],
                "Second Group": ["more_settings_a", "more_settings_b", "more_settings_c", "more_settings_d"],
            }
        )
        
        if self.is_enabled():
            print(self.__class__.__name__ + " is enabled")

            # disable OSC processing so the Plugin can take it over:
            settings.SetOption("osc_auto_processing_enabled", False)
            # disable TTS so the Plugin can take it over:
            settings.SetOption("tts_answer", False)

            # disable websocket final messages processing so the Plugin can take it over:
            # this is really only needed if you want to use the websocket to send your own messages.
            # for the Websocket clients to understand the messages, you must follow the format. (see the LLM Plugin for a good example)
            ## settings.SetOption("websocket_final_messages", False)
        else:
            print(self.__class__.__name__ + " is disabled")

    ## OPTIONAL. called every x seconds (defined in plugin_timer)
    def timer(self):
        # get the settings from the global app settings
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")

        hello_world = self.get_plugin_setting("hello_world", "default foo bar")
        hello_world2 = self.get_plugin_setting("hello_world2")
        print(hello_world2)

        if self.is_enabled():
            VRC_OSCLib.Chat(hello_world, True, False, osc_address, IP=osc_ip, PORT=osc_port,
                            convert_ascii=False)
        pass

    ## OPTIONAL. called when the STT engine returns a result
    def stt(self, text, result_obj):
        if self.is_enabled():
            print("Plugin Example")
            print(result_obj['language'])
        return

    ## OPTIONAL. only called when the STT engine returns an intermediate live result
    def stt_intermediate(self, text, result_obj):
        if self.is_enabled():
            print("Plugin Example")
            print(result_obj['language'])
        return

    ## OPTIONAL. called when the "send TTS" function is called
    def tts(self, text, device_index, websocket_connection=None, download=False):
        return
    
    ## OPTIONAL - called when audio is finished recording and the audio is sent to the STT model
    def sts(self, wavefiledata, sample_rate):
        return

    ## OPTIONAL - called when translation is requested and no other translator is selected. must return a tuple consisting of text, from_code, to_code.
    def text_translate(self, text, from_code, to_code) -> tuple:
        return text, from_code, to_code
    
    ## OPTIONAL - called when a websocket message is received.
    ## formats are: (where 'ExamplePlugin' is the plugin class name)
    ## {"name": "ExamplePlugin", "type": "plugin_button_press", "value": "button_name"}
    ## {"name": "ExamplePlugin", "type": "plugin_custom_event", "value": []}
    def on_event_received(self, message, websocket_connection=None):
        if "type" not in message:
            return
        if message["type"] == "plugin_button_press":
            if message["value"] == "button_name":
                print("button pressed")
        if message["type"] == "plugin_custom_event":
            if message["value"] == "other_event_name":
                print("custom event received")
        pass
    
    ## OPTIONAL
    def on_enable(self):
        pass
    
    ## OPTIONAL
    def on_disable(self):
        pass

    ## OPTIONAL - custom event call function.
    # def on_{event_name}_call(self, data_obj):
    #     if self.is_enabled(False):
    #         return data_obj
    # return None
```

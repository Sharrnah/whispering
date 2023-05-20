# Plugin Creation

Plugins are a way to extend the functionality of Whispering Tiger.

There mignt still be changes to the plugin system in the future.

## How to use

Plugins are loaded from the `Plugins` directory in the root of the project. The directory is scanned for `.py` files and each file is loaded as a plugin.

## How to write

Plugins are written as Python classes. The class must inherit from `Plugins.Base` and implement the `init`, `timer`, `stt` and `tts` methods.

The `timer` method is called every x seconds (defined in `plugin_timer`) and is paused for x seconds (defined in `plugin_timer_timeout`) when the STT engine returned a result.

The `stt` method is called when the STT engine returns a result.

The `tts` method is called when the TTS engine is about to play a result, except when called by the sst engine.
So if you want to play a sound when the STT engine returns a result, you should do it in the `stt` method as well.

The `init` method is called at the initialization of whispering tiger, right after the settings file is loaded.

The _optional_ methods `on_enable` and `on_disable` are called when the plugin is enabled or disabled.

The _optional_ method `stt_intermediate` is only called when a live transcription result is available. Make sure to use the `stt` function for final results.

## Helper methods

The `Base` class provides some helper methods to make it easier to write plugins.

`init_plugin_settings(self, settings, settings_groups=None)` - Prepare all possible plugin settings and their default values. This method must be called in the `init` method of the plugin. The `settings` parameter is a dictionary with the settings and their default values. The `settings_groups` parameter is an optional dictionary with the settings groups and the settings that belong to that group. The settings groups are used in the settings window to group the settings. If the `settings_groups` parameter is not provided no groups are displayed in the UI.

**IMPORTANT: Settings that are not initialized with `init_plugin_settings` are deleted when calling `init_plugin_settings`, so make sure to define every setting your Plugin needs.**

`get_plugin_setting(self, setting, default=None)` - Get a plugin setting from the settings file. If the setting is not yet in the settings file, the default value is used.

`set_plugin_setting(self, setting, value)` - Set a plugin setting in the settings file.

__Note:__ _When using the *_plugin_setting methods, the settings are saved with the class name as the section name. So if you have a plugin called `ExamplePlugin`, the settings will be saved in the `ExamplePlugin` section._

`is_enabled(self, default=True)` - Check if the plugin is enabled. If the plugin is not yet in the settings file, the default value is used. So by default, plugins will be enabled. Use this around your main functionality to allow enabling/disabling of your plugin even at runtime.


## Example plugin
```python
import Plugins
import settings
import VRC_OSCLib

class ExamplePlugin(Plugins.Base):
    def init(self):
        # prepare all possible plugin settings and their default values
        self.init_plugin_settings(
            {
                "osc_auto_processing_enabled": False,
                "tts_answer": False,

                "more_settings_a": "default value",
                "more_settings_b": "default value\nmultiline",
                "more_settings_c": 0.15,
                "more_settings_d": 60,
            },
            settings_groups={
                "General": ["osc_auto_processing_enabled", "tts_answer"],
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

    # called every x seconds (defined in plugin_timer)
    def timer(self):
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")

        hello_world = self.get_plugin_setting("hello_world", "default foo bar")

        if self.is_enabled():
            VRC_OSCLib.Chat(hello_world, True, False, osc_address, IP=osc_ip, PORT=osc_port,
                            convert_ascii=False)
        pass

    # called when the STT engine returns a result
    def stt(self, text, result_obj):
        if self.is_enabled():
            print("Plugin Example")
            print(result_obj['language'])
        return

    # OPTIONAL. only called when the STT engine returns an intermediate live result
    def stt_intermediate(self, text, result_obj):
        if self.is_enabled():
            print("Plugin Example")
            print(result_obj['language'])
        return

    # called when the "send TTS" function is called
    def tts(self, text, device_index, websocket_connection=None, download=False):
        return
    
    # OPTIONAL
    def on_enable(self):
        pass
    
    # OPTIONAL
    def on_disable(self):
        pass
```

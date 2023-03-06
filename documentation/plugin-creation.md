# Plugin Creation

Plugins are a way to extend the functionality of Whispering Tiger.

## How to use

Plugins are loaded from the `Plugins` directory in the root of the project. The directory is scanned for `.py` files and each file is loaded as a plugin.

## How to write

Plugins are written as Python classes. The class must inherit from `Plugins.Base` and implement the `timer` and `stt` methods. The `__init__` method is called once when the plugin is loaded.

Settings are __not__ available in the `__init__` method.

The `timer` method is called every x seconds (defined in `plugin_timer`). The `stt` method is called when the STT engine returns a result.

The `timer` method is paused for x seconds (defined in `plugin_timer_timeout`) when the STT engine returned a result.

## Helper methods

The `Base` class provides some helper methods to make it easier to write plugins.

`get_plugin_setting(self, setting, default=None)` - Get a plugin setting from the settings file. If the setting is not yet in the settings file, the default value is used.

`set_plugin_setting(self, setting, value)` - Set a plugin setting in the settings file.

__Note:__ _When using the *_plugin_setting methods, the settings are saved with the class name as the section name. So if you have a plugin called `ExamplePlugin`, the settings will be saved in the `ExamplePlugin` section._

`is_enabled(self, default=True)` - Check if the plugin is enabled. If the plugin is not yet in the settings file, the default value is used. So by default, plugins will be enabled.


## Example plugin
```python
import Plugins
import settings
import VRC_OSCLib

class ExamplePlugin(Plugins.Base):
    # No settings are available in __init__
    def __init__(self):
        print(self.__class__.__name__ + " loaded")
        pass

    def timer(self):
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")

        hello_world = self.get_plugin_setting("hello_world", "default foo bar")

        if self.is_enabled():
            VRC_OSCLib.Chat(hello_world, True, False, osc_address, IP=osc_ip, PORT=osc_port,
                                            convert_ascii=False)
        pass

    def stt(self, text, result_obj):
        if self.is_enabled():
            print("Plugin Example")
            print(result_obj['language'])
        return
```

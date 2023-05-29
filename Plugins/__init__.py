import os
import traceback
from abc import abstractmethod
from importlib import util
from pathlib import Path
import copy

import settings

SUPPORTED_WIDGET_TYPES = ["button", "slider", "select", "textarea", "hyperlink", "label", "file_open", "file_save",
                          "folder_open", "dir_open"]


class Base:
    """Basic resource class. Concrete resources will inherit from this one
    """
    plugins = []

    plugin_settings_default = {}

    # For every class that inherits from the current,
    # the class name will be added to plugins
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.plugins.append(cls)

    def __init__(self):
        pass

    def is_enabled(self, default=False):
        if self.__class__.__name__ not in settings.GetOption("plugins"):
            setting = settings.GetOption("plugins")
            setting[self.__class__.__name__] = default
            settings.SetOption("plugins", setting)

        return settings.GetOption("plugins")[self.__class__.__name__]

    # set plugins that do not yet exist and delete settings that are not given to init_plugin_settings where key is the settings_name and value the settings default value
    def init_plugin_settings(self, init_settings=None, settings_groups=None):
        if init_settings is None:
            init_settings = {}

        self.plugin_settings_default = copy.deepcopy(init_settings)

        # set plugins that do not yet exist
        for settings_name, default in init_settings.items():
            self.get_plugin_setting(settings_name, default)

        plugin_settings = copy.deepcopy(settings.GetOption("plugin_settings"))
        if self.__class__.__name__ in plugin_settings:
            for settings_name in list(plugin_settings[self.__class__.__name__]):
                # delete settings that are not given to init_plugin_settings
                if settings_name not in init_settings and settings_name != "settings_groups":
                    del plugin_settings[self.__class__.__name__][settings_name]

                # set event widgets
                if settings_name in init_settings and isinstance(init_settings[settings_name], dict) and \
                        "type" in init_settings[settings_name] and init_settings[settings_name]["type"] in SUPPORTED_WIDGET_TYPES:
                    if "value" in init_settings[settings_name] and isinstance(
                            plugin_settings[self.__class__.__name__][settings_name], dict) and "value" in \
                            plugin_settings[self.__class__.__name__][settings_name]:
                        # keep value
                        init_settings[settings_name]["value"] = plugin_settings[self.__class__.__name__][settings_name][
                            "value"]
                    elif "value" in init_settings[settings_name] and type(
                            plugin_settings[self.__class__.__name__][settings_name]) == type(
                        init_settings[settings_name]["value"]):
                        # keep value and convert to widget
                        init_settings[settings_name]["value"] = plugin_settings[self.__class__.__name__][settings_name]

                    plugin_settings[self.__class__.__name__][settings_name] = init_settings[settings_name]

            # set settings_groups
            plugin_settings[self.__class__.__name__]["settings_groups"] = settings_groups

        settings.SetOption("plugin_settings", plugin_settings)

    # fetch plugin settings value, also supporting widgets
    def _get_plugin_setting_value(self, settings_value):
        if isinstance(settings_value, dict) and "type" in settings_value and \
                settings_value["type"] in SUPPORTED_WIDGET_TYPES and "value" in settings_value:
            return settings_value["value"]
        return settings_value

    def get_plugin_setting(self, *args):
        if len(args) == 1:
            if args[0] in self.plugin_settings_default:
                return self._get_plugin_setting(args[0], self.plugin_settings_default[args[0]])
            else:
                return self._get_plugin_setting(args[0])
        return self._get_plugin_setting(*args)

    def _get_plugin_setting(self, settings_name, default=None):
        if self.__class__.__name__ in settings.GetOption("plugin_settings") and \
                settings.GetOption("plugin_settings")[self.__class__.__name__] is not None and \
                settings_name in settings.GetOption("plugin_settings")[self.__class__.__name__]:
            return self._get_plugin_setting_value(
                settings.GetOption("plugin_settings")[self.__class__.__name__][settings_name]
            )
        else:
            setting = copy.deepcopy(settings.GetOption("plugin_settings"))
            if self.__class__.__name__ not in setting or setting[self.__class__.__name__] is None:
                setting[self.__class__.__name__] = {settings_name: default}
            elif settings_name not in setting[self.__class__.__name__]:
                setting[self.__class__.__name__][settings_name] = default
            settings.SetOption("plugin_settings", setting)
            return self._get_plugin_setting_value(default)

    def set_plugin_setting(self, settings_name, value):
        setting = copy.deepcopy(settings.GetOption("plugin_settings"))
        if self.__class__.__name__ not in setting:
            setting[self.__class__.__name__] = {}
        setting[self.__class__.__name__][settings_name] = value
        settings.SetOption("plugin_settings", setting)

    def on_enable(self):
        pass

    def on_disable(self):
        pass

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def stt(self, text, result_obj):
        pass

    @abstractmethod
    def tts(self, text, device_index, websocket_connection=None, download=False):
        pass

    @abstractmethod
    def timer(self):
        pass


# Small utility to automatically load modules
def load_module(path):
    name = os.path.split(path)[-1]
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Get current path
plugin_path = Path(Path.cwd() / "Plugins")
os.makedirs(plugin_path, exist_ok=True)

# path = os.path.abspath(__file__)
# dirpath = os.path.dirname(path)
dirpath = str(plugin_path.resolve())

for fname in os.listdir(dirpath):
    # Load only "real modules"
    if not fname.startswith('.') and \
            not fname.startswith('__') and fname.endswith('.py'):
        try:
            load_module(os.path.join(dirpath, fname))
        except Exception:
            traceback.print_exc()

# load plugins into array
plugins = []
for plugin in Base.plugins:
    plugins.append(plugin())

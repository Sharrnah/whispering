import os
import traceback
from abc import abstractmethod
from importlib import util
from pathlib import Path
import copy

from audio_tools import get_audio_device_index_by_name_and_api, get_audio_api_index_by_name
import settings

SUPPORTED_WIDGET_TYPES = ["button", "slider", "select", "select_textvalue", "textarea", "textfield", "hyperlink", "label", "file_open", "file_save",
                          "folder_open", "dir_open", "select_audio", "select_completion"]


class Base:
    """Basic resource class. Concrete resources will inherit from this one
    """
    base_plugins_list = []

    plugin_settings_default = {}

    _settings = None

    # For every class that inherits from the current,
    # the class name will be added to plugins
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.base_plugins_list.append(cls)

    def __init__(self, init_settings=settings.SETTINGS):
        self._settings = init_settings
        if hasattr(self, '__plugin_init__'):
            self.__plugin_init__()
        pass

    def is_enabled(self, default=False):
        if self.__class__.__name__ not in self._settings.GetOption("plugins"):
            setting = self._settings.GetOption("plugins")
            setting[self.__class__.__name__] = default
            self._settings.SetOption("plugins", setting)

        return self._settings.GetOption("plugins")[self.__class__.__name__]

    # set plugins that do not yet exist and delete settings that are not given to init_plugin_settings where key is the settings_name and value the settings default value
    def init_plugin_settings(self, init_settings=None, settings_groups=None):
        if init_settings is None:
            init_settings = {}

        self.plugin_settings_default = copy.deepcopy(init_settings)

        # set plugins that do not yet exist
        for settings_name, default in init_settings.items():
            self.get_plugin_setting(settings_name, default)

        plugin_settings = copy.deepcopy(self._settings.GetOption("plugin_settings"))
        if self.__class__.__name__ in plugin_settings:
            for settings_name in list(plugin_settings[self.__class__.__name__]):
                # delete settings that are not given to init_plugin_settings
                if settings_name not in init_settings and settings_name != "settings_groups":
                    del plugin_settings[self.__class__.__name__][settings_name]

                # set event widgets
                if settings_name in init_settings and isinstance(init_settings[settings_name], dict) and \
                        "type" in init_settings[settings_name] and init_settings[settings_name]["type"] in SUPPORTED_WIDGET_TYPES:
                    if "value" in init_settings[settings_name] and init_settings[settings_name].get("type") != "button" and isinstance(
                            plugin_settings[self.__class__.__name__][settings_name], dict) and "value" in \
                            plugin_settings[self.__class__.__name__][settings_name]:
                        # keep value
                        init_settings[settings_name]["value"] = plugin_settings[self.__class__.__name__][settings_name][
                            "value"]
                        # keep keys that start with an underscore
                        for key in plugin_settings[self.__class__.__name__][settings_name]:
                            if key.startswith('_'):
                                init_settings[settings_name][key] = plugin_settings[self.__class__.__name__][settings_name][key]
                    elif "value" in init_settings[settings_name] and type(
                            plugin_settings[self.__class__.__name__][settings_name]) == type(
                        init_settings[settings_name]["value"]):
                        # keep value and convert to widget
                        init_settings[settings_name]["value"] = plugin_settings[self.__class__.__name__][settings_name]

                    plugin_settings[self.__class__.__name__][settings_name] = init_settings[settings_name]

            # set settings_groups
            plugin_settings[self.__class__.__name__]["settings_groups"] = settings_groups

        self._settings.SetOption("plugin_settings", plugin_settings)

    def _audio_widget_device_getter(self, settings_value):
        device_api = 0
        is_input = True
        device_api_name = ""
        device_type_name = ""
        if "device_api" in settings_value and settings_value["device_api"] != "" and settings_value["device_api"] != "all":
            device_api_name = settings_value["device_api"]
        if "device_type" in settings_value:
            device_type_name = settings_value["device_type"]
        value = settings_value["value"]
        widget_value = settings_value["value"]
        if widget_value.find("#|") != -1:
            value = widget_value.split("#|")[0]
            special_values = widget_value.split("#|")[1]
            if special_values != "":
                if device_type_name == "":
                    device_type_name = special_values.split(",")[1]
                if device_api_name == "":
                    device_api_name = special_values.split(",")[0]

        if device_type_name == "input":
            is_input = True
        if device_type_name == "output":
            is_input = False

        if device_api_name != "":
            device_api, _ = get_audio_api_index_by_name(device_api_name)
        return device_api, is_input, int(value)

    # fetch plugin settings value, also supporting widgets
    def _get_plugin_setting_value(self, settings_value):
        if isinstance(settings_value, dict) and "type" in settings_value and \
                settings_value["type"] in SUPPORTED_WIDGET_TYPES and "value" in settings_value:
            # special case for select_audio widget
            if settings_value["type"] == "select_audio" and "_value_text" in settings_value and settings_value["_value_text"] != "":
                device_api, is_input, value = self._audio_widget_device_getter(settings_value)
                return get_audio_device_index_by_name_and_api(settings_value["_value_text"], device_api, is_input, value)
            # special case for select_completion and select_textvalue widget
            if settings_value["type"] == "select_completion" or settings_value["type"] == "select_textvalue":
                if "_value_real" in settings_value:
                    return settings_value["_value_real"]
                else:
                    # in case there is no _value_real, try to find value in values list
                    for item in settings_value["values"]:
                        if item[0] == settings_value["value"]:
                            return item[1]

            # regular widgets
            return settings_value["value"]
        # non widget setting
        return settings_value

    def get_plugin_setting(self, *args):
        if len(args) == 1:
            if args[0] in self.plugin_settings_default:
                return self._get_plugin_setting(args[0], self.plugin_settings_default[args[0]])
            else:
                return self._get_plugin_setting(args[0])
        return self._get_plugin_setting(*args)

    def _get_plugin_setting(self, settings_name, default=None):
        if self.__class__.__name__ in self._settings.GetOption("plugin_settings") and \
                self._settings.GetOption("plugin_settings")[self.__class__.__name__] is not None and \
                settings_name in self._settings.GetOption("plugin_settings")[self.__class__.__name__]:
            return self._get_plugin_setting_value(
                self._settings.GetOption("plugin_settings")[self.__class__.__name__][settings_name]
            )
        else:
            setting = copy.deepcopy(self._settings.GetOption("plugin_settings"))
            if self.__class__.__name__ not in setting or setting[self.__class__.__name__] is None:
                setting[self.__class__.__name__] = {settings_name: default}
            elif settings_name not in setting[self.__class__.__name__]:
                setting[self.__class__.__name__][settings_name] = default
            self._settings.SetOption("plugin_settings", setting)
            return self._get_plugin_setting_value(default)

    def set_plugin_setting(self, settings_name, value):
        setting = copy.deepcopy(self._settings.GetOption("plugin_settings"))
        if self.__class__.__name__ not in setting:
            setting[self.__class__.__name__] = {}
        setting[self.__class__.__name__][settings_name] = value
        self._settings.SetOption("plugin_settings", setting)

    def reset_plugin_setting(self, settings_name):
        setting = copy.deepcopy(self._settings.GetOption("plugin_settings"))
        if self.__class__.__name__ in setting and settings_name in setting[self.__class__.__name__]:
            setting[self.__class__.__name__][settings_name] = self.plugin_settings_default[settings_name]
            self._settings.SetOption("plugin_settings", setting)

    def reset_plugin_all_settings(self):
        setting = copy.deepcopy(self._settings.GetOption("plugin_settings"))
        if self.__class__.__name__ in setting:
            for settings_name in setting[self.__class__.__name__]:
                if settings_name in self.plugin_settings_default:
                    setting[self.__class__.__name__][settings_name] = self.plugin_settings_default[settings_name]
            self._settings.SetOption("plugin_settings", setting)

    @abstractmethod
    def init(self):
        pass

    # def on_enable(self):
    #     pass

    # def on_disable(self):
    #     pass

    # def text_translate(self, text, from_code, to_code) -> tuple:
    #     pass

    # def stt(self, text, result_obj):
    #     pass

    # def tts(self, text, device_index, websocket_connection=None, download=False):
    #     pass

    # def timer(self):
    #     pass


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
for plugin in Base.base_plugins_list:
    plugins.append(plugin(init_settings=settings.SETTINGS))


def get_plugin(class_name):
    for plugin_inst in plugins:
        if plugin_inst.__class__.__name__ == class_name:
            return plugin_inst  # return plugin instance
    return None

# Add a new plugin at runtime and check if it is not already added.
# Check the provided className if given, otherwise add the first class found that inherits from Base. And check if className is provided if it is already added.
def add_plugin(plugin_file, className=None):
    global plugins
    plugin_file_path_obj = Path(plugin_path / plugin_file)
    if not plugin_file_path_obj.exists() or not plugin_file_path_obj.is_file():
        print(f"Plugin file {plugin_file} does not exist.")
        return
    plugin_file_path = str(plugin_file_path_obj.resolve())
    try:
        # Check if className is already added
        if className is not None:
            existing_plugin = get_plugin(className)
            if existing_plugin is not None:
                print(f"Plugin {className} is already added.")
                return

        # Only load the module if className not provided or not already added
        module = None
        if className is None or get_plugin(className) is None:
            module = load_module(plugin_file_path)
        else:
            print(f"Plugin {className} is already loaded.")
            return

        # If className is provided, try to load that specific class
        if className is not None:
            plugin_class = getattr(module, className, None)
            if plugin_class is None:
                print(f"Class {className} not found in {plugin_file_path}.")
                return
            if not issubclass(plugin_class, Base):
                print(f"Class {className} does not inherit from Base.")
                return

            plugin_instance = plugin_class(init_settings=settings.SETTINGS)
            plugins.append(plugin_instance)
            print(f"Plugin {className} added successfully.")
            return

        # Otherwise, add the first class found that inherits from Base
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Base) and attr is not Base:
                if get_plugin(attr.__name__) is None:
                    plugin_instance = attr(init_settings=settings.SETTINGS)
                    plugins.append(plugin_instance)
                    print(f"Plugin {attr.__name__} added successfully.")
                    break
                else:
                    print(f"Plugin {attr.__name__} is already added.")
                    break

    except Exception as e:
        print(f"Error loading plugin from {plugin_file_path}: {e}")
        traceback.print_exc()

def internal_plugin_custom_event_call(plugins_list, event_name, data_obj):
    plugin_event_name = event_name
    for plugin_inst in plugins_list:
        call_func_name = 'on_'+plugin_event_name+'_call'
        if hasattr(plugin_inst, call_func_name):
            try:
                yield getattr(plugin_inst, call_func_name)(copy.deepcopy(data_obj))
            except Exception as e:
                print(f"Error in plugin {plugin_inst.__class__.__name__} on {call_func_name}: {e}")
                traceback.print_exc()


def call_internal_custom_event_with_plugins(plugins_list, event_name, data_obj):
    for result in internal_plugin_custom_event_call(plugins_list, event_name, data_obj):
        return result

def plugin_custom_event_call(event_name, data_obj):
    # Return the first item from the generator
    for result in internal_plugin_custom_event_call(plugins, event_name, data_obj):
        if result is not None:
            return result
    return None

def plugin_custom_event_call_all(event_name, data_obj):
    return list(internal_plugin_custom_event_call(plugins, event_name, data_obj))

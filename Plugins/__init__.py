import os
import traceback
from importlib import util
from pathlib import Path

import settings

class Base:
    """Basic resource class. Concrete resources will inherit from this one
    """
    plugins = []

    # For every class that inherits from the current,
    # the class name will be added to plugins
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.plugins.append(cls)


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

#path = os.path.abspath(__file__)
#dirpath = os.path.dirname(path)
dirpath = str(plugin_path.resolve())

for fname in os.listdir(dirpath):
    # Load only "real modules"
    if not fname.startswith('.') and \
            not fname.startswith('__') and fname.endswith('.py'):
        try:
            load_module(os.path.join(dirpath, fname))
        except Exception:
            traceback.print_exc()

# hook-flash_attn.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Triton & mamba_ssm are pulled by flash_attn
hiddenimports = []
hiddenimports += collect_submodules("flash_attn")
hiddenimports += collect_submodules("triton")
hiddenimports += collect_submodules("mamba_ssm")

# Include .py sources so inspect.getsourcelines can read them
datas = []
datas += collect_data_files("flash_attn", includes=["**/*.py"])
datas += collect_data_files("triton", includes=["**/*.py"])
datas += collect_data_files("mamba_ssm", includes=["**/*.py"])

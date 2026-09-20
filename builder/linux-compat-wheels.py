"""Build labeled Linux wheels with explicit, runtime-tested metadata overrides.

Upstream sources/licenses are unchanged. NeMo's Transformers constraint predates
this application's tested Transformers 5 bridge; AudioTools' protobuf constraint
predates the protobuf version required by current NeMo/TensorBoard. Validation
must include NeMo imports, AudioSignal operations and TensorBoard serialization.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
patched = root / 'patched'
patched.mkdir(exist_ok=True)
packages = {
    'Resemblyzer': ('0.1.4', {
        'Requires-Dist: typing\n': 'Requires-Dist: typing; python_version < "3.5"\n',
    }),
    'descript_audiotools': ('0.7.2', {
        'Requires-Dist: protobuf (<3.20,>=3.9.2)': 'Requires-Dist: protobuf (>=5.29.5,<6)',
    }),
    'nemo_toolkit': ('2.6.0', {
        'Requires-Dist: transformers~=4.53.0;': 'Requires-Dist: transformers==5.14.1;',
        'Requires-Dist: numba-cuda[cu13]>=0.20.0;': 'Requires-Dist: numba-cuda==0.22.1;',
    }),
}
for name, (version, replacements) in packages.items():
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--dest', str(root),
                    f'{name}=={version}'], check=True)
    wheel = next(root.glob(f'{name}-{version}-*.whl'))
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        subprocess.run([sys.executable, '-m', 'wheel', 'unpack', str(wheel), '-d', temporary], check=True)
        unpacked = Path(temporary) / f'{name}-{version}'
        info = unpacked / f'{name}-{version}.dist-info'
        metadata = info / 'METADATA'
        content = metadata.read_text(encoding='utf-8')
        for old, new in replacements.items():
            if old not in content:
                raise RuntimeError(f'Unexpected {name} metadata: missing {old!r}')
            content = content.replace(old, new)
        content = content.replace(f'Version: {version}\n', f'Version: {version}+wtlinux1\n', 1)
        metadata.write_text(content, encoding='utf-8')
        info.rename(unpacked / f'{name}-{version}+wtlinux1.dist-info')
        subprocess.run([sys.executable, '-m', 'wheel', 'pack', str(unpacked), '-d', str(patched)], check=True)

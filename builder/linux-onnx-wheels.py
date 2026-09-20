"""Select ONNX Runtime GPU in dependent wheel metadata without changing code.

onnxruntime and onnxruntime-gpu expose the same Python namespace. Installing
both is order-dependent and can silently replace the GPU runtime with CPU.
Build the exact application source requirements with only that dependency name
changed, keeping their runtime code, version constraints, and licenses intact.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

requirements, root = map(Path, sys.argv[1:])
patched = root / 'patched'
patched.mkdir(parents=True, exist_ok=True)
for name in ('faster-whisper', 'ruaccent'):
    requirement = next(line.strip() for line in requirements.read_text().splitlines()
                       if line.strip().startswith(name + ' @ '))
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        work = Path(temporary)
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                        '--wheel-dir', str(work), requirement], check=True)
        wheel, = work.glob('*.whl')
        subprocess.run([sys.executable, '-m', 'wheel', 'unpack', str(wheel),
                        '-d', str(work)], check=True)
        unpacked, = (path for path in work.iterdir() if path.is_dir())
        info, = unpacked.glob('*.dist-info')
        metadata = info / 'METADATA'
        content, count = re.subn(r'(?m)^(Requires-Dist: )onnxruntime(?=[\s<>=!~;\[]|$)',
                                r'\1onnxruntime-gpu', metadata.read_text())
        if count != 1:
            raise RuntimeError(f'Expected exactly one ONNX dependency in {name}, got {count}')
        version = re.search(r'(?m)^Version: (.+)$', content).group(1)
        new_version = version + '+wtlinux1'
        content = content.replace(f'Version: {version}\n', f'Version: {new_version}\n', 1)
        metadata.write_text(content)
        info.rename(info.with_name(info.name.replace(version + '.dist-info', new_version + '.dist-info')))
        subprocess.run([sys.executable, '-m', 'wheel', 'pack', str(unpacked),
                        '-d', str(patched)], check=True)

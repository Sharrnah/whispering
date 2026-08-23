import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PyInstallerInspectFallbackTests(unittest.TestCase):
    def test_index_tts_sources_are_collected_for_torchscript(self):
        spec = (PROJECT_ROOT / "audioWhisper.spec").read_text(encoding="utf-8")

        self.assertIn("collect_data_files('indextts', include_py_files=True", spec)

    def test_fallback_scripts_function_from_collected_source_file(self):
        script = textwrap.dedent(
            f"""
            import importlib.util
            import inspect
            import sys
            import tempfile
            import types
            from pathlib import Path

            import torch

            with tempfile.TemporaryDirectory() as temp_dir:
                source_path = Path(temp_dir) / "frozen_example.py"
                source = (
                    "import torch\\n\\n"
                    "def add_one(value: torch.Tensor) -> torch.Tensor:\\n"
                    "    return value + 1\\n\\n"
                    "def unrelated():\\n"
                    "    return 'TorchScript must not receive this definition'\\n"
                )
                source_path.write_text(source, encoding="utf-8")

                module = types.ModuleType("frozen_example")
                module.__file__ = str(source_path)
                module.__spec__ = types.SimpleNamespace(loader=object())
                sys.modules[module.__name__] = module
                exec(compile(source, str(source_path), "exec"), module.__dict__)

                def unavailable(_):
                    raise OSError("frozen bytecode has no source")

                inspect.getsourcelines = unavailable
                hook_spec = importlib.util.spec_from_file_location(
                    "test_rt_inspect_fallback",
                    {str(PROJECT_ROOT / 'rthooks' / 'rt_inspect_fallback.py')!r},
                )
                hook = importlib.util.module_from_spec(hook_spec)
                hook_spec.loader.exec_module(hook)

                lines, start_line = inspect.getsourcelines(module.add_one)
                assert start_line == 3, start_line
                recovered = "".join(lines)
                assert "def add_one" in recovered, recovered
                assert "def unrelated" not in recovered, recovered

                scripted = torch.jit.script(module.add_one)
                torch.testing.assert_close(
                    scripted(torch.tensor([1.0])),
                    torch.tensor([2.0]),
                )
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

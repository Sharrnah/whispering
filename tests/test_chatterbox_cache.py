import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cache_harness_class():
    """Compile the two cache methods without importing application plugins."""
    source = (PROJECT_ROOT / "Models" / "TTS" / "chatterbox_tts.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    chatterbox = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Chatterbox"
    )
    methods = [
        node
        for node in chatterbox.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_voice_cache_key", "_ensure_voice_conditionals"}
    ]
    harness = ast.ClassDef(
        name="ChatterboxCacheHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {"os": os}
    exec(compile(module, "chatterbox-cache-methods", "exec"), namespace)
    return namespace["ChatterboxCacheHarness"]


class ChatterboxConditioningCacheTests(unittest.TestCase):
    def test_cache_tracks_exaggeration_and_reference_file_version(self):
        adapter = _cache_harness_class()()
        adapter.compute_device_str = "cpu"
        adapter.voice_conds_cache = {}
        adapter._voice_cache_order = []
        adapter._voice_cache_max_entries = 8
        adapter.model = mock.Mock()

        def prepare_conditionals(path, exaggeration):
            adapter.model.conds = (path, exaggeration)

        adapter.model.prepare_conditionals.side_effect = prepare_conditionals

        with tempfile.TemporaryDirectory() as temp_dir:
            voice = Path(temp_dir) / "voice.wav"
            voice.write_bytes(b"voice")

            adapter._ensure_voice_conditionals(str(voice), exaggeration=0.5)
            first = adapter.model.conds
            adapter._ensure_voice_conditionals(str(voice), exaggeration=0.5)
            self.assertIs(adapter.model.conds, first)
            self.assertEqual(adapter.model.prepare_conditionals.call_count, 1)

            adapter._ensure_voice_conditionals(str(voice), exaggeration=0.8)
            self.assertEqual(adapter.model.prepare_conditionals.call_count, 2)

            voice.write_bytes(b"voice changed")
            adapter._ensure_voice_conditionals(str(voice), exaggeration=0.8)

        self.assertEqual(adapter.model.prepare_conditionals.call_count, 3)
        self.assertEqual(len(adapter.voice_conds_cache), 3)


if __name__ == "__main__":
    unittest.main()

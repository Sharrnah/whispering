import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PluginHostTests(unittest.TestCase):
    def test_timers_run_without_transcripts_and_quit_with_open_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Plugins").mkdir()
            (root / "Plugins" / "ticker.py").write_text('''
import Plugins
from pathlib import Path
class Ticker(Plugins.Base):
    def init(self): self.init_plugin_settings({})
    def timer(self): Path("tick.txt").write_text("timer ran")
    def on_disable(self): Path("closed.txt").write_text("closed")
''', encoding="utf-8")
            config = root / "local.yaml"
            config.write_text("plugins: {Ticker: true}\nplugin_settings: {}\nplugin_timer: 0.05\n", encoding="utf-8")
            process = subprocess.Popen([sys.executable,str(ROOT/"audioWhisper.py"),"--plugin_host","--config",str(config)],
                cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                deadline = time.monotonic() + 10
                while not (root / "tick.txt").exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(process.stderr.read())
                    time.sleep(0.02)
                self.assertTrue((root / "tick.txt").exists(), "idle timer was never dispatched")
                process.stdin.write('{"type":"quit"}\n')
                process.stdin.flush()
                self.assertEqual(process.wait(timeout=5), 0)
                self.assertEqual((root / "closed.txt").read_text(), "closed")
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)

    def test_all_plugin_classes_are_discovered_without_import_or_metadata(self):
        from plugin_host import catalog
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "steamvr_overlay_plugin.py").write_text("raise RuntimeError('must not execute during discovery')\nclass SteamVROverlayPlugin(Plugins.Base): pass\n", encoding="utf-8")
            (root / "inference_plugin.py").write_text("raise RuntimeError('must not import')\nclass InferencePlugin(Plugins.Base): pass\n", encoding="utf-8")
            self.assertEqual(sorted(catalog(root)), ["InferencePlugin", "SteamVROverlayPlugin"])

    def test_bundle_entrypoint_routes_callbacks_and_flushes_on_eof_without_ai(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugins = root / "Plugins"
            plugins.mkdir()
            (plugins / "do_not_import.py").write_text("raise AssertionError('unselected plugin imported')\nclass Unselected(Plugins.Base): pass\n", encoding="utf-8")
            (plugins / "fixture.py").write_text('''
import json, sys
from pathlib import Path
import Plugins
class LocalFixture(Plugins.Base):
    def record(self, event, result=None):
        with Path("events.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"event": event, "result": result, "heavy": [name for name in sys.modules if name.split(".")[0] in ("torch", "transformers", "audioprocessor", "audio_tools", "websocket")]}) + "\\n")
    def init(self):
        self.init_plugin_settings({"color":"white"})
        print("plugin stdout is diagnostic output")
        self.record("init")
    def stt(self, text, result): self.record("final", result)
    def stt_intermediate(self, text, result): self.record("intermediate", result)
    def on_event_received(self, event, websocket_connection=None): self.record("button", event)
    def on_disable(self): self.record("disabled")
''', encoding="utf-8")
            config = root / "local.yaml"
            requests = [
                {"type":"configure", "enabled":["LocalFixture"]},
                {"type":"transcript", "final":False, "result":{"text":"hello", "txt_translation":"Hallo"}},
                {"type":"transcript", "final":True, "result":{"text":"hello", "txt_translation":"Hallo", "language":"en", "audio_source_id":"game"}},
                {"type":"plugin_button_press", "name":"LocalFixture", "value":"test"},
            ]
            result = subprocess.run([sys.executable, str(ROOT/"audioWhisper.py"), "--plugin_host", "--config", str(config)],
                cwd=root, input="".join(json.dumps(request)+"\n" for request in requests), capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            replies = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(replies[0]["available"], ["LocalFixture", "Unselected"])
            self.assertIn("plugin stdout is diagnostic output", result.stderr)
            events = [json.loads(line) for line in (root/"events.jsonl").read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["init","intermediate","final","button","disabled"])
            self.assertEqual(events[2]["result"]["txt_translation"],"Hallo")
            self.assertEqual(events[2]["result"]["audio_source_id"],"game")
            self.assertTrue(all(not event["heavy"] for event in events), events)
            saved = config.read_text()
            self.assertIn("color: white",saved)
            self.assertIn("process_id: 0",saved)
            self.assertFalse((root/"Profiles"/"settings.yaml").exists())

    @unittest.skipUnless((ROOT / "Plugins/steamvr_overlay_plugin.py").is_file(), "SteamVR plugin is not installed")
    def test_real_steamvr_callbacks_load_without_ai_modules(self):
        # Exercise the installed plugin while replacing only its hardware thread.
        code = r'''
import os, sys, tempfile
os.environ["WT_PLUGIN_HOST"]="1"
import settings, Plugins, plugin_host
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    config=Path(directory)/"local.yaml"
    config.write_text("plugins: {}\nplugin_settings: {}\n")
    settings.SETTINGS.load_yaml(str(config))
    settings.SetOption("ui_download",False)
    module=Plugins.load_module("Plugins/steamvr_overlay_plugin.py")
    module.SteamVROverlayPlugin.start_overlay=lambda self: None
    host=plugin_host.Host(settings.SETTINGS,Plugins,"Plugins")
    plugin=module.SteamVROverlayPlugin(init_settings=settings.SETTINGS)
    host.loaded["SteamVROverlayPlugin"]=plugin
    host.configure(["SteamVROverlayPlugin"])
    host.handle({"type":"transcript","final":False,"result":{"text":"Hello","txt_translation":"Hallo"}})
    assert plugin._intermediate == ("Hello","Hallo")
    host.handle({"type":"transcript","final":True,"result":{"text":"Hello","txt_translation":"Hallo"}})
    assert plugin._history[-1] == ("Hello","Hallo")
    assert plugin._intermediate is None
    assert not {"torch","transformers","audio_tools","audioprocessor","websocket"}.intersection(sys.modules)
    host.close()
print("STEAMVR_CALLBACKS_WITHOUT_AI_OK")
'''
        result = subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("STEAMVR_CALLBACKS_WITHOUT_AI_OK",result.stdout)

    def test_unavailable_plugin_is_rejected_without_import(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable,str(ROOT/"audioWhisper.py"),"--plugin_host","--config",str(Path(directory)/"local.yaml")],
                cwd=directory,input='{"type":"configure","enabled":["../bad"]}\n{"type":"quit"}\n',capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout.splitlines()[-1])["type"],"error")

    def test_settings_reset_reinitialize_and_failed_import_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Plugins").mkdir()
            (root / "Plugins" / "simple.py").write_text('''
import Plugins
class Simple(Plugins.Base):
    def init(self): self.init_plugin_settings({"color":"white"})
''', encoding="utf-8")
            (root / "Plugins" / "broken.py").write_text("raise RuntimeError('missing dependency')\nclass Broken(Plugins.Base): pass\n", encoding="utf-8")
            requests = [
                {"type":"configure", "enabled":["Simple"]},
                {"type":"configure", "enabled":["Simple"], "settings":{"Simple":{"color":"blue"}}},
                {"type":"setting_reinit", "name":"plugin", "value":"Simple"},
                {"type":"setting_reset_all", "name":"plugin", "value":"Simple"},
                {"type":"configure", "enabled":["Simple", "Broken"]},
            ]
            result = subprocess.run([sys.executable,str(ROOT/"audioWhisper.py"),"--plugin_host","--config",str(root/"local.yaml")],
                cwd=root,input="".join(json.dumps(value)+"\n" for value in requests),capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            replies = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(replies[2]["settings"]["Simple"]["color"], "blue")
            self.assertEqual(replies[3]["settings"]["Simple"]["color"], "blue")
            self.assertEqual(replies[4]["settings"]["Simple"]["color"], "white")
            self.assertEqual(replies[-2]["enabled"], ["Simple"])
            self.assertIn("missing dependency", replies[-1]["message"])

    @unittest.skipUnless((ROOT / "Plugins/write_transcript_plugin.py").is_file(), "Write Transcript plugin is not installed")
    def test_existing_transcript_plugin_works_without_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Plugins").mkdir()
            (root / "Plugins" / "write_transcript_plugin.py").write_bytes((ROOT / "Plugins/write_transcript_plugin.py").read_bytes())
            requests = [
                {"type":"configure", "enabled":["WriteTranscriptPlugin"]},
                {"type":"transcript", "final":True, "result":{"text":"A normal plugin works locally.", "language":"en"}},
            ]
            result = subprocess.run([sys.executable,str(ROOT/"audioWhisper.py"),"--plugin_host","--config",str(root/"local.yaml")],
                cwd=root,input="".join(json.dumps(value)+"\n" for value in requests),capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("A normal plugin works locally.", (root / "transcript.txt").read_text())


if __name__ == "__main__":
    unittest.main()

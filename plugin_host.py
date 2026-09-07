"""Local integration mode of the existing backend executable.

Private newline-delimited JSON over parent-owned stdio; no network listener.
Only explicitly enabled plugins are imported. Model processing stays remote.
"""
import argparse
import ast
import copy
import json
import os
import queue
import threading
import time
from pathlib import Path
import sys
import traceback

MAX_MESSAGE = 1024 * 1024


def catalog(directory):
    found = {}
    for path in sorted(Path(directory).glob("*.py")):
        if path.name.startswith((".", "__")):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                is_plugin = any(isinstance(base, ast.Attribute) and
                    isinstance(base.value, ast.Name) and base.value.id == "Plugins" and
                    base.attr == "Base" for base in node.bases)
                if is_plugin:
                    if node.name in found:
                        raise ValueError("Duplicate local plugin class: " + node.name)
                    found[node.name] = path
        except (SyntaxError, UnicodeError):
            continue
    return found


class Host:
    def __init__(self, settings, plugins, directory):
        self.settings = settings
        self.plugins = plugins
        self.directory = directory
        self.available = catalog(directory)
        self.loaded = {}

    def state(self):
        return {"type": "state", "available": sorted(self.available),
                "enabled": [name for name, enabled in self.settings.GetOption("plugins").items() if enabled and name in self.available],
                "settings": self.settings.GetOption("plugin_settings")}

    def configure(self, names, values=None):
        if not isinstance(names, list) or any(not isinstance(name, str) or name not in self.available for name in names):
            raise ValueError("Select only installed plugins")
        if values is not None and not isinstance(values, dict):
            raise ValueError("Plugin settings must be an object")
        previously_active = {name for name, instance in self.loaded.items() if instance.is_enabled(False)}
        self.settings.SetOption("plugins", {name: True for name in names})
        for name, instance in self.loaded.items():
            if name not in names and name in previously_active:
                if hasattr(instance, "on_disable"):
                    instance.on_disable()
        if values is not None:
            self.settings.SetOption("plugin_settings", copy.deepcopy(values))
        for name in names:
            try:
                if name not in self.loaded:
                    module = self.plugins.load_module(str(self.available[name]))
                    cls = getattr(module, name)
                    if not issubclass(cls, self.plugins.Base):
                        raise ValueError("Invalid local plugin class: " + name)
                    self.loaded[name] = cls(init_settings=self.settings)
                    self.plugins.plugins.append(self.loaded[name])
                instance = self.loaded[name]
                if name not in previously_active:
                    instance.init()
                    if hasattr(instance, "on_enable"):
                        instance.on_enable()
            except Exception:
                enabled = dict(self.settings.GetOption("plugins"))
                enabled[name] = False
                self.settings.SetOption("plugins", enabled)
                instance = self.loaded.get(name)
                if instance is not None and hasattr(instance, "on_disable"):
                    instance.on_disable()
                self.settings.flush_pending_save()
                raise
        self.settings.flush_pending_save()
        return self.state()

    def handle(self, request):
        kind = request.get("type")
        if kind == "configure":
            return self.configure(request.get("enabled", []), request.get("settings"))
        if kind == "refresh":
            self.available = catalog(self.directory)
            return self.state()
        if kind in ("setting_reset_all", "setting_reinit"):
            instance = self.loaded.get(request.get("value"))
            if instance is None:
                raise ValueError("Enable the plugin before resetting or reinitializing it")
            if kind == "setting_reset_all":
                instance.reset_plugin_all_settings()
            else:
                instance.init()
            self.settings.flush_pending_save()
            return self.state()
        if kind == "state":
            return self.state()
        if kind == "transcript":
            result = request.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                raise ValueError("Invalid transcript")
            method = "stt" if request.get("final") is True else "stt_intermediate"
            for instance in self.loaded.values():
                if instance.is_enabled(False) and hasattr(instance, method):
                    try:
                        getattr(instance, method)(result["text"], copy.deepcopy(result))
                    except Exception:
                        traceback.print_exc()
            return None
        if kind == "plugin_button_press":
            instance = self.loaded.get(request.get("name"))
            if instance and instance.is_enabled(False) and hasattr(instance, "on_event_received"):
                instance.on_event_received(request, None)
            return self.state()
        raise ValueError("Unsupported local plugin command")

    def timer(self):
        for name, instance in self.loaded.items():
            if instance.is_enabled(False) and hasattr(instance, "timer"):
                try:
                    instance.timer()
                except Exception as exc:
                    traceback.print_exc()
                    raise RuntimeError(f"{name}: {exc}") from exc

    def close(self):
        for instance in self.loaded.values():
            try:
                if hasattr(instance, "on_disable"):
                    instance.on_disable()
            except Exception:
                traceback.print_exc()
        self.settings.SetOption("process_id", 0)
        self.settings.flush_pending_save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin_host", action="store_true")
    parser.add_argument("--config", required=True, help="Dedicated local-plugin YAML, never the AI profile")
    args = parser.parse_args(argv)
    output = sys.stdout
    sys.stdout = sys.stderr  # Plugin prints cannot corrupt the protocol stream.
    os.environ["WT_PLUGIN_HOST"] = "1"
    import settings
    config = Path(args.config).resolve()
    config.parent.mkdir(parents=True, exist_ok=True)
    if not config.exists():
        with config.open("x", encoding="utf-8") as file:
            file.write("plugins: {}\nplugin_settings: {}\n")
    settings.SETTINGS.load_yaml(str(config))
    settings.SetOption("process_id", os.getpid())
    settings.SetOption("ui_download", False)
    import Plugins
    host = Host(settings.SETTINGS, Plugins, Path.cwd() / "Plugins")
    def send(value):
        output.write(json.dumps(value, ensure_ascii=True) + "\n")
        output.flush()
    try:
        startup_error = None
        try:
            enabled = [name for name, enabled in settings.GetOption("plugins").items() if enabled and name in host.available]
            host.configure(enabled)
        except Exception as exc:
            traceback.print_exc()
            startup_error = str(exc)
        send(host.state())
        if startup_error:
            send({"type": "error", "message": startup_error})
        incoming = queue.Queue(maxsize=64)
        def read_input():
            # Raw reads avoid holding Python's buffered-stdin lock at interpreter
            # shutdown when the parent sends quit without closing its pipe yet.
            pending = b""
            try:
                while True:
                    chunk = os.read(sys.stdin.fileno(), 65536)
                    if not chunk:
                        if pending:
                            incoming.put(pending)
                        incoming.put(b"")
                        return
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        incoming.put(line + b"\n")
                        if len(line) > MAX_MESSAGE:
                            return
                    if len(pending) > MAX_MESSAGE:
                        incoming.put(pending)
                        return
            except OSError:
                incoming.put(b"")
        threading.Thread(target=read_input, daemon=True).start()
        interval = max(0.05, float(settings.GetOption("plugin_timer")))
        next_tick = time.monotonic() + interval
        while True:
            # Serialize timers with settings and transcript callbacks. Plugins
            # cannot be disabled or reset halfway through a timer callback.
            if time.monotonic() >= next_tick:
                try:
                    host.timer()
                except Exception as exc:
                    send({"type": "error", "message": str(exc)})
                next_tick = time.monotonic() + interval
            try:
                raw = incoming.get(timeout=max(0, next_tick - time.monotonic()))
            except queue.Empty:
                continue
            if not raw:
                break
            if len(raw) > MAX_MESSAGE:
                raise ValueError("Plugin host message too large")
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise ValueError("Expected an object")
                if request.get("type") == "quit":
                    break
                reply = host.handle(request)
                if reply is not None:
                    send(reply)
            except Exception as exc:
                traceback.print_exc()
                send(host.state())
                send({"type": "error", "message": str(exc)})
    finally:
        host.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

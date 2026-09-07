"""Run the Go client transport against the actual Python server (no AI/audio devices)."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_audio import Server


async def main():
    def factory(session, hello):
        class Source:
            def feed(self, pcm):
                assert len(pcm) == 1024
                session.result({"text": "Python to Go"}, True)
            def close(self):
                pass
        return Source()

    server = await Server(factory, lambda session, text: session.audio(b"\1\0\2\0", 24000),
                          "interop-test").start("127.0.0.1", 0)
    try:
        port = server.listener.sockets[0].getsockname()[1]
        env = {**os.environ, "WT_REMOTE_TEST_URL": f"ws://127.0.0.1:{port}"}
        ui = sys.argv[1] if len(sys.argv) > 1 else "G:/Projekte/Repositories/whispering-tiger-ui"
        result = await asyncio.to_thread(subprocess.run,
            ["go", "test", "-count=1", "-run", "TestPythonInterop", "-v", "./RemoteAudio"],
            cwd=ui, env=env, timeout=60)
        if result.returncode:
            raise RuntimeError("Python/Go interoperability failed")
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())

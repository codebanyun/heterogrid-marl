import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from heterogrid import EnvConfig, parallel_env
from heterogrid.io import SnapshotWriter
from heterogrid.viewer import ViewerSession, ViewerServer, make_handler


class ViewerTests(unittest.TestCase):
    def test_controls_and_invalid_configuration_are_atomic(self):
        session = ViewerSession(EnvConfig(max_steps=5))
        session.command({"action": "step"})
        self.assertEqual(session.state()["frame"]["step"], 1)
        with self.assertRaises(ValueError):
            session.command({"action": "configure", "config": {"map_size": [0, 4, 2]}})
        self.assertEqual(session.state()["frame"]["step"], 1)
        session.command({"action": "reset", "seed": 3})
        self.assertEqual(session.state()["frame"]["step"], 0)
        session.command({"action": "play"})
        self.assertTrue(session.running)
        session.command({"action": "pause"})
        self.assertFalse(session.running)
        session.close()

    def test_snapshot_mode_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            session = ViewerSession(snapshot_path=path)
            self.assertIsNone(session.state()["frame"])
            e = parallel_env()
            e.reset(1)
            SnapshotWriter(path).write(e)
            self.assertEqual(session.state()["frame"]["step"], 0)
            with self.assertRaises(ValueError):
                session.command({"action": "step"})
            self.assertEqual(e.core.step_count, 0)
            session.close()

    def test_http_page_state_and_control(self):
        session = ViewerSession(EnvConfig(max_steps=2))
        server = ViewerServer(("127.0.0.1", 0), make_handler(session))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/") as response:
                self.assertIn("三维战场", response.read().decode("utf-8"))
            with urllib.request.urlopen(base + "/api/state") as response:
                self.assertEqual(json.load(response)["frame"]["step"], 0)
            request = urllib.request.Request(base + "/api/control", data=b'{"action":"step"}',
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(json.load(response)["frame"]["step"], 1)
            request.add_header("Origin", "http://unrelated.invalid")
            with self.assertRaises(urllib.error.HTTPError) as result:
                urllib.request.urlopen(request)
            self.assertEqual(result.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            session.close()


if __name__ == "__main__":
    unittest.main()

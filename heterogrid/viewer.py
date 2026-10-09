"""Independent, dependency-free browser viewer and scripted evaluation console."""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import EnvConfig
from .env import HeteroGridParallelEnv
from .policies import ScriptedPolicy, load_policy


class ViewerSession:
    def __init__(self, config=None, seed=42, snapshot_path=None, opponent_factory=None, checkpoint=None):
        self.lock = threading.RLock()
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.running = False
        self.fps = 5.0
        self.seed = seed
        self.last_error = None
        self.config = config or EnvConfig()
        self.opponent_factory, self.checkpoint = opponent_factory, checkpoint
        self.environment = None
        self.policy = None
        if self.snapshot_path is None:
            self._replace_environment(self.config, seed)

    def _replace_environment(self, config, seed):
        opponent = load_policy(self.opponent_factory, config, "blue", self.checkpoint) if config.opponent == "pretrained" and self.opponent_factory else None
        replacement = HeteroGridParallelEnv(config, opponent_policy=opponent)
        obs, info = replacement.reset(seed=seed)
        policy = ScriptedPolicy(config, "red", "rule")
        policy.reset(seed)
        old = self.environment
        self.environment, self.policy = replacement, policy
        self.obs, self.info = obs, info
        self.config, self.seed = config, seed
        self.running = False
        self.last_error = None
        if old is not None:
            old.close()

    def step(self):
        with self.lock:
            if self.snapshot_path is not None:
                raise ValueError("Snapshot monitoring is read-only")
            if not self.environment.agents:
                self.running = False
                return
            self.obs, self.info = self.environment.share_visible_targets()
            obs, _, _, _, info = self.environment.step(self.policy.act(self.obs, self.info))
            self.obs = {n: obs[n] for n in self.environment.agents}
            self.info = {n: info[n] for n in self.environment.agents}
            if not self.environment.agents:
                self.running = False

    def state(self):
        with self.lock:
            if self.snapshot_path is not None:
                try:
                    frame = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
                    if not isinstance(frame, dict) or frame.get("schema_version") != 1 or "units" not in frame:
                        raise ValueError("Not a HeteroGrid snapshot")
                    return {"frame": frame, "mode": "snapshot", "running": False, "error": None}
                except (OSError, ValueError) as exc:
                    return {"frame": None, "mode": "snapshot", "running": False,
                            "error": f"Waiting for a valid snapshot: {exc}"}
            return {"frame": self.environment.snapshot(), "mode": "demo", "running": self.running,
                    "config": self.config.to_dict(), "seed": self.seed, "fps": self.fps, "error": self.last_error}

    def command(self, data):
        with self.lock:
            if self.snapshot_path is not None:
                raise ValueError("Snapshot monitoring is read-only; control the evaluation process separately")
            action = data.get("action")
            if action == "play":
                if not self.environment.agents:
                    raise ValueError("Episode has ended; reset first")
                self.running = True
            elif action == "pause":
                self.running = False
            elif action == "step":
                self.running = False
                self.step()
            elif action == "reset":
                seed = data.get("seed", self.seed)
                if type(seed) is not int or seed < 0:
                    raise ValueError("seed must be a nonnegative integer")
                self._replace_environment(self.config, seed)
            elif action == "configure":
                seed = data.get("seed", self.seed)
                if type(seed) is not int or seed < 0:
                    raise ValueError("seed must be a nonnegative integer")
                config = EnvConfig.from_dict(data["config"])
                # Limit only the interactive viewer, not the headless environment.
                if len(config.red) + len(config.blue) > 200:
                    raise ValueError("Viewer supports at most 200 units")
                self._replace_environment(config, seed)
            elif action == "speed":
                fps = float(data["fps"])
                if not math.isfinite(fps) or not 0.5 <= fps <= 30:
                    raise ValueError("fps must be between 0.5 and 30")
                self.fps = fps
            else:
                raise ValueError("Unknown control action")
        return self.state()

    def close(self):
        with self.lock:
            self.running = False
            if self.environment is not None:
                self.environment.close()


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _reply(self, status, payload, content_type="application/json; charset=utf-8"):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self._reply(200, (Path(__file__).parent / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path.split("?")[0] == "/api/state":
                self._reply(200, session.state())
            else:
                self._reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/control":
                self._reply(404, {"error": "Not found"})
                return
            # Reject browser cross-origin writes; no permissive CORS headers.
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://{self.headers.get('Host')}", f"https://{self.headers.get('Host')}"):
                self._reply(403, {"error": "Cross-origin control is disabled"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 100_000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Control payload must be an object")
                self._reply(200, session.command(data))
            except (ValueError, TypeError, KeyError) as exc:
                self._reply(400, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot", help="Read-only latest-frame JSON published by another process")
    parser.add_argument("--opponent-factory")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    config = EnvConfig.load(args.config) if args.config else EnvConfig()
    session = ViewerSession(config, args.seed, args.snapshot, args.opponent_factory, args.checkpoint)
    stop = threading.Event()

    def simulate():
        while not stop.is_set():
            started = time.monotonic()
            with session.lock:
                if session.running:
                    try:
                        session.step()
                    except Exception as exc:
                        session.running = False
                        session.last_error = f"Simulation stopped: {type(exc).__name__}: {exc}"
                interval = 1 / session.fps
            stop.wait(max(0.01, interval - (time.monotonic() - started)))

    worker = threading.Thread(target=simulate, name="viewer-evaluation", daemon=True)
    worker.start()
    server = ViewerServer((args.host, args.port), make_handler(session))
    print(f"HeteroGrid viewer: http://{args.host}:{server.server_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=2)
        session.close()


if __name__ == "__main__":
    main()

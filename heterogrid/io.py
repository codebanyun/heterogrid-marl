"""Opt-in snapshots; no filesystem/network activity in reset()/step() by default."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class SnapshotWriter:
    """Publish an atomic latest-frame JSON at an explicit chosen frequency.

    One writer per path. A separate browser server reads it. Prefer publishing
    from an independent evaluation process rather than every training environment.
    """
    def __init__(self, path, every=10):
        if type(every) is not int or every < 1:
            raise ValueError("every must be a positive integer")
        self.path = Path(path)
        self.every = every
        self.calls = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, env, *, force=False):
        self.calls += 1
        if not force and (self.calls - 1) % self.every:
            return False
        frame = env.snapshot()
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(frame, stream, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return True

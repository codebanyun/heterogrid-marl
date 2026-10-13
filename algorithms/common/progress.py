"""Budget alignment and safe progress reads for detached experiments."""
import json


def align_steps(steps, batch=512):
    return ((steps + batch - 1) // batch) * batch


def latest_progress(path):
    try:
        with (path / "metrics.jsonl").open("rb") as file:
            file.seek(0, 2)
            file.seek(max(0, file.tell() - 16384))
            lines = file.read().splitlines()
        # A process may be part-way through appending its newest line.
        for line in reversed(lines):
            try:
                row = json.loads(line)
                return {k: row[k] for k in ("environment_steps", "episodes", "recent_return", "recent_win_rate")}
            except (ValueError, KeyError):
                continue
    except FileNotFoundError:
        pass
    return {"environment_steps": 0, "episodes": 0}

"""Lossless streaming episode records, optionally compressed without duplicates."""
import gzip
import json
from collections import deque
from pathlib import Path


def open_episode_log(output, compressed=False):
    path = Path(output) / ("training_episodes.jsonl.gz" if compressed else "training_episodes.jsonl")
    if compressed:
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=1)
    return path.open("w", encoding="utf-8")


def episode_rows(output):
    output = Path(output)
    compressed = output / "training_episodes.jsonl.gz"
    opener, path = (gzip.open, compressed) if compressed.exists() else (open, output / "training_episodes.jsonl")
    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def recent_buffer(compact):
    return deque(maxlen=32) if compact else []

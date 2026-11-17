"""Compact finished runs; dry-run unless --apply is explicitly supplied."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path


def digest(path, compressed=False):
    result = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def active_outputs():
    outputs = set()
    for file in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = file.read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        for index, arg in enumerate(args):
            if arg == "--output" and index + 1 < len(args):
                outputs.add(str(Path(args[index + 1]).resolve()))
            elif arg.startswith("--output="):
                outputs.add(str(Path(arg.partition("=")[2]).resolve()))
    return outputs


def replay_selection(run):
    keep = set()
    for choices in (("heldout_latest_greedy", "greedy", "after"), ("heldout_best_greedy",)):
        for name in choices:
            episodes = sorted((run / "replays" / name).glob("episode_*.json"))
            if not episodes:
                continue
            episode = episodes[0]
            value = json.loads(episode.read_text())
            assert value["schema_version"] == 1
            assert len(value["frames"]) == len(value["transitions"]) + 1
            assert value["transitions"][-1]["done"]
            keep.add(episode)
            actions = episode.with_name(episode.name.replace("episode_", "actions_", 1))
            if actions.exists():
                json.loads(actions.read_text())
                keep.add(actions)
            break
    return keep


def plan(run, include_interrupted=False):
    metadata = json.loads((run / "run.json").read_text())
    allowed = {"COMPLETED", "INTERRUPTED"} if include_interrupted else {"COMPLETED"}
    if metadata["status"] not in allowed or str(run.resolve()) in active_outputs():
        return None
    if any(file.is_symlink() for file in run.rglob("*")):
        raise ValueError("Refusing a run containing symbolic links: " + str(run))
    models = run / "models"
    if not all((models / name).is_file() for name in ("best.pt", "latest.pt")):
        return None
    keep_replays = replay_selection(run)
    remove = [f for f in models.glob("*.pt") if f.name not in {"best.pt", "latest.pt"}]
    remove += [f for f in (run / "replays").rglob("*.json")
               if f.name.startswith(("episode_", "actions_")) and f not in keep_replays]
    log = run / "training_episodes.jsonl"
    duplicate = run / "training_episodes.json"
    if duplicate.exists() and (log.exists() or (run / "training_episodes.jsonl.gz").exists()):
        remove.append(duplicate)
    protected = [f for f in run.iterdir() if f.is_file() and f.suffix == ".json"
                 and f.name not in {"training_episodes.json", "retention.json", "audit.json"}]
    protected += [models / "best.pt", models / "latest.pt", *keep_replays]
    if not log.exists() and (run / "training_episodes.jsonl.gz").exists():
        protected.append(run / "training_episodes.jsonl.gz")
    return {"run": run, "state": metadata["status"], "remove": sorted(remove),
            "compress": log if log.exists() else None, "protected": protected}


def compact(item):
    run = item["run"]
    before = plan(run, include_interrupted=item["state"] == "INTERRUPTED")
    if before is None or before["remove"] != item["remove"]:
        raise ValueError("Run changed after planning: " + str(run))
    protected = {str(f.relative_to(run)): digest(f) for f in item["protected"]}
    original_bytes = sum(f.stat().st_blocks * 512 for f in item["remove"])
    log = item["compress"]
    log_report = None
    if log:
        target = log.with_suffix(log.suffix + ".gz")
        source_hash = digest(log)
        if target.exists():
            if digest(target, compressed=True) != source_hash:
                raise ValueError("Compressed log differs from original: " + str(run))
        else:
            temporary = target.with_name(target.name + ".compacting")
            try:
                with log.open("rb") as source, gzip.open(temporary, "wb", compresslevel=6) as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                if digest(temporary, compressed=True) != source_hash or digest(log) != source_hash:
                    raise ValueError("Episode log verification failed: " + str(run))
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        original_bytes += log.stat().st_blocks * 512 - target.stat().st_blocks * 512
        log_report = {"file": target.name, "uncompressed_sha256": source_hash}
    canonical = run / "training_episodes.jsonl.gz"
    if canonical.exists() and log:
        count = steps = 0
        with gzip.open(canonical, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                steps += row["length"]
        summary = run / "summary.json"
        if summary.exists():
            expected = json.loads(summary.read_text())
            assert count == expected["training_episodes"], (run, count)
        if log_report is None:
            log_report = {"file": canonical.name, "uncompressed_sha256": digest(canonical, compressed=True)}
        log_report.update(episodes=count, episode_steps=steps)
    elif canonical.exists():
        log_report = {"file": canonical.name, "compressed_sha256": digest(canonical),
                      "preserved_without_modification": True}
    removed = [{"path": str(f.relative_to(run)), "bytes": f.stat().st_size} for f in item["remove"]]
    if log:
        removed.append({"path": log.name, "bytes": log.stat().st_size, "reason": "lossless gzip"})
    report = {"cleaned_at": datetime.now(timezone.utc).isoformat(),
              "state_at_cleanup": item["state"], "models": ["models/latest.pt", "models/best.pt"],
              "protected_sha256": protected, "episode_log": log_report, "removed": removed,
              "estimated_freed_bytes": original_bytes,
              "historical_audit_note": "Existing audit.json records pre-cleanup artifact counts; rerun audit for current counts."}
    temporary = run / "retention.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, run / "retention.json")
    for file in item["remove"]:
        file.unlink()
    if log:
        log.unlink()
    for file, expected in protected.items():
        assert digest(run / file) == expected, (run, file)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-interrupted", action="store_true")
    cli = parser.parse_args()
    root = cli.root.resolve()
    items = []
    for models in sorted(root.rglob("models")):
        run = models.parent
        if not models.is_dir() or not (run / "run.json").exists():
            continue
        item = plan(run, cli.include_interrupted)
        if item and (item["remove"] or item["compress"]):
            items.append(item)
    print(json.dumps({"apply": cli.apply, "runs": len(items),
                      "delete_files": sum(len(x["remove"]) for x in items),
                      "compress_logs": sum(x["compress"] is not None for x in items),
                      "delete_bytes": sum(f.stat().st_blocks * 512 for x in items for f in x["remove"])}), flush=True)
    if cli.apply:
        for item in items:
            report = compact(item)
            print(json.dumps({"run": str(item["run"].relative_to(root)),
                              "freed_bytes": report["estimated_freed_bytes"]}), flush=True)


if __name__ == "__main__":
    main()

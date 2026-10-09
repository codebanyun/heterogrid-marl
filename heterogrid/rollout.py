"""Run scripted rollouts and export replays/snapshots; this does not train a network."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from .config import EnvConfig
from .env import HeteroGridParallelEnv
from .io import SnapshotWriter
from .policies import ScriptedPolicy, load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--red-policy", choices=["random", "rule", "static"], default="rule")
    parser.add_argument("--opponent", choices=["dummy", "static", "random", "rule", "pretrained"])
    parser.add_argument("--opponent-factory", help="module:factory for a saved actor")
    parser.add_argument("--checkpoint")
    parser.add_argument("--communication", choices=["none", "full", "distance", "random"])
    parser.add_argument("--replay-dir")
    parser.add_argument("--snapshot", help="Latest-frame JSON for an independent viewer")
    parser.add_argument("--publish-every", type=int, default=10)
    parser.add_argument("--fps", type=float, default=0, help="0=unlimited; >0 only for live demonstrations")
    args = parser.parse_args()
    if args.episodes < 1 or args.fps < 0:
        parser.error("episodes must be positive and fps nonnegative")
    config = EnvConfig.load(args.config) if args.config else EnvConfig()
    overrides = {k: getattr(args, k) for k in ("opponent", "communication") if getattr(args, k) is not None}
    config = replace(config, **overrides)
    opponent = load_policy(args.opponent_factory, config, "blue", args.checkpoint) if args.opponent_factory else None
    environment = HeteroGridParallelEnv(config, opponent_policy=opponent, record_replay=bool(args.replay_dir))
    policy = ScriptedPolicy(config, "red", args.red_policy)
    publisher = SnapshotWriter(args.snapshot, args.publish_every) if args.snapshot else None
    counts = {"red": 0, "blue": 0, "draw": 0}
    start = time.perf_counter()
    steps = 0
    for episode in range(args.episodes):
        obs, info = environment.reset(seed=args.seed + episode)
        policy.reset(seed=args.seed + episode)
        if publisher:
            publisher.write(environment, force=True)
        while environment.agents:
            # This is an explicitly selected broadcast baseline, not learned communication.
            obs, info = environment.share_visible_targets()
            obs, rewards, terminated, truncated, info = environment.step(policy.act(obs, info))
            obs = {n: obs[n] for n in environment.agents}
            info = {n: info[n] for n in environment.agents}
            steps += 1
            if publisher:
                publisher.write(environment, force=not environment.agents)
            if args.fps:
                time.sleep(1 / args.fps)
        winner = environment.core.winner or "draw"
        counts[winner] += 1
        print(json.dumps({"episode": episode, "seed": args.seed + episode, "steps": environment.core.step_count,
                          "winner": winner, "reason": environment.core.reason,
                          "returns": environment.core.episode_returns}, ensure_ascii=False))
        if args.replay_dir:
            environment.save_replay(Path(args.replay_dir) / f"episode_{episode:03d}.json")
    elapsed = time.perf_counter() - start
    print(json.dumps({"summary": counts, "environment_steps": steps, "seconds": round(elapsed, 3),
                      "steps_per_second": round(steps / max(elapsed, 1e-9), 1)}, ensure_ascii=False))
    environment.close()


if __name__ == "__main__":
    main()

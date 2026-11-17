"""One-shot, audited replay-kernel change at an existing learner boundary.

No policy/learner/optimizer is reconstructed, and no process is restarted.
arm() only replaces the train module's next global learn call; an update
already in progress continues with its old code. Deployment is opt-in through
a coordinator manifest and a separately tested CPython pending call.
"""
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from algorithms.common.experiment_log import write_atomic


VERSION = "equivalent_replay_v3"


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            visit(item.detach().cpu().contiguous().numpy())
        elif isinstance(item, np.ndarray):
            h.update(str(item.dtype).encode()); h.update(repr(item.shape).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            h.update((type(item).__name__ + ":" + repr(item)).encode())
        h.update(b"\0")
    visit(value)
    return h.hexdigest()


def find_frame(name, suffix):
    for frame in sys._current_frames().values():
        while frame is not None:
            if frame.f_code.co_name == name and frame.f_code.co_filename.endswith(suffix):
                return frame
            frame = frame.f_back
    raise RuntimeError("Expected training/manager frame not found")


def source_check(manifest):
    runtime = Path(manifest["runtime"])
    for relative, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((runtime / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Optimized source changed after freeze: " + relative)


def load_kernel(runtime, name):
    fullname = "algorithms.dib_mappo." + name
    path = Path(runtime) / "algorithms/dib_mappo" / (name + ".py")
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    setattr(sys.modules["algorithms.dib_mappo"], name, module)
    spec.loader.exec_module(module)
    return module


def audit(policy, batch, frame):
    values = frame.f_locals
    models = (policy.mac.agent, policy.learner.old_mac.agent,
              policy.learner.critic, policy.learner.target_critic)
    return {
        "networks": digest([model.state_dict() for model in models]),
        "optimizers": digest([policy.learner.agent_optimiser.state_dict(),
                              policy.learner.critic_optimiser.state_dict()]),
        "parameter_identity": digest([(id(p), p.requires_grad) for model in models for p in model.parameters()]),
        "optimizer_identity": [id(policy.learner.agent_optimiser), id(policy.learner.critic_optimiser)],
        "rng": digest((random.getstate(), np.random.get_state(), torch.get_rng_state(),
                        values["rng"].bit_generator.state)),
        "batch": digest(batch.data.transition_data),
        "collection": digest((values["hidden"], values["observations"], values["available"],
                              values["histories"], values["totals"], values["components"],
                              values["episode_seeds"], sorted(values["active"]))),
        "environment_identity": [id(env) for env in values["envs"]],
        "counters": {key: values[key] for key in ("steps", "updates", "completed", "episode_index", "learned_steps")},
        "learner_counters": [policy.learner.critic_training_steps, policy.learner.last_target_update_step,
                             policy.learner.log_stats_t],
        "gates": [policy.mac.agent.residual_actor_gate, policy.learner.old_mac.agent.residual_actor_gate],
        "batch_start_step": policy.batch_start_step,
        "threads": torch.get_num_threads(),
    }


def arm_worker(manifest):
    frame = find_frame("main", "/algorithms/dib_mappo/train.py")
    namespace = frame.f_globals
    values = frame.f_locals
    output = Path(values["cli"].output).resolve()
    allowed = {str(Path(item["output"]).resolve()) for item in manifest["workers"]}
    if str(output) not in allowed:
        raise RuntimeError("Worker output is outside the deployment manifest")
    marker = output / "optimization_switch.json"
    existing = json.loads(marker.read_text()) if marker.exists() else {}
    if existing.get("version") == VERSION and existing.get("status") in ("ARMED", "APPLIED", "VERIFIED"):
        return
    if values.get("interrupted") or values.get("phase") not in ("TRAINING", "VALIDATION"):
        raise RuntimeError("Worker is not in an eligible active training phase")
    original_learn = namespace["learn"]
    original_write = namespace["write_atomic"]
    record = {"version": VERSION, "status": "ARMED", "pid": os.getpid(), "armed": now(),
              "runtime": manifest["runtime"], "source_sha256": manifest["source_sha256"],
              "method": "same-process next-complete-batch boundary; no checkpoint reload or restart"}
    write_atomic(marker, record)

    def boundary(policy, batch, clock, episodes):
        saved = {}
        for label, mac in (("mac", policy.mac), ("old_mac", policy.learner.old_mac)):
            if getattr(mac, "_replay_enabled", False):
                raise RuntimeError("Replay was still active at the migration boundary")
            saved[label] = (mac.init_hidden, mac.forward, mac.compute_auxiliary_losses)
        saved["teacher"] = policy.mac.agent._compute_teacher_losses
        saved["mhf"] = policy.mac.agent._compute_mhf_losses
        saved["returns"] = policy.learner.nstep_returns
        before = audit(policy, batch, frame)
        if frame.f_locals["active"]:
            raise RuntimeError("Migration must occur after all fresh collection episodes complete")
        try:
            source_check(manifest)
            auxiliary = load_kernel(manifest["runtime"], "auxiliary")
            returns = load_kernel(manifest["runtime"], "returns")
            sequence = load_kernel(manifest["runtime"], "sequence")
            sequence.install(policy.mac)
            sequence.install(policy.learner.old_mac)
            auxiliary.install(policy.mac.agent)
            policy.learner.nstep_returns = lambda rewards, mask, target, nsteps: returns.nstep_returns(
                policy.args, rewards, mask, target, nsteps)
            after = audit(policy, batch, frame)
            if before != after:
                raise RuntimeError("Training-state audit changed during kernel installation")
        except BaseException:
            for label, mac in (("mac", policy.mac), ("old_mac", policy.learner.old_mac)):
                mac.init_hidden, mac.forward, mac.compute_auxiliary_losses = saved[label]
            policy.mac.agent._compute_teacher_losses = saved["teacher"]
            policy.mac.agent._compute_mhf_losses = saved["mhf"]
            policy.learner.nstep_returns = saved["returns"]
            namespace["learn"] = original_learn
            record.update(status="ROLLED_BACK", error=traceback.format_exc(), failed=now())
            write_atomic(marker, record)
            if audit(policy, batch, frame) != before:
                raise RuntimeError("State audit failed after kernel rollback")
            return original_learn(policy, batch, clock, episodes)
        record.update(status="APPLIED", applied=now(), environment_steps=frame.f_locals["steps"],
                      collection_batch_start=clock, completed_episodes=episodes,
                      state_before=before, state_after=after, exact_state_preserved=True)
        write_atomic(marker, record)
        def versioned_write(path, data):
            if Path(path) == output / "progress.json":
                data = {**data, "replay_version": VERSION,
                        "optimization_switch_steps": record["environment_steps"]}
            return original_write(path, data)
        namespace["write_atomic"] = versioned_write
        start = time.monotonic()
        result = sequence.learn(policy, batch, clock, episodes)
        metrics = policy.logger.values
        if metrics.get("agent_update_skipped", 0) or any(not math.isfinite(float(v)) for v in metrics.values()):
            raise FloatingPointError("Non-finite first optimized update")
        if policy.learner.critic_training_steps != before["learner_counters"][0] + 1:
            raise RuntimeError("Unexpected learner counter after first optimized update")
        record.update(status="VERIFIED", verified=now(), first_update_seconds=time.monotonic() - start,
                      first_update_metrics=dict(metrics), learner_counter_after=policy.learner.critic_training_steps)
        write_atomic(marker, record)
        namespace["learn"] = sequence.learn
        print(json.dumps({"optimization_switch": VERSION, "status": "VERIFIED",
                          "environment_steps": record["environment_steps"],
                          "first_update_seconds": record["first_update_seconds"]}), flush=True)
        return result
    namespace["learn"] = boundary


def arm_manager(manifest, coordinator):
    frame = find_frame("supervise", "/algorithms/common/value_decomposition/campaign.py")
    plan = frame.f_locals["plan"]
    if Path(frame.f_locals["coordinator"]).resolve() != coordinator.resolve():
        raise RuntimeError("Manager coordinator does not match deployment")
    pending = verified = 0
    for job in plan["jobs"]:
        if job["algorithm"] != "dib_mappo":
            raise RuntimeError("Only the DIB campaign can be upgraded")
        if job["status"] == "PENDING":
            job["runtime"] = manifest["runtime"]
            job["replay_version"] = VERSION
            pending += 1
        else:
            marker = Path(job["output"]) / "optimization_switch.json"
            ack = json.loads(marker.read_text()) if marker.exists() else {}
            if ack.get("status") == "VERIFIED" and ack.get("version") == VERSION:
                job["effective_replay_runtime"] = manifest["runtime"]
                job["replay_version"] = VERSION
                job["optimization_switch_steps"] = ack["environment_steps"]
                verified += 1
    plan["optimization_upgrade"] = {"version": VERSION, "runtime": manifest["runtime"],
        "pending_routed": pending, "live_verified": verified, "updated": now(),
        "method": "same-process complete-batch boundary; original runtime retained"}
    # Do not re-enter persist(): the pending callback may have interrupted its
    # atomic write. Its normal next iteration persists the changed plan.
    write_atomic(coordinator / "optimization_manager_ack.json", {**plan["optimization_upgrade"],
        "pid": os.getpid(), "status": "APPLIED"})


def arm():
    coordinator = Path(__file__).resolve().parents[3]
    manifest = json.loads((coordinator / "optimization_upgrade.json").read_text())
    source_check(manifest)
    if "--run" in sys.argv:
        arm_manager(manifest, coordinator)
    elif "--output" in sys.argv:
        arm_worker(manifest)
    else:
        raise RuntimeError("Not an allowlisted DIB worker or manager")


if __name__ == "__dib_live_upgrade__":
    try:
        arm()
    except BaseException:
        coordinator = Path(__file__).resolve().parents[3]
        write_atomic(coordinator / ("optimization_arm_error_" + str(os.getpid()) + ".json"),
                     {"pid": os.getpid(), "error": traceback.format_exc(), "time": now()})
        traceback.print_exc()

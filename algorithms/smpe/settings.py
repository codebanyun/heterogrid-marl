"""Portable external-backend discovery and native SMPE hyperparameters."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/scenarios/6v6_rule_distance.json"


def add_backend_argument(parser):
    parser.add_argument("--smpe-root", type=Path, default=os.environ.get("SMPE_ROOT"))


def validate_backend(parser, root):
    if root is None:
        parser.error("Set SMPE_ROOT or pass --smpe-root")
    root = Path(root).expanduser().resolve()
    for name in ("learners/actor_critic_learner.py", "model_learners/vae_controller.py",
                 "config/default.yaml", "config/algs/smpe_lbf.yaml"):
        if not (root / name).is_file():
            parser.error("Incomplete SMPE backend: " + str(root / name))
    return root


def hyperparameters(root, device):
    import yaml
    with (root / "config/default.yaml").open() as file:
        settings = yaml.safe_load(file)
    with (root / "config/algs/smpe_lbf.yaml").open() as file:
        settings.update(yaml.safe_load(file))
    settings.update(
        env="sc2", game="heterogrid", device=device, use_cuda=device == "cuda",
        add_value_last_step=False,
        # The native VAE otherwise writes all concurrent runs into the same saves/.
        save_period=2 ** 60,
    )
    if (settings["batch_size_run"] != settings["batch_size"] or settings["buffer_size"] != settings["batch_size"]
            or not all(settings[name] for name in ("use_dynamics", "use_w", "use_intrinsic", "use_z_inputs", "use_w_critic"))):
        raise ValueError("The SMPE LBF preset must use complete on-policy batches and all SMPE components")
    return settings

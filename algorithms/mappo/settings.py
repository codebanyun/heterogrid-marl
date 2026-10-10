"""Portable defaults for source-checkout MAPPO entry points."""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/scenarios/6v6_rule_distance.json"


def add_backend_argument(parser):
    parser.add_argument("--mappo-root", type=Path, default=os.environ.get("MAPPO_ROOT"),
                        help="MAPPO checkout containing onpolicy/; defaults to MAPPO_ROOT")


def validate_backend(parser, path):
    if path is None:
        parser.error("Provide --mappo-root or set MAPPO_ROOT to an external MAPPO checkout")
    path = Path(path).expanduser().resolve()
    if not (path / "onpolicy/config.py").is_file():
        parser.error("--mappo-root must contain onpolicy/config.py")
    return path

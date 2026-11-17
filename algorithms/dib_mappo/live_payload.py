"""Small ASCII bootstrap scheduled through CPython's pending-call queue."""
import sys
import traceback
from pathlib import Path

try:
    _coordinator = Path(sys.modules["__main__"].__file__).resolve().parents[3]
    _manifest = __import__("json").loads((_coordinator / "optimization_upgrade.json").read_text())
    _source = Path(_manifest["runtime"]) / "algorithms/dib_mappo/live_upgrade.py"
    _scope = {"__name__": "__dib_live_upgrade__", "__file__": str(_source)}
    exec(compile(_source.read_text(), str(_source), "exec"), _scope)
except BaseException:
    traceback.print_exc()

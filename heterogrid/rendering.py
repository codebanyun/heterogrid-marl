"""Optional offscreen renderer; never imported on the headless training path."""
import numpy as np


def render_rgb(frame):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise ImportError("rgb_array rendering needs: pip install 'heterogrid-marl[render]'") from exc
    fig = Figure(figsize=(7, 5), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    for unit in frame["units"]:
        if not unit["alive"]:
            continue
        ax.scatter(*unit["position"], color="#ef4444" if unit["team"] == "red" else "#3b82f6",
                   marker={"scout": "^", "strike": "s", "hybrid": "o"}[unit["role"]], s=70)
        ax.text(*unit["position"], unit["id"], fontsize=7)
    for attack in frame["attacks"]:
        points = np.asarray([attack["origin"], attack["aim"]])
        ax.plot(*points.T, color="#f59e0b", linewidth=1)
    x, y, z = frame["map_size"]
    ax.set(xlim=(0, max(1, x - 1)), ylim=(0, max(1, y - 1)), zlim=(0, max(1, z - 1)),
           xlabel="X", ylabel="Y", zlabel="Z", title=f"HeteroGrid | step {frame['step']} | {frame['winner'] or 'ongoing'}")
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

"""Display small regret on an expanded linear segment without modifying data."""
from __future__ import annotations

import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter


def expand_lower_regret(ax, threshold=0.05, fraction=0.65):
    """Map [0, threshold] to fraction of the positive plotting height.

    The upper segment remains linear too; all observations remain visible.
    A dotted guide marks the change in scale, not a missing interval.
    """
    if threshold <= 0 or not 0 < fraction < 1:
        raise ValueError("Require threshold > 0 and 0 < fraction < 1")
    _, upper = ax.get_ylim()
    if upper <= threshold:
        return
    old_ticks = ax.get_yticks()

    def forward(y):
        y = np.asarray(y)
        return np.where(y <= threshold, fraction * y / threshold,
                        fraction + (1 - fraction) * (y - threshold) / (upper - threshold))

    def inverse(z):
        z = np.asarray(z)
        return np.where(z <= fraction, threshold * z / fraction,
                        threshold + (upper - threshold) * (z - fraction) / (1 - fraction))

    ax.set_yscale("function", functions=(forward, inverse))
    # Keep a small inset below the genuine zero, with no offset added to data.
    ax.set_ylim(float(inverse(-0.02)), upper)
    upper_ticks = [float(t) for t in old_ticks if threshold < t <= upper]
    top_tick = max(upper_ticks) if upper_ticks else upper
    # Place the middle tick near the visual midpoint, using a readable value.
    middle = float(inverse(float(forward(top_tick)) / 2))
    step = threshold / 10
    middle = round(middle / step) * step
    ax.yaxis.set_major_locator(FixedLocator([0, middle, top_tick]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.axhline(threshold, color="0.4", linewidth=0.7,
               linestyle=(0, (2, 3)), alpha=0.65, zorder=1)
    # Mark the scale transition on both spines, without removing any interval.
    slash_marker = Path(
        [(-1, -0.8), (1, 0.0), (-1, 0.0), (1, 0.8)],
        [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO],
    )
    marks, = ax.plot(
        [0, 1], [threshold, threshold], transform=ax.get_yaxis_transform(),
        linestyle="none", marker=slash_marker, markersize=10,
        markeredgecolor="black", markeredgewidth=1.1, markerfacecolor="none",
        clip_on=False, zorder=10,
    )
    marks.set_path_effects([
        path_effects.Stroke(linewidth=3.5, foreground=ax.get_facecolor()),
        path_effects.Normal(),
    ])

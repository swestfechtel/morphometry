#!/usr/bin/env python3
"""Illustrate the golden-angle ray layout used by the subchondral-distance cone.

Writes golden_angle.png (four panels) and golden_angle.gif (the rays placed one at
a time) next to this script. Every ray position comes from the measurement's own
morphometry.measurements.hip._sample_cone_directions, so the figure shows the real
layout rather than an idealised one.

Everything is drawn looking straight down the cone axis: a ray tilted theta from the axis
at azimuth phi lands at sin(theta) * (cos phi, sin phi), so the 45 degree cone rim is a
circle of radius sin 45. Needs no sample data.

    python docs/illustrations/plot_golden_angle.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arc, Circle
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # repo root
from morphometry.measurements.hip import _sample_cone_directions

OUT = str(HERE)
N, CONE = 200, 45.0
GOLDEN = np.degrees(np.pi * (3 - np.sqrt(5)))          # 137.507...
d = _sample_cone_directions(np.array([0, 0, 1.0]), N, CONE)
tilt = np.degrees(np.arccos(d[:, 2]))
xy = d[:, :2]                                           # top-down view of each ray
RIM = np.sin(np.radians(CONE))
SLAB = 1.5 / 30.0          # a ray within 1.5 mm of a slice at ~30 mm out, in these units
cmap = plt.get_cmap("viridis")


def rim(ax, label=True):
    ax.add_patch(Circle((0, 0), RIM, fill=False, lw=1.5, ec="0.35"))
    if label:
        ax.text(RIM * 0.72, RIM * 0.75, "cone rim\n(45°)", color="0.35", fontsize=8)
    ax.plot(0, 0, "+", color="k", ms=10, mew=1.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    lim = RIM * 1.12
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)


fig, axes = plt.subplots(2, 2, figsize=(13, 13))

# --- A: the turn alone, tilt ignored -------------------------------------------------
ax = axes[0, 0]
ax.add_patch(Circle((0, 0), 1, fill=False, lw=1, ec="0.7"))
K = 13
for i in range(K):
    a = np.radians(i * GOLDEN)
    ax.plot([0, np.cos(a)], [0, np.sin(a)], color=cmap(i / K), lw=1.2, alpha=0.6)
    ax.plot(np.cos(a), np.sin(a), "o", ms=17, color=cmap(i / K))
    ax.text(np.cos(a), np.sin(a), str(i), color="white", ha="center", va="center",
            fontsize=9, fontweight="bold")
for i, rad in ((0, 0.34), (1, 0.50), (2, 0.66)):          # the first three steps, as arcs
    start = (i * GOLDEN) % 360
    ax.add_patch(Arc((0, 0), 2 * rad, 2 * rad, theta1=start, theta2=start + GOLDEN,
                     color="#DC2626", lw=2))
    mid = np.radians(start + GOLDEN / 2)
    ax.text(rad * 1.12 * np.cos(mid), rad * 1.12 * np.sin(mid), f"{i}→{i+1}\n+137.5°",
            color="#DC2626", fontsize=8, ha="center", va="center", fontweight="bold")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
ax.set_title("A. The turn only (tilt ignored)\nEach ray is rotated a further 137.5° around the axis.\n"
             "Every new ray lands in one of the widest gaps left so far.", fontsize=10)

# --- B: the real layout: turn + steadily increasing tilt -----------------------------
ax = axes[0, 1]
rim(ax)
ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(N), cmap=cmap, s=22, zorder=3)
for i in range(8):
    ax.annotate(str(i), xy[i], xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.plot(xy[:8, 0], xy[:8, 1], color="#DC2626", lw=0.9, alpha=0.8)
ax.set_title("B. The real 200 rays, looking down the axis\nSame 137.5° turn, and each ray also tilts a little further out.\n"
             "Result: an even 'sunflower' with no spokes and no big gaps.", fontsize=10)

# --- C: why not a round-number turn? --------------------------------------------------
ax = axes[1, 0]
rim(ax, label=False)
phi90 = np.radians(np.arange(N) * 90.0)
r = np.sin(np.radians(tilt))
ax.scatter(r * np.cos(phi90), r * np.sin(phi90), c=np.arange(N), cmap=cmap, s=22, zorder=3)
ax.set_title("C. Same tilts with a 90° turn instead\nThe rays stack into 4 spokes and leave whole wedges unsampled.\n"
             "An irrational turn (golden angle) never lines up like this.", fontsize=10)

# --- D: consequence for a 2D slice through the axis -----------------------------------
ax = axes[1, 1]
rim(ax, label=False)
ax.axhspan(-SLAB, SLAB, color="#22D3EE", alpha=0.25, zorder=1)
in_slab = np.abs(xy[:, 1]) <= SLAB
ax.scatter(xy[~in_slab, 0], xy[~in_slab, 1], color="0.75", s=18, zorder=2)
ax.scatter(xy[in_slab, 0], xy[in_slab, 1], color="#DC2626", s=34, zorder=3)
edge_l, edge_r = tilt[in_slab & (xy[:, 0] < 0)].max(), tilt[in_slab & (xy[:, 0] > 0)].max()
ax.text(0, -RIM * 1.05, f"slice through the axis (cyan band)", ha="center", color="#0E7490", fontsize=9)
ax.set_title(f"D. What a 2D slice through the axis sees\nOnly {in_slab.sum()} of {N} rays fall in a thin slice, and the widest ones\n"
             f"reach {edge_l:.1f}° / {edge_r:.1f}°, not 45° — so edges drawn from in-slice rays look narrower.",
             fontsize=10)

fig.suptitle("How the subchondral cone spreads its 200 rays: the golden-angle turn", fontsize=14, y=0.995)
fig.tight_layout()
fig.savefig(f"{OUT}/golden_angle.png", dpi=130)
print("wrote golden_angle.png  (in-slab rays:", in_slab.sum(), " edges:", round(edge_l, 1), round(edge_r, 1), ")")

# --- animation: rays placed one at a time ----------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 6.6))
FRAMES = list(range(1, 41)) + list(range(44, N + 1, 4)) + [N] * 12


def draw(frame):
    ax.clear()
    n = FRAMES[frame]
    rim(ax)
    ax.scatter(xy[:n - 1, 0], xy[:n - 1, 1], c=np.arange(n - 1), cmap=cmap,
               vmin=0, vmax=N, s=26, zorder=3)
    i = n - 1
    a0, a1 = (i - 1) * GOLDEN if i else 0.0, i * GOLDEN
    ax.plot([0, xy[i, 0] / max(np.linalg.norm(xy[i]), 1e-9) * RIM],
            [0, xy[i, 1] / max(np.linalg.norm(xy[i]), 1e-9) * RIM], color="#DC2626", lw=1.2, alpha=0.6)
    if i:
        ax.add_patch(Arc((0, 0), 0.5 * RIM, 0.5 * RIM, theta1=a0 % 360, theta2=(a0 % 360) + GOLDEN,
                         color="#DC2626", lw=2))
    ax.scatter(*xy[i], s=90, color="#DC2626", zorder=4, edgecolor="k")
    ax.set_title(f"ray {i}: turned {i} × 137.5° = {a1 % 360:.1f}° around, tilted {tilt[i]:.1f}°",
                 fontsize=10)


anim = FuncAnimation(fig, draw, frames=len(FRAMES), interval=180)
anim.save(f"{OUT}/golden_angle.gif", writer=PillowWriter(fps=6))
print("wrote golden_angle.gif")

"""
Figures for the knee-cartilage thickness delta maps (:mod:`morphometry.cartilage.deformation`).

Axis conventions (both figures): the map coordinates are those of the reference scan. For the
tibia, x is the plateau e1 axis (patient right, increasing to the right) and y is e2 (anterior,
increasing upwards). For the femur, x is the medial-lateral coordinate (patient right to the
right) and y is the articular arc length in mm with the trochlea (negative) at the TOP and the
posterior condyle (positive) at the bottom. All axes are in mm. Blue is thinner than the
reference, red thicker.
"""
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from morphometry.cartilage.deformation import COMPARISONS, REFERENCE, BoneDeltaMaps

#: Rows (loads) and columns (flexion angles) of the per-knee figure.
LOADS = ('unbelastet', 'belastet')
ANGLES = (0, 30, 60, 90)
#: Colours of the three compartments in the cohort scatter (patient side).
COMPARTMENT_COLOURS = {'both': 'black', 'left': 'tab:red', 'right': 'tab:blue'}
#: Minimum half-range (mm) of the shared symmetric delta colour scale.
MIN_LIMIT_MM = 0.1


def _extent(grid) -> list:
    """
    The imshow extent (mm) of a map grid, for the transposed array with ``origin='lower'``.

    :param grid: A :class:`~morphometry.cartilage.deformation.MapGrid`.
    :return: ``[x0, x1, y0, y1]`` cell edges: column 0 horizontal, column 1 vertical.
    """
    h = grid.grid_mm / 2.0
    o, (n0, n1) = grid.origin_mm, grid.shape
    return [o[0] - h, o[0] + (n0 - 1) * grid.grid_mm + h,
            o[1] - h, o[1] + (n1 - 1) * grid.grid_mm + h]


def _style_axes(ax, bone: str, extent: list) -> None:
    """
    Apply the shared axis limits, orientation and labels.

    :param ax: Matplotlib axes.
    :param bone: ``'tibia'`` (y up = anterior) or ``'femur'`` (y down = posterior).
    :param extent: ``[x0, x1, y0, y1]`` in mm.
    """
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim((extent[3], extent[2]) if bone == 'femur' else (extent[2], extent[3]))
    ax.set_aspect('equal')
    ax.set_xlabel('patient right (mm)' if bone == 'tibia' else 'medial-lateral (mm)', fontsize=7)
    ax.set_ylabel('anterior (mm)' if bone == 'tibia' else 'arc, trochlea at top (mm)', fontsize=7)
    ax.tick_params(labelsize=6)


def _region_borders(ax, maps: BoneDeltaMaps) -> None:
    """
    Draw the reference's subregion borders: lines between neighbouring cells of different code.

    :param ax: Matplotlib axes.
    :param maps: The bone's maps (``region_codes``, ``grid``).
    """
    codes, g = maps.region_codes, maps.grid
    segs = []
    x0, y0 = g.origin_mm - g.grid_mm / 2.0

    def edge(i, j, horizontal_step):
        # horizontal_step: border between (i, j) and (i + 1, j) is vertical at x = i + 1/2.
        if horizontal_step:
            x = x0 + (i + 1) * g.grid_mm
            segs.append([(x, y0 + j * g.grid_mm), (x, y0 + (j + 1) * g.grid_mm)])
        else:
            y = y0 + (j + 1) * g.grid_mm
            segs.append([(x0 + i * g.grid_mm, y), (x0 + (i + 1) * g.grid_mm, y)])

    for i, j in zip(*np.nonzero(codes[:-1, :] != codes[1:, :])):
        if codes[i, j] >= 0 or codes[i + 1, j] >= 0:
            edge(i, j, True)
    for i, j in zip(*np.nonzero(codes[:, :-1] != codes[:, 1:])):
        if codes[i, j] >= 0 or codes[i, j + 1] >= 0:
            edge(i, j, False)
    if segs:
        ax.add_collection(LineCollection(segs, colors='0.15', linewidths=0.4))


def plot_knee_deltas(maps: BoneDeltaMaps, knee: str, hotspots: Optional[Dict[tuple, dict]] = None,
                     comparisons: Sequence[tuple] = COMPARISONS, reference_key: tuple = REFERENCE,
                     panel_titles: Optional[Dict[tuple, str]] = None,
                     path: Optional[str] = None) -> Figure:
    """
    Draw one knee's 2 x 4 grid of thickness maps for one bone.

    Rows are ``unbelastet`` (unloaded) and ``belastet`` (loaded), columns 0/30/60/90 degrees.
    The reference cell shows the reference's absolute thickness (``viridis``, own colourbar);
    the other panels show the delta to it (``RdBu_r``, blue = thinner) on one shared symmetric
    scale (95th percentile of |delta| over all drawn delta panels, at least 0.1 mm) with one
    colourbar. The reference's subregion borders are drawn on every panel and hotspots are
    marked with an ``x``. A configuration absent from ``maps.configurations`` gets a blank
    panel titled "no scan"; if ``panel_titles`` names it (e.g. ``'failed'``) that text is used.
    Axis conventions are in the module docstring.

    :param maps: The bone's :class:`BoneDeltaMaps`.
    :param knee: Knee name, used in the figure title.
    :param hotspots: ``(angle, load)`` -> hotspot dict (``map_mm`` = x, y) or ``None``.
    :param comparisons: The configurations drawn as delta panels.
    :param reference_key: The ``(angle, load)`` of the reference.
    :param panel_titles: ``(angle, load)`` -> text for a panel with no map (default "no scan").
    :param path: If given, the figure is saved there (PNG) .
    :return: The matplotlib figure.
    """
    hotspots = hotspots or {}
    panel_titles = panel_titles or {}
    bone = maps.bone
    extent = _extent(maps.grid)
    shown = [k for k in comparisons if k in maps.deltas]
    stack = np.concatenate([np.abs(maps.deltas[k]).ravel() for k in shown]) if shown else np.array([])
    stack = stack[np.isfinite(stack)]
    limit = max(float(np.percentile(stack, 95)), MIN_LIMIT_MM) if stack.size else MIN_LIMIT_MM

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), squeeze=False)
    delta_im = ref_im = None
    for r, load in enumerate(LOADS):
        for c, angle in enumerate(ANGLES):
            key, ax = (angle, load), axes[r, c]
            if key == reference_key:
                ref_im = ax.imshow(maps.reference.T, origin='lower', extent=extent, cmap='viridis')
                ax.set_title(f'{angle}° {_load_name(load)} (reference, mm)', fontsize=9)
            elif key in maps.deltas and key in comparisons:
                delta_im = ax.imshow(maps.deltas[key].T, origin='lower', extent=extent,
                                     cmap='RdBu_r', vmin=-limit, vmax=limit)
                ax.set_title(f'{angle}° {_load_name(load)}', fontsize=9)
            else:
                ax.set_title(f'{angle}° {_load_name(load)}: {panel_titles.get(key, "no scan")}',
                             fontsize=9)
                ax.set_facecolor('0.93')
            _style_axes(ax, bone, extent)
            _region_borders(ax, maps)
            hs = hotspots.get(key)
            if hs is not None and key != reference_key:
                x, y = hs['map_mm']
                if np.isfinite(x) and np.isfinite(y):
                    ax.plot([x], [y], marker='x', color='k', markersize=9, mew=2)
    fig.suptitle(f'{knee} {bone}: thickness change relative to {reference_key[0]}° '
                 f'{_load_name(reference_key[1])} (blue = thinner)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    if delta_im is not None:
        cax = fig.add_axes([0.92, 0.12, 0.012, 0.35])
        fig.colorbar(delta_im, cax=cax).set_label('Δ thickness (mm)', fontsize=8)
    if ref_im is not None:
        cax = fig.add_axes([0.92, 0.55, 0.012, 0.3])
        fig.colorbar(ref_im, cax=cax).set_label('reference thickness (mm)', fontsize=8)
    if path is not None:
        fig.savefig(str(path), dpi=130)
    return fig


def _load_name(load: str) -> str:
    """
    English name of a load state.

    :param load: ``'belastet'`` or ``'unbelastet'``.
    :return: ``'loaded'`` or ``'unloaded'``.
    """
    return 'loaded' if load == 'belastet' else 'unloaded'


def plot_cohort_hotspots(rows: Sequence[dict], bone: str, path: Optional[str] = None) -> Figure:
    """
    Scatter the cohort's hotspot positions, one panel per comparison (seven panels).

    Each panel plots ``map_x_mm`` / ``map_y_mm`` of the rows of that ``(angle, load)`` and bone,
    coloured by compartment (``both`` black, ``left`` red, ``right`` blue; patient side).
    Rows with a non-finite coordinate are ignored. Axis conventions are in the module docstring.

    :param rows: Dicts with at least ``angle``, ``load``, ``bone``, ``compartment``,
        ``map_x_mm``, ``map_y_mm`` (already mirrored by the caller when a laterality table is
        used).
    :param bone: ``'tibia'`` or ``'femur'``; other bones' rows are ignored.
    :param path: If given, the figure is saved there (PNG).
    :return: The matplotlib figure.
    """
    kept = []
    for row in rows:
        if row.get('bone') != bone:
            continue
        try:
            x, y = float(row['map_x_mm']), float(row['map_y_mm'])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            kept.append((int(row['angle']), row['load'], row.get('compartment'), x, y))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), squeeze=False)
    xs = [k[3] for k in kept] or [0.0]
    ys = [k[4] for k in kept] or [0.0]
    pad = 2.0
    extent = [min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad]
    for r, load in enumerate(LOADS):
        for c, angle in enumerate(ANGLES):
            ax = axes[r, c]
            key = (angle, load)
            ax.set_title(f'{angle}° {_load_name(load)}', fontsize=9)
            if key not in COMPARISONS:
                ax.axis('off')
                ax.set_title(f'{angle}° {_load_name(load)} (reference)', fontsize=9)
                continue
            for comp, colour in COMPARTMENT_COLOURS.items():
                pts = [(k[3], k[4]) for k in kept if (k[0], k[1]) == key and k[2] == comp]
                if pts:
                    p = np.array(pts)
                    ax.scatter(p[:, 0], p[:, 1], s=18, c=colour, alpha=0.7, label=comp)
            _style_axes(ax, bone, extent)
            ax.set_aspect('auto')
    handles = [plt.Line2D([], [], marker='o', ls='', color=col, label=name)
               for name, col in COMPARTMENT_COLOURS.items()]
    fig.legend(handles=handles, loc='lower right', fontsize=8)
    fig.suptitle(f'{bone}: largest-thinning hotspot positions (patient side)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if path is not None:
        fig.savefig(str(path), dpi=130)
    return fig

"""
Figures for the knee-cartilage thickness delta maps (:mod:`morphometry.cartilage.deformation`).

Axis conventions (both figures): the map coordinates are those of the reference scan. For the
tibia, x is the plateau e1 axis (patient right, increasing to the right) and y is e2 (anterior,
increasing upwards). For the femur, x is the medial-lateral coordinate (patient right to the
right) and y is the articular arc length in mm with the trochlea (negative) at the TOP and the
posterior condyle (positive) at the bottom. All axes are in mm. Blue is thinner than the
reference, red thicker.

The maps show the **one-sided** cAB -> AC distance per position (the cMe direction,
``ThicknessMaps.triangle_thickness``), which is what the colour-bar labels say; the
per-subregion tables of the deformation script are in aMe (both directions averaged), so the
magnitudes of the two are not directly comparable.
"""
import textwrap
from typing import Dict, List, Optional, Sequence

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from morphometry.cartilage.deformation import (
    COMPARISONS, DEFAULT_HOTSPOT_EDGE_MARGIN_MM, REFERENCE, BoneDeltaMaps,
)
from morphometry.cartilage.femur_regions import FEMUR_REGION_KEYS
from morphometry.cartilage.tibia_regions import TIBIA_REGION_KEYS

#: Rows (loads) and columns (flexion angles) of the per-knee figure.
LOADS = ('unbelastet', 'belastet')
ANGLES = (0, 30, 60, 90)
#: Colours of the three compartments in the cohort scatter (patient side).
COMPARTMENT_COLOURS = {'both': 'black', 'left': 'tab:red', 'right': 'tab:blue'}
#: The same, once a laterality table has renamed the compartments (medial red, lateral blue).
ANATOMICAL_COMPARTMENT_COLOURS = {'both': 'black', 'medial': 'tab:red', 'lateral': 'tab:blue'}
#: Minimum half-range (mm) of the shared symmetric delta colour scale.
MIN_LIMIT_MM = 0.1
#: Region key per ``BoneDeltaMaps.region_codes`` value, per bone.
REGION_KEYS = {'tibia': TIBIA_REGION_KEYS, 'femur': FEMUR_REGION_KEYS}
#: Background of every panel: a map cell with no value (NaN) shows it, so "not measured" is
#: never confused with the white of a zero delta in ``RdBu_r``.
NO_DATA_GREY = '0.82'
#: Characters per caption line (each caption column is about half the figure width).
CAPTION_WIDTH = 118
#: Font size (pt) of the caption block.
CAPTION_FONTSIZE = 7.5
#: Height (inches) reserved under the panels for the caption block.
CAPTION_HEIGHT_IN = 3.0
#: Figure width (inches) of both 2 x 4 figures.
FIGURE_WIDTH_IN = 15.0


def _figure_height(extent: list, width_fraction: float) -> float:
    """
    Figure height (inches) that fits two rows of equal-aspect panels plus the caption block.

    :param extent: ``[x0, x1, y0, y1]`` of a panel (mm); its aspect sets the panel height.
    :param width_fraction: Fraction of the figure width the four panel columns share.
    :return: The height in inches (at least 7).
    """
    aspect = abs(extent[3] - extent[2]) / max(abs(extent[1] - extent[0]), 1e-9)
    panel_w = FIGURE_WIDTH_IN * width_fraction / 4.0 * 0.85  # minus tick labels and spacing
    # Two rows of panels with titles and axis labels, plus the suptitle and the caption.
    return max(2.0 * (panel_w * aspect + 1.3) + 0.7 + CAPTION_HEIGHT_IN, 7.0)


def region_display_name(key: str) -> str:
    """
    The short display name of a subregion key, in the Wirth & Eckstein capitalisation.

    The letters before the side code stay lower case and the side code (``lt``/``rt`` for a
    tibial plate, ``lf``/``rf`` for a femoral condyle) is upper-cased: ``'clt'`` -> ``'cLT'``,
    ``'iclf'`` -> ``'icLF'``, ``'alf'`` -> ``'aLF'``.

    :param key: A key of ``TIBIA_REGION_KEYS`` or ``FEMUR_REGION_KEYS``.
    :return: The display name.
    """
    return key[:-2] + key[-2:].upper()


def _wrap(paragraphs: Sequence[str], width: int = CAPTION_WIDTH) -> str:
    """
    Wrap caption paragraphs to a fixed width, keeping a hanging indent after a leading bullet.

    :param paragraphs: The paragraphs; one starting with ``'• '`` is indented under its text.
    :param width: Characters per line.
    :return: The wrapped text, paragraphs separated by newlines.
    """
    lines: List[str] = []
    for para in paragraphs:
        indent = '   ' if para.startswith('• ') else ''
        lines.extend(textwrap.wrap(para, width, subsequent_indent=indent) or [''])
    return '\n'.join(lines)


def _axes_caption(bone: str) -> List[str]:
    """
    Caption paragraphs explaining a bone's map axes and its subregion abbreviations.

    :param bone: ``'tibia'`` or ``'femur'``.
    :return: The paragraphs (the first is a heading).
    """
    if bone == 'tibia':
        return [
            'AXES AND ABBREVIATIONS (tibia)',
            '• View from above onto the tibial plateau, in the reference scan\'s plateau frame: '
            'x = towards the patient\'s right (mm), y = anterior (mm, up). Both scans\' '
            'cartilage is projected along the plateau normal.',
            '• Subregions (Wirth & Eckstein 2008), labelled on the reference panel: c = central '
            '(20% of the plate), i = internal (the wedge facing the other plate), e = external, '
            'a = anterior, p = posterior. LT / RT = plate on the patient\'s LEFT / RIGHT side. '
            'This is the patient side, not medial / lateral: the knees\' laterality is not yet '
            'recorded, so LT is the medial plate of a right knee but the lateral plate of a left one.',
        ]
    return [
        'AXES AND ABBREVIATIONS (femur)',
        '• The femoral cartilage unrolled onto a plane, in the reference scan\'s femoral frame: '
        'x = medial-lateral position, towards the patient\'s right (mm); y = arc length (mm) '
        'along the bone surface in each sagittal cut, 0 at the trochlear notch plane, '
        'negative (top) = trochlea / anterior, positive (down) = weight-bearing condyle, then '
        'posterior condyle. Distances along y are along the curved bone, not straight lines.',
        '• Subregions, labelled on the reference panel: the central (weight-bearing) femur cF '
        'runs from the notch plane to 60% of the way to the posterior end of the condyle and is '
        'split across the condyle into thirds ic = internal (notch side), cc = central, '
        'ec = external (Wirth & Eckstein 2008); the map labels them ic/cc/ec, the tables '
        'icLF, ccLF, ecLF, icRF, .... Extension zones (not in the paper): a = trochlea '
        '(anterior of the notch plane), p = posterior condyle (behind cF). LF / RF = condyle on '
        'the patient\'s LEFT / RIGHT side. This is the patient side, not medial / lateral: the '
        'knees\' laterality is not yet recorded.',
    ]


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
    ax.set_xlabel('patient right (mm)', fontsize=7)
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
        lc = LineCollection(segs, colors='black', linewidths=1.0)
        # White halo under the black line keeps the border visible on dark and light cells.
        lc.set_path_effects([pe.withStroke(linewidth=2.6, foreground='white')])
        ax.add_collection(lc)


def knee_caption(bone: str, limit_mm: float, grid_mm: float,
                 reference_key: tuple = REFERENCE, neighbourhood_mm: float = 3.0,
                 edge_margin_mm: float = DEFAULT_HOTSPOT_EDGE_MARGIN_MM) -> List[str]:
    """
    The two caption columns of the per-knee figure: what is shown, and the axes/abbreviations.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param limit_mm: Half-range of the shared delta colour scale (mm).
    :param grid_mm: Map cell size (mm).
    :param reference_key: ``(angle, load)`` of the reference.
    :param neighbourhood_mm: Smoothing neighbourhood of the hotspot search (mm).
    :param edge_margin_mm: Edge margin of the hotspot search (mm).
    :return: ``[left_text, right_text]``, each already wrapped.
    """
    ref = f'{reference_key[0]}° {_load_name(reference_key[1])}'
    what = [
        'WHAT IS SHOWN',
        f'• One knee, one panel per scan: rows = load (unloaded / loaded), columns = flexion '
        f'angle. The {ref} scan is the REFERENCE; its panel shows its absolute cartilage '
        f'thickness (viridis, upper colour bar).',
        f'• Every other panel shows Δ = thickness(this scan) − thickness(reference), in mm, at '
        f'the SAME SPOT OF THE BONE: the scan\'s {bone} is rigidly registered onto the '
        f'reference {bone} (ICP on the bone surface), so each cell compares the cartilage '
        f'lying over the same piece of bone, not the same contact area.',
        f'• SIGN: Δ < 0 (BLUE) = cartilage THINNER than in the reference (e.g. compressed). '
        f'Δ > 0 (RED) = thicker. White = no change. One symmetric scale ±{limit_mm:.2f} mm '
        f'for all Δ panels (95th percentile of |Δ|); larger changes saturate.',
        f'• Thickness = distance from the cartilage-covered bone surface (cAB) to the '
        f'articular cartilage surface (AC), averaged per {grid_mm:g} mm cell. This one-sided '
        f'distance is not the aMe of the subregion tables, so their magnitudes differ.',
        '• Grey = no value: no cartilage in this scan or in the reference at that spot (or '
        'not mappable). A grey panel titled "no scan" / "failed" has no comparison.',
        f'• Black lines = the reference\'s subregion borders. × = largest-thinning hotspot of '
        f'the whole bone: the most negative Δ after median smoothing over {neighbourhood_mm:g} '
        f'mm, at least {edge_margin_mm:g} mm inside both scans\' cartilage edge (per '
        f'plate/condyle hotspots: hotspots.csv).',
    ]
    return [_wrap(what), _wrap(_axes_caption(bone))]


def _draw_caption(fig: Figure, columns: Sequence[str], top: float) -> None:
    """
    Draw caption columns side by side under the panels.

    :param fig: The figure.
    :param columns: The column texts (already wrapped), left to right.
    :param top: Figure-fraction height of the top of the caption block.
    """
    n = len(columns)
    for i, text in enumerate(columns):
        fig.text(0.01 + i * 0.99 / n, top, text, ha='left', va='top',
                 fontsize=CAPTION_FONTSIZE, linespacing=1.3)


def _region_labels(ax, maps: BoneDeltaMaps) -> None:
    """
    Write each subregion's display name at a cell of that region near its centroid.

    :param ax: Matplotlib axes (the reference panel).
    :param maps: The bone's maps (``region_codes``, ``grid``, ``bone``).
    """
    keys = REGION_KEYS.get(maps.bone, ())
    g = maps.grid
    for code in np.unique(maps.region_codes):
        if code < 0 or code >= len(keys):
            continue
        name = region_display_name(keys[code])
        if maps.bone == 'femur' and len(keys[code]) == 4:
            # A condylar third (icLF ...) is too narrow for its full name; its condyle is named
            # on the neighbouring a/p zones and the caption gives the full key.
            name = keys[code][:2]
        cells = np.argwhere(maps.region_codes == code)
        # The cell nearest the centroid, so a curved region's label stays on the region.
        nearest = cells[np.argmin(((cells - cells.mean(axis=0)) ** 2).sum(axis=1))]
        x, y = g.origin_mm + nearest * g.grid_mm
        ax.text(x, y, name, ha='center', va='center', fontsize=6.5,
                fontweight='bold', color='black',
                path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])


def plot_knee_deltas(maps: BoneDeltaMaps, knee: str, hotspots: Optional[Dict[tuple, dict]] = None,
                     comparisons: Sequence[tuple] = COMPARISONS, reference_key: tuple = REFERENCE,
                     panel_titles: Optional[Dict[tuple, str]] = None,
                     path: Optional[str] = None, neighbourhood_mm: float = 3.0,
                     edge_margin_mm: float = DEFAULT_HOTSPOT_EDGE_MARGIN_MM) -> Figure:
    """
    Draw one knee's 2 x 4 grid of thickness maps for one bone.

    Rows are ``unbelastet`` (unloaded) and ``belastet`` (loaded), columns 0/30/60/90 degrees.
    The reference cell shows the reference's absolute thickness (``viridis``, own colourbar);
    the other panels show the delta to it (``RdBu_r``, blue = thinner) on one shared symmetric
    scale (95th percentile of |delta| over all drawn delta panels, at least 0.1 mm) with one
    colourbar. The reference's subregion borders are drawn on every panel and hotspots are
    marked with an ``x``; the reference panel also carries the subregion names. A
    configuration absent from ``maps.configurations`` gets a blank panel titled "no scan"; if
    ``panel_titles`` names it (e.g. ``'failed'``) that text is used. Every panel's background is
    :data:`NO_DATA_GREY`, so unmeasured cells read grey rather than as a white zero. Under the
    panels a caption (:func:`knee_caption`) explains the setup, the sign of the delta, the colour
    scales, the markers, the axes and the abbreviations. Axis conventions are in the module
    docstring.

    :param maps: The bone's :class:`BoneDeltaMaps`.
    :param knee: Knee name, used in the figure title.
    :param hotspots: ``(angle, load)`` -> hotspot dict (``map_mm`` = x, y) or ``None``.
    :param comparisons: The configurations drawn as delta panels.
    :param reference_key: The ``(angle, load)`` of the reference.
    :param panel_titles: ``(angle, load)`` -> text for a panel with no map (default "no scan").
    :param path: If given, the figure is saved there (PNG) .
    :param neighbourhood_mm: Smoothing neighbourhood the hotspots were found with (mm); only
        quoted in the caption.
    :param edge_margin_mm: Edge margin the hotspots were found with (mm); only quoted in the
        caption.
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

    height = _figure_height(extent, 0.9)
    bottom = CAPTION_HEIGHT_IN / height  # figure fraction taken by the caption block
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_WIDTH_IN, height), squeeze=False)
    delta_im = ref_im = None
    for r, load in enumerate(LOADS):
        for c, angle in enumerate(ANGLES):
            key, ax = (angle, load), axes[r, c]
            if key == reference_key:
                ref_im = ax.imshow(maps.reference.T, origin='lower', extent=extent, cmap='viridis')
                ax.set_title(f'{angle}° {_load_name(load)}: REFERENCE (thickness, mm)',
                             fontsize=9)
            elif key in maps.deltas and key in comparisons:
                delta_im = ax.imshow(maps.deltas[key].T, origin='lower', extent=extent,
                                     cmap='RdBu_r', vmin=-limit, vmax=limit)
                ax.set_title(f'{angle}° {_load_name(load)}: Δ vs reference', fontsize=9)
            else:
                ax.set_title(f'{angle}° {_load_name(load)}: {panel_titles.get(key, "no scan")}',
                             fontsize=9)
            ax.set_facecolor(NO_DATA_GREY)
            _style_axes(ax, bone, extent)
            _region_borders(ax, maps)
            if key == reference_key:
                _region_labels(ax, maps)
            hs = hotspots.get(key)
            if hs is not None and key != reference_key:
                x, y = hs['map_mm']
                if np.isfinite(x) and np.isfinite(y):
                    ax.plot([x], [y], marker='x', color='k', markersize=9, mew=2)
    fig.suptitle(f'{knee} {bone}: cartilage thickness change relative to {reference_key[0]}° '
                 f'{_load_name(reference_key[1])} (Δ = scan − reference; blue = thinner, '
                 f'red = thicker)', fontsize=11)
    # Panels above, the caption in the bottom CAPTION_HEIGHT_IN inches.
    top = 1.0 - 0.5 / height
    fig.tight_layout(rect=(0, bottom, 0.9, top))
    span = top - bottom  # the two colour bars split the panel band: reference above, delta below
    if delta_im is not None:
        cax = fig.add_axes([0.92, bottom + 0.06 * span, 0.012, 0.38 * span])
        bar = fig.colorbar(delta_im, cax=cax, extend='both')
        bar.set_label('Δ thickness, cAB→AC (mm)\n< 0 thinner (blue) · > 0 thicker (red)',
                      fontsize=8)
    if ref_im is not None:
        cax = fig.add_axes([0.92, bottom + 0.56 * span, 0.012, 0.38 * span])
        fig.colorbar(ref_im, cax=cax).set_label('reference thickness, cAB→AC (mm)', fontsize=8)
    _draw_caption(fig, knee_caption(bone, limit, maps.grid.grid_mm, reference_key,
                                    neighbourhood_mm, edge_margin_mm),
                  top=bottom - 0.15 / height)
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


def cohort_caption(bone: str, anatomical: bool = False, reference_key: tuple = REFERENCE,
                   neighbourhood_mm: float = 3.0,
                   edge_margin_mm: float = DEFAULT_HOTSPOT_EDGE_MARGIN_MM) -> List[str]:
    """
    The two caption columns of the cohort hotspot figure.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param anatomical: ``True`` when a laterality table mapped the compartments to
        medial/lateral and mirrored x to medial-positive.
    :param reference_key: ``(angle, load)`` of the reference.
    :param neighbourhood_mm: Smoothing neighbourhood of the hotspot search (mm).
    :param edge_margin_mm: Edge margin of the hotspot search (mm).
    :return: ``[left_text, right_text]``, each already wrapped.
    """
    ref = f'{reference_key[0]}° {_load_name(reference_key[1])}'
    if anatomical:
        colours = ('black ring = searched over the whole bone (it usually coincides with one '
                   'of the coloured dots), red = only the MEDIAL '
                   'plate/condyle, blue = only the LATERAL one (from the laterality table)')
        xaxis = ('x is mirrored so that positive = MEDIAL for every knee (left and right knees '
                 'overlay).')
    else:
        colours = ('black ring = searched over the whole bone (it usually coincides with one '
                   'of the coloured dots), red = only the plate/condyle on the '
                   'patient\'s LEFT, blue = only the one on the patient\'s RIGHT (patient side, '
                   'not medial/lateral: laterality not yet recorded)')
        xaxis = 'x = towards the patient\'s right, as in the per-knee figures.'
    what = [
        'WHAT IS SHOWN',
        f'• One panel per comparison (rows = load, columns = flexion angle); the {ref} '
        f'reference has none. Each dot is ONE KNEE: where its cartilage THINNED MOST in that '
        f'scan compared with its own {ref} scan, at the same spot of the bone.',
        f'• Colours: {colours}.',
        f'• Thinned most = the most negative Δ = thickness(scan) − thickness(reference) after '
        f'median smoothing over {neighbourhood_mm:g} mm, at least {edge_margin_mm:g} mm inside '
        f'both scans\' cartilage edge. The dots give the position only; the amounts are in '
        f'hotspots.csv.',
        '• Each knee\'s dot is in its own reference map, and the knees are not registered to '
        'each other: positions are comparable through the anatomical frame only (tibia: '
        'plateau frame; femur: arc from the trochlear notch), not millimetre-exact.',
        f'• {xaxis}',
    ]
    return [_wrap(what), _wrap(_axes_caption(bone)[:2])]


def plot_cohort_hotspots(rows: Sequence[dict], bone: str, path: Optional[str] = None,
                         anatomical: bool = False, neighbourhood_mm: float = 3.0,
                         edge_margin_mm: float = DEFAULT_HOTSPOT_EDGE_MARGIN_MM) -> Figure:
    """
    Scatter the cohort's hotspot positions, one panel per comparison (seven panels).

    Each panel plots ``map_x_mm`` / ``map_y_mm`` of the rows of that ``(angle, load)`` and bone,
    coloured by compartment (``both`` black, ``left`` red, ``right`` blue; patient side).
    Rows with a non-finite coordinate are ignored. Axis conventions are in the module docstring,
    except with ``anatomical=True``: the compartments are then ``medial`` (red) / ``lateral``
    (blue) and the horizontal axis is the medial direction (the caller mirrored ``map_x_mm``).

    :param rows: Dicts with at least ``angle``, ``load``, ``bone``, ``compartment``,
        ``map_x_mm``, ``map_y_mm`` (already mirrored by the caller when a laterality table is
        used).
    :param bone: ``'tibia'`` or ``'femur'``; other bones' rows are ignored.
    :param path: If given, the figure is saved there (PNG).
    :param anatomical: ``True`` when the rows carry medial/lateral compartments and a
        medial-positive ``map_x_mm`` (a laterality table was applied).
    :param neighbourhood_mm: Smoothing neighbourhood of the hotspot search (mm); only quoted
        in the caption (:func:`cohort_caption`) drawn under the panels.
    :param edge_margin_mm: Edge margin of the hotspot search (mm); only quoted in the caption.
    :return: The matplotlib figure.
    """
    colours = ANATOMICAL_COMPARTMENT_COLOURS if anatomical else COMPARTMENT_COLOURS
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
    xs = [k[3] for k in kept] or [0.0]
    ys = [k[4] for k in kept] or [0.0]
    pad = 2.0
    extent = [min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad]
    height = 7.5 + CAPTION_HEIGHT_IN  # the scatter panels are not equal-aspect
    bottom = CAPTION_HEIGHT_IN / height
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_WIDTH_IN, height), squeeze=False)
    for r, load in enumerate(LOADS):
        for c, angle in enumerate(ANGLES):
            ax = axes[r, c]
            key = (angle, load)
            ax.set_title(f'{angle}° {_load_name(load)}', fontsize=9)
            if key not in COMPARISONS:
                ax.axis('off')
                ax.set_title(f'{angle}° {_load_name(load)} (reference)', fontsize=9)
                continue
            # The whole-bone hotspot is usually one of the compartments' hotspots too, so it is
            # drawn last, as a ring around that dot, instead of a dot hidden under it.
            for comp in sorted(colours, key=lambda name: name == 'both'):
                pts = [(k[3], k[4]) for k in kept if (k[0], k[1]) == key and k[2] == comp]
                if pts:
                    p = np.array(pts)
                    if comp == 'both':
                        ax.scatter(p[:, 0], p[:, 1], s=90, facecolors='none',
                                   edgecolors=colours[comp], linewidths=1.4, label=comp)
                    else:
                        ax.scatter(p[:, 0], p[:, 1], s=22, c=colours[comp], alpha=0.8,
                                   label=comp)
            _style_axes(ax, bone, extent)
            if anatomical:
                ax.set_xlabel('medial (mm)', fontsize=7)
            ax.set_aspect('auto')
    names = {'both': 'whole bone', 'left': 'patient-left plate/condyle',
             'right': 'patient-right plate/condyle', 'medial': 'medial plate/condyle',
             'lateral': 'lateral plate/condyle'}
    handles = [plt.Line2D([], [], marker='o', ls='', color=col, label=names.get(name, name),
                          markersize=10 if name == 'both' else 6,
                          markerfacecolor='none' if name == 'both' else col)
               for name, col in colours.items()]
    side = 'medial positive' if anatomical else 'patient side'
    fig.suptitle(f'{bone}: where the cartilage thinned most, per knee and comparison ({side})',
                 fontsize=11)
    fig.tight_layout(rect=(0, bottom, 1, 1.0 - 0.5 / height))
    # The legend sits in the empty reference panel's slot when there is one.
    ref_axes = [axes[r, c] for r, load in enumerate(LOADS) for c, angle in enumerate(ANGLES)
                if (angle, load) not in COMPARISONS]
    if ref_axes:
        ref_axes[0].legend(handles=handles, loc='center', fontsize=9, title='hotspot searched in',
                           title_fontsize=9, frameon=False)
    else:
        fig.legend(handles=handles, loc='lower right', fontsize=8)
    _draw_caption(fig, cohort_caption(bone, anatomical, REFERENCE, neighbourhood_mm,
                                      edge_margin_mm), top=bottom - 0.15 / height)
    if path is not None:
        fig.savefig(str(path), dpi=130)
    return fig

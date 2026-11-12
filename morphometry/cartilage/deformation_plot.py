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


# --------------------------------------------------------------------------------------------
# Registration figures
# --------------------------------------------------------------------------------------------

#: Fixed half-range (mm) of the residual colour scale: fixed, not per figure, so knees compare.
RESIDUAL_LIMIT_MM = 0.5
#: Colour map of the residual maps; deliberately not the Δ maps' ``RdBu_r``.
RESIDUAL_CMAP = 'PuOr_r'
#: Line styles of the cross-section overlays.
SECTION_STYLES = {
    'reference': dict(color='black', lw=1.6, zorder=3),
    'initial': dict(color='tab:orange', lw=1.0, ls='--', zorder=2),
    'registered': dict(color='tab:red', lw=1.0, zorder=4),
}


def _rms_title(maps: BoneDeltaMaps, key: tuple) -> str:
    """
    A panel title line with the registration's residual and rotation change.

    :param maps: The bone's maps.
    :param key: ``(angle, load)``.
    :return: e.g. ``'rms 0.12 mm · ICP turned 3.1°'``, or ``''`` without a registration.
    """
    result = maps.registrations.get(key)
    if result is None:
        return ''
    rotation = maps.rotation_change_deg.get(key, float('nan'))
    return f'rms {result.rms_mm:.2f} mm · ICP turned {rotation:.1f}°'


def residual_caption(bone: str, reference_key: tuple = REFERENCE) -> List[str]:
    """
    The two caption columns of the per-knee registration residual figure.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param reference_key: ``(angle, load)`` of the reference.
    :return: ``[left_text, right_text]``, each already wrapped.
    """
    ref = f'{reference_key[0]}° {_load_name(reference_key[1])}'
    what = [
        'WHAT IS SHOWN',
        f'• How well each scan\'s {bone} was aligned onto the {ref} reference {bone} before '
        f'the thickness comparison. Same panels and map coordinates as the Δ-thickness '
        f'figure, so the two can be read side by side.',
        f'• Alignment: the scan\'s {bone} is first placed by matching the two anatomical '
        f'frames, then rigidly registered by ICP on the bone surface only (no cartilage), '
        f'using the articular end of the bone. Panel titles: rms = root-mean-square distance '
        f'of the registered bone to the reference bone (best 90% of points); ICP turned = '
        f'how far ICP rotated away from the frame-based start (flagged above 15°).',
        '• Colour = residual at each spot: signed distance (mm) from the scan\'s registered '
        'bone surface UNDER ITS CARTILAGE to the reference bone surface. Orange/brown (> 0) = '
        'the scan\'s bone lies outside the reference bone there; purple (< 0) = inside. Fixed '
        f'scale ±{RESIDUAL_LIMIT_MM:g} mm for every knee; larger values saturate.',
        '• HOW TO USE IT: where |residual| is much smaller than the |Δ thickness| at the same '
        'spot, the Δ is not an alignment error. A residual pattern that mirrors the Δ pattern '
        'is a warning sign.',
        '• Grey = no value (no cartilage-covered bone of this scan there). The reference panel '
        'shows the subregions only (it is the target). Black lines = reference subregion '
        'borders.',
    ]
    return [_wrap(what), _wrap(_axes_caption(bone))]


def plot_knee_residuals(maps: BoneDeltaMaps, knee: str,
                        comparisons: Sequence[tuple] = COMPARISONS,
                        reference_key: tuple = REFERENCE,
                        panel_titles: Optional[Dict[tuple, str]] = None,
                        path: Optional[str] = None) -> Figure:
    """
    Draw one knee's 2 x 4 grid of registration residual maps for one bone.

    Same layout, coordinates and caption style as :func:`plot_knee_deltas`, showing
    ``maps.residuals`` (signed bone-to-bone distance after registration) on a fixed symmetric
    scale of :data:`RESIDUAL_LIMIT_MM` in :data:`RESIDUAL_CMAP`. Each comparison panel's title
    carries the registration's rms residual and rotation change; the reference panel shows only
    the subregion borders and names.

    :param maps: The bone's :class:`BoneDeltaMaps` (with ``residuals``).
    :param knee: Knee name, used in the figure title.
    :param comparisons: The configurations drawn.
    :param reference_key: The ``(angle, load)`` of the reference.
    :param panel_titles: ``(angle, load)`` -> text for a panel with no map (default "no scan").
    :param path: If given, the figure is saved there (PNG).
    :return: The matplotlib figure.
    """
    panel_titles = panel_titles or {}
    bone = maps.bone
    extent = _extent(maps.grid)
    height = _figure_height(extent, 0.9)
    bottom = CAPTION_HEIGHT_IN / height
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_WIDTH_IN, height), squeeze=False)
    image = None
    for r, load in enumerate(LOADS):
        for c, angle in enumerate(ANGLES):
            key, ax = (angle, load), axes[r, c]
            label = f'{angle}° {_load_name(load)}'
            if key == reference_key:
                ax.set_title(f'{label}: REFERENCE (registration target)', fontsize=9)
            elif key in maps.residuals and key in comparisons:
                image = ax.imshow(maps.residuals[key].T, origin='lower', extent=extent,
                                  cmap=RESIDUAL_CMAP, vmin=-RESIDUAL_LIMIT_MM,
                                  vmax=RESIDUAL_LIMIT_MM)
                ax.set_title(f'{label}\n{_rms_title(maps, key)}', fontsize=8.5)
            else:
                ax.set_title(f'{label}: {panel_titles.get(key, "no scan")}', fontsize=9)
            ax.set_facecolor(NO_DATA_GREY)
            _style_axes(ax, bone, extent)
            _region_borders(ax, maps)
            if key == reference_key:
                _region_labels(ax, maps)
    fig.suptitle(f'{knee} {bone}: registration residual onto {reference_key[0]}° '
                 f'{_load_name(reference_key[1])} (bone-to-bone distance after alignment)',
                 fontsize=11)
    top = 1.0 - 0.5 / height
    fig.tight_layout(rect=(0, bottom, 0.9, top))
    if image is not None:
        span = top - bottom
        cax = fig.add_axes([0.92, bottom + 0.2 * span, 0.012, 0.6 * span])
        fig.colorbar(image, cax=cax, extend='both').set_label(
            'registration residual (mm)\n> 0 scan bone outside · < 0 inside', fontsize=8)
    _draw_caption(fig, residual_caption(bone, reference_key), top=bottom - 0.15 / height)
    if path is not None:
        fig.savefig(str(path), dpi=130)
    return fig


def _section_segments(mesh, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """
    Cut a surface with a sagittal plane and return the cut in in-plane 2-D coordinates.

    :param mesh: A ``pv.PolyData`` surface (world mm).
    :param origin: ``(3,)`` a point of the plane (world mm).
    :param axes: ``(3, 3)`` rows ``(normal, u, v)``: the plane normal (frame row 0, M-L) and
        the two in-plane directions (frame rows 1 and 2: anterior and the long axis).
    :return: ``(S, 2, 2)`` line segments ``(u, v)`` in mm relative to ``origin``; empty if the
        plane misses the surface.
    """
    cut = mesh.slice(normal=axes[0], origin=origin)
    if cut.n_points == 0 or cut.n_lines == 0:
        return np.zeros((0, 2, 2))
    pts = (np.asarray(cut.points) - origin) @ axes[1:].T
    lines = np.asarray(cut.lines)
    segs, i = [], 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i + 1:i + 1 + n]
        segs.extend(zip(ids[:-1], ids[1:]))
        i += n + 1
    segs = np.asarray(segs, dtype=int)
    return pts[segs] if len(segs) else np.zeros((0, 2, 2))


def _transformed(mesh, transform):
    """
    A copy of a surface with a rigid transform applied to its points.

    :param mesh: A ``pv.PolyData``.
    :param transform: A :class:`~morphometry.cartilage.registration.RigidTransform`.
    :return: The moved copy.
    """
    moved = mesh.copy()
    moved.points = transform.apply(np.asarray(mesh.points))
    return moved


def section_origins(reference) -> Dict[str, np.ndarray]:
    """
    Where the sagittal cuts of the section figure go: one per plate / condyle.

    The area-weighted centroid of the reference's covered tAB triangles of each patient side
    (region keys ending ``lt``/``lf`` = left, ``rt``/``rf`` = right).

    :param reference: The reference :class:`~morphometry.cartilage.deformation.BoneMeasurement`.
    :return: ``{'left': (3,), 'right': (3,)}`` (world mm); a side with no covered triangle is
        left out.
    """
    from morphometry.cartilage.surface import triangle_areas, triangle_centroids
    keys = REGION_KEYS[reference.bone]
    mesh = reference.tab_mesh
    region = np.asarray(mesh.cell_data['region'])
    covered = np.asarray(mesh.cell_data['covered'], dtype=bool)
    centroids, areas = triangle_centroids(mesh), triangle_areas(mesh)
    out = {}
    for side in ('left', 'right'):
        codes = [i for i, k in enumerate(keys) if k[-2] == side[0]]
        sel = covered & np.isin(region, codes)
        if sel.any():
            out[side] = np.average(centroids[sel], axis=0, weights=areas[sel])
    return out


def sections_caption(bone: str, reference_key: tuple = REFERENCE) -> List[str]:
    """
    The two caption columns of the per-knee registration cross-section figure.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param reference_key: ``(angle, load)`` of the reference.
    :return: ``[left_text, right_text]``, each already wrapped.
    """
    ref = f'{reference_key[0]}° {_load_name(reference_key[1])}'
    part = 'plate' if bone == 'tibia' else 'condyle'
    up = 'superior' if bone == 'tibia' else 'proximal (towards the hip)'
    what = [
        'WHAT IS SHOWN',
        f'• Two sagittal cuts per scan, one through the centre of each {part}\'s cartilage in the '
        f'{ref} reference. Rows: unloaded patient-left {part}, unloaded patient-right {part}, '
        f'loaded left, loaded right; columns: flexion angle. Every '
        f'scan is cut with the SAME reference plane, after being moved into the reference '
        f'scan\'s coordinates.',
        f'• BLACK = reference {bone} bone outline (the part the registration aligns onto). '
        f'ORANGE dashed = this scan\'s {bone} placed by matching anatomical frames only (the '
        f'ICP start). RED = after ICP (the pose the thickness comparison uses). Only the '
        f'articular end of the bone used for the registration is drawn.',
        '• A good registration puts red on black. Orange far from black is fine: it only shows '
        'how much ICP had to correct. Titles: rms residual and how far ICP rotated away from '
        'the frame-based start (flagged above 15°).',
    ]
    axes_text = [
        'AXES',
        f'• Coordinates in the reference {bone}\'s anatomical frame, in mm, relative to the cut '
        f'centre: x = anterior (to the right), y = {up}. Equal scale on both axes. Bone only; '
        f'cartilage is not drawn.',
        '• Patient-left / -right is the patient side, not medial / lateral (laterality not yet '
        'recorded).',
    ]
    return [_wrap(what), _wrap(axes_text)]


def plot_registration_sections(maps: BoneDeltaMaps, reference, configurations: Dict[tuple, object],
                               knee: str, crop_mm: Optional[float] = None,
                               reference_key: tuple = REFERENCE,
                               panel_titles: Optional[Dict[tuple, str]] = None,
                               path: Optional[str] = None) -> Figure:
    """
    Overlay sagittal bone outlines of each scan, before and after ICP, on the reference.

    For each configuration (2 rows of loads x 4 flexion angles) two subpanels: a sagittal cut
    (plane normal = the reference frame's M-L axis) through the centre of the patient-left and
    of the patient-right plate/condyle (:func:`section_origins`). Drawn in the reference's
    frame: the reference's registration surface (black), the configuration's registration
    surface at the ICP start (frame match, orange dashed) and after ICP (red). The surfaces are
    exactly those :func:`~morphometry.cartilage.deformation.knee_delta_maps` registers
    (:func:`~morphometry.cartilage.deformation.reference_registration_surface`,
    :func:`~morphometry.cartilage.deformation.configuration_registration_start`).

    :param maps: The bone's :class:`BoneDeltaMaps` (``registrations``, ``rotation_change_deg``).
    :param reference: The reference scan's ``BoneMeasurement`` of this bone.
    :param configurations: ``{(angle, load): BoneMeasurement}`` of this bone.
    :param knee: Knee name for the title.
    :param crop_mm: The registration crop (mm); default
        :data:`~morphometry.cartilage.registration.REGISTRATION_CROP_MM`.
    :param reference_key: The ``(angle, load)`` of the reference.
    :param panel_titles: ``(angle, load)`` -> text for a configuration with no registration.
    :param path: If given, the figure is saved there (PNG).
    :return: The matplotlib figure.
    """
    from morphometry.cartilage.deformation import (
        configuration_registration_start, reference_registration_surface,
    )
    from morphometry.cartilage.registration import REGISTRATION_CROP_MM
    crop_mm = REGISTRATION_CROP_MM if crop_mm is None else crop_mm
    panel_titles = panel_titles or {}
    bone = maps.bone
    axes3 = np.asarray(reference.frame_axes, dtype=float)
    origins = section_origins(reference)
    fixed = reference_registration_surface(bone, reference, crop_mm)
    ref_segs = {side: _section_segments(fixed, o, axes3) for side, o in origins.items()}

    # Common limits per side: the reference cut plus a margin, so every panel shows the same
    # window and offsets are comparable by eye.
    limits = {}
    for side, segs in ref_segs.items():
        if len(segs):
            pts = segs.reshape(-1, 2)
            lo, hi = pts.min(axis=0) - 4.0, pts.max(axis=0) + 4.0
            limits[side] = (lo, hi)
    span_u = max((hi[0] - lo[0] for lo, hi in limits.values()), default=60.0)
    span_v = max((hi[1] - lo[1] for lo, hi in limits.values()), default=40.0)
    part = 'plate' if bone == 'tibia' else 'condyle'
    # Rows: (unloaded, left), (unloaded, right), (loaded, left), (loaded, right); columns:
    # flexion angle. Each panel is one sagittal cut of one configuration.
    panel_w = FIGURE_WIDTH_IN / 4.0 * 0.8
    height = 4.0 * (panel_w * span_v / span_u + 0.8) + 1.0 + CAPTION_HEIGHT_IN
    bottom = CAPTION_HEIGHT_IN / height
    fig, axes = plt.subplots(4, 4, figsize=(FIGURE_WIDTH_IN, height), squeeze=False)
    for r, load in enumerate(LOADS):
        for c, angle in enumerate(ANGLES):
            key = (angle, load)
            label = f'{angle}° {_load_name(load)}'
            moving = initial = result = None
            if key != reference_key and key in maps.registrations and key in configurations:
                result = maps.registrations[key]
                moving, initial = configuration_registration_start(
                    bone, reference, configurations[key], crop_mm)
            for s, side in enumerate(('left', 'right')):
                ax = axes[2 * r + s, c]
                ax.set_aspect('equal')
                ax.tick_params(labelsize=6)
                where = f'patient-{side} {part}'
                if key == reference_key:
                    head = f'{label}, {where}: REFERENCE'
                elif result is not None:
                    head = f'{label}, {where}\n{_rms_title(maps, key)}'
                else:
                    head = f'{label}, {where}: {panel_titles.get(key, "no scan")}'
                ax.set_title(head, fontsize=8)
                if side not in origins:
                    ax.set_axis_off()
                    continue
                o = origins[side]
                layers = [('reference', ref_segs[side])]
                if result is not None:
                    layers.append(('initial', _section_segments(
                        _transformed(moving, initial), o, axes3)))
                    layers.append(('registered', _section_segments(
                        _transformed(moving, result.transform), o, axes3)))
                for name, segs in layers:
                    if len(segs):
                        ax.add_collection(LineCollection(segs, **SECTION_STYLES[name]))
                if side in limits:
                    lo, hi = limits[side]
                    ax.set_xlim(lo[0], hi[0])
                    ax.set_ylim(lo[1], hi[1])
                if 2 * r + s == 3:  # x label on the bottom row only; the axis is shared
                    ax.set_xlabel('anterior (mm)', fontsize=7)
                if c == 0:
                    ax.set_ylabel('proximal (mm)' if bone == 'femur' else 'superior (mm)',
                                  fontsize=7)
    handles = [plt.Line2D([], [], label=text, **{k: v for k, v in SECTION_STYLES[name].items()
                                                 if k != 'zorder'})
               for name, text in (('reference', 'reference bone'),
                                  ('initial', 'scan, frames matched (ICP start)'),
                                  ('registered', 'scan after ICP'))]
    fig.suptitle(f'{knee} {bone}: registration onto {reference_key[0]}° '
                 f'{_load_name(reference_key[1])}, sagittal bone outlines', fontsize=11)
    top = 1.0 - 0.9 / height
    fig.tight_layout(rect=(0, bottom, 1, top))
    # Legend on its own line under the title.
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.5 / height))
    _draw_caption(fig, sections_caption(bone, reference_key), top=bottom - 0.15 / height)
    if path is not None:
        fig.savefig(str(path), dpi=130)
    return fig


#: Columns of ``registration.csv`` drawn in the cohort registration figure: column, axis label,
#: QC threshold (``None`` for none).
COHORT_REGISTRATION_METRICS = (
    ('rms_mm', 'rms residual (mm)', 'rms'),
    ('rotation_change_deg', 'ICP rotation away from frame start (°)', 'rotation'),
    ('dropped_fraction', 'covered area off the map (%)', None),
)


def registration_cohort_caption(max_rms_mm: float, max_rotation_deg: float) -> List[str]:
    """
    The caption columns of the cohort registration QC figure.

    :param max_rms_mm: rms QC threshold (mm).
    :param max_rotation_deg: Rotation-change QC threshold (degrees).
    :return: ``[left_text, right_text]``, each already wrapped.
    """
    ref = f'{REFERENCE[0]}° {_load_name(REFERENCE[1])}'
    left = [
        'WHAT IS SHOWN',
        f'• One dot per knee and comparison: how well that scan\'s bone was aligned onto the '
        f'same knee\'s {ref} reference bone (rows: tibia, femur). The knee with the largest '
        f'value per comparison is named.',
        f'• rms residual = root-mean-square distance (mm) of the registered bone surface to the '
        f'reference bone surface, over the closest 90% of points. Dashed line = QC flag '
        f'threshold ({max_rms_mm:g} mm). For scale: one in-plane voxel is 0.31 mm, one slice '
        f'0.75 mm.',
    ]
    right = [
        '',
        f'• ICP rotation = how far the ICP rotated the bone away from the starting pose given '
        f'by matching the anatomical frames. Large values can mean the ICP slid into a wrong '
        f'(e.g. flipped) pose that a small rms cannot reveal; flagged above '
        f'{max_rotation_deg:g}° (dashed line).',
        '• Covered area off the map = share of the scan\'s cartilage-covered bone that lands '
        'outside the reference map after registration (femur: beyond the unrolled surface), so '
        'it is missing from the Δ maps. The reference itself already loses ~2% this way '
        '(registration.csv: reference_unmapped_fraction), so only the excess over that is due '
        'to the comparison. Always 0 for the tibia (every point projects onto the plateau).',
        '• x jitter only separates the dots.',
    ]
    return [_wrap(left), _wrap(right)]


def plot_registration_cohort(rows: Sequence[dict], path: Optional[str] = None,
                             max_rms_mm: Optional[float] = None,
                             max_rotation_deg: Optional[float] = None) -> Figure:
    """
    Cohort registration QC: residual, rotation change and dropped area per comparison.

    One row of panels per bone, one column per metric of
    :data:`COHORT_REGISTRATION_METRICS`; x = the seven comparisons, one jittered dot per knee,
    the QC thresholds as dashed lines and the knee with the largest value of each comparison
    named. Rows with a non-finite value are skipped.

    :param rows: ``registration.csv`` rows (``knee``, ``angle``, ``load``, ``bone`` and the
        metric columns).
    :param path: If given, the figure is saved there (PNG).
    :param max_rms_mm: rms threshold drawn (default
        :data:`~morphometry.cartilage.registration.REGISTRATION_MAX_RMS_MM`).
    :param max_rotation_deg: Rotation threshold drawn (default 15).
    :return: The matplotlib figure.
    """
    from morphometry.cartilage.registration import REGISTRATION_MAX_RMS_MM
    max_rms_mm = REGISTRATION_MAX_RMS_MM if max_rms_mm is None else max_rms_mm
    max_rotation_deg = 15.0 if max_rotation_deg is None else max_rotation_deg
    thresholds = {'rms': max_rms_mm, 'rotation': max_rotation_deg}
    height = 7.0 + 2.2
    bottom = 2.2 / height
    fig, axes = plt.subplots(2, len(COHORT_REGISTRATION_METRICS),
                             figsize=(FIGURE_WIDTH_IN, height), squeeze=False)
    labels = [f'{a}° {"L" if l == "belastet" else "U"}' for a, l in COMPARISONS]
    rng = np.random.default_rng(0)
    for r, bone in enumerate(('tibia', 'femur')):
        for c, (column, ylabel, threshold) in enumerate(COHORT_REGISTRATION_METRICS):
            ax = axes[r, c]
            if bone == 'tibia' and column == 'dropped_fraction':
                # Every tibial point projects onto the plateau plane: nothing to show.
                ax.set_axis_off()
                ax.text(0.5, 0.5, 'tibia: covered area off the map\nalways 0 (every point projects '
                        'onto the plateau)', ha='center', va='center', fontsize=8,
                        color='0.4', transform=ax.transAxes)
                continue
            for x, key in enumerate(COMPARISONS):
                vals = []
                for row in rows:
                    if (row.get('bone') == bone and int(row.get('angle', -1)) == key[0]
                            and row.get('load') == key[1]):
                        try:
                            v = float(row[column])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if np.isfinite(v):
                            vals.append((100.0 * v if column == 'dropped_fraction' else v,
                                         str(row.get('knee', ''))))
                if not vals:
                    continue
                ys = np.array([v for v, _ in vals])
                xs = x + rng.uniform(-0.18, 0.18, len(ys))
                ax.scatter(xs, ys, s=16, color='tab:blue', alpha=0.75, zorder=3)
                top_i = int(np.argmax(ys))
                ax.annotate(vals[top_i][1], (xs[top_i], ys[top_i]), xytext=(3, 3),
                            textcoords='offset points', fontsize=6.5)
            if threshold is not None:
                ax.axhline(thresholds[threshold], color='tab:red', ls='--', lw=1)
            ax.set_xticks(range(len(COMPARISONS)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f'{bone}: {ylabel}', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Registration QC per comparison (U = unloaded, L = loaded; flexion angle)',
                 fontsize=11)
    fig.tight_layout(rect=(0, bottom, 1, 1.0 - 0.45 / height))
    _draw_caption(fig, registration_cohort_caption(max_rms_mm, max_rotation_deg),
                  top=bottom - 0.15 / height)
    if path is not None:
        fig.savefig(str(path), dpi=130)
    return fig

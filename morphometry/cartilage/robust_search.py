"""
Robust smoothed-argmin search over a dense 2-D grid.

Shared by :func:`morphometry.cartilage.knee.find_thinnest_area` (thinnest area of the
tibiofemoral contact) and the thickness-change (delta) maps, which look for the largest
thinning hotspot. A plain ``argmin`` is dominated by single erroneous cells and by the rim
of the measured footprint, so the search is done on a NaN-aware median-smoothed copy, only
at cells whose whole smoothing window lies inside the measured footprint, and is then
refined to the raw minimum within a disc around the smoothed minimum.
"""
import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, generic_filter, minimum_filter


def edge_distance(footprint: np.ndarray, spacing: float) -> np.ndarray:
    """
    Distance (mm) of every footprint cell to the nearest cell outside the footprint.

    A Euclidean distance transform over a copy padded with one empty cell, so that a footprint
    touching the array border is treated as ending there: a boundary cell reads ``spacing``,
    the next ring ``2 * spacing`` (or a diagonal distance), and so on. Cells outside the
    footprint read 0.

    :param footprint: ``(H, W)`` bool mask of the cells a map covers.
    :param spacing: Cell size in mm.
    :return: ``(H, W)`` float array of distances in mm.
    """
    padded = np.pad(np.asarray(footprint, dtype=bool), 1, mode='constant',
                    constant_values=False)
    return distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1]


def interior_cells(footprint: np.ndarray, margin_mm: float, spacing: float) -> np.ndarray:
    """
    The cells of a footprint more than ``margin_mm`` from its boundary.

    The boundary is measured with :func:`edge_distance`, so a footprint touching the array
    border is still treated as ending there. Used by
    :func:`~morphometry.cartilage.knee.find_thinnest_area` (``edge_margin_mm``) and by the
    delta-map hotspot search (:func:`~morphometry.cartilage.deformation.find_largest_thinning`).

    :param footprint: ``(H, W)`` bool mask of the cells a map covers.
    :param margin_mm: Required distance to the nearest cell outside the footprint, in mm
        (strictly greater). ``<= 0`` returns the footprint unchanged (the same object).
    :param spacing: Cell size in mm.
    :return: ``(H, W)`` bool mask of the interior cells.
    """
    if margin_mm <= 0 or not footprint.any():
        return footprint
    return edge_distance(footprint, spacing) > margin_mm


def smoothed_argmin(grid: np.ndarray, valid: np.ndarray, spacing: float,
                    neighbourhood_mm: float, radius_mm: Optional[float] = None,
                    restrict: Optional[np.ndarray] = None) -> dict:
    """
    Find the minimum *area* of a grid, robust to single-cell outliers and footprint rims.

    The grid is median-smoothed (NaN-aware, reflection padding) with an odd kernel of
    ``round(neighbourhood_mm / spacing)`` cells (at least 3). The neighbourhood centre is the
    smoothed minimum over the searchable cells: those whose whole kernel window lies inside
    ``valid`` (and inside ``restrict`` if given) and whose smoothed value is defined. If no
    cell is searchable, ``fallback`` is set and every valid cell with a defined smoothed
    value is searched instead. The centre is then refined to the minimum of the raw grid
    within a disc around it, again over searchable cells only.

    No logging is done here; callers report ``fallback`` themselves.

    :param grid: ``(H, W)`` float grid; NaN where undefined.
    :param valid: ``(H, W)`` bool mask of the measured cells.
    :param spacing: Cell size in mm.
    :param neighbourhood_mm: Diameter (mm) of the median-smoothing neighbourhood.
    :param radius_mm: Radius (mm) of the refinement disc; defaults to ``neighbourhood_mm / 2``.
        The radius in cells is ``max(1, round(radius_mm / spacing))``.
    :param restrict: Optional ``(H, W)`` bool mask further restricting the searchable cells.
    :return: Dict with ``'center_index'`` and ``'point_index'`` (``(i, j)`` int tuples),
        ``'smoothed'`` (``(H, W)`` smoothed grid, NaN where undefined), ``'searchable'``
        (``(H, W)`` bool mask actually searched), ``'kernel_size'`` (int) and ``'fallback'``
        (bool).
    """
    k = int(round(neighbourhood_mm / spacing))
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    # Median smooth with reflection padding, ignoring NaN neighbours. A window holding no
    # measured cell at all yields NaN and warns; that is expected (the footprint is
    # irregular) and such cells are excluded from the search below.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        smoothed = generic_filter(grid, np.nanmedian, size=k, mode='reflect')

    # Restrict the neighbourhood-centre search to cells whose whole k x k neighbourhood is
    # inside the footprint. We are looking for the minimum *area*, so the area has to be
    # fully measured: a cell on the footprint boundary has its median taken over the few
    # neighbours that exist, and the boundary is exactly where the cartilage runs out and the
    # measured thickness falls towards zero, so a partially-observed window would win the
    # argmin on nearly every knee.
    searchable = minimum_filter(valid, size=k, mode='constant', cval=False) & ~np.isnan(smoothed)
    if restrict is not None:
        searchable &= restrict
    fallback = not searchable.any()
    if fallback:
        searchable = valid & ~np.isnan(smoothed)
    search = np.where(searchable, smoothed, np.inf)
    ci, cj = np.unravel_index(np.argmin(search), search.shape)

    # Refine to the actual minimal cell within a disc around the centre, on the raw grid. The
    # disc radius defaults to half the neighbourhood so it matches the smoothing extent (k is
    # a diameter) rather than doubling it. The refinement obeys the same restrictions, or it
    # would hand back a rim cell from just inside the disc.
    radius = neighbourhood_mm / 2.0 if radius_mm is None else radius_mm
    r = max(1, int(round(radius / spacing)))
    ii, jj = np.indices(grid.shape)
    disc = (ii - ci) ** 2 + (jj - cj) ** 2 <= r ** 2
    restricted = np.where(disc & searchable, grid, np.inf)
    pi, pj = np.unravel_index(np.argmin(restricted), restricted.shape)

    return {
        'center_index': (int(ci), int(cj)),
        'point_index': (int(pi), int(pj)),
        'smoothed': smoothed,
        'searchable': searchable,
        'kernel_size': k,
        'fallback': fallback,
    }

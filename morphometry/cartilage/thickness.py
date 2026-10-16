"""
Area-weighted surface cartilage thickness after Wirth & Eckstein 2008 (IEEE TMI 27(6), II-C).

The paper measures regional cartilage thickness as **Euclidean distances between two triangle
meshes**, each triangle weighted by its own area:

- ``ThC.cAB.cMe`` -- for every triangle of the cartilage-covered subchondral bone area (cAB),
  the distance to the articular cartilage surface (AC);
- ``ThC.cAB.bMe`` -- for every triangle of AC, the distance back to cAB;
- ``ThC.cAB.aMe = (cMe + bMe) / 2`` -- the robust average of the two directions, which is the
  number to quote;
- ``ThC.tAB.*`` -- the same over the *total* area of subchondral bone, with denuded
  (cartilage-free) triangles counted as 0.0 mm, i.e. diluted by the denuded fraction;
- ``aMax`` (single largest value), ``aMav`` (mean of the largest 1%), ``aSD`` (area-weighted
  standard deviation, a heterogeneity measure), ``aMin`` and ``aMiv`` (mean of the smallest 1%).

This replaced the voxel-column thickness outright (see
``docs/wirth_eckstein_2008_plan.md`` step 4): there is no longer a choice of thickness
"method", and rows from the two schemes must not be pooled.

Shared in-plane frame
---------------------
The per-position maps that :func:`morphometry.cartilage.knee.find_thinnest_area` overlaps are
keyed in **one frame shared by both bones**: the *tibial plateau frame*
(:class:`KneeFrame`, built by :func:`knee_frame_from_tibia`), with ``(u, v)`` integer grid
cells of ``grid_mm`` in the plateau plane. Its third axis -- the direction the projection
collapses -- is the mean tibial plate normal, which is the direction along which the
tibiofemoral joint makes contact at *every* flexion angle. The femoral contact patch migrates
from the distal condyle at full extension to the posterior condyle at 90 degrees of flexion,
but it stays over the plateau in this frame, so the tibial and femoral maps keep overlapping.
That overlap is where the two *projections* meet, which is close to but not the same as where
the joint touches: the trochlea projects onto the plateau too, and it supplies the femoral
value at the thinnest point in 16 of 93 whole-knee searches (31 of 93 on the patient-left
compartment). A femoral frame cannot even do this much: at 90 degrees the contact
surface faces along the femur's anterior-posterior axis, not along its shaft axis, so a
projection along either one loses the overlap at one end of the flexion range. The old
per-voxel-column keys were worse still -- they were array axes 0 and 1 whatever axis a
subregion was measured along, so the posterior femoral zones were keyed by ``(x, depth)``
and their columns collided.

A bone measured on its own (no tibia) gets a frame from its own subdivision, marked
``source != 'tibia'``; maps in different frames are not comparable and
:func:`~morphometry.cartilage.knee.find_thinnest_area` refuses to overlap them.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from morphometry.image_io import Image
from morphometry.cartilage import surface as S

# Side length (mm) of one in-plane cell of the shared thickness maps. 0.5 mm is finer than the
# through-plane sampling of a knee MRI (0.75 mm on dfg_knees) and a little coarser than its
# in-plane sampling (0.31 mm), so a cell holds a handful of tAB triangles (~0.1 mm^2 each) --
# enough to average over without blurring the map. It is a parameter everywhere it is used.
DEFAULT_GRID_MM = 0.5

# Share of the area at either end of a region's thickness distribution that aMav / aMiv
# average over (the paper's "top 1%" / "bottom 1%").
EXTREME_FRACTION = 0.01

# The statistics every region carries, in the order they are reported. There are two
# families and they must not be mixed: the bare names are **cAB** (over the cartilage that is
# there) and the ``tab_`` ones **tAB** (the same with denuded bone counted as 0.0 mm). So
# `aMe +- aSD` is self-consistent, `tab_aMin` is 0 for any region with any denuded bone while
# `aMin` is the thinnest cartilage actually measured, and `tab_aMax` / `tab_aMav` differ from
# their cAB twins only when a region is almost entirely denuded.
STATISTIC_KEYS = ('cMe', 'bMe', 'aMe', 'aSD', 'aMax', 'aMav', 'aMin', 'aMiv',
                  'tab_cMe', 'tab_bMe', 'tab_aMe', 'tab_aSD', 'tab_aMax', 'tab_aMav',
                  'tab_aMin', 'tab_aMiv',
                  'tab_area_mm2', 'cab_area_mm2', 'ac_area_mm2', 'denuded_fraction',
                  'n_tab_triangles', 'n_ac_triangles')

# Default single-linkage gap (mm) separating two sheets of a folded surface projected onto the
# shared plane (see _joint_facing_sheet). A cartilage-bearing surface is nowhere near 5 mm
# thick along the projection axis, while the two sheets of the femoral horseshoe are tens of
# millimetres apart, so the cut is not delicate.
DEFAULT_SHEET_GAP_MM = 5.0


# ----------------------------------------------------------------------------------------
# The shared in-plane frame
# ----------------------------------------------------------------------------------------

@dataclass(eq=False)
class KneeFrame:
    """
    The in-plane coordinate frame the tibial and femoral thickness maps are keyed in.

    A point's map key is the integer cell ``(u, v)`` of its projection onto the frame's
    ``e1 - e2`` plane, on a square grid of ``grid_mm``. The frame is anatomical (built from the
    subchondral bone surfaces, in world mm), so the keys do not depend on the voxel grid, on
    the image's direction matrix, or on where the knee sits in the scanner.

    :ivar origin: World (RAS, mm) origin of the frame: the midpoint of the two tibial plates'
        centres of gravity (or, for a femur-only frame, the trochlear notch point).
    :ivar axes: ``(3, 3)``, rows ``e1, e2, e3``, orthonormal and right-handed
        (``e1 x e2 = e3``). ``e1`` points to the patient's **right**, ``e2`` **anterior** and
        ``e3`` **superior** -- superior in the sense of the tibial plateau normal, i.e. the
        direction the joint is compressed along, not the world's.
    :ivar grid_mm: Side length of one grid cell in mm.
    :ivar source: Which bone the frame was derived from, ``'tibia'`` or ``'femur'``. Only
        frames from the same source (and with the same geometry) may be overlapped.
    """
    origin: np.ndarray
    axes: np.ndarray
    grid_mm: float = DEFAULT_GRID_MM
    source: str = 'tibia'

    def in_plane(self, points_world: np.ndarray) -> np.ndarray:
        """
        In-plane coordinates (mm) of world points, relative to the frame's origin.

        :param points_world: ``(N, 3)`` world mm (RAS).
        :return: ``(N, 2)`` coordinates along ``e1`` (patient right) and ``e2`` (anterior).
        """
        points = np.asarray(points_world, dtype=float).reshape(-1, 3) - self.origin
        return points @ self.axes[:2].T

    def cells(self, points_world: np.ndarray) -> np.ndarray:
        """
        Integer map cells of world points.

        Rounding is half-up (``floor(x + 0.5)``) rather than numpy's round-half-to-even, so
        that the cell of a point is a monotone function of its coordinate and two mirrored
        knees discretise the same way.

        :param points_world: ``(N, 3)`` world mm (RAS).
        :return: ``(N, 2)`` int array of ``(u, v)`` cell indices.
        """
        return np.floor(self.in_plane(points_world) / self.grid_mm + 0.5).astype(np.int64)

    def cell_centre(self, cell) -> np.ndarray:
        """
        The world (RAS, mm) point at the centre of a map cell, in the frame's own plane.

        The cell's real surface position is recorded per bone in
        :attr:`ThicknessMaps.positions`; this is the fallback for a cell with no surface
        recorded, and the in-plane position is exact either way.

        :param cell: A ``(u, v)`` cell index pair.
        :return: ``(3,)`` world coordinates.
        """
        u, v = float(cell[0]), float(cell[1])
        return self.origin + self.grid_mm * (u * self.axes[0] + v * self.axes[1])

    def matches(self, other: 'KneeFrame', tolerance: float = 1e-6) -> bool:
        """
        Whether two frames key points identically, so their maps may be overlapped.

        :param other: The other frame.
        :param tolerance: Absolute tolerance on the origin (mm) and the axes.
        :return: ``True`` if source, grid size, origin and axes agree.
        """
        if not isinstance(other, KneeFrame):
            return False
        return (self.source == other.source
                and abs(self.grid_mm - other.grid_mm) <= tolerance
                and np.allclose(self.origin, other.origin, atol=tolerance)
                and np.allclose(self.axes, other.axes, atol=tolerance))


def _right_handed_frame(origin: np.ndarray, right: np.ndarray, superior: np.ndarray,
                        grid_mm: float, source: str) -> KneeFrame:
    """
    Build a :class:`KneeFrame` from a superior and a (not necessarily perpendicular) right axis.

    ``e3`` is ``superior`` normalised, ``e1`` the component of ``right`` perpendicular to it,
    and ``e2 = e3 x e1``, which points anteriorly for an anatomical right/superior pair and
    makes ``(e1, e2, e3)`` right-handed.

    :param origin: ``(3,)`` world origin in mm.
    :param right: A direction towards the patient's right (need not be unit or perpendicular).
    :param superior: The plane normal, towards superior.
    :param grid_mm: Grid cell size in mm.
    :param source: ``'tibia'`` or ``'femur'``.
    :return: The frame.
    :raises ValueError: If the two directions are (numerically) parallel.
    """
    e3 = np.asarray(superior, dtype=float)
    e3 = e3 / np.linalg.norm(e3)
    e1 = np.asarray(right, dtype=float)
    e1 = e1 - (e1 @ e3) * e3
    norm = np.linalg.norm(e1)
    if norm < 1e-9:
        raise ValueError('the knee frame needs a right direction not parallel to the plate normal')
    e1 = e1 / norm
    return KneeFrame(origin=np.asarray(origin, dtype=float),
                     axes=np.stack([e1, np.cross(e3, e1), e3]),
                     grid_mm=float(grid_mm), source=source)


def knee_frame_from_tibia(subdivision, grid_mm: float = DEFAULT_GRID_MM) -> KneeFrame:
    """
    The shared in-plane frame of a knee, derived from its tibial plateau.

    The origin is the midpoint of the two plates' centres of gravity S, the superior axis the
    mean of the two plate normals ``b3`` (both oriented superior by
    :func:`~morphometry.cartilage.tibia_regions.principal_frame`) and the right axis the line
    from the patient-left to the patient-right plate's S. This is the plane the tibiofemoral
    joint is compressed along, which is why both bones' maps are keyed in it -- see the module
    docstring.

    :param subdivision: A :class:`~morphometry.cartilage.tibia_regions.TibiaSubdivision`.
    :param grid_mm: Side length of one map cell in mm.
    :return: The :class:`KneeFrame` with ``source='tibia'``.
    """
    left, right = subdivision.plates['left'], subdivision.plates['right']
    return _right_handed_frame(origin=0.5 * (left.centroid + right.centroid),
                               right=right.centroid - left.centroid,
                               superior=left.axes[2] + right.axes[2],
                               grid_mm=grid_mm, source='tibia')


def knee_frame_from_femur(subdivision, grid_mm: float = DEFAULT_GRID_MM) -> KneeFrame:
    """
    A fallback in-plane frame derived from the femur alone, for a femur measured without a tibia.

    It is the femoral frame itself (origin at the trochlear notch, ``e1`` the posterior
    condylar axis, ``e3`` the shaft axis), which is *not* the tibiofemoral contact plane at
    anything but full extension. Maps keyed in it are therefore marked ``source='femur'`` and
    :func:`~morphometry.cartilage.knee.find_thinnest_area` refuses to overlap them with tibial
    maps; they are still a valid per-position map of the femoral cartilage on its own.

    :param subdivision: A :class:`~morphometry.cartilage.femur_regions.FemurSubdivision`.
    :param grid_mm: Side length of one map cell in mm.
    :return: The :class:`KneeFrame` with ``source='femur'``.
    """
    frame = subdivision.frame
    return _right_handed_frame(origin=frame.notch_point, right=frame.ml_axis,
                               superior=frame.proximal_axis, grid_mm=grid_mm, source='femur')


# ----------------------------------------------------------------------------------------
# Area-weighted statistics
# ----------------------------------------------------------------------------------------

def _extreme_mean(values: np.ndarray, weights: np.ndarray, fraction: float,
                  largest: bool) -> float:
    """
    Area-weighted mean of the largest (or smallest) ``fraction`` of a weighted sample.

    The sample is sorted and weights accumulated until ``fraction`` of the total weight is
    reached; the triangle that straddles the cut contributes only the part of its area that is
    still needed, so the result is a continuous function of the weights (the paper's "average
    of the top 1% of maximal values").

    :param values: ``(N,)`` thickness values in mm.
    :param weights: ``(N,)`` non-negative areas in mm^2.
    :param fraction: Share of the total weight to average over, in ``(0, 1]``.
    :param largest: ``True`` for the top fraction, ``False`` for the bottom one.
    :return: The weighted mean of that tail (``NaN`` for an empty or zero-weight sample).
    """
    if len(values) == 0 or weights.sum() <= 0:
        return float('nan')
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    v, w = values[order], weights[order]
    cumulative = np.cumsum(w)
    target = fraction * cumulative[-1]
    last = min(int(np.searchsorted(cumulative, target)), len(v) - 1)
    taken = w[:last + 1].astype(float).copy()
    taken[-1] -= max(0.0, cumulative[last] - target)
    if taken.sum() <= 0:                    # the first triangle alone exceeds the fraction
        return float(v[0])
    return float((v[:last + 1] * taken).sum() / taken.sum())


def weighted_statistics(values, weights, extreme_fraction: float = EXTREME_FRACTION) -> dict:
    """
    The paper's thickness parameters of one area-weighted thickness sample.

    Entries with a non-positive weight are dropped before anything is computed -- they carry no
    area, so they must not reach ``aMax`` / ``aMin`` either (the denuded placeholder of
    :func:`_region_statistics` has weight 0 on a fully covered subregion, and would otherwise
    read that subregion's minimum as 0 mm). So are non-finite values.

    :param values: ``(N,)`` thickness values in mm.
    :param weights: ``(N,)`` areas in mm^2 (any positive scaling; only ratios matter).
    :param extreme_fraction: Share of the area aMav / aMiv average over (paper: 0.01).
    :return: ``{'aMe', 'aSD', 'aMax', 'aMav', 'aMin', 'aMiv'}`` in mm, where ``aMe`` is the
        area-weighted mean, ``aSD`` the area-weighted (population) standard deviation, ``aMax``
        / ``aMin`` the single extreme values and ``aMav`` / ``aMiv`` the means of the extreme
        ``extreme_fraction`` of the area. All ``NaN`` for an empty or zero-weight sample.
    """
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if len(values) == 0 or weights.sum() <= 0:
        return {key: float('nan') for key in ('aMe', 'aSD', 'aMax', 'aMav', 'aMin', 'aMiv')}
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return {'aMe': mean, 'aSD': float(np.sqrt(max(0.0, variance))),
            'aMax': float(values.max()), 'aMin': float(values.min()),
            'aMav': _extreme_mean(values, weights, extreme_fraction, largest=True),
            'aMiv': _extreme_mean(values, weights, extreme_fraction, largest=False)}


# ----------------------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------------------

class ThicknessMaps(dict):
    """
    Per-subregion cartilage thickness of one bone: the position maps plus the paper's statistics.

    It *is* a ``{subregion: {(u, v): thickness_mm}}`` dict, so every existing consumer (the
    plotting code, :func:`~morphometry.cartilage.knee.find_thinnest_area`, the scripts' CSV /
    JSON output) keeps working unchanged. The keys ``(u, v)`` are integer cells of the shared
    in-plane :class:`KneeFrame` (**not** voxel indices any more, and not array axes 0 and 1).
    The value of a cell is the area-weighted mean of the ``cAB -> AC`` distances of the tAB
    triangles whose centroids fall in it, with denuded triangles counted as 0 mm -- a
    bone-anchored thickness map, the ``cMe`` direction only. The two-directional
    :attr:`statistics` are the numbers to quote.

    :ivar frame: The :class:`KneeFrame` the keys live in.
    :ivar bone: ``'tibia'`` or ``'femur'`` (which bone this is), for messages.
    :ivar statistics: ``{subregion: {statistic: value}}`` with the keys of
        :data:`STATISTIC_KEYS`.
    :ivar positions: ``{(u, v): (3,) world mm}`` -- the articular-surface point of each cell
        (area-weighted mean of the AC triangle centroids in it, falling back to the tAB
        centroids for a fully denuded cell). Used to place the thinnest-point marker in 3D.
    :ivar triangle_thickness: ``(n_cells,)`` ``cAB -> AC`` distance per triangle of the
        subdivision mesh, aligned with its cells, 0 mm on denuded triangles. This is what
        :meth:`morphometry.cartilage.knee.Tibia.plot_thickness` colours the tAB with.
    """

    def __init__(self, maps: dict, frame: KneeFrame, bone: str = '', statistics: dict = None,
                 positions: dict = None, triangle_thickness: np.ndarray = None):
        """
        :param maps: ``{subregion: {(u, v): thickness_mm}}``.
        :param frame: The in-plane frame the keys live in.
        :param bone: ``'tibia'`` or ``'femur'``.
        :param statistics: ``{subregion: {statistic: value}}``.
        :param positions: ``{(u, v): world point}`` of the articular surface.
        :param triangle_thickness: Per-triangle distances aligned with the subdivision mesh.
        """
        super().__init__(maps)
        self.frame = frame
        self.bone = bone
        self.statistics = {} if statistics is None else statistics
        self.positions = {} if positions is None else positions
        self.triangle_thickness = triangle_thickness

    def subset(self, keys) -> 'ThicknessMaps':
        """
        The same maps restricted to some subregions, keeping the frame and the statistics.

        Used by :func:`~morphometry.cartilage.knee.split_tibia_thickness_by_compartment` so a
        per-compartment subset is still recognisably keyed in the shared frame.

        :param keys: The subregion keys to keep (missing ones are skipped).
        :return: A new :class:`ThicknessMaps` sharing the per-subregion dicts (not copied).
        """
        keys = [k for k in keys if k in self]
        return ThicknessMaps({k: self[k] for k in keys}, frame=self.frame, bone=self.bone,
                             statistics={k: self.statistics[k] for k in keys
                                         if k in self.statistics},
                             positions=self.positions,
                             triangle_thickness=self.triangle_thickness)


def thickness_frame(maps) -> Optional[KneeFrame]:
    """
    The in-plane frame of a thickness dict, or ``None`` for a plain dict without one.

    :param maps: A :class:`ThicknessMaps` or any ``{subregion: {key: value}}`` dict.
    :return: The :class:`KneeFrame`, or ``None``.
    """
    frame = getattr(maps, 'frame', None)
    return frame if isinstance(frame, KneeFrame) else None


# ----------------------------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------------------------

def _cell_means(cells: np.ndarray, values: np.ndarray, weights: np.ndarray) -> dict:
    """
    Area-weighted mean value per in-plane grid cell.

    :param cells: ``(N, 2)`` int cell indices.
    :param values: ``(N,)`` values (mm).
    :param weights: ``(N,)`` areas (mm^2).
    :return: ``{(u, v): mean}`` over the cells present.
    """
    sums: Dict[Tuple[int, int], float] = {}
    areas: Dict[Tuple[int, int], float] = {}
    for cell, value, weight in zip(map(tuple, cells.tolist()), values, weights):
        sums[cell] = sums.get(cell, 0.0) + float(value) * float(weight)
        areas[cell] = areas.get(cell, 0.0) + float(weight)
    return {cell: sums[cell] / areas[cell] for cell in sums if areas[cell] > 0}


def _cell_positions(cells: np.ndarray, points: np.ndarray, weights: np.ndarray) -> dict:
    """
    Area-weighted mean world position per in-plane grid cell.

    :param cells: ``(N, 2)`` int cell indices.
    :param points: ``(N, 3)`` world mm.
    :param weights: ``(N,)`` areas (mm^2).
    :return: ``{(u, v): (3,) world mm}``.
    """
    sums: Dict[Tuple[int, int], np.ndarray] = {}
    areas: Dict[Tuple[int, int], float] = {}
    for cell, point, weight in zip(map(tuple, cells.tolist()), points, weights):
        sums[cell] = sums.get(cell, 0.0) + point * float(weight)
        areas[cell] = areas.get(cell, 0.0) + float(weight)
    return {cell: sums[cell] / areas[cell] for cell in sums if areas[cell] > 0}


def _joint_facing_sheet(cells: np.ndarray, along: np.ndarray, facing_sign: int,
                        sheet_gap_mm: float) -> np.ndarray:
    """
    Keep, per in-plane cell, only the surface sheet facing the other bone.

    Projecting a bone's whole subchondral area onto the shared plane folds a curved surface
    onto itself: the femoral horseshoe wraps round from the trochlea to the posterior condyle,
    so one cell can collect triangles from two sheets tens of millimetres apart along the
    projection axis. Averaging those into one map value mixes two different places, and the
    cell's recorded articular position then lands *inside* the bone, halfway between the
    sheets -- which is where the thinnest-point marker and the femoral arc coordinate read it
    from. Measured on the 96 ``dfg_knees``: cells whose contributing centroids span more than
    5 mm are 10 / 7 / 13 / 22% of the femoral cells at 0 / 30 / 60 / 90 degrees of flexion
    (26-47% of the tAB *area*, since the folded parts are large), and 14 / 5 / 2 / 18% of the
    cells the thinnest-point search actually uses; the largest spreads are 30-60 mm.

    The sheet kept is the one nearest the joint: for the femur the lowest along the frame's
    third axis (the tibia is below it), for the tibia the highest. A triangle belongs to it
    when it is within ``sheet_gap_mm`` of that extreme within its own cell -- a single-linkage
    cut, which is safe because the two sheets of a folded surface are far further apart than
    the thickness of either.

    This only shapes the per-position maps and positions; the regional statistics are computed
    per triangle and are untouched by it.

    :param cells: ``(N, 2)`` int in-plane cell of each triangle.
    :param along: ``(N,)`` coordinate of each triangle along the frame's projection axis.
    :param facing_sign: ``-1`` to keep the lowest sheet (a femur, whose joint is below it),
        ``+1`` the highest (a tibia). ``0`` keeps everything.
    :param sheet_gap_mm: How far from its cell's extreme a triangle may be and still count as
        the same sheet.
    :return: ``(N,)`` bool, True for the triangles of the joint-facing sheet.
    """
    if facing_sign == 0 or len(cells) == 0:
        return np.ones(len(cells), dtype=bool)
    signed = -facing_sign * np.asarray(along, dtype=float)      # smallest = nearest the joint
    _, inverse = np.unique(np.asarray(cells), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    nearest = np.full(inverse.max() + 1, np.inf)
    np.minimum.at(nearest, inverse, signed)
    return signed <= nearest[inverse] + sheet_gap_mm


def _region_statistics(tab_distances: np.ndarray, tab_areas: np.ndarray, covered: np.ndarray,
                       ac_distances: np.ndarray, ac_areas: np.ndarray,
                       extreme_fraction: float) -> dict:
    """
    The paper's thickness parameters of one subregion, from both measurement directions.

    The two directions are pooled into a single area-weighted sample in which each direction
    carries half the total weight, so the pooled mean is exactly ``(cMe + bMe) / 2 = aMe`` and
    ``aSD`` / ``aMax`` / ``aMav`` / ``aMin`` / ``aMiv`` describe one distribution rather than
    being averaged over two. For the tAB variants the denuded tAB triangles enter the
    ``cAB -> AC`` side with a thickness of 0.0 mm, and the ``AC -> cAB`` side -- which has no
    denuded triangles of its own -- is diluted by the same denuded area fraction through one
    0.0 mm entry, so that ``tab_aMe = aMe * cAB / tAB`` holds exactly, as it does for the
    ``cMe`` direction by construction.

    :param tab_distances: ``(T,)`` ``cAB -> AC`` distance of each of the subregion's tAB
        triangles in mm, already 0.0 on the denuded ones.
    :param tab_areas: ``(T,)`` their areas in mm^2.
    :param covered: ``(T,)`` bool, True for the cAB (cartilage-covered) triangles.
    :param ac_distances: ``(A,)`` ``AC -> cAB`` distance of each of the subregion's AC
        triangles in mm.
    :param ac_areas: ``(A,)`` their areas in mm^2.
    :param extreme_fraction: Share of the area aMav / aMiv average over.
    :return: ``{statistic: value}`` with the keys of :data:`STATISTIC_KEYS`.
    """
    tab_area = float(tab_areas.sum())
    cab_area = float(tab_areas[covered].sum())
    ac_area = float(ac_areas.sum())
    empty = {key: float('nan') for key in STATISTIC_KEYS}
    empty.update(tab_area_mm2=tab_area, cab_area_mm2=cab_area, ac_area_mm2=ac_area,
                 n_tab_triangles=int(len(tab_distances)), n_ac_triangles=int(len(ac_distances)),
                 denuded_fraction=(1.0 - cab_area / tab_area) if tab_area > 0 else float('nan'))
    if cab_area <= 0 or ac_area <= 0:
        return empty

    # Each direction normalised to half the weight, so the pooled mean is (cMe + bMe) / 2.
    covered_fraction = cab_area / tab_area
    cab_stats = weighted_statistics(
        np.concatenate([tab_distances[covered], ac_distances]),
        np.concatenate([0.5 * tab_areas[covered] / cab_area, 0.5 * ac_areas / ac_area]),
        extreme_fraction)
    tab_stats = weighted_statistics(
        np.concatenate([tab_distances, ac_distances, [0.0]]),
        np.concatenate([0.5 * tab_areas / tab_area, 0.5 * covered_fraction * ac_areas / ac_area,
                        [0.5 * (1.0 - covered_fraction)]]),
        extreme_fraction)

    cme = float(np.average(tab_distances[covered], weights=tab_areas[covered]))
    bme = float(np.average(ac_distances, weights=ac_areas))
    out = dict(empty)
    out.update(cMe=cme, bMe=bme,
               tab_cMe=cme * covered_fraction, tab_bMe=bme * covered_fraction)
    # Both families, each internally consistent: the cAB one describes the cartilage that is
    # there, the tAB one the same diluted by denuded bone counted as 0.0 mm. Mixing them --
    # quoting aMe +- a tAB SD, say -- is exactly what keeping them apart prevents.
    out.update({key: cab_stats[key] for key in ('aMe', 'aSD', 'aMax', 'aMav', 'aMin', 'aMiv')})
    out.update({f'tab_{key}': tab_stats[key]
                for key in ('aMe', 'aSD', 'aMax', 'aMav', 'aMin', 'aMiv')})
    return out


def measure_thickness(image: Image, subdivision, region_keys: Sequence[str], bone_label: int,
                      cartilage_label: int, frame: KneeFrame, bone: str = '',
                      map_keys: Optional[Dict[str, str]] = None,
                      coverage_tolerance_mm: Optional[float] = None,
                      extreme_fraction: float = EXTREME_FRACTION,
                      facing_sign: int = 0,
                      sheet_gap_mm: float = DEFAULT_SHEET_GAP_MM,
                      max_rim_angle_deg: float = S.DEFAULT_RIM_ANGLE_DEG,
                      probe_depths: Sequence[float] = S.DEFAULT_PROBE_DEPTHS) -> ThicknessMaps:
    """
    Measure the area-weighted surface cartilage thickness of one bone, per subregion.

    Steps (all in world mm):

    1. The labelled tAB is the subdivision's own mesh, whose ``'region'`` cell array says which
       subregion each triangle belongs to and whose ``'covered'`` array separates cAB from
       denuded bone.
    2. The articular cartilage surface AC is reconstructed with
       :func:`~morphometry.cartilage.surface.articular_surface_mesh`, and each of its triangles
       takes the subregion of its nearest tAB triangle (the paper's transfer of the regions
       from the bone to the cartilage surface).
    3. Every cAB triangle's exact distance to the AC mesh and every AC triangle's exact
       distance to the cAB mesh are computed
       (:func:`~morphometry.cartilage.surface.surface_distances`); denuded tAB triangles are
       assigned 0.0 mm rather than measured.
    4. Per subregion those two area-weighted samples give ``cMe``, ``bMe``, ``aMe`` and the
       distribution parameters (:func:`_region_statistics`).
    5. The per-position map is the ``cAB -> AC`` distance averaged by area within each cell of
       the shared in-plane ``frame`` (see the module docstring), over the joint-facing sheet
       only (:func:`_joint_facing_sheet`).

    :param image: The (LPI-oriented) knee ``Segmentation``; either image backend.
    :param subdivision: The bone's subdivision -- a
        :class:`~morphometry.cartilage.tibia_regions.TibiaSubdivision` or
        :class:`~morphometry.cartilage.femur_regions.FemurSubdivision`. Only its ``mesh`` and
        ``region_codes_at`` are used.
    :param region_keys: The subregion key per integer region code, i.e.
        :data:`~morphometry.cartilage.tibia_regions.TIBIA_REGION_KEYS` or
        :data:`~morphometry.cartilage.femur_regions.FEMUR_REGION_KEYS`.
    :param bone_label: Label of the bone in the segmentation.
    :param cartilage_label: Label of that bone's cartilage.
    :param frame: The shared in-plane :class:`KneeFrame` the maps are keyed in.
    :param bone: ``'tibia'`` or ``'femur'``, carried on the result for messages.
    :param map_keys: Optional ``{region_key: output_key}`` renaming for the returned dicts (the
        femoral extension zones keep the names the thickness dicts have always used for them).
    :param coverage_tolerance_mm: cAB / AC distance tolerance in mm, see
        :func:`~morphometry.cartilage.surface.cartilage_coverage` (``None`` = its default).
    :param extreme_fraction: Share of the area aMav / aMiv average over (paper: 0.01).
    :param facing_sign: Which sheet of a surface folded by the projection enters the maps:
        ``-1`` the lowest along the frame's third axis (a femur), ``+1`` the highest (a tibia),
        ``0`` (the default, for a caller that knows the surface does not fold) keeps all of it.
        See :func:`_joint_facing_sheet`.
    :param sheet_gap_mm: Single-linkage gap separating the sheets, see
        :func:`_joint_facing_sheet`.
    :param max_rim_angle_deg: AC rim-exclusion angle, see
        :func:`~morphometry.cartilage.surface.articular_surface_mesh`.
    :param probe_depths: AC under-probe depths in voxels, same reference.
    :return: A :class:`ThicknessMaps`.
    :raises ValueError: If the subdivision mesh is empty or no articular surface can be
        reconstructed (the cartilage label is absent or nowhere on this bone).
    """
    mesh = subdivision.mesh
    if mesh is None or mesh.n_cells == 0:
        raise ValueError(f'the {bone or "bone"} subdivision has no subchondral bone area to '
                         'measure the cartilage thickness on')
    tab_areas = S.triangle_areas(mesh)
    tab_centroids = S.triangle_centroids(mesh)
    tab_regions = np.asarray(mesh.cell_data['region'])
    covered = np.asarray(mesh.cell_data['covered'], dtype=bool)

    ac = S.articular_surface_mesh(image, bone_label, cartilage_label,
                                  bone_mesh=getattr(subdivision, 'bone_mesh', None),
                                  tolerance_mm=coverage_tolerance_mm,
                                  max_rim_angle_deg=max_rim_angle_deg,
                                  probe_depths=probe_depths)
    if ac.n_cells == 0:
        raise ValueError(f'no articular cartilage surface could be reconstructed for the '
                         f'{bone or "bone"} (cartilage label {cartilage_label})')
    ac_areas = S.triangle_areas(ac)
    ac_centroids = S.triangle_centroids(ac)
    ac_regions = subdivision.region_codes_at(ac_centroids)

    cab = S.extract_faces(mesh, covered)
    # Denuded tAB triangles are not measured: the paper assigns them 0.0 mm thickness.
    tab_distances = np.zeros(mesh.n_cells)
    tab_distances[covered] = S.surface_distances(tab_centroids[covered], ac)
    ac_distances = S.surface_distances(ac_centroids, cab)

    tab_cells = frame.cells(tab_centroids)
    # Only the joint-facing sheet enters the per-position maps (see _joint_facing_sheet).
    tab_sheet = _joint_facing_sheet(tab_cells, tab_centroids @ frame.axes[2], facing_sign,
                                    sheet_gap_mm)
    maps, statistics = {}, {}
    for code, key in enumerate(region_keys):
        in_tab = tab_regions == code
        in_ac = ac_regions == code
        out_key = (map_keys or {}).get(key, key)
        statistics[out_key] = _region_statistics(
            tab_distances[in_tab], tab_areas[in_tab], covered[in_tab],
            ac_distances[in_ac], ac_areas[in_ac], extreme_fraction)
        keep = in_tab & tab_sheet
        maps[out_key] = _cell_means(tab_cells[keep], tab_distances[keep], tab_areas[keep])

    # The articular surface position of every cell, for placing markers in 3D; cells that hold
    # only denuded bone fall back to the subchondral surface itself. Both are restricted to
    # the joint-facing sheet, so a folded surface never averages a position into the bone.
    ac_cells = frame.cells(ac_centroids)
    ac_sheet = _joint_facing_sheet(ac_cells, ac_centroids @ frame.axes[2], facing_sign,
                                   sheet_gap_mm)
    positions = _cell_positions(tab_cells[tab_sheet], tab_centroids[tab_sheet],
                                tab_areas[tab_sheet])
    positions.update(_cell_positions(ac_cells[ac_sheet], ac_centroids[ac_sheet],
                                     ac_areas[ac_sheet]))
    return ThicknessMaps(maps, frame=frame, bone=bone, statistics=statistics,
                         positions=positions, triangle_thickness=tab_distances)

"""
Map coordinates for knee-cartilage deformation (thickness-change) maps.

Why this coordinate exists
--------------------------
A thickness-change ("delta") map compares one knee's femoral cartilage between two
configurations (flexion angle x load). Each configuration's thickness is carried into the
**reference** scan's world frame by a rigid registration of the femoral *bone*
(:mod:`morphometry.cartilage.registration`), and must then be drawn on a 2-D map in which one
pixel is one anatomical spot of the reference femur. The femoral articular surface is a
horseshoe that wraps round the distal condyle, so no single projection plane shows it without
folding: in the tibial plateau plane the trochlea and the posterior condyle land on top of
each other. The map is therefore **unrolled**: horizontal = medial-lateral position, vertical =
arc length along the condyle's sagittal bone profile, from the trochlea round the distal
condyle (the 0 degree contact) to the posterior condyle (the 90 degree contact), so each of
those appears exactly once.

Anchoring. The zero of the arc is a **bone** landmark -- the trochlear-notch plane of the
femoral frame (:class:`~morphometry.cartilage.femur_regions.FemoralFrame`, ``notch_ap``, read
off the bone by :func:`~morphometry.cartilage.femur_regions.find_trochlear_notch`) -- and arc
lengths are absolute millimetres, never normalised. The labelled cartilage's own extent is not
reproducible between acquisitions of one knee (how far the trochlea is labelled varies), so
anchoring at, or normalising by, the cartilage edge
(as :func:`~morphometry.cartilage.femur_regions.articular_arc_fraction` does) would shift every
point of the map by an amount that has nothing to do with deformation.

Per knee, computed once. The coordinate is built on the **reference** scan's bone and frame
only; every other configuration is first registered into that scan's world frame and then
looked up in the same :class:`UnrolledFemur`. It is not a cohort atlas: millimetres of arc on
two different knees are different anatomy.

Construction (:meth:`UnrolledFemur.from_reference`)
---------------------------------------------------
1. The reference bone surface is cropped to its distal ``REGISTRATION_CROP_MM`` along the shaft
   axis (:func:`~morphometry.cartilage.registration.crop_to_articular_end`), so the proximal
   field-of-view edge of the shaft never enters a contour.
2. At every medial-lateral level ``notch_ml + k * slice_mm`` strictly inside the cropped
   bone's extent the mesh is cut by the sagittal plane (normal ``ml_axis``). The arc origin of
   the level is the profile's **crossing of the notch plane** ``ap = notch_ap`` with the
   smallest proximal coordinate, over all contour components -- the distal crossing. Only if
   there is no crossing, or the lowest vertex within ``NOTCH_SNAP_MM`` in front of the plane lies
   more than ``NOTCH_CORNER_DROP_MM`` below the lowest crossing (the profile merely touches the
   plane at a corner, e.g. an intercondylar wall lying nearly in it, and the lowest crossing is
   the shaft's underside), is that corner taken instead (:func:`_origin_candidates`). A condyle
   whose lowest point lies anterior of the plane keeps its origin at the crossing. The component
   holding the origin is the profile of that level; along it, signed arc length is measured
   from the origin, **positive posteriorly** (the direction in which ``ap`` decreases: round
   the distal condyle to the posterior condyle) and negative anteriorly (up the trochlea). A
   closed loop (a level whose whole profile lies inside the crop) is opened at its most
   proximal point, which is non-articular, so the arc runs from there one way round to the
   other.
3. Every vertex of the cropped bone mesh takes the arc of its nearest contour point (in the
   sagittal plane) on the two levels that bracket it, linearly interpolated in the
   medial-lateral coordinate. A level gives no value for the vertex if that nearest point
   belongs to a *different* component than the profile (the shaft, say) or lies further than
   ``max_distance_mm`` from it (the vertex sits on a side wall the plane misses, e.g. the
   condyle's wall into the fossa next to a fossa level); then the other level's value is
   used alone, and with neither the vertex is ``NaN``. Beyond the outermost levels a vertex
   takes the outermost value up to one ``slice_mm`` out (the rounded end of a condyle).

:meth:`UnrolledFemur.coordinates` then reads, for any world point, column 0 exactly as
``(p - notch_point) . ml_axis`` and column 1 from the nearest bone vertex (``cKDTree``);
points further than ``max_distance_mm`` from the reference bone, or outside the range of
computed levels, are ``NaN``. Callers count and drop those points rather than mapping them to
a wrong spot (cartilage of another configuration that extends beyond the reference's bone).

The intercondylar region
------------------------
Medial-lateral levels through the trochlear groove / intercondylar fossa have a profile that
runs from the groove into the fossa roof: the notch plane is the fossa's anterior border, so
the origin there is where the groove meets the fossa and positive arc climbs the fossa roof
(no articular cartilage). These levels get finite coordinates, continuous with their
neighbours, but they are only as reproducible as the fossa's shape at the notch plane: on the
test phantom, whose fossa wall is a plane 0.5 mm in front of and *parallel to* the notch plane,
the origin is a crossing up to ``NOTCH_CORNER_DROP_MM`` above the foot of that wall in one
voxelisation and the foot itself in another, so the arc moves by up to ~2 mm between two
voxelisations of the same bone (against <= 0.6 mm on the condyles). A real fossa roof is oblique to the notch plane,
so this is the phantom's worst case, not a measured property of real knees. Before the
corner rule above, such a level could jump to a crossing ~20 mm proximal (the underside of the
shaft); that is pinned against by ``tests/unit/test_unrolled_femur.py``.

Delta maps (:func:`knee_delta_maps`)
------------------------------------
Per knee and bone, every configuration's measurement (:class:`BoneMeasurement`, built from a
subdivision and its :class:`~morphometry.cartilage.thickness.ThicknessMaps` by
:func:`bone_measurement`) is rigidly registered onto the reference's bone
(:data:`REFERENCE`, unloaded 30 degrees), its **covered** tAB triangle centroids are carried
into the reference's world frame and binned, area-weighted, into the reference's map
coordinates: the tibial plateau plane of the reference :class:`KneeFrame` for the tibia, the
:class:`UnrolledFemur` of the reference femur for the femur. Denuded triangles are excluded
(their cells stay empty), never counted as 0. All maps of one knee and bone share one
:class:`MapGrid`; ``delta = configuration - reference`` per cell, defined only where both are.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from morphometry.cartilage.femur_regions import FEMUR_REGION_KEYS, FemoralFrame, _polyline_components
from morphometry.cartilage.registration import (
    REGISTRATION_CROP_MM, RegistrationResult, RigidTransform, crop_to_articular_end,
    register_bone,
)
from morphometry.cartilage.robust_search import smoothed_argmin
from morphometry.cartilage.surface import triangle_areas, triangle_centroids
from morphometry.cartilage.thickness import KneeFrame, ThicknessMaps, knee_frame_from_tibia
from morphometry.cartilage.tibia_regions import TIBIA_REGION_KEYS

logger = logging.getLogger('morphometry')

#: Largest distance (mm) anterior of the notch plane at which a profile vertex can serve as the
#: arc origin (a corner) when the profile has no distal crossing of the plane (see
#: :func:`_origin_candidates`). Vertices further in front never compete with a crossing.
NOTCH_SNAP_MM = 1.0

#: How far (mm) below the lowest notch-plane crossing such a corner must lie to replace it.
#: Near a crossing the profile within ``NOTCH_SNAP_MM`` in front of the plane is at most a
#: fraction of a millimetre lower (a condyle of radius 20 mm whose lowest point is 3 mm anterior
#: of the plane: ~0.1 mm), whereas a crossing that is not distal at all (the shaft's underside
#: behind a fossa wall) lies ~20 mm above the corner on the phantom.
NOTCH_CORNER_DROP_MM = 2.0


@dataclass
class _Level:
    """
    One sagittal cross-section of the reference bone, parametrised for the arc lookup.

    :ivar ml: Medial-lateral coordinate of the cutting plane (absolute, ``p . ml_axis``).
    :ivar tree: KD-tree over the in-plane coordinates ``(ap, proximal)`` of **all** contour
        points of the cross-section (every component, not only the profile).
    :ivar arc: ``(M,)`` signed arc length (mm) of each of those points; ``NaN`` for points on
        components other than the profile.
    """
    ml: float
    tree: cKDTree
    arc: np.ndarray


def _edge_set(lines: np.ndarray) -> set:
    """
    The undirected edges of a polyline connectivity array.

    :param lines: A ``pv.PolyData.lines`` connectivity array.
    :return: A set of ``(min_id, max_id)`` point-index pairs.
    """
    edges, i = set(), 0
    lines = np.asarray(lines).ravel()
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1:i + 1 + n]
        for a, b in zip(ids[:-1], ids[1:]):
            edges.add((min(int(a), int(b)), max(int(a), int(b))))
        i += n + 1
    return edges


def _origin_candidates(points_2d: np.ndarray, notch_ap: float):
    """
    Candidate arc origins on one ordered polyline: notch-plane crossings and corner vertices.

    The arc origin is the profile's **crossing of the notch plane** ``ap = notch_ap`` with the
    smallest proximal coordinate (the distal crossing). The vertices lying on or at most
    ``NOTCH_SNAP_MM`` in front of the plane are returned as a second, fallback kind: a level
    whose profile only *touches* the plane at a corner -- an intercondylar wall lying (nearly)
    in it, as on the phantom, whose smoothed wall crosses the plane in one voxelisation and not
    in another -- has no distal crossing, and its lowest genuine crossing can be far proximal
    (the underside of the shaft). :func:`_level_profile` uses a corner only in that case (see
    ``NOTCH_CORNER_DROP_MM``); vertices further in front never compete, so a condyle whose
    lowest point lies a few mm anterior of the notch plane keeps its origin at the crossing.

    :param points_2d: ``(M, 2)`` ordered in-plane coordinates ``(ap, proximal)``.
    :param notch_ap: The notch plane's ``ap`` coordinate.
    :return: ``(crossings, corners)``: two lists of ``(proximal, segment_index, t)`` -- the
        candidate's proximal coordinate, the index ``j`` of the segment ``(j, j + 1)`` it lies
        on and the fraction ``t`` in ``[0, 1]`` along it.
    """
    f = points_2d[:, 0] - notch_ap
    crossings, corners = [], []
    last = len(f) - 1
    for j in range(len(f)):
        if 0.0 <= f[j] <= NOTCH_SNAP_MM:
            segment, t = (j, 0.0) if j < last else (j - 1, 1.0)
            corners.append((float(points_2d[j, 1]), segment, t))
        if j < last and f[j] * f[j + 1] < 0:
            t = f[j] / (f[j] - f[j + 1])
            proximal = points_2d[j, 1] + t * (points_2d[j + 1, 1] - points_2d[j, 1])
            crossings.append((float(proximal), j, float(t)))
    return crossings, corners


def _profile_arc(ordered_2d: np.ndarray, segment: int, t: float,
                 probe_mm: float = 1.0) -> np.ndarray:
    """
    Signed arc length along an ordered polyline from a point on one of its segments.

    :param ordered_2d: ``(M, 2)`` ordered in-plane coordinates ``(ap, proximal)``.
    :param segment: Index ``j`` of the segment ``(j, j + 1)`` holding the origin.
    :param t: Fraction along that segment at which the origin lies.
    :param probe_mm: Arc distance on either side of the origin at which the direction is read:
        the side whose point lies further posterior (smaller ``ap``) is the positive one. A
        probe rather than the origin's own segment, so that an origin at a corner (the foot of
        an intercondylar wall) is oriented by where the profile goes, not by one short edge.
    :return: ``(M,)`` arc length (mm), 0 at the origin, positive posteriorly.
    """
    steps = np.linalg.norm(np.diff(ordered_2d, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    origin = cumulative[segment] + t * steps[segment]
    arc = cumulative - origin
    ap = ordered_2d[:, 0]
    forward = np.interp(min(origin + probe_mm, cumulative[-1]), cumulative, ap)
    backward = np.interp(max(origin - probe_mm, 0.0), cumulative, ap)
    return arc if forward < backward else -arc


def _level_profile(mesh: pv.PolyData, frame: FemoralFrame, ml: float) -> Optional[_Level]:
    """
    Cut one sagittal cross-section and parametrise its distal profile by signed arc length.

    :param mesh: The cropped reference bone surface (world mm).
    :param frame: The reference femoral frame.
    :param ml: The cutting plane's coordinate along ``frame.ml_axis``.
    :return: A :class:`_Level`, or ``None`` if the cross-section is empty or has neither a
        notch-plane crossing nor a vertex within ``NOTCH_SNAP_MM`` in front of the plane.
    """
    cut = mesh.slice(normal=frame.ml_axis, origin=ml * frame.ml_axis)
    if cut is None or cut.n_points < 2 or cut.n_cells == 0:
        return None
    points = np.asarray(cut.points, dtype=float)
    plane = np.stack([points @ frame.ap_axis, points @ frame.proximal_axis], axis=1)
    edges = _edge_set(cut.lines)

    crossing = corner = None  # (proximal, component index, segment, t)
    components = []
    for sequence in _polyline_components(points, cut.lines):
        if len(sequence) < 2:
            continue
        ids = list(sequence)
        closed = len(ids) > 2 and (min(ids[0], ids[-1]), max(ids[0], ids[-1])) in edges
        if closed:
            # Open the loop at its most proximal (non-articular) point.
            top = int(np.argmax(plane[ids, 1]))
            ids = ids[top:] + ids[:top] + [ids[top]]
        components.append(ids)
        crossings, corners = _origin_candidates(plane[ids], frame.notch_ap)
        for proximal, segment, t in crossings:
            if crossing is None or proximal < crossing[0]:
                crossing = (proximal, len(components) - 1, segment, t)
        for proximal, segment, t in corners:
            if corner is None or proximal < corner[0]:
                corner = (proximal, len(components) - 1, segment, t)
    # The brief's rule: the crossing with the smallest proximal coordinate. A corner within the
    # snap band replaces it only if there is no crossing at all or the corner lies more than
    # NOTCH_CORNER_DROP_MM below it -- i.e. the lowest crossing is not on the distal side.
    best = crossing
    if corner is not None and (crossing is None or corner[0] < crossing[0] - NOTCH_CORNER_DROP_MM):
        best = corner
    if best is None:
        return None

    _, index, segment, t = best
    all_ids, all_arc = [], []
    for k, ids in enumerate(components):
        all_ids.extend(ids)
        all_arc.append(_profile_arc(plane[ids], segment, t) if k == index
                       else np.full(len(ids), np.nan))
    return _Level(ml=float(ml), tree=cKDTree(plane[all_ids]), arc=np.concatenate(all_arc))


class UnrolledFemur:
    """
    The unrolled 2-D coordinate of one reference femur (see the module docstring).

    Build it with :meth:`from_reference`; query it with :meth:`coordinates`.

    :ivar frame: The reference :class:`FemoralFrame` (world mm).
    :ivar vertices: ``(V, 3)`` vertices of the cropped reference bone surface (world mm).
    :ivar arc: ``(V,)`` signed arc length (mm) of each vertex; ``NaN`` where no profile reaches.
    :ivar ml_range: ``(lo, hi)`` absolute medial-lateral range (``p . ml_axis``) covered by
        the computed levels: the outermost levels widened by one ``slice_mm`` each (each level
        stands for its slab; the outermost value is held constant over that last slice).
        Points outside are ``NaN``.
    :ivar max_distance_mm: Largest distance from a bone vertex at which a point is mapped.
    :ivar slice_mm: Spacing of the sagittal levels (mm).
    """

    def __init__(self, frame: FemoralFrame, vertices: np.ndarray, arc: np.ndarray,
                 ml_range: Tuple[float, float], max_distance_mm: float, slice_mm: float):
        """
        Store a precomputed parametrisation (use :meth:`from_reference` to compute one).

        :param frame: The reference femoral frame.
        :param vertices: ``(V, 3)`` cropped reference bone vertices (world mm).
        :param arc: ``(V,)`` signed arc length per vertex (mm), ``NaN`` allowed.
        :param ml_range: ``(lo, hi)`` absolute M-L range covered by the computed levels.
        :param max_distance_mm: Distance cut-off of :meth:`coordinates` (mm).
        :param slice_mm: Level spacing (mm), kept for the record.
        """
        self.frame = frame
        self.vertices = np.asarray(vertices, dtype=float)
        self.arc = np.asarray(arc, dtype=float)
        self.ml_range = (float(ml_range[0]), float(ml_range[1]))
        self.max_distance_mm = float(max_distance_mm)
        self.slice_mm = float(slice_mm)
        self._tree = cKDTree(self.vertices)

    @classmethod
    def from_reference(cls, bone_mesh: pv.PolyData, frame: FemoralFrame, slice_mm: float = 0.5,
                       max_distance_mm: float = 2.0) -> 'UnrolledFemur':
        """
        Build the unrolled coordinate on a reference femur.

        :param bone_mesh: The reference femoral bone surface (all-triangle ``pv.PolyData``,
            world mm), e.g. ``FemurSubdivision.bone_mesh``. It is cropped here to its distal
            ``REGISTRATION_CROP_MM`` along ``frame.proximal_axis``.
        :param frame: The reference :class:`FemoralFrame` (axes, ``notch_point``,
            ``notch_ap``), in the same world frame as ``bone_mesh``.
        :param slice_mm: Spacing of the sagittal levels (mm); levels sit at
            ``notch_point . ml_axis + k * slice_mm``.
        :param max_distance_mm: :meth:`coordinates` returns ``NaN`` for points further than
            this from the nearest bone vertex (mm).
        :return: An :class:`UnrolledFemur`.
        :raises ValueError: If ``slice_mm`` is not positive or fewer than two levels have a
            profile crossing the notch plane.
        """
        if not slice_mm > 0:
            raise ValueError(f'slice_mm must be positive, got {slice_mm!r}')
        mesh = crop_to_articular_end(bone_mesh, frame.proximal_axis, REGISTRATION_CROP_MM, 'min')
        vertices = np.asarray(mesh.points, dtype=float)
        across = vertices @ frame.ml_axis
        origin = float(frame.notch_point @ frame.ml_axis)
        # Levels strictly inside the bone's M-L extent (a plane through the very edge cuts
        # nothing).
        k_lo = int(np.floor((across.min() - origin) / slice_mm)) + 1
        k_hi = int(np.ceil((across.max() - origin) / slice_mm)) - 1
        levels = [_level_profile(mesh, frame, origin + k * slice_mm)
                  for k in range(k_lo, k_hi + 1)]
        levels = [level for level in levels if level is not None]
        if len(levels) < 2:
            raise ValueError('the reference femur has too few sagittal cross-sections crossing '
                             'the trochlear-notch plane to unroll it')
        level_ml = np.array([level.ml for level in levels])

        # Per vertex: arc on the two bracketing levels, linear in M-L.
        in_plane = np.stack([vertices @ frame.ap_axis, vertices @ frame.proximal_axis], axis=1)
        upper = np.clip(np.searchsorted(level_ml, across), 1, len(levels) - 1)
        lower = upper - 1
        values = np.full((2, len(vertices)), np.nan)
        for which, index in enumerate((lower, upper)):
            for i in np.unique(index):
                rows = np.flatnonzero(index == i)
                distance, nearest = levels[i].tree.query(in_plane[rows])
                # A level whose contour passes far from the vertex (the vertex lies on a side
                # wall the plane misses, e.g. the condyle's wall into the fossa next to a
                # fossa level) says nothing about it.
                values[which, rows] = np.where(distance <= max_distance_mm,
                                               levels[i].arc[nearest], np.nan)
        weight = np.clip((across - level_ml[lower]) / (level_ml[upper] - level_ml[lower]), 0, 1)
        arc = (1 - weight) * values[0] + weight * values[1]
        # One level without a profile point: fall back to the other one.
        arc = np.where(np.isnan(values[0]), values[1], np.where(np.isnan(values[1]), values[0],
                                                                arc))
        # Each level stands for the slab of one slice around it: beyond the outermost levels a
        # vertex takes the outermost level's value up to one slice out (the rounded end face
        # of a condyle lies there, and a plane through the very edge cuts nothing), and is
        # NaN further out.
        ml_range = (level_ml[0] - slice_mm, level_ml[-1] + slice_mm)
        arc[(across < ml_range[0]) | (across > ml_range[1])] = np.nan
        return cls(frame, vertices, arc, ml_range, max_distance_mm, slice_mm)

    def coordinates(self, points_world: np.ndarray) -> np.ndarray:
        """
        Unrolled map coordinates of world points (in the reference scan's world frame).

        :param points_world: ``(N, 3)`` (or a single ``(3,)``) points in world mm, already in
            the reference scan's frame (registered, for another configuration's points).
        :return: ``(N, 2)`` mm. Column 0 = ``(p - notch_point) . ml_axis`` (patient right
            positive); column 1 = signed arc length along the sagittal bone profile from the
            notch plane's distal crossing, positive posteriorly, read off the nearest reference
            bone vertex. Both columns are ``NaN`` for a point further than
            ``max_distance_mm`` from the reference bone, outside the medial-lateral range of
            the computed levels, or whose nearest vertex has no arc.
        """
        points = np.asarray(points_world, dtype=float).reshape(-1, 3)
        out = np.full((len(points), 2), np.nan)
        if len(points) == 0:
            return out
        across = points @ self.frame.ml_axis
        distance, nearest = self._tree.query(points)
        arc = self.arc[nearest]
        ok = ((distance <= self.max_distance_mm) & (across >= self.ml_range[0])
              & (across <= self.ml_range[1]) & np.isfinite(arc))
        out[ok, 0] = across[ok] - float(self.frame.notch_point @ self.frame.ml_axis)
        out[ok, 1] = arc[ok]
        return out


# ----------------------------------------------------------------------------------------
# Reference-anchored thickness maps and their deltas
# ----------------------------------------------------------------------------------------

#: The reference configuration ``(flexion angle in degrees, load)`` every other configuration
#: of a knee is compared against: unloaded, 30 degrees of flexion.
REFERENCE = (30, 'unbelastet')

#: The configurations compared against :data:`REFERENCE`, in reporting order.
COMPARISONS = ((0, 'unbelastet'), (60, 'unbelastet'), (90, 'unbelastet'),
               (0, 'belastet'), (30, 'belastet'), (60, 'belastet'), (90, 'belastet'))

#: How much longer (mm) the reference (fixed) bone is cropped than a configuration's (moving)
#: bone before registration. The registration residual is one-sided (moving -> fixed) and
#: counts moving surface beyond the fixed surface's extent at full distance, so the moving
#: crop must lie inside the fixed one (see :mod:`morphometry.cartilage.registration`).
REFERENCE_CROP_MARGIN_MM = 10.0

#: Which end of the bone's long axis (``frame_axes[2]``) holds the articular surface:
#: the tibial plateau normal points superior (keep its maximum), the femoral shaft axis points
#: proximal (keep its minimum, the distal end).
_ARTICULAR_END = {'tibia': 'max', 'femur': 'min'}


@dataclass
class BoneMeasurement:
    """
    Everything the delta maps need from one bone of one scan, detached from the image.

    Plain arrays, meshes and frame dataclasses only, so instances pickle (results of the
    expensive subdivision + thickness step are cached to disk per scan).

    :ivar bone: ``'tibia'`` or ``'femur'``.
    :ivar tab_mesh: The labelled tAB of the subdivision (``subdivision.mesh``, world mm), with
        cell data ``'region'`` (index into ``TIBIA_REGION_KEYS`` / ``FEMUR_REGION_KEYS``) and
        ``'covered'`` (bool, ``False`` = denuded).
    :ivar bone_mesh: The full bone surface the subdivision was built on (world mm); the
        registration uses this, never the cartilage.
    :ivar triangle_thickness: ``(tab_mesh.n_cells,)`` per-triangle ``cAB -> AC`` distance (mm),
        ``ThicknessMaps.triangle_thickness`` (0 on denuded triangles, which the maps exclude).
    :ivar statistics: ``ThicknessMaps.statistics`` (per-subregion area-weighted statistics).
    :ivar frame_origin: ``(3,)`` origin of the bone's anatomical frame (world mm), used for the
        initial registration estimate.
    :ivar frame_axes: ``(3, 3)`` row-orthonormal, right-handed axes of that frame. Row 2 is the
        bone's long axis (tibia: plateau normal, superior; femur: shaft axis, proximal), along
        which the bone is cropped to its articular end.
    :ivar femoral_frame: The :class:`FemoralFrame` (femur only; needed for a femoral reference,
        whose :class:`UnrolledFemur` is built on it).
    :ivar knee_frame: The tibial :class:`KneeFrame` (tibia only; needed for a tibial reference,
        whose plateau plane keys the map).
    """
    bone: str
    tab_mesh: pv.PolyData
    bone_mesh: pv.PolyData
    triangle_thickness: np.ndarray
    statistics: dict
    frame_origin: np.ndarray
    frame_axes: np.ndarray
    femoral_frame: Optional[FemoralFrame] = None
    knee_frame: Optional[KneeFrame] = None

    def __post_init__(self):
        """Coerce the array fields to float arrays of the right shape."""
        self.triangle_thickness = np.asarray(self.triangle_thickness, dtype=float).ravel()
        self.frame_origin = np.asarray(self.frame_origin, dtype=float).reshape(3)
        self.frame_axes = np.asarray(self.frame_axes, dtype=float).reshape(3, 3)


def bone_measurement(bone: str, subdivision, thickness: ThicknessMaps) -> BoneMeasurement:
    """
    Collect one bone's subdivision and thickness result into a :class:`BoneMeasurement`.

    The anatomical frame used to initialise the registration is, for the femur, the femoral
    frame (rows ``ml_axis, ap_axis, proximal_axis``, origin the trochlear-notch point); for the
    tibia, the shared knee frame of :func:`~morphometry.cartilage.thickness.knee_frame_from_tibia`
    (rows ``e1`` patient right, ``e2`` anterior, ``e3`` superior; origin the midpoint of the two
    plates' centres of gravity). Both are right-handed with the long axis in row 2.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param subdivision: A :class:`~morphometry.cartilage.tibia_regions.TibiaSubdivision` or
        :class:`~morphometry.cartilage.femur_regions.FemurSubdivision` carrying ``bone_mesh``.
    :param thickness: The :class:`~morphometry.cartilage.thickness.ThicknessMaps` measured on
        that subdivision (its ``triangle_thickness`` must be aligned with ``subdivision.mesh``).
    :return: A :class:`BoneMeasurement`. The meshes and frames are shared with ``subdivision``,
        not copied.
    :raises ValueError: For an unknown ``bone``, a subdivision without ``bone_mesh``, or a
        thickness result without (or misaligned) ``triangle_thickness``.
    """
    if bone not in _ARTICULAR_END:
        raise ValueError(f"bone must be 'tibia' or 'femur', got {bone!r}")
    if getattr(subdivision, 'bone_mesh', None) is None:
        raise ValueError(f'the {bone} subdivision carries no bone_mesh')
    triangles = getattr(thickness, 'triangle_thickness', None)
    if triangles is None or len(triangles) != subdivision.mesh.n_cells:
        raise ValueError(f'the {bone} thickness result has no triangle_thickness aligned with '
                         'the subdivision mesh')
    femoral_frame = knee_frame = None
    if bone == 'femur':
        femoral_frame = subdivision.frame
        axes = np.stack([femoral_frame.ml_axis, femoral_frame.ap_axis,
                         femoral_frame.proximal_axis])
        origin = femoral_frame.notch_point
    else:
        knee_frame = knee_frame_from_tibia(subdivision)
        axes, origin = knee_frame.axes, knee_frame.origin
    return BoneMeasurement(bone=bone, tab_mesh=subdivision.mesh, bone_mesh=subdivision.bone_mesh,
                           triangle_thickness=triangles,
                           statistics=dict(getattr(thickness, 'statistics', None) or {}),
                           frame_origin=origin, frame_axes=axes, femoral_frame=femoral_frame,
                           knee_frame=knee_frame)


@dataclass(eq=False)
class MapGrid:
    """
    A square grid of map cells over 2-D map coordinates (mm).

    Cell ``(i, j)`` is centred at ``origin_mm + grid_mm * (i, j)``: index ``i`` runs along map
    coordinate column 0 and ``j`` along column 1, so a map array ``a`` of this grid has
    ``a[i, j]`` for the cell at ``(column 0, column 1) ~ centre_mm(i, j)``. (For an image with
    column 0 horizontal, plot ``a.T`` with ``origin='lower'``.) The map coordinates are those
    of the reference scan: the tibial :meth:`KneeFrame.in_plane` (patient right, anterior) or
    the :meth:`UnrolledFemur.coordinates` (patient right, arc posteriorly).

    :ivar origin_mm: ``(2,)`` map coordinates (mm) of the centre of cell ``(0, 0)``.
    :ivar grid_mm: Side length of one cell (mm).
    :ivar shape: ``(H, W)``: number of cells along column 0 and column 1.
    """
    origin_mm: np.ndarray
    grid_mm: float
    shape: Tuple[int, int]

    def __post_init__(self):
        """Coerce the fields and check the cell size."""
        self.origin_mm = np.asarray(self.origin_mm, dtype=float).reshape(2)
        self.grid_mm = float(self.grid_mm)
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        if not self.grid_mm > 0:
            raise ValueError(f'grid_mm must be positive, got {self.grid_mm!r}')

    def __eq__(self, other) -> bool:
        """
        Whether two grids have the same origin, cell size and shape.

        :param other: Another object.
        :return: ``True`` for an identical :class:`MapGrid`.
        """
        return (isinstance(other, MapGrid) and self.grid_mm == other.grid_mm
                and self.shape == other.shape
                and np.array_equal(self.origin_mm, other.origin_mm))

    @classmethod
    def covering(cls, coordinate_sets: Sequence[np.ndarray], grid_mm: float,
                 pad_cells: int = 2) -> 'MapGrid':
        """
        The smallest grid holding every finite point of several coordinate sets, plus padding.

        Cell boundaries lie on the absolute lattice ``(k + 1/2) * grid_mm`` (cell centres at
        integer multiples of ``grid_mm``), so two grids of the same cell size over the same
        coordinates discretise identically wherever their extents differ.

        :param coordinate_sets: Sequence of ``(N_k, 2)`` map coordinates (mm); rows with a NaN
            are ignored.
        :param grid_mm: Side length of one cell (mm).
        :param pad_cells: Empty cells added on every side (room for a smoothing kernel).
        :return: A :class:`MapGrid`.
        :raises ValueError: If no set holds a finite point, or ``grid_mm`` is not positive.
        """
        if not grid_mm > 0:
            raise ValueError(f'grid_mm must be positive, got {grid_mm!r}')
        points = [np.asarray(c, dtype=float).reshape(-1, 2) for c in coordinate_sets]
        points = np.vstack(points) if points else np.zeros((0, 2))
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            raise ValueError('no finite map coordinates to build a grid over')
        lo = np.floor(points.min(axis=0) / grid_mm + 0.5).astype(np.int64)
        hi = np.floor(points.max(axis=0) / grid_mm + 0.5).astype(np.int64)
        shape = hi - lo + 1 + 2 * int(pad_cells)
        return cls((lo - int(pad_cells)) * float(grid_mm), grid_mm, (shape[0], shape[1]))

    def cells(self, coords: np.ndarray) -> np.ndarray:
        """
        Integer cell indices of map coordinates.

        Rounding is half-up (``floor(x + 0.5)``), as in :meth:`KneeFrame.cells`, so the cell of
        a point is a monotone function of its coordinate. Rows with a NaN have no cell; filter
        them out first (:func:`rasterise` does). Indices may fall outside ``shape``.

        :param coords: ``(N, 2)`` map coordinates (mm), finite.
        :return: ``(N, 2)`` int array of ``(i, j)``.
        """
        coords = np.asarray(coords, dtype=float).reshape(-1, 2)
        return np.floor((coords - self.origin_mm) / self.grid_mm + 0.5).astype(np.int64)

    def centre_mm(self, i, j) -> np.ndarray:
        """
        Map coordinates of a cell centre.

        :param i: Index along coordinate column 0.
        :param j: Index along coordinate column 1.
        :return: ``(2,)`` map coordinates (mm).
        """
        return self.origin_mm + self.grid_mm * np.array([float(i), float(j)])


def _cell_index(grid: MapGrid, coords: np.ndarray, *arrays: np.ndarray):
    """
    Flat cell indices of the usable rows of point data.

    A row is usable if its coordinates and every given array are finite, its weight (the last
    array) is positive, and its cell lies inside the grid.

    :param grid: The grid.
    :param coords: ``(N, 2)`` map coordinates (mm).
    :param arrays: ``(N,)`` per-point arrays; the last one is the weight.
    :return: ``(flat_index, selected_arrays...)``: the ravelled cell index of each usable row
        and the arrays restricted to those rows.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    arrays = [np.asarray(a).ravel() for a in arrays]
    ok = np.all(np.isfinite(coords), axis=1)
    for a in arrays:
        if a.dtype.kind == 'f':
            ok &= np.isfinite(a)
    ok &= arrays[-1] > 0
    cells = grid.cells(coords[ok])
    inside = np.all((cells >= 0) & (cells < np.array(grid.shape)), axis=1)
    flat = np.ravel_multi_index((cells[inside, 0], cells[inside, 1]), grid.shape)
    rows = np.flatnonzero(ok)[inside]
    return (flat,) + tuple(a[rows] for a in arrays)


def rasterise(grid: MapGrid, coords: np.ndarray, values: np.ndarray,
              weights: np.ndarray) -> np.ndarray:
    """
    Bin point values into a grid as a weighted mean per cell.

    :param grid: The grid.
    :param coords: ``(N, 2)`` map coordinates (mm) of the points.
    :param values: ``(N,)`` values (e.g. thickness in mm).
    :param weights: ``(N,)`` weights (e.g. triangle areas in mm^2).
    :return: ``(H, W)`` float array: ``sum(w * v) / sum(w)`` over the points in each cell,
        ``NaN`` where no point falls. Points with a NaN coordinate or value, a non-positive
        weight, or a cell outside the grid are dropped.
    """
    flat, v, w = _cell_index(grid, coords, np.asarray(values, dtype=float),
                             np.asarray(weights, dtype=float))
    size = grid.shape[0] * grid.shape[1]
    total = np.bincount(flat, weights=w, minlength=size)
    weighted = np.bincount(flat, weights=w * v, minlength=size)
    out = np.full(size, np.nan)
    filled = total > 0
    out[filled] = weighted[filled] / total[filled]
    return out.reshape(grid.shape)


def rasterise_labels(grid: MapGrid, coords: np.ndarray, labels: np.ndarray,
                     weights: np.ndarray) -> np.ndarray:
    """
    Bin integer point labels into a grid by weighted majority per cell.

    :param grid: The grid.
    :param coords: ``(N, 2)`` map coordinates (mm) of the points.
    :param labels: ``(N,)`` integer labels (e.g. region codes, non-negative).
    :param weights: ``(N,)`` weights (e.g. triangle areas in mm^2).
    :return: ``(H, W)`` int64 array: per cell, the label with the largest total weight (ties go
        to the smaller label), ``-1`` where no point falls. Points with a NaN coordinate, a
        non-positive weight or a cell outside the grid are dropped.
    """
    flat, lab, w = _cell_index(grid, coords, np.asarray(labels, dtype=np.int64),
                               np.asarray(weights, dtype=float))
    size = grid.shape[0] * grid.shape[1]
    out = np.full(size, -1, dtype=np.int64)
    if len(flat):
        unique = np.unique(lab)
        totals = np.stack([np.bincount(flat[lab == u], weights=w[lab == u], minlength=size)
                           for u in unique])
        best = np.argmax(totals, axis=0)
        filled = totals.max(axis=0) > 0
        out[filled] = unique[best[filled]]
    return out.reshape(grid.shape)


@dataclass
class BoneDeltaMaps:
    """
    One knee's thickness maps of one bone on the reference's grid, and their deltas.

    Keys of the per-configuration dicts are ``(angle, load)`` tuples. A configuration whose
    registration failed appears only in ``errors``.

    :ivar bone: ``'tibia'`` or ``'femur'``.
    :ivar grid: The :class:`MapGrid` all maps share.
    :ivar reference: ``(H, W)`` area-weighted mean thickness (mm) of the reference's covered
        tAB triangles per cell, ``NaN`` where none.
    :ivar region_codes: ``(H, W)`` int: per cell, the area-majority ``'region'`` code of the
        reference's tAB triangles (covered or not), ``-1`` where none.
    :ivar configurations: Per configuration, ``(H, W)`` thickness (mm) of its covered triangles
        after registration into the reference, ``NaN`` where none.
    :ivar deltas: Per configuration, ``configuration - reference`` (mm; negative = thinner),
        ``NaN`` where either is ``NaN``.
    :ivar registrations: Per configuration, the bone :class:`RegistrationResult`
        (configuration -> reference).
    :ivar dropped_fraction: Per configuration, the area fraction of its covered triangles whose
        registered centroid has no map coordinate (beyond the reference bone, see
        :meth:`UnrolledFemur.coordinates`) and is therefore missing from the map.
    :ivar reference_positions: ``(H, W, 3)`` area-weighted mean world position (RAS mm, the
        reference scan's frame) of the reference's covered tAB triangle centroids per cell,
        ``NaN`` where none -- where a map cell lies on the reference bone.
    :ivar rotation_change_deg: Per configuration, the angle (degrees) between the rotation of
        the frame-derived initial transform and the final ICP rotation. The trimmed residual
        cannot detect a ~180 degree flip of a near-symmetric bone; a large change from the
        frame-derived start means the ICP slid off it.
    :ivar errors: Per configuration whose registration raised, ``'<ExceptionType>: <message>'``.
    :ivar reference_dropped_fraction: The same as ``dropped_fraction`` for the reference's own
        covered triangles (non-zero only for a femur, at the unrolling's edges).
    """
    bone: str
    grid: MapGrid
    reference: np.ndarray
    region_codes: np.ndarray
    configurations: Dict[tuple, np.ndarray]
    deltas: Dict[tuple, np.ndarray]
    registrations: Dict[tuple, RegistrationResult]
    dropped_fraction: Dict[tuple, float]
    reference_positions: np.ndarray
    rotation_change_deg: Dict[tuple, float]
    errors: Dict[tuple, str] = field(default_factory=dict)
    reference_dropped_fraction: float = 0.0


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """
    The angle of the relative rotation between two rotation matrices.

    :param a: ``(3, 3)`` rotation.
    :param b: ``(3, 3)`` rotation.
    :return: The angle (degrees, in ``[0, 180]``) of ``a @ b.T``.
    """
    cosine = (np.trace(np.asarray(a) @ np.asarray(b).T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _covered_triangles(m: BoneMeasurement):
    """
    Centroids, areas and thickness of a measurement's covered, measured tAB triangles.

    :param m: The measurement.
    :return: ``(centroids (K, 3), areas (K,), thickness (K,), selected (n_cells,) bool)``.
    :raises ValueError: If the mesh has no ``'covered'`` cell data or the thickness is not
        aligned with the mesh.
    """
    if 'covered' not in m.tab_mesh.cell_data:
        raise ValueError(f"the {m.bone} tAB mesh has no 'covered' cell data")
    if len(m.triangle_thickness) != m.tab_mesh.n_cells:
        raise ValueError(f'the {m.bone} triangle_thickness is not aligned with its tAB mesh')
    selected = (np.asarray(m.tab_mesh.cell_data['covered'], dtype=bool)
                & np.isfinite(m.triangle_thickness))
    centroids = triangle_centroids(m.tab_mesh)[selected]
    areas = triangle_areas(m.tab_mesh)[selected]
    return centroids, areas, m.triangle_thickness[selected], selected


def _dropped(coords: np.ndarray, areas: np.ndarray) -> float:
    """
    Area fraction of points without a map coordinate.

    :param coords: ``(K, 2)`` map coordinates, ``NaN`` rows for unmappable points.
    :param areas: ``(K,)`` areas.
    :return: The dropped area fraction, ``NaN`` if the total area is zero.
    """
    total = float(np.sum(areas))
    if not total > 0:
        return float('nan')
    return float(np.sum(areas[~np.all(np.isfinite(coords), axis=1)]) / total)


def knee_delta_maps(bone: str, reference: BoneMeasurement,
                    configurations: Dict[tuple, BoneMeasurement], grid_mm: float = 1.0,
                    crop_mm: float = REGISTRATION_CROP_MM) -> BoneDeltaMaps:
    """
    Register every configuration of one knee's bone onto the reference and build the Δ maps.

    Per configuration:

    1. Initial estimate ``RigidTransform.from_frames(cfg frame -> reference frame)`` (the
       frames may rest on cartilage; only the start does).
    2. Rigid ICP on **bone only**: the configuration's bone cropped to ``crop_mm`` of its
       articular end (along ``frame_axes[2]``) onto the reference's bone cropped to
       ``crop_mm + REFERENCE_CROP_MARGIN_MM`` (longer, so the moving surface lies within the
       fixed one's coverage, as the one-sided residual requires).
    3. The configuration's **covered** tAB triangle centroids are carried into the reference's
       world frame and given map coordinates: the reference :class:`KneeFrame`'s
       ``in_plane`` (tibia) or the reference's :class:`UnrolledFemur` (femur, built once).
       Points with no coordinate are dropped and their area share is recorded.

    One :class:`MapGrid` of ``grid_mm`` covers the reference's whole tAB and every
    configuration's mapped triangles; each map is the area-weighted mean thickness per cell
    (:func:`rasterise`). Denuded triangles never enter a map, so a cell with only denuded bone
    is empty, not 0.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param reference: The reference scan's :class:`BoneMeasurement` (needs ``knee_frame`` for a
        tibia, ``femoral_frame`` for a femur).
    :param configurations: ``{(angle, load): BoneMeasurement}`` of the configurations to
        compare (same bone, same knee).
    :param grid_mm: Map cell size (mm).
    :param crop_mm: Length of the articular end of the configuration's bone used for the
        registration (mm).
    :return: A :class:`BoneDeltaMaps`. A configuration whose registration raises
        ``RuntimeError`` / ``ValueError`` / ``AssertionError`` is left out of every dict but
        ``errors``, so one bad scan does not lose the knee.
    :raises ValueError: For invalid input -- an unknown bone, a measurement of another bone,
        a reference without the frame its map needs, a non-positive ``grid_mm`` -- or a
        reference that cannot be mapped (e.g. a femur too small to unroll).
    """
    if bone not in _ARTICULAR_END:
        raise ValueError(f"bone must be 'tibia' or 'femur', got {bone!r}")
    if not grid_mm > 0:
        raise ValueError(f'grid_mm must be positive, got {grid_mm!r}')
    for key, m in [('reference', reference)] + list(configurations.items()):
        if m.bone != bone:
            raise ValueError(f'measurement {key!r} is a {m.bone}, expected a {bone}')
    end = _ARTICULAR_END[bone]
    if bone == 'tibia':
        if reference.knee_frame is None:
            raise ValueError('the tibial reference measurement has no knee_frame')
        to_map = reference.knee_frame.in_plane
    else:
        if reference.femoral_frame is None:
            raise ValueError('the femoral reference measurement has no femoral_frame')
        to_map = UnrolledFemur.from_reference(reference.bone_mesh,
                                              reference.femoral_frame).coordinates

    # The reference: its covered triangles for the thickness and positions, its whole tAB for
    # the region codes.
    ref_centroids, ref_areas, ref_thickness, _ = _covered_triangles(reference)
    ref_coords = to_map(ref_centroids)
    all_coords = to_map(triangle_centroids(reference.tab_mesh))
    all_areas = triangle_areas(reference.tab_mesh)
    fixed = crop_to_articular_end(reference.bone_mesh, reference.frame_axes[2],
                                  crop_mm + REFERENCE_CROP_MARGIN_MM, end)

    mapped, registrations, rotation_change, dropped, errors = {}, {}, {}, {}, {}
    for key, cfg in configurations.items():
        try:
            initial = RigidTransform.from_frames(cfg.frame_origin, cfg.frame_axes,
                                                 reference.frame_origin, reference.frame_axes)
            moving = crop_to_articular_end(cfg.bone_mesh, cfg.frame_axes[2], crop_mm, end)
            result = register_bone(moving, fixed, initial)
            centroids, areas, thickness, _ = _covered_triangles(cfg)
        except (RuntimeError, ValueError, AssertionError) as error:
            errors[key] = f'{type(error).__name__}: {error}'
            continue
        coords = to_map(result.transform.apply(centroids))
        mapped[key] = (coords, thickness, areas)
        registrations[key] = result
        rotation_change[key] = _rotation_angle_deg(result.transform.rotation, initial.rotation)
        dropped[key] = _dropped(coords, areas)

    grid = MapGrid.covering([all_coords, ref_coords] + [c for c, _, _ in mapped.values()],
                            grid_mm)
    reference_map = rasterise(grid, ref_coords, ref_thickness, ref_areas)
    positions = np.stack([rasterise(grid, ref_coords, ref_centroids[:, k], ref_areas)
                          for k in range(3)], axis=-1)
    region_codes = rasterise_labels(grid, all_coords,
                                    np.asarray(reference.tab_mesh.cell_data['region']),
                                    all_areas)
    config_maps = {key: rasterise(grid, *value) for key, value in mapped.items()}
    deltas = {key: value - reference_map for key, value in config_maps.items()}
    return BoneDeltaMaps(bone=bone, grid=grid, reference=reference_map,
                         region_codes=region_codes, configurations=config_maps, deltas=deltas,
                         registrations=registrations, dropped_fraction=dropped,
                         reference_positions=positions, rotation_change_deg=rotation_change,
                         errors=errors,
                         reference_dropped_fraction=_dropped(ref_coords, ref_areas))


#: Compartments of the hotspot search; patient-side, ``'both'`` = the whole bone.
COMPARTMENTS = ('both', 'left', 'right')


def compartment_codes(bone: str, compartment: str) -> Tuple[int, ...]:
    """
    Region codes (indices into the bone's region-key tuple) of one compartment.

    Tibia: patient-left keys end in ``'lt'``, patient-right in ``'rt'``. Femur: the patient
    side letter is the second-to-last character (``iclf``/``alf`` -> ``l``, ``icrf``/``arf``
    -> ``r``).

    :param bone: ``'tibia'`` or ``'femur'``.
    :param compartment: One of :data:`COMPARTMENTS`.
    :return: Tuple of int region codes, in key order.
    :raises ValueError: On an unknown bone or compartment.
    """
    if bone not in ('tibia', 'femur'):
        raise ValueError(f"unknown bone {bone!r}")
    if compartment not in COMPARTMENTS:
        raise ValueError(f"unknown compartment {compartment!r}; expected one of {COMPARTMENTS}")
    keys = TIBIA_REGION_KEYS if bone == 'tibia' else FEMUR_REGION_KEYS
    if compartment == 'both':
        return tuple(range(len(keys)))
    side = compartment[0]
    return tuple(c for c, k in enumerate(keys) if k[-2] == side)


def find_largest_thinning(maps: BoneDeltaMaps, key: tuple, compartment: str = 'both',
                          neighbourhood_mm: float = 3.0,
                          radius_mm: Optional[float] = None) -> Optional[dict]:
    """
    Locate the centre of the area that thinned most in one configuration's delta map.

    Runs :func:`smoothed_argmin` on ``maps.deltas[key]`` (delta = configuration - reference,
    negative = thinner), so the result is the minimum of a median-smoothed map, searched only
    where the whole smoothing window is measured: a single outlier cell or the rim where one
    scan's cartilage ends (an edge mismatch reads strongly negative) cannot win. The search is
    restricted to the cells whose reference region belongs to the compartment.

    :param maps: The bone's :class:`BoneDeltaMaps`.
    :param key: ``(angle, load)`` of the configuration.
    :param compartment: ``'both'``, ``'left'`` or ``'right'`` (patient side).
    :param neighbourhood_mm: Diameter (mm) of the smoothing neighbourhood.
    :param radius_mm: Refinement-disc radius (mm); default half the neighbourhood.
    :return: ``None`` if the configuration is absent from ``maps.deltas`` or the compartment
        has no finite delta cell; else a dict with ``'cell'`` (refined ``(i, j)``),
        ``'centre'`` (neighbourhood centre ``(i, j)``), ``'map_mm'`` (map coordinates of
        ``cell``), ``'smoothed_delta_mm'`` (smoothed delta at the centre), ``'delta_mm'`` (raw
        delta at ``cell``), ``'subregion'`` (region key at ``cell``, ``''`` if none),
        ``'kernel_size'`` and ``'fallback'`` (True if no fully measured window existed; a
        WARNING is logged then).
    """
    if key not in maps.deltas:
        return None
    delta = np.asarray(maps.deltas[key], dtype=float)
    valid = np.isfinite(delta)
    restrict = np.isin(maps.region_codes, compartment_codes(maps.bone, compartment))
    if not (valid & restrict).any():
        return None
    res = smoothed_argmin(delta, valid, maps.grid.grid_mm, neighbourhood_mm,
                          radius_mm=radius_mm, restrict=restrict)
    if res['fallback']:
        logger.warning("thinning hotspot %s/%s %s: no fully measured %d-cell window; searched "
                       "all measured cells (the hotspot may lie on a footprint rim)",
                       maps.bone, compartment, key, res['kernel_size'])
    ci, cj = res['center_index']
    pi, pj = res['point_index']
    code = int(maps.region_codes[pi, pj])
    keys = TIBIA_REGION_KEYS if maps.bone == 'tibia' else FEMUR_REGION_KEYS
    return {
        'cell': (int(pi), int(pj)),
        'centre': (int(ci), int(cj)),
        'map_mm': tuple(float(v) for v in maps.grid.centre_mm(pi, pj)),
        'smoothed_delta_mm': float(res['smoothed'][ci, cj]),
        'delta_mm': float(delta[pi, pj]),
        'subregion': keys[code] if code >= 0 else '',
        'kernel_size': int(res['kernel_size']),
        'fallback': bool(res['fallback']),
    }

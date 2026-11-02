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
   the level is the **distal-most point of the cross-section on or anterior of the notch plane**
   ``ap = notch_ap``, over all contour components (:func:`_origin_candidates`). Wherever the
   profile crosses the plane distally -- every condyle -- that point is exactly the notch-plane
   crossing with the smallest proximal coordinate, because the bone rises proximally as it
   runs anteriorly up the trochlea. Where the profile only reaches the plane at a corner (an
   intercondylar wall lying nearly in it), the foot of that corner is taken if it lies within
   ``NOTCH_SNAP_MM`` of the plane; further away the level has no profile. The component
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
the origin slides along the foot of that wall by up to ~1.3 mm between two voxelisations of the
same bone (against <= 0.6 mm on the condyles). A real fossa roof is oblique to the notch plane,
so this is the phantom's worst case, not a measured property of real knees. Before the
corner rule above, such a level could jump to a crossing ~20 mm proximal (the underside of the
shaft); that is pinned against by ``tests/unit/test_unrolled_femur.py``.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from morphometry.cartilage.femur_regions import FemoralFrame, _polyline_components
from morphometry.cartilage.registration import REGISTRATION_CROP_MM, crop_to_articular_end

#: Largest distance (mm) anterior of the notch plane at which the distal-most anterior point of
#: a sagittal profile is still accepted as the arc origin when the profile does not cross the
#: plane there (see :func:`_origin_candidates`). Beyond it the level has no profile (``NaN``).
NOTCH_SNAP_MM = 1.0


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


def _origin_candidates(points_2d: np.ndarray,
                       notch_ap: float) -> List[Tuple[float, float, int, float]]:
    """
    Candidate arc origins on one ordered polyline: everything on or anterior of the notch plane.

    The origin of the arc is the **distal-most point of the profile on or anterior of the notch
    plane** ``ap = notch_ap``. Wherever the profile crosses the plane on its distal side (every
    condyle, every trochlear level whose groove runs into a sloping fossa roof) that point *is*
    the distal crossing, because the bone surface rises proximally as it runs anteriorly up the
    trochlea. The vertices are candidates as well so that a level whose profile only *touches*
    the plane -- an intercondylar wall lying (nearly) in it, as on the phantom, whose smoothed
    wall crosses the plane in one voxelisation and not in another -- gets its origin at the
    foot of that wall instead of jumping to some crossing far proximal.

    :param points_2d: ``(M, 2)`` ordered in-plane coordinates ``(ap, proximal)``.
    :param notch_ap: The notch plane's ``ap`` coordinate.
    :return: A list of ``(proximal, ap, segment_index, t)``: the candidate's proximal and
        anterior-posterior coordinates, the index ``j`` of the segment ``(j, j + 1)`` it lies
        on and the fraction ``t`` in ``[0, 1]`` along it.
    """
    f = points_2d[:, 0] - notch_ap
    candidates = []
    last = len(f) - 1
    for j in range(len(f)):
        if f[j] >= 0:
            segment, t = (j, 0.0) if j < last else (j - 1, 1.0)
            candidates.append((float(points_2d[j, 1]), float(points_2d[j, 0]), segment, t))
        if j < last and f[j] * f[j + 1] < 0:
            t = f[j] / (f[j] - f[j + 1])
            point = points_2d[j] + t * (points_2d[j + 1] - points_2d[j])
            candidates.append((float(point[1]), float(notch_ap), j, float(t)))
    return candidates


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
    :return: A :class:`_Level`, or ``None`` if the cross-section is empty, has no point on or
        anterior of the notch plane, or its distal-most such point lies more than
        ``NOTCH_SNAP_MM`` in front of the plane.
    """
    cut = mesh.slice(normal=frame.ml_axis, origin=ml * frame.ml_axis)
    if cut is None or cut.n_points < 2 or cut.n_cells == 0:
        return None
    points = np.asarray(cut.points, dtype=float)
    plane = np.stack([points @ frame.ap_axis, points @ frame.proximal_axis], axis=1)
    edges = _edge_set(cut.lines)

    best = None  # (proximal, ap, component index, segment, t)
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
        for proximal, ap, segment, t in _origin_candidates(plane[ids], frame.notch_ap):
            if best is None or proximal < best[0]:
                best = (proximal, ap, len(components) - 1, segment, t)
    # No profile reaches the notch plane, or its distal-most anterior point is a corner too far
    # in front of it to be read as the notch-plane crossing.
    if best is None or best[1] - frame.notch_ap > NOTCH_SNAP_MM:
        return None

    _, _, index, segment, t = best
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

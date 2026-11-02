"""
Rigid registration of bone surfaces (trimmed point-to-plane ICP).

Used by the knee-cartilage thickness-change (delta) maps: every configuration of a knee
(flexion angle x load) is a separate acquisition, so before two scans' cartilage thickness can
be compared at the same anatomical spot, each scan's femur and tibia are rigidly aligned onto
the reference scan's bone of the same knee. Bone does not deform between the acquisitions, so a
rigid motion is the right model, and the fit uses **bone geometry only** (the meshes passed in
are bone surfaces from :func:`morphometry.cartilage.surface.bone_surface_mesh`); cartilage may
at most enter the *initial* guess, built by the caller with :meth:`RigidTransform.from_frames`.

The knee field of view cuts the shafts at different heights in different acquisitions, so the
two surfaces only partly overlap. That is handled twice: callers restrict both meshes to the
articular end of the bone with :func:`crop_to_articular_end` (``REGISTRATION_CROP_MM``), and
the ICP itself is trimmed -- pairs further apart than ``max_pair_mm`` are ignored and only the
``trim_fraction`` closest of the rest drive each update.

**What the residual means.** The reported residual is the exact point-to-triangle distance
(:func:`morphometry.cartilage.surface.surface_distances`) from the final transformed moving
points to the fixed surface, over the ``trim_fraction`` closest of *all* sampled moving points
-- not only of those that happened to find a partner within ``max_pair_mm``. A registration
stuck in a wrong local minimum therefore cannot hide behind the pair cut-off: at most the
trimmed ``1 - trim_fraction`` of the points are excluded from the residual, so a wrong pose
shows up as a large residual (compare with ``REGISTRATION_MAX_RMS_MM``), never as a small one.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from morphometry.cartilage.surface import surface_distances

#: Length (mm) of the articular end of a bone kept for registration, measured from the bone's
#: extreme along its long axis (distal femur / proximal tibia). Long enough to include the
#: whole articular surface and some metaphysis, short enough to stay inside every knee FOV.
REGISTRATION_CROP_MM = 40.0

#: QC threshold on the registration residual RMS (mm). Provisional: to be confirmed against
#: the observed residual distribution (never against the thickness deltas).
REGISTRATION_MAX_RMS_MM = 1.0


@dataclass
class RigidTransform:
    """
    A rigid motion ``p -> rotation @ p + translation`` of world (mm) points.

    Both fields are plain numpy arrays, so instances pickle (results are cached to disk).

    :param rotation: ``(3, 3)`` proper rotation matrix.
    :param translation: ``(3,)`` translation in mm.
    """
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        """Coerce the fields to float arrays of the right shape."""
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points.

        :param points: ``(N, 3)`` points, or a single ``(3,)`` point, in world mm.
        :return: The transformed points, same shape as ``points``.
        """
        p = np.asarray(points, dtype=float)
        return p @ self.rotation.T + self.translation

    def inverse(self) -> 'RigidTransform':
        """
        The inverse motion, ``p -> rotation.T @ (p - translation)``.

        :return: A new :class:`RigidTransform`.
        """
        r_inv = self.rotation.T
        return RigidTransform(r_inv, -r_inv @ self.translation)

    def compose(self, first: 'RigidTransform') -> 'RigidTransform':
        """
        The motion that applies ``first`` and then ``self``.

        :param first: The transform applied first.
        :return: A new :class:`RigidTransform` equal to ``self o first``.
        """
        return RigidTransform(self.rotation @ first.rotation,
                              self.rotation @ first.translation + self.translation)

    @classmethod
    def identity(cls) -> 'RigidTransform':
        """
        The identity motion.

        :return: ``RigidTransform(eye(3), zeros(3))``.
        """
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_frames(cls, moving_origin: np.ndarray, moving_axes: np.ndarray,
                    fixed_origin: np.ndarray, fixed_axes: np.ndarray) -> 'RigidTransform':
        """
        The motion that carries one anatomical frame onto another.

        A point with local coordinates ``c = moving_axes @ (p_m - moving_origin)`` in the moving
        frame is mapped to the point with the same local coordinates in the fixed frame,
        ``p_f = fixed_origin + fixed_axes.T @ c``. Used to initialise :func:`register_bone`
        from the frames the subdivisions compute (femoral frame, tibial plate frame).

        :param moving_origin: ``(3,)`` origin of the moving frame (world mm).
        :param moving_axes: ``(3, 3)`` row-orthonormal axes of the moving frame (row ``i`` is
            axis ``i`` in world coordinates).
        :param fixed_origin: ``(3,)`` origin of the fixed frame (world mm).
        :param fixed_axes: ``(3, 3)`` row-orthonormal axes of the fixed frame, in the same
            order and with the same handedness as ``moving_axes``.
        :return: The :class:`RigidTransform` moving -> fixed.
        """
        mo = np.asarray(moving_origin, dtype=float).reshape(3)
        ma = np.asarray(moving_axes, dtype=float).reshape(3, 3)
        fo = np.asarray(fixed_origin, dtype=float).reshape(3)
        fa = np.asarray(fixed_axes, dtype=float).reshape(3, 3)
        rotation = fa.T @ ma
        return cls(rotation, fo - rotation @ mo)


@dataclass
class RegistrationResult:
    """
    Outcome of :func:`register_bone`.

    :param transform: The fitted moving -> fixed :class:`RigidTransform`.
    :param rms_mm: Root-mean-square residual (mm): exact distance of the final transformed
        moving sample points to the fixed surface, over the ``trim_fraction`` closest of all
        sample points.
    :param median_mm: Median of those residuals (mm).
    :param p95_mm: 95th percentile of those residuals (mm).
    :param n_points: Number of sample points the residual statistics are taken over.
    :param iterations: ICP iterations performed.
    :param converged: ``True`` if the last update moved every sample point by less than
        ``tolerance_mm``; ``False`` if the iteration limit was hit or too few pairs were found.
        A converged registration can still be wrong (a local minimum) -- judge it by the
        residual.
    """
    transform: RigidTransform
    rms_mm: float
    median_mm: float
    p95_mm: float
    n_points: int
    iterations: int
    converged: bool


def crop_to_articular_end(mesh: pv.PolyData, axis: np.ndarray, length_mm: float,
                          end: str) -> pv.PolyData:
    """
    Keep the part of a bone mesh within ``length_mm`` of its extreme along an axis.

    A cell is kept if its centroid's coordinate along ``axis`` lies within ``length_mm`` of the
    smallest (``end='min'``) or largest (``end='max'``) centroid coordinate. For the femur pass
    the proximal shaft axis with ``end='min'`` (keeps the distal end); for the tibia the plateau
    normal ``e3`` (superior) with ``end='max'`` (keeps the proximal end). Restricting both scans
    to the articular end removes the FOV-dependent shaft from the registration.

    :param mesh: All-triangle bone surface (world mm).
    :param axis: ``(3,)`` direction; normalised here, so its length does not matter.
    :param length_mm: Length of the kept band (mm).
    :param end: ``'min'`` or ``'max'``: which extreme along ``axis`` to keep.
    :return: A new ``pv.PolyData`` with the kept triangles (unused points removed).
    :raises ValueError: If ``end`` is not ``'min'`` / ``'max'`` or ``axis`` has zero length.
    """
    if end not in ('min', 'max'):
        raise ValueError(f"end must be 'min' or 'max', got {end!r}")
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis)
    if not norm > 0:
        raise ValueError('axis must be a non-zero vector')
    coord = mesh.cell_centers().points @ (axis / norm)
    if end == 'min':
        keep = coord <= coord.min() + length_mm
    else:
        keep = coord >= coord.max() - length_mm
    cropped = mesh.extract_cells(np.flatnonzero(keep)).extract_surface()
    return cropped.triangulate()


def _residual_statistics(points: np.ndarray, fixed: pv.PolyData, trim_fraction: float):
    """
    Exact residuals of transformed sample points against the fixed surface, trimmed.

    :param points: ``(N, 3)`` transformed moving sample points (world mm).
    :param fixed: The fixed surface.
    :param trim_fraction: Fraction of the closest points kept.
    :return: ``(rms, median, p95, n)`` over the kept points.
    """
    d = np.sort(surface_distances(points, fixed))
    n = max(1, int(np.ceil(trim_fraction * len(d))))
    d = d[:n]
    return (float(np.sqrt(np.mean(d ** 2))), float(np.median(d)),
            float(np.percentile(d, 95)), int(n))


def register_bone(moving: pv.PolyData, fixed: pv.PolyData,
                  initial: Optional[RigidTransform] = None, max_iterations: int = 60,
                  trim_fraction: float = 0.9, max_pair_mm: float = 5.0,
                  tolerance_mm: float = 1e-4, max_points: int = 20000,
                  seed: int = 0) -> RegistrationResult:
    """
    Rigidly register one bone surface onto another by trimmed point-to-plane ICP.

    1. Sample up to ``max_points`` vertices of ``moving`` (without replacement, NumPy
       generator seeded with ``seed``, so the result is deterministic).
    2. Per iteration: transform the sample by the current estimate (starting from
       ``initial``), pair each point with its nearest ``fixed`` *vertex* (KD-tree), drop pairs
       further apart than ``max_pair_mm``, keep the ``trim_fraction`` closest of the rest, and
       solve the linearised point-to-plane least squares
       ``min sum(((w x p) + t + p - q) . n)^2`` for a small rotation ``w`` and translation
       ``t``, with ``n`` the fixed surface's point normals and ``p`` taken about the paired
       points' centroid for conditioning. The rotation is formed exactly (Rodrigues) from
       ``w``, composed onto the estimate and re-orthonormalised (SVD).
    3. Stop when the update moves every sample point by less than ``tolerance_mm``
       (``converged=True``), after ``max_iterations``, or when fewer than six pairs remain.

    The residual is the exact surface distance of the final transformed sample to ``fixed``,
    over the ``trim_fraction`` closest of all sample points (see the module docstring for why
    the pair cut-off is not applied there).

    :param moving: All-triangle bone surface to move (world mm).
    :param fixed: All-triangle reference bone surface (world mm).
    :param initial: Initial moving -> fixed estimate; identity if ``None``.
    :param max_iterations: Maximum number of ICP iterations.
    :param trim_fraction: Fraction (0, 1] of the closest pairs used per update, and of the
        closest sample points the residual statistics are taken over.
    :param max_pair_mm: Pairs further apart than this (mm) are ignored in the updates.
    :param tolerance_mm: Convergence threshold on the largest point displacement of an update.
    :param max_points: Maximum number of moving vertices sampled.
    :param seed: Seed of the sampling RNG.
    :return: A :class:`RegistrationResult`.
    :raises ValueError: If either mesh has no points or ``trim_fraction`` is outside (0, 1].
    """
    if moving.n_points == 0 or fixed.n_points == 0:
        raise ValueError('register_bone needs two non-empty meshes')
    if not 0 < trim_fraction <= 1:
        raise ValueError(f'trim_fraction must be in (0, 1], got {trim_fraction}')

    rng = np.random.default_rng(seed)
    source = np.asarray(moving.points, dtype=float)
    if len(source) > max_points:
        source = source[np.sort(rng.choice(len(source), size=max_points, replace=False))]

    fixed_points = np.asarray(fixed.points, dtype=float)
    normals_mesh = fixed.compute_normals(cell_normals=False, point_normals=True,
                                         auto_orient_normals=False, split_vertices=False)
    fixed_normals = np.asarray(normals_mesh.point_data['Normals'], dtype=float)
    tree = cKDTree(fixed_points)

    transform = RigidTransform.identity() if initial is None else \
        RigidTransform(initial.rotation, initial.translation)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        p = transform.apply(source)
        dist, idx = tree.query(p, distance_upper_bound=max_pair_mm)
        paired = np.flatnonzero(np.isfinite(dist))
        if len(paired) < 6:
            break
        n_keep = max(6, int(np.ceil(trim_fraction * len(paired))))
        paired = paired[np.argsort(dist[paired], kind='stable')[:n_keep]]
        pp, q, n = p[paired], fixed_points[idx[paired]], fixed_normals[idx[paired]]

        centre = pp.mean(axis=0)
        pc = pp - centre
        a = np.hstack([np.cross(pc, n), n])
        b = -np.einsum('ij,ij->i', pp - q, n)
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
        r_inc = Rotation.from_rotvec(x[:3]).as_matrix()
        # Increment about the centroid: p -> r_inc (p - c) + c + t.
        increment = RigidTransform(r_inc, centre - r_inc @ centre + x[3:])
        step = np.max(np.linalg.norm(increment.apply(p) - p, axis=1))

        transform = increment.compose(transform)
        u, _, vt = np.linalg.svd(transform.rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:          # keep it a proper rotation
            u[:, -1] *= -1
            rotation = u @ vt
        transform = RigidTransform(rotation, transform.translation)

        if step < tolerance_mm:
            converged = True
            break

    rms, median, p95, n_res = _residual_statistics(transform.apply(source), fixed,
                                                   trim_fraction)
    return RegistrationResult(transform=transform, rms_mm=rms, median_mm=median, p95_mm=p95,
                              n_points=n_res, iterations=iterations, converged=converged)

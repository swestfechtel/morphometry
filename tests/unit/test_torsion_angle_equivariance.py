"""Mirror equivariance of the femoral-torsion angle composition (pure, no image data).

A right image half is the mirror image of a left one: the voxel at ``x`` corresponds to
``n - 1 - x``. The side-dependent conventions in ``_proximal_angle``,
``_order_knee_points`` and ``_femoral_torsion_angle`` must therefore give the same angle
for left-side landmarks under ``side='left'`` and for their mirrored counterparts under
``side='right'``. If this holds, any left/right asymmetry of the femoral torsion lies in
the landmark extraction, not in how the angle is composed.
"""
import numpy as np
import pytest

from morphometry.measurements.femur import _femoral_torsion_angle, _order_knee_points, _proximal_angle


def _mirror_x(point: np.ndarray, n: int) -> np.ndarray:
    """Map a voxel index into the mirrored image: ``x -> n - 1 - x``, y and z unchanged.

    :param point: A length-3 index.
    :param n: Width of the image along x.
    :return: The mirrored index (same dtype family, as a float array for exactness checks).
    """
    point = np.asarray(point, dtype=float).copy()
    point[0] = n - 1 - point[0]
    return point


def _random_landmarks(rng: np.random.Generator):
    """Plausible integer landmarks: a head centre, a neck point, two condylar points.

    :param rng: The random generator.
    :return: ``(hip_start, hip_end, knee_start, knee_end)`` as float arrays.
    """
    hip_start = rng.integers(20, 180, 3).astype(float)
    hip_end = rng.integers(20, 180, 3).astype(float)
    hip_start[2] = hip_end[2]
    knee_start = rng.integers(20, 180, 3).astype(float)
    knee_end = rng.integers(20, 180, 3).astype(float)
    knee_end[2] = knee_start[2]
    return hip_start, hip_end, knee_start, knee_end


@pytest.mark.parametrize('seed', range(200))
def test_torsion_angle_is_invariant_under_mirroring_with_the_side_swapped(seed):
    """``angle(left landmarks, 'left') == angle(mirrored landmarks, 'right')`` and vice versa."""
    rng = np.random.default_rng(seed)
    n_hip, n_knee = 200, 210
    hs, he, ks, ke = _random_landmarks(rng)
    if ks[0] == ke[0]:
        # a condylar line exactly along y is a tie for ``_order_knee_points``, which keeps the
        # given order on either side and so is not mirror-consistent; the right-side torsion
        # therefore composes its angle in the mirrored (left-sided) frame instead of relying on this
        ks[0] += 1
    ms, me = _mirror_x(hs, n_hip), _mirror_x(he, n_hip)
    mks, mke = _mirror_x(ks, n_knee), _mirror_x(ke, n_knee)

    assert _femoral_torsion_angle(ms, me, mks, mke, 'right') == pytest.approx(
        _femoral_torsion_angle(hs, he, ks, ke, 'left'), abs=1e-9)
    assert _femoral_torsion_angle(ms, me, mks, mke, 'left') == pytest.approx(
        _femoral_torsion_angle(hs, he, ks, ke, 'right'), abs=1e-9)


@pytest.mark.parametrize('seed', range(50))
def test_proximal_angle_and_knee_ordering_are_mirror_consistent(seed):
    """The proximal angle swaps its axis with the side; the knee ordering swaps its endpoints."""
    rng = np.random.default_rng(seed)
    n = 200
    hs, he, ks, ke = _random_landmarks(rng)
    assert _proximal_angle(_mirror_x(hs, n), _mirror_x(he, n), 'right') == pytest.approx(
        _proximal_angle(hs, he, 'left'), abs=1e-9)

    ks[0] = ke[0] + 1 + rng.integers(0, 50)  # strictly lateral start, so the order is unambiguous
    start, end = _order_knee_points(_mirror_x(ks, n), _mirror_x(ke, n))
    assert np.array_equal(start, _mirror_x(ke, n)) and np.array_equal(end, _mirror_x(ks, n))

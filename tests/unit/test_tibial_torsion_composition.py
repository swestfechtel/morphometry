"""Rotation invariance and sign of the tibial-torsion angle composition (pure, no image data).

Tibial torsion is the signed angle, in the axial plane, from the posterior tibial condylar line to the
distal tibia -> fibula centroid line: external torsion (the distal axis turned from lateral towards
posterior relative to the proximal one) positive. In the LPI index frame, x increases towards the
patient's left and y posteriorly, so a left image half holds a patient-right leg with lateral = -x and
posterior = +y. The landmarks are synthesised from a known torsion and a rotation of the whole leg
about its long axis, which turns both lines together and so must not change the composed angle. The
former composition folded both lines to acute angles against the image x-axis and combined them by the
signs of their y-offsets, which returned ``-TT`` whenever the fibula centroid lay level with or
anterior to the tibia centroid in the image, and a wrong magnitude past 90 degrees.

The landmark search is patched out, so ``calculate_tibial_torsion`` composes the given points.
"""
import itertools

import numpy as np
import pytest

from morphometry.measurements import tibia as T


def _direction(deg: float) -> np.ndarray:
    """In-plane unit vector ``deg`` degrees from lateral (-x) towards posterior (+y), left image half.

    :param deg: Angle in degrees, positive towards posterior.
    :return: The ``(x, y)`` unit vector.
    """
    rad = np.radians(deg)
    return np.array([-np.cos(rad), np.sin(rad)])


def _synthetic_landmarks(torsion: float, rotation: float, condyle_width: float = 70., tibia_fibula: float = 25.):
    """Landmarks of a patient-right leg (left image half) with known tibial torsion and whole-leg rotation.

    :param torsion: True tibial torsion in degrees (external positive).
    :param rotation: Rotation of the whole leg in degrees, turning both lines by the same amount
        (lateral towards posterior for positive values).
    :param condyle_width: Distance between the two posterior-condylar points, in voxels.
    :param tibia_fibula: Distance between the tibia and fibula centroids, in voxels.
    :return: ``(knee_start, knee_end, ankle_start, ankle_end)``: the condylar points ordered as in
        production (``knee_start`` has the larger x), the tibia centroid and the fibula centroid.
    """
    knee_centre = np.array([100., 120., 10.])
    a, b = knee_centre.copy(), knee_centre.copy()
    a[:2] += condyle_width / 2 * _direction(rotation)
    b[:2] -= condyle_width / 2 * _direction(rotation)
    knee_start, knee_end = (a, b) if a[0] >= b[0] else (b, a)
    tibia = np.array([100., 110., 5.])
    fibula = tibia.copy()
    fibula[:2] += tibia_fibula * _direction(rotation + torsion)
    return knee_start, knee_end, tibia, fibula


def _mirror(points, n: int = 200):
    """Mirror landmarks into the left-right mirrored image (``x -> n - 1 - x``) and re-order the condyles.

    :param points: ``(knee_start, knee_end, ankle_start, ankle_end)``.
    :param n: Width of the image along x.
    :return: The mirrored landmarks, ``knee_start`` again having the larger x.
    """
    mirrored = []
    for point in points:
        point = np.asarray(point, dtype=float).copy()
        point[0] = n - 1 - point[0]
        mirrored.append(point)
    if mirrored[0][0] < mirrored[1][0]:
        mirrored[0], mirrored[1] = mirrored[1], mirrored[0]
    return mirrored


def _compose(monkeypatch, points, side: str) -> float:
    """Run ``calculate_tibial_torsion`` on given landmarks, bypassing the landmark search.

    :param monkeypatch: The pytest monkeypatch fixture.
    :param points: ``(knee_start, knee_end, ankle_start, ankle_end)``.
    :param side: Image side, 'left' or 'right'.
    :return: The composed tibial torsion in degrees.
    """
    monkeypatch.setattr(T, '_tibial_torsion_reference_points', lambda *args, **kwargs: tuple(points))
    return T.calculate_tibial_torsion(None, None, 2, 1, 2, side=side)


TORSIONS = [-20, -5, 0, 10, 25, 35, 45, 60]
ROTATIONS = [-60, -40, -20, -10, -5, 0, 5, 10, 20, 30, 40, 60]


@pytest.mark.parametrize('torsion,rotation', list(itertools.product(TORSIONS, ROTATIONS)))
def test_torsion_is_recovered_at_any_whole_leg_rotation(monkeypatch, torsion, rotation):
    """The composed angle equals the true torsion for every whole-leg rotation, on both image sides."""
    points = _synthetic_landmarks(torsion, rotation)
    assert _compose(monkeypatch, points, 'left') == pytest.approx(torsion, abs=1e-9)
    # the image-right half holds the mirror-image (patient-left) leg
    assert _compose(monkeypatch, _mirror(points), 'right') == pytest.approx(torsion, abs=1e-9)


@pytest.mark.parametrize('torsion,rotation,description', [
    (-5, 0, 'internal torsion with a horizontal condylar line'),
    (30, -30, 'fibula centroid exactly level with the tibia centroid'),
    (35, -40, 'fibula centroid anterior to the tibia centroid'),
    (45, 60, 'distal line past 90 degrees in the image'),
])
def test_edge_cases(monkeypatch, torsion, rotation, description):
    """Configurations the former fold-and-sign composition got wrong or only got right by accident."""
    points = _synthetic_landmarks(torsion, rotation)
    assert _compose(monkeypatch, points, 'left') == pytest.approx(torsion, abs=1e-9), description


def test_condylar_point_order_does_not_matter(monkeypatch):
    """The posterior condylar line has no direction: swapping its endpoints leaves the angle unchanged."""
    knee_start, knee_end, tibia, fibula = _synthetic_landmarks(30, 15)
    assert _compose(monkeypatch, (knee_end, knee_start, tibia, fibula), 'left') == pytest.approx(30, abs=1e-9)


@pytest.mark.parametrize('seed', range(100))
def test_rigid_in_plane_rotation_of_arbitrary_landmarks_leaves_the_angle_unchanged(monkeypatch, seed):
    """Rotating all four landmarks about any in-plane centre leaves the composed angle unchanged."""
    rng = np.random.default_rng(seed)
    points = [rng.uniform(20, 180, 3) for _ in range(4)]
    theta = rng.uniform(-np.pi, np.pi)
    centre = rng.uniform(50, 150, 2)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = []
    for point in points:
        moved = point.copy()
        moved[:2] = rotation @ (point[:2] - centre) + centre
        rotated.append(moved)
    for side in ('left', 'right'):
        before = _compose(monkeypatch, points, side)
        after = _compose(monkeypatch, rotated, side)
        # the result lives in (-90, 90]; compare on the circle so a value at the +-90 seam cannot fail
        assert np.cos(np.radians(2 * (after - before))) == pytest.approx(1, abs=1e-9)
        assert -90 < after <= 90

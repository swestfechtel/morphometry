"""Rotation invariance and sign of the femoral-torsion angle composition (pure, no image data).

Femoral torsion is the signed angle, in the axial plane, between the posterior condylar line and
the femoral neck axis: antetorsion (the neck turned anteriorly relative to the condyles) positive,
retrotorsion negative. In the LPI index frame, x increases towards the patient's left and y
increases posteriorly, so a left image half holds a patient-right femur with medial = +x and
anterior = -y. The landmarks are synthesised from a known torsion and a whole-leg rotation about
the long axis: rotating the whole leg rotates the neck and the condylar line together, so the
composed angle must not change. The former composition folded both lines to acute angles against
the image x-axis and combined them by the signs of their y-offsets, which returned ``-AT`` whenever
the head lay level with or posterior to the neck point in the image (e.g. a -10 degree retrotorsion
with a horizontal condylar line read +10), and a wrong magnitude once the neck turned past 90
degrees in the image.
"""
import itertools

import numpy as np
import pytest

from morphometry.measurements.femur import _femoral_torsion_angle


def _direction(deg: float) -> np.ndarray:
    """In-plane unit vector ``deg`` degrees from medial (+x) towards anterior (-y), left image half.

    :param deg: Angle in degrees, positive towards anterior.
    :return: The ``(x, y)`` unit vector.
    """
    rad = np.radians(deg)
    return np.array([np.cos(rad), -np.sin(rad)])


def _synthetic_landmarks(torsion: float, rotation: float, neck_length: float = 40., condyle_width: float = 60.):
    """Landmarks of a patient-right femur (left image half) with known torsion and whole-leg rotation.

    :param torsion: True femoral torsion in degrees (antetorsion positive).
    :param rotation: Rotation of the whole leg about its long axis in degrees; it turns the condylar
        line (medial end towards anterior for positive values) and the neck by the same amount.
    :param neck_length: Distance from the neck point to the head centre, in voxels.
    :param condyle_width: Distance between the two posterior-condylar points, in voxels.
    :return: ``(hip_start, hip_end, knee_start, knee_end)``: head centre, neck point and the two
        condylar points (``knee_start`` medial), as float arrays.
    """
    hip_end = np.array([100., 100., 30.])
    hip_start = hip_end.copy()
    hip_start[:2] += neck_length * _direction(rotation + torsion)
    knee_centre = np.array([100., 120., 10.])
    knee_start, knee_end = knee_centre.copy(), knee_centre.copy()
    knee_start[:2] += condyle_width / 2 * _direction(rotation)
    knee_end[:2] -= condyle_width / 2 * _direction(rotation)
    return hip_start, hip_end, knee_start, knee_end


def _mirror_x(point: np.ndarray, n: int = 200) -> np.ndarray:
    """Map a voxel index into the left-right mirrored image (``x -> n - 1 - x``).

    :param point: A length-3 index.
    :param n: Width of the image along x.
    :return: The mirrored index as a float array.
    """
    point = np.asarray(point, dtype=float).copy()
    point[0] = n - 1 - point[0]
    return point


TORSIONS = [-30, -10, -3, 0, 3, 10, 15, 25, 40]
ROTATIONS = [-80, -60, -40, -20, -15.5, -10, -5, 0, 5, 10, 20, 40, 60, 75, 80, 85]


@pytest.mark.parametrize('torsion,rotation', list(itertools.product(TORSIONS, ROTATIONS)))
def test_torsion_is_recovered_at_any_whole_leg_rotation(torsion, rotation):
    """The composed angle equals the true torsion for every whole-leg rotation, on both image sides."""
    points = _synthetic_landmarks(torsion, rotation)
    assert _femoral_torsion_angle(*points, 'left') == pytest.approx(torsion, abs=1e-9)
    # the image-right half holds the mirror-image (patient-left) femur
    assert _femoral_torsion_angle(*[_mirror_x(p) for p in points], 'right') == pytest.approx(torsion, abs=1e-9)


@pytest.mark.parametrize('torsion,rotation,description', [
    (-10, 0, 'retrotorsion with a horizontal condylar line'),
    (15, -15, 'head exactly level with the neck point'),
    (15, -15.5, 'head just posterior to the neck point'),
    (20, 70, 'neck at 90 degrees in the image'),
    (20, 75, 'neck past 90 degrees in the image'),
    (25, 85, 'neck at 110 degrees in the image'),
])
def test_edge_cases(torsion, rotation, description):
    """Configurations the former fold-and-sign composition got wrong or only got right by accident."""
    points = _synthetic_landmarks(torsion, rotation)
    assert _femoral_torsion_angle(*points, 'left') == pytest.approx(torsion, abs=1e-9), description


def test_condylar_point_order_does_not_matter():
    """The posterior condylar line has no direction: swapping its endpoints leaves the angle unchanged."""
    hip_start, hip_end, knee_start, knee_end = _synthetic_landmarks(12, 30)
    assert _femoral_torsion_angle(hip_start, hip_end, knee_end, knee_start, 'left') == pytest.approx(12, abs=1e-9)


@pytest.mark.parametrize('seed', range(100))
def test_rigid_in_plane_rotation_of_arbitrary_landmarks_leaves_the_angle_unchanged(seed):
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
        before = _femoral_torsion_angle(*points, side)
        after = _femoral_torsion_angle(*rotated, side)
        # the result lives in (-90, 90]; compare on the circle so a value at the +-90 seam cannot fail
        assert np.cos(np.radians(2 * (after - before))) == pytest.approx(1, abs=1e-9)
        assert -90 < after <= 90

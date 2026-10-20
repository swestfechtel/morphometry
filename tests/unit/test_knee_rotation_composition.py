"""Rotation invariance, sign and mirror symmetry of the knee-rotation composition (pure, no image data).

The knee rotation angle is the signed angle, in the axial plane, from the posterior FEMORAL
condylar line to the posterior TIBIAL condylar line: with the provisional convention
(``measurements.knee.KNEE_ROTATION_POSITIVE_SIGN``, backlog task #16), external rotation of the
tibia relative to the femur (its lateral aspect turned posteriorly) is positive. In the LPI index
frame x increases towards the patient's left and y posteriorly, so a left image half holds a
patient-right knee with lateral = -x and posterior = +y. The tests below pin the *invariances*,
which hold under either direction; the one test that depends on the direction says so.

The landmarks are synthesised from a known rotation mismatch plus an in-plane rotation of the whole
knee, which turns both lines together and so must not change the composed angle. The former
composition folded both lines to acute angles against the image x-axis and combined them by the
signs of their y-offsets, which returned the NEGATED angle whenever the medial end of the femoral
condylar line lay anterior to its lateral end in the image, and the supplement rather than the
folded angle past 90 degrees. It also ignored ``side`` entirely (negating both y-offsets leaves
their relative sign unchanged).

These tests exercise the composition directly (``_knee_rotation_angle``); the endpoint ordering
production applies (``geometry.order_condylar_points``) is irrelevant to the angle and is asserted
to be so.
"""
import itertools

import numpy as np
import pytest

from morphometry import geometry as G
from morphometry.measurements import knee as K


def _direction(deg: float) -> np.ndarray:
    """In-plane unit vector ``deg`` degrees from lateral (-x) towards posterior (+y), left image half.

    :param deg: Angle in degrees, positive towards posterior.
    :return: The ``(x, y)`` unit vector.
    """
    rad = np.radians(deg)
    return np.array([-np.cos(rad), np.sin(rad)])


def _synthetic_landmarks(rotation_angle: float, limb_rotation: float,
                         femoral_width: float = 70., tibial_width: float = 60.):
    """Condylar landmarks of a patient-right knee (left image half) with a known rotation mismatch.

    :param rotation_angle: True knee rotation angle in degrees (external tibial rotation positive).
    :param limb_rotation: In-plane rotation of the whole knee in degrees, turning both condylar
        lines by the same amount (lateral towards posterior for positive values).
    :param femoral_width: Distance between the two posterior femoral condyle points, in voxels.
    :param tibial_width: Distance between the two posterior tibial condyle points, in voxels.
    :return: ``(femur_start, femur_end, tibia_start, tibia_end)``, each pair ordered as in
        production (``start`` has the larger x index).
    """
    def _pair(centre, width, deg):
        a, b = np.array(centre, dtype=float), np.array(centre, dtype=float)
        a[:2] += width / 2 * _direction(deg)
        b[:2] -= width / 2 * _direction(deg)
        return G.order_condylar_points(a, b)

    femur = _pair([100., 120., 10.], femoral_width, limb_rotation)
    tibia = _pair([100., 125., 20.], tibial_width, limb_rotation + rotation_angle)
    return femur[0], femur[1], tibia[0], tibia[1]


def _mirror(points, n: int = 200):
    """Mirror condylar landmarks into the left-right mirrored image (``x -> n - 1 - x``).

    :param points: ``(femur_start, femur_end, tibia_start, tibia_end)``.
    :param n: Width of the image along x.
    :return: The mirrored landmarks, each condylar pair re-ordered in the mirrored frame.
    """
    f = G.order_condylar_points(G.mirror_index_x(points[0], n), G.mirror_index_x(points[1], n))
    t = G.order_condylar_points(G.mirror_index_x(points[2], n), G.mirror_index_x(points[3], n))
    return f[0], f[1], t[0], t[1]


ROTATION_ANGLES = [-75., -40., -12.5, -3., 0., 3., 12.5, 40., 75.]
LIMB_ROTATIONS = [-80., -45., -20., -5., 0., 5., 20., 45., 80., 120., 175.]


@pytest.mark.parametrize("rotation_angle", ROTATION_ANGLES)
@pytest.mark.parametrize("limb_rotation", LIMB_ROTATIONS)
def test_recovers_the_true_angle_at_any_limb_rotation(rotation_angle, limb_rotation):
    """The composed angle is the synthesised one, whatever the in-plane rotation of the knee."""
    points = _synthetic_landmarks(rotation_angle, limb_rotation)
    assert K._knee_rotation_angle(*points, 'left') == pytest.approx(rotation_angle, abs=1e-9)


@pytest.mark.parametrize("rotation_angle", ROTATION_ANGLES)
def test_invariant_to_in_plane_rotation(rotation_angle):
    """Rotating the whole knee in the axial plane does not move the angle at all."""
    values = [K._knee_rotation_angle(*_synthetic_landmarks(rotation_angle, r), 'left')
              for r in np.arange(0., 180., 2.5)]
    assert np.ptp(values) == pytest.approx(0., abs=1e-9)


@pytest.mark.parametrize("rotation_angle,limb_rotation",
                         list(itertools.product(ROTATION_ANGLES, LIMB_ROTATIONS)))
def test_mirror_equivariance(rotation_angle, limb_rotation):
    """A mirrored knee measured as the other image side gives exactly the same angle."""
    points = _synthetic_landmarks(rotation_angle, limb_rotation)
    left = K._knee_rotation_angle(*points, 'left')
    right = K._knee_rotation_angle(*_mirror(points), 'right')
    assert right == pytest.approx(left, abs=1e-9)


@pytest.mark.parametrize("rotation_angle", ROTATION_ANGLES)
def test_endpoint_order_is_irrelevant(rotation_angle):
    """Both condylar lines are undirected: swapping either pair's endpoints changes nothing."""
    fs, fe, ts, te = _synthetic_landmarks(rotation_angle, 17.)
    base = K._knee_rotation_angle(fs, fe, ts, te, 'left')
    for swapped in ((fe, fs, ts, te), (fs, fe, te, ts), (fe, fs, te, ts)):
        assert K._knee_rotation_angle(*swapped, 'left') == pytest.approx(base, abs=1e-9)


def test_sign_is_external_tibial_rotation():
    """A tibia whose lateral aspect is turned posteriorly relative to the femur reads positive."""
    # left image half = patient-right knee: lateral = -x, posterior = +y
    femur_start, femur_end = np.array([130., 120., 10.]), np.array([70., 120., 10.])
    # tibial line with its lateral (-x) end displaced posteriorly (+y) -> external rotation
    tibia_start, tibia_end = np.array([130., 115., 20.]), np.array([70., 125., 20.])
    assert K._knee_rotation_angle(femur_start, femur_end, tibia_start, tibia_end, 'left') > 0
    # the same configuration in a right image half (mirror twin) is external rotation too
    mirrored = _mirror((femur_start, femur_end, tibia_start, tibia_end))
    assert K._knee_rotation_angle(*mirrored, 'right') > 0


def test_parallel_lines_read_zero():
    """Parallel condylar lines are a zero rotation mismatch, not 180 degrees."""
    points = _synthetic_landmarks(0., 33.)
    assert K._knee_rotation_angle(*points, 'left') == pytest.approx(0., abs=1e-9)


@pytest.mark.parametrize("rotation_angle", ROTATION_ANGLES)
def test_direction_convention_is_isolated_in_one_constant(monkeypatch, rotation_angle):
    """Flipping ``KNEE_ROTATION_POSITIVE_SIGN`` negates the value and changes nothing else.

    The clinical direction is provisional (backlog task #16), so it must be decided in exactly one
    place: negating that constant must negate every reported angle while leaving the magnitude, the
    rotation invariance and the mirror equivariance intact.
    """
    points = _synthetic_landmarks(rotation_angle, 23.)
    base = K._knee_rotation_angle(*points, 'left')
    monkeypatch.setattr(K, "KNEE_ROTATION_POSITIVE_SIGN", -1)
    flipped = K._knee_rotation_angle(*points, 'left')
    assert flipped == pytest.approx(-base, abs=1e-9)
    # still rotation-invariant and mirror-equivariant with the other convention
    rotated = K._knee_rotation_angle(*_synthetic_landmarks(rotation_angle, 61.), 'left')
    assert rotated == pytest.approx(flipped, abs=1e-9)
    mirrored = K._knee_rotation_angle(*_mirror(points), 'right')
    assert mirrored == pytest.approx(flipped, abs=1e-9)

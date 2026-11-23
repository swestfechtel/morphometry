"""Unit tests for the femoral antetorsion measured against the inter-head line.

``calculate_anteversion_head_line`` measures the Murphy neck axis of both image halves against the
line through the two femoral head centres (the NaKo readers' reference), instead of the image
x-axis that ``calculate_anteversion`` uses. These tests need no data: the pure composition helper
``_head_line_antetorsion`` is exercised on hand-placed landmarks, and the fallback logic of the
measurement is exercised by monkeypatching the landmark search on a tiny synthetic mask.

Landmark frame used below (LPI index frame): x increases towards the patient's left, y posteriorly.
A left image half holds a patient-right femur, so its head centre is medial (+x) of its neck base;
a right image half is the mirror image.
"""
import logging

import nibabel as nib
import numpy as np
import pytest

from morphometry.image_io import Segmentation
from morphometry.measurements import hip as H
from morphometry.measurements.hip import _head_line_antetorsion, calculate_anteversion_head_line

X_AXIS = np.array([1., 0., 0.])


def _landmarks():
    """A plausible pair of Murphy lines in the parent index frame (antetorted: bases posterior).

    :return: ``(left_head, left_base, right_head, right_base)``, both heads on the same y and z, so
        the inter-head line is parallel to the image x-axis.
    """
    return (np.array([60., 50, 30]), np.array([40., 56, 20]),
            np.array([140., 50, 30]), np.array([160., 58, 20]))


def _rotate_z(point: np.ndarray, angle_deg: float, origin: np.ndarray) -> np.ndarray:
    """Rotate a 3-D point about the z axis through ``origin`` by ``angle_deg``.

    :param point: The point to rotate.
    :param angle_deg: The rotation angle in degrees.
    :param origin: A point on the rotation axis.
    :return: The rotated point.
    """
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return rot @ (point - origin) + origin


def _tiny_mask(spacing=(1., 1., 1.), width: int = 20) -> Segmentation:
    """A small two-sided mask with one femur voxel per half; the landmark search is monkeypatched.

    :param spacing: Voxel spacing written into the affine.
    :param width: Size of the sagittal axis; an odd width makes the right half one voxel wider than
        the left (the split is at ``width // 2``).
    :return: A Segmentation of shape (width, 10, 10).
    """
    a = np.zeros((width, 10, 10), dtype=np.uint8)
    a[4, 5, 5] = 1
    a[width - 5, 5, 5] = 1
    return Segmentation.from_nibabel(nib.Nifti1Image(a, np.diag([*spacing, 1.])))


def _fake_reference_line(left=None, right=None):
    """Build a stand-in for ``get_proximal_reference_line`` returning fixed half-frame landmarks.

    :param left: ``(head, base)`` for the left half, or an exception instance to raise.
    :param right: ``(head, base)`` for the right half (in the *right half's* index frame), or an
        exception instance to raise.
    :return: A function with the ``get_proximal_reference_line`` call signature.
    """
    def fake(hip_image, side='left', method='lee', segmentation_label=1, x_ratio=1., isotropic=False,
             return_notch_distance=False):
        assert method == 'murphy'
        result = left if side == 'left' else right
        if isinstance(result, Exception):
            raise result
        head, base = result
        return np.array(head, dtype=float), np.array(base, dtype=float), 10.
    return fake


def test_head_line_parallel_to_x_equals_x_axis_reference():
    """With both head centres on one y, the inter-head line is the x-axis and nothing changes."""
    lh, lb, rh, rb = _landmarks()
    assert _head_line_antetorsion(lh, lb, rh, rb) == pytest.approx(
        _head_line_antetorsion(lh, lb, rh, rb, reference=X_AXIS))


def test_sign_matches_calculate_anteversion_convention():
    """Neck base anterior (smaller y) to the head centre reads negative, as calculate_anteversion."""
    left, right = _head_line_antetorsion(np.array([60., 50, 0]), np.array([40., 44, 0]),
                                         np.array([140., 50, 0]), np.array([160., 44, 0]))
    assert left < 0 and right < 0
    assert left == pytest.approx(right)
    assert left == pytest.approx(-np.degrees(np.arctan2(6, 20)))


def test_antetorsion_is_positive():
    """Neck base posterior to the head centre (head turned anteriorly) reads positive."""
    left, right = _head_line_antetorsion(*_landmarks())
    assert left > 0 and right > 0


def test_head_line_angle_is_rotation_invariant():
    """A pelvis rotated in the scanner leaves the head-line values unchanged, the x-axis ones not."""
    points = _landmarks()
    origin = np.array([100., 50, 0])
    rotated = [_rotate_z(p, 17., origin) for p in points]
    np.testing.assert_allclose(_head_line_antetorsion(*rotated), _head_line_antetorsion(*points),
                               atol=1e-9)
    assert not np.allclose(_head_line_antetorsion(*rotated, reference=X_AXIS),
                           _head_line_antetorsion(*points, reference=X_AXIS), atol=1e-3)


def test_mirror_swaps_sides():
    """Mirroring every landmark (x -> W - 1 - x) and swapping the sides swaps the two values."""
    width = 200
    lh, lb, rh, rb = _landmarks()
    rb = rb + np.array([0., 3, 0])  # break the left/right symmetry so a swap is observable

    def mirror(p):
        return np.array([width - 1 - p[0], p[1], p[2]])

    left, right = _head_line_antetorsion(lh, lb, rh, rb)
    m_left, m_right = _head_line_antetorsion(mirror(rh), mirror(rb), mirror(lh), mirror(lb))
    assert left != pytest.approx(right)
    assert m_left == pytest.approx(right, abs=1e-12)
    assert m_right == pytest.approx(left, abs=1e-12)


def test_head_line_reference_is_used(monkeypatch):
    """The measurement maps the right landmarks into the parent frame and uses the head line."""
    lh, lb, rh, rb = (np.array([6., 5, 5]), np.array([3., 6, 4]),
                      np.array([14., 3, 5]), np.array([17., 4, 4]))
    half = 10
    monkeypatch.setattr(H, 'get_proximal_reference_line', _fake_reference_line(
        left=(lh, lb), right=(rh - [half, 0, 0], rb - [half, 0, 0])))
    left, right, reference = calculate_anteversion_head_line(_tiny_mask())
    assert reference == 'head_line'
    np.testing.assert_allclose((left, right), _head_line_antetorsion(lh, lb, rh, rb), atol=1e-12)


def test_right_landmarks_are_shifted_by_the_left_half_width(monkeypatch):
    """On an odd-width mask (left half 10, right half 11) the shift is the LEFT half's width.

    With equal halves, shifting by the right half's width would go unnoticed; here it moves the right
    head centre by one voxel and changes both head-line values.
    """
    lh, lb, rh, rb = (np.array([6., 5, 5]), np.array([3., 6, 4]),
                      np.array([14., 3, 5]), np.array([17., 4, 4]))
    half = 10  # 21 // 2
    monkeypatch.setattr(H, 'get_proximal_reference_line', _fake_reference_line(
        left=(lh, lb), right=(rh - [half, 0, 0], rb - [half, 0, 0])))
    left, right, reference = calculate_anteversion_head_line(_tiny_mask(width=21))
    assert reference == 'head_line'
    np.testing.assert_allclose((left, right), _head_line_antetorsion(lh, lb, rh, rb), atol=1e-12)
    wrong = _head_line_antetorsion(lh, lb, rh + [1, 0, 0], rb + [1, 0, 0])
    assert not np.allclose((left, right), wrong, atol=1e-6)


@pytest.mark.parametrize('bad', ['head', 'base'])
def test_non_finite_landmark_counts_as_failed_side(monkeypatch, bad):
    """A NaN landmark is a failed side (NaN value, x-axis for the other), never a 'head_line' row."""
    lh, lb = np.array([6., 5, 5]), np.array([3., 6, 4])
    rh, rb = np.array([4., 3, 5]), np.array([7., 4, 4])
    if bad == 'head':
        rh = np.array([np.nan, 3, 5])
    else:
        rb = np.array([7., np.nan, 4])
    monkeypatch.setattr(H, 'get_proximal_reference_line', _fake_reference_line(
        left=(lh, lb), right=(rh, rb)))
    left, right, reference = calculate_anteversion_head_line(_tiny_mask())
    assert reference == 'x_axis'
    assert np.isnan(right)
    assert left == pytest.approx(G_signed(X_AXIS, lh - lb, 'left'))


def test_one_side_failing_falls_back_to_x_axis(monkeypatch, caplog):
    """A failed right side is NaN; the left side is measured against the image x-axis."""
    lh, lb = np.array([6., 5, 5]), np.array([3., 6, 4])
    monkeypatch.setattr(H, 'get_proximal_reference_line', _fake_reference_line(
        left=(lh, lb), right=ValueError('no lesser trochanter')))
    with caplog.at_level(logging.WARNING, logger=H.logger.name):
        left, right, reference = calculate_anteversion_head_line(_tiny_mask())
    assert reference == 'x_axis'
    assert np.isnan(right)
    assert left == pytest.approx(G_signed(X_AXIS, lh - lb, 'left'))
    assert any('x-axis' in r.getMessage() and 'right' in r.getMessage() for r in caplog.records)


def test_left_side_failing_falls_back_to_x_axis(monkeypatch):
    """The mirror case: a failed left side is NaN, the right side gets an x-axis value."""
    rh, rb = np.array([4., 3, 5]), np.array([7., 4, 4])
    monkeypatch.setattr(H, 'get_proximal_reference_line', _fake_reference_line(
        left=IndexError('empty slice'), right=(rh, rb)))
    left, right, reference = calculate_anteversion_head_line(_tiny_mask())
    assert reference == 'x_axis'
    assert np.isnan(left)
    assert right == pytest.approx(G_signed(X_AXIS, rh - rb, 'right'))


def test_both_sides_failing_raises(monkeypatch):
    """With neither side's landmarks there is nothing to measure: ValueError."""
    monkeypatch.setattr(H, 'get_proximal_reference_line', _fake_reference_line(
        left=RuntimeError('left'), right=AssertionError('right')))
    with pytest.raises(ValueError):
        calculate_anteversion_head_line(_tiny_mask())


def test_degenerate_head_line_falls_back(monkeypatch, caplog):
    """Head centres coinciding in the axial plane give no line: both sides use the x-axis."""
    lh, lb = np.array([10., 5, 5]), np.array([7., 6, 4])
    rh, rb = np.array([10., 5, 9]), np.array([13., 7, 4])  # same axial (x, y) as lh, other z
    half = 10
    monkeypatch.setattr(H, 'get_proximal_reference_line', _fake_reference_line(
        left=(lh, lb), right=(rh - [half, 0, 0], rb - [half, 0, 0])))
    with caplog.at_level(logging.WARNING, logger=H.logger.name):
        left, right, reference = calculate_anteversion_head_line(_tiny_mask())
    assert reference == 'x_axis'
    np.testing.assert_allclose((left, right), _head_line_antetorsion(lh, lb, rh, rb, reference=X_AXIS),
                               atol=1e-12)
    assert np.isfinite(left) and np.isfinite(right)
    assert any('coincide' in r.getMessage() for r in caplog.records)


def test_anisotropic_in_plane_spacing_is_refused(monkeypatch):
    """Unequal in-plane voxel edges shear an index-space angle: ValueError before any landmark."""
    monkeypatch.setattr(H, 'get_proximal_reference_line', _fake_reference_line(
        left=AssertionError('must not be reached'), right=AssertionError('must not be reached')))
    with pytest.raises(ValueError, match='in-plane'):
        calculate_anteversion_head_line(_tiny_mask(spacing=(1., 2., 1.)))


def G_signed(reference, direction, side):
    """Shorthand for :func:`morphometry.geometry.signed_axial_angle` in the expectations above.

    :param reference: The reference line vector.
    :param direction: The measured direction.
    :param side: Image side.
    :return: The signed axial angle in degrees.
    """
    from morphometry.geometry import signed_axial_angle
    return signed_axial_angle(reference, direction, side)

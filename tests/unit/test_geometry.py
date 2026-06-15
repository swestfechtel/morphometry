"""Unit tests for the shared geometry helpers."""
import numpy as np
import pytest

from morphometry import geometry as G


@pytest.mark.parametrize("angle,expected", [
    (0, 0), (30, 30), (90, 90), (120, 60), (180, 0), (135, 45),
])
def test_fold_to_acute(angle, expected):
    assert G.fold_to_acute(angle) == pytest.approx(expected)


@pytest.mark.parametrize("angle,expected", [
    (0, 180), (30, 150), (90, 90), (120, 120), (180, 180), (45, 135),
])
def test_fold_to_obtuse(angle, expected):
    assert G.fold_to_obtuse(angle) == pytest.approx(expected)


def test_fold_matches_legacy_idioms():
    for a in np.linspace(0, 180, 37):
        assert G.fold_to_acute(a) == pytest.approx(180 - a if a > 90 else a)
        assert G.fold_to_obtuse(a) == pytest.approx(180 - a if a < 90 else a)


def test_split_left_right():
    arr = np.arange(10 * 4 * 3).reshape(10, 4, 3)
    left, right = G.split_left_right(arr)
    assert left.shape == (5, 4, 3)
    assert right.shape == (5, 4, 3)
    np.testing.assert_array_equal(np.concatenate([left, right], axis=0), arr)


def test_mirror_sagittal_coordinate_matches_legacy():
    # legacy: center[0] + (mask.shape[0] // 2 - center[0]) * 2
    for n in (16, 17, 100, 101):
        half = n // 2
        for c in (0, 3, half, n - 1):
            assert G.mirror_sagittal_coordinate(c, half) == c + (half - c) * 2


def test_slice_centroid_to_point():
    sl = np.zeros((5, 5), dtype=int)
    sl[1:4, 1:4] = 1  # centroid at (2, 2)
    pt = G.slice_centroid_to_point(sl, 7)
    np.testing.assert_allclose(pt, [2.0, 2.0, 7])


def test_validate_side():
    G.validate_side("left")
    G.validate_side("right")
    with pytest.raises(AssertionError):
        G.validate_side("patient_left")

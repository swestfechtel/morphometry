"""Unit tests for ``find_thinnest_area`` (robust thinnest-cartilage-area search).

The function combines the tibial and femoral thickness maps on their overlapping
axial coordinates, median-smooths the combined map to suppress single-voxel outliers,
locates the thinnest *neighbourhood*, then refines to the actual thinnest voxel within a
disc around that neighbourhood centre. These tests use synthetic thickness dicts where
the correct answer is known, without needing real data.
"""
import numpy as np
import nibabel as nib
import pytest

from morphometry.image_io import Segmentation
from morphometry.cartilage.knee import find_thinnest_area, _thinnest_point_physical


class _Bone:
    """Minimal stand-in exposing the attributes _thinnest_point_physical reads."""
    def __init__(self, point_cloud, image=None):
        self.point_cloud = point_cloud
        self.image = image


def _image(spacing=(1.0, 1.0, 1.0)):
    """A tiny segmentation whose only role is to provide in-plane voxel spacing."""
    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4))
    img.header.set_zooms(spacing)
    return Segmentation.from_nibabel(img)


def _uniform_map(value=2.0, size=12):
    """A single-subregion thickness dict filling a size x size grid with ``value``."""
    return {'c': {(float(x), float(y)): value for x in range(size) for y in range(size)}}


def test_returns_thin_patch_not_single_voxel_outlier():
    """A genuine thin patch is preferred over a lower but isolated single-voxel outlier."""
    tibia = _uniform_map(1.0)   # baseline T' = 2.0 everywhere
    femur = _uniform_map(1.0)
    # A 3x3 thin patch around (8, 8): combined thickness 1.0.
    for x in (7, 8, 9):
        for y in (7, 8, 9):
            tibia['c'][(float(x), float(y))] = 0.5
            femur['c'][(float(x), float(y))] = 0.5
    # A single-voxel outlier at (2, 2): combined thickness 0.2 (lower, but isolated).
    tibia['c'][(2.0, 2.0)] = 0.1
    femur['c'][(2.0, 2.0)] = 0.1

    result = find_thinnest_area(tibia, femur, _image(), neighbourhood_mm=3.0)

    assert result['point'] != (2.0, 2.0), 'the isolated outlier must not be chosen'
    px, py = result['point']
    assert 7 <= px <= 9 and 7 <= py <= 9, f'expected the patch, got {result["point"]}'
    assert result['combined_thickness'] == pytest.approx(1.0)
    cx, cy = result['center']
    assert 7 <= cx <= 9 and 7 <= cy <= 9
    # the robust (smoothed) thickness of the thin area is reported and outlier-free
    assert result['center_thickness'] == pytest.approx(1.0)


def test_only_overlapping_coordinates_are_searched():
    """Coordinates present in only one bone are ignored, even if extremely thin."""
    tibia = _uniform_map(1.0)
    femur = _uniform_map(1.0)
    # A very thin coordinate that exists only in the femur map -> not an overlap coord.
    femur['c'][(50.0, 50.0)] = 0.001

    result = find_thinnest_area(tibia, femur, _image(), neighbourhood_mm=3.0)
    assert result['point'] != (50.0, 50.0)
    px, py = result['point']
    assert 0 <= px < 12 and 0 <= py < 12


def test_no_overlap_raises():
    """Disjoint coordinate sets raise ValueError."""
    tibia = {'c': {(0.0, 0.0): 1.0, (1.0, 0.0): 1.0}}
    femur = {'c': {(100.0, 100.0): 1.0}}
    with pytest.raises(ValueError):
        find_thinnest_area(tibia, femur, _image())


def test_kernel_size_is_odd_and_scaled_by_spacing():
    """k is derived from physical size / spacing, forced odd and >= 3."""
    tibia = _uniform_map(1.0)
    femur = _uniform_map(1.0)
    # 4 mm neighbourhood at 1 mm spacing -> 4 -> forced odd -> 5.
    r1 = find_thinnest_area(tibia, femur, _image((1.0, 1.0, 1.0)), neighbourhood_mm=4.0)
    assert r1['kernel_size'] == 5
    # 3 mm neighbourhood at 2 mm spacing -> round(1.5)=2 -> forced odd/min -> 3.
    r2 = find_thinnest_area(tibia, femur, _image((2.0, 2.0, 2.0)), neighbourhood_mm=3.0)
    assert r2['kernel_size'] == 3


def test_marker_is_contact_midpoint():
    """The 3D marker sits at the midpoint of the femur/tibia contact surfaces at (x,y)."""
    image = _image()  # identity affine -> physical coords equal index coords
    # femur at column (5,5) spans z 10..12 (tibia-facing surface = max z = 12);
    # tibia at (5,5) spans z 16..17 (femur-facing surface = min z = 16); midpoint z = 14.
    femur = _Bone(np.array([[5, 5, 10], [5, 5, 11], [5, 5, 12]], float))
    tibia = _Bone(np.array([[5, 5, 16], [5, 5, 17]], float), image=image)
    marker = _thinnest_point_physical(tibia, femur, (5.0, 5.0))
    assert np.allclose(marker, [5.0, 5.0, 14.0])


def test_marker_falls_back_to_available_bone():
    """If only one bone has voxels at (x,y), the marker uses that bone's surface."""
    image = _image()
    femur = _Bone(np.array([[5, 5, 10], [5, 5, 12]], float))
    tibia = _Bone(np.empty((0, 3)), image=image)  # no tibial voxels at (5,5)
    marker = _thinnest_point_physical(tibia, femur, (5.0, 5.0))
    assert np.allclose(marker, [5.0, 5.0, 12.0])  # femur tibia-facing surface

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
from morphometry.cartilage.knee import (
    Tibia, find_thinnest_area, find_thinnest_area_by_compartment,
    split_tibia_thickness_by_compartment, _thinnest_point_physical,
)


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


# --- per-compartment (patient-side) search -------------------------------------------

_LT_KEYS = ('clt', 'ilt', 'elt', 'alt', 'plt')
_RT_KEYS = ('crt', 'irt', 'ert', 'art', 'prt')


def _plateau(x_range, y_range, value, patch_center=None, patch_value=None):
    """A flat ``{(x, y): value}`` plateau map with an optional 3x3 thin patch."""
    plateau = {(float(x), float(y)): value for x in x_range for y in y_range}
    if patch_center is not None:
        cx, cy = patch_center
        for x in (cx - 1, cx, cx + 1):
            for y in (cy - 1, cy, cy + 1):
                plateau[(float(x), float(y))] = patch_value
    return plateau


def _two_plateau_maps():
    """
    Synthetic two-plateau knee: the patient-right tibial plateau (``*rt`` keys) spans
    x 0..11 with a moderately thin patch at (5, 8); the patient-left plateau (``*lt``
    keys) spans x 20..31 with a *globally* thinner patch at (25, 5). The femoral map
    covers both plateaus with the same values, so combined = 2 * tibial.
    """
    right = _plateau(range(12), range(12), 1.0, patch_center=(5, 8), patch_value=0.7)
    left = _plateau(range(20, 32), range(12), 1.0, patch_center=(25, 5), patch_value=0.3)
    # Spread each plateau over its five subregion keys (split on x) so the partition by
    # key, not by a single dict, is exercised.
    tibia = {key: {} for key in _LT_KEYS + _RT_KEYS}
    for coord, value in right.items():
        tibia[_RT_KEYS[int(coord[0]) % 5]][coord] = value
    for coord, value in left.items():
        tibia[_LT_KEYS[int(coord[0]) % 5]][coord] = value
    femur = {'c': {**right, **left}}
    return tibia, femur


def test_by_compartment_finds_each_sides_own_minimum():
    """Each side finds its own thin patch, even when the other side is globally thinner."""
    tibia, femur = _two_plateau_maps()
    result = find_thinnest_area_by_compartment(tibia, femur, _image(), neighbourhood_mm=3.0)

    assert set(result) == {'left', 'right'}
    lx, ly = result['left']['point']
    rx, ry = result['right']['point']
    assert 24 <= lx <= 26 and 4 <= ly <= 6, f'left should find (25, 5), got {(lx, ly)}'
    assert 4 <= rx <= 6 and 7 <= ry <= 9, f'right should find (5, 8), got {(rx, ry)}'
    assert result['left']['combined_thickness'] == pytest.approx(0.6)
    assert result['right']['combined_thickness'] == pytest.approx(1.4)


def test_by_compartment_maps_lt_to_left_and_rt_to_right():
    """``*LT`` subregions are the patient-left compartment, ``*RT`` the patient-right one."""
    tibia, femur = _two_plateau_maps()
    result = find_thinnest_area_by_compartment(tibia, femur, _image())
    assert 20 <= result['left']['point'][0] <= 31      # the LT plateau
    assert 0 <= result['right']['point'][0] <= 11      # the RT plateau


def test_by_compartment_equals_find_thinnest_area_on_subset():
    """Each side's result is exactly ``find_thinnest_area`` on that side's tibial subset."""
    tibia, femur = _two_plateau_maps()
    result = find_thinnest_area_by_compartment(tibia, femur, _image(), neighbourhood_mm=4.0,
                                               radius_mm=1.5)
    for side, keys in (('left', _LT_KEYS), ('right', _RT_KEYS)):
        subset = {k: tibia[k] for k in keys}
        expected = find_thinnest_area(subset, femur, _image(), neighbourhood_mm=4.0,
                                      radius_mm=1.5)
        assert result[side] == expected


def test_by_compartment_accepts_display_case_keys():
    """Display labels (``cLT`` / ``cRT``) partition the same way as attribute names."""
    tibia, femur = _two_plateau_maps()
    relabelled = {key[0] + key[1:].upper(): value for key, value in tibia.items()}
    assert find_thinnest_area_by_compartment(relabelled, femur, _image()) == \
        find_thinnest_area_by_compartment(tibia, femur, _image())


def test_by_compartment_empty_side_raises_naming_side():
    """A compartment with no tibial thickness values raises a ValueError naming it."""
    tibia, femur = _two_plateau_maps()
    for key in _RT_KEYS:
        tibia[key] = {}
    with pytest.raises(ValueError, match='right'):
        find_thinnest_area_by_compartment(tibia, femur, _image())


def test_by_compartment_no_overlap_raises_naming_side():
    """A compartment whose tibial map does not overlap the femoral map raises, naming it."""
    tibia, femur = _two_plateau_maps()
    femur = {'c': {c: v for c, v in femur['c'].items() if c[0] < 20}}  # femur only over RT
    with pytest.raises(ValueError, match='left'):
        find_thinnest_area_by_compartment(tibia, femur, _image())


def test_split_tibia_thickness_by_compartment():
    """The public split keys the tibial subregion dicts by patient side, without copying values."""
    tibia, _ = _two_plateau_maps()
    split = split_tibia_thickness_by_compartment(tibia)
    assert set(split) == {'left', 'right'}
    assert set(split['left']) == set(_LT_KEYS)
    assert set(split['right']) == set(_RT_KEYS)
    assert split['left']['clt'] is tibia['clt']


def test_real_tibia_high_x_plateau_is_patient_left():
    """
    Pin the side mapping through the real ``Tibia`` pipeline, not hand-labelled keys.

    In LPI orientation increasing index x points toward patient left, so the plateau at
    higher x must come back under ``'left'`` (``*LT`` subregions). A flip in
    ``Tibia.calculate_landmarks`` (plate swap) or ``classify_point`` would break this.
    Elliptical slabs are used because box-shaped ones leave subregions empty.
    """
    array = np.zeros((80, 40, 20), np.uint8)
    X, Y = np.mgrid[:80, :40]
    for cx in (18, 58):
        array[((X - cx) / 14.0) ** 2 + ((Y - 20) / 16.0) ** 2 <= 1, 9:12] = 4
    seg = Segmentation.from_nibabel(nib.Nifti1Image(array, np.eye(4)))

    split = split_tibia_thickness_by_compartment(Tibia(seg, 4).calculate_thickness('knn'))

    left_x = [coord[0] for subregion in split['left'].values() for coord in subregion]
    right_x = [coord[0] for subregion in split['right'].values() for coord in subregion]
    assert left_x and right_x
    assert min(left_x) >= 44, 'patient-left compartment must be the high-x plateau (x 44..72)'
    assert max(right_x) <= 32, 'patient-right compartment must be the low-x plateau (x 4..32)'


def test_by_compartment_rejects_unknown_subregion_keys():
    """Keys that belong to neither compartment raise instead of being silently dropped."""
    tibia, femur = _two_plateau_maps()
    tibia['mystery'] = {(0.0, 0.0): 1.0}
    with pytest.raises(ValueError, match='mystery'):
        find_thinnest_area_by_compartment(tibia, femur, _image())

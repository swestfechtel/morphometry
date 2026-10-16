"""Unit tests for ``find_thinnest_area`` (robust thinnest-cartilage-area search).

The function combines the tibial and femoral thickness maps on the in-plane positions they
have in common, median-smooths the combined map to suppress single-cell outliers, locates the
thinnest *neighbourhood* (whose window must lie wholly inside the overlap), then refines to
the actual thinnest cell within a disc around that neighbourhood centre. Most of these tests
use synthetic thickness dicts where the correct answer is known, without needing real data;
such a dict carries no in-plane frame, so the search falls back to the image's voxel spacing
as its step (the frame-aware path is covered in ``test_cartilage_thickness.py``).
"""
import numpy as np
import nibabel as nib
import pytest

from morphometry.image_io import Segmentation
from morphometry.cartilage.knee import (
    Tibia, find_thinnest_area, find_thinnest_area_by_compartment,
    split_tibia_thickness_by_compartment, thinnest_point_position,
)
from morphometry.cartilage.thickness import KneeFrame, ThicknessMaps


def _frame(grid_mm=1.0):
    """An axis-aligned in-plane frame at the origin, so cell (u, v) sits at world (u, v, 0)."""
    return KneeFrame(origin=np.zeros(3), axes=np.eye(3), grid_mm=grid_mm, source='tibia')


def _maps(positions, frame=None):
    """A ThicknessMaps carrying only the per-cell articular-surface positions."""
    return ThicknessMaps({}, frame=frame or _frame(), positions=positions)


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
    """The 3D marker sits midway between the two bones' articular surfaces at that cell."""
    # The two bones recorded their articular surface of cell (5, 5) at z = 12 (femur) and
    # z = 16 (tibia); the joint contact is halfway between them.
    femur = _maps({(5, 5): np.array([5.0, 5.0, 12.0])})
    tibia = _maps({(5, 5): np.array([5.0, 5.0, 16.0])})
    marker = thinnest_point_position(tibia, femur, (5.0, 5.0))
    assert np.allclose(marker, [5.0, 5.0, 14.0])


def test_marker_falls_back_to_available_bone_then_to_the_frame():
    """With one surface recorded the marker uses it; with none, the cell's in-plane centre."""
    femur = _maps({(5, 5): np.array([5.0, 5.0, 12.0])})
    tibia = _maps({})
    assert np.allclose(thinnest_point_position(tibia, femur, (5.0, 5.0)), [5.0, 5.0, 12.0])
    # No surface at all: the frame still places the cell in its own plane (z = 0 here).
    assert np.allclose(thinnest_point_position(_maps({}), _maps({}), (5.0, 5.0)),
                       [5.0, 5.0, 0.0])
    # Without any frame the point cannot be placed at all.
    with pytest.raises(ValueError, match='no in-plane frame'):
        thinnest_point_position({}, {}, (5.0, 5.0))


def test_edge_margin_keeps_the_search_off_each_bones_own_rim():
    """A thin band at one bone's map edge must not win once the edge margin is on.

    This is the rim artefact :func:`find_thinnest_area` warns about: a cartilage map stops
    rather than fades, so a band of its last cells reads near zero. The always-on full-window
    rule only protects the outermost cell of that band, which is why the margin exists. Here
    the femoral map's last three columns are the rim and the genuine thin patch is at (4, 4).
    """
    tibia = _uniform_map(1.0, size=12)
    femur = _uniform_map(1.0, size=12)
    for x in (9, 10, 11):                     # the femoral map's rim band, reading ~0
        for y in range(12):
            femur['c'][(float(x), float(y))] = 0.05
    for x in (3, 4, 5):                       # a genuine thin patch, well inside both maps
        for y in (3, 4, 5):
            tibia['c'][(float(x), float(y))] = 0.6
            femur['c'][(float(x), float(y))] = 0.6

    rim = find_thinnest_area(tibia, femur, _image(), neighbourhood_mm=3.0, edge_margin_mm=0.0)
    assert rim['point'][0] >= 9.0, 'without a margin the rim band wins'

    # The margin is a distance to the first cell *outside* the map, so clearing a band three
    # cells deep needs a little more than 3 mm here.
    inside = find_thinnest_area(tibia, femur, _image(), neighbourhood_mm=3.0,
                                edge_margin_mm=3.5)
    px, py = inside['point']
    assert 3 <= px <= 5 and 3 <= py <= 5, f'expected the real patch, got {inside["point"]}'
    assert inside['femoral_thickness'] == pytest.approx(0.6)


def test_minimum_thickness_floor_excludes_implausibly_thin_cells():
    """``min_thickness_mm`` attacks the same artefact from the thickness side."""
    tibia = _uniform_map(1.0, size=12)
    femur = _uniform_map(1.0, size=12)
    for x in (5, 6, 7):                       # implausibly thin, but deep inside both maps
        for y in (5, 6, 7):
            femur['c'][(float(x), float(y))] = 0.1
    for x in (2, 3, 4):                       # plausible cartilage, merely the thinnest of it
        for y in (2, 3, 4):
            femur['c'][(float(x), float(y))] = 0.6
    thin = find_thinnest_area(tibia, femur, _image(), neighbourhood_mm=3.0)
    assert thin['femoral_thickness'] == pytest.approx(0.1)
    kept = find_thinnest_area(tibia, femur, _image(), neighbourhood_mm=3.0,
                              min_thickness_mm=0.5)
    assert kept['femoral_thickness'] == pytest.approx(0.6)


def test_edge_margin_measures_each_bones_own_boundary_not_the_overlap():
    """A cell deep in the overlap can still be at one bone's own edge; the margin sees that."""
    # The tibia covers x 0..11, the femur only x 0..7, so the femur's own rim band (x 5..7)
    # lies inside the tibial map. Only a per-bone margin can see it.
    tibia = _uniform_map(1.0, size=12)
    femur = {'c': {(float(x), float(y)): 1.0 for x in range(8) for y in range(12)}}
    for x in (5, 6, 7):
        for y in range(12):
            femur['c'][(float(x), float(y))] = 0.05
    assert find_thinnest_area(tibia, femur, _image(), edge_margin_mm=0.0)['point'][0] >= 5.0
    assert find_thinnest_area(tibia, femur, _image(), edge_margin_mm=3.5)['point'][0] < 5.0


def test_falls_back_and_warns_when_nothing_survives(caplog):
    """An over-strict restriction searches the whole overlap rather than dropping the scan."""
    tibia = _uniform_map(1.0, size=6)
    femur = _uniform_map(1.0, size=6)
    with caplog.at_level('WARNING', logger='morphometry'):
        result = find_thinnest_area(tibia, femur, _image(), edge_margin_mm=50.0)
    assert np.isfinite(result['combined_thickness'])
    assert any('falling back' in r.getMessage() for r in caplog.records)


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
    ``Tibia.calculate_landmarks`` / ``tibia_regions.find_plates`` (plate swap) would break
    this. The plates are named from the world (RAS) position, so the phantom carries an LPI
    affine (``-I``, as after ``transform_coordinate_system``) and a tibial bone block
    (label 2) under the cartilage, on whose surface the subregions are defined (index z
    grows inferiorly, so the bone is at higher z than the cartilage).

    The thickness maps are keyed in the shared in-plane frame, whose ``u`` axis points to the
    patient's **right** from the midpoint of the two plates, so the patient-left compartment
    must come back at negative ``u`` and the patient-right one at positive ``u``.
    """
    array = np.zeros((80, 40, 24), np.uint8)
    array[1:79, 1:39, 12:22] = 2
    X, Y = np.mgrid[:80, :40]
    for cx in (18, 58):
        array[((X - cx) / 14.0) ** 2 + ((Y - 20) / 16.0) ** 2 <= 1, 9:12] = 4
    seg = Segmentation.from_nibabel(nib.Nifti1Image(array, np.diag([-1.0, -1.0, -1.0, 1.0])))

    thickness = Tibia(seg, 4).calculate_thickness()
    split = split_tibia_thickness_by_compartment(thickness)

    left_u = [coord[0] for subregion in split['left'].values() for coord in subregion]
    right_u = [coord[0] for subregion in split['right'].values() for coord in subregion]
    assert left_u and right_u
    assert max(left_u) < 0, 'patient-left compartment must be at negative u (towards -world x)'
    assert min(right_u) > 0, 'patient-right compartment must be at positive u'
    # The per-compartment subsets keep the frame, so they are still overlappable.
    assert split['left'].frame.matches(thickness.frame)


def test_by_compartment_rejects_unknown_subregion_keys():
    """Keys that belong to neither compartment raise instead of being silently dropped."""
    tibia, femur = _two_plateau_maps()
    tibia['mystery'] = {(0.0, 0.0): 1.0}
    with pytest.raises(ValueError, match='mystery'):
        find_thinnest_area_by_compartment(tibia, femur, _image())

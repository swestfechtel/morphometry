"""Unit tests for the parameter-free fixes of the NMDID outlier-review code defects (2026-11-16).

See ``docs/evidence/nmdid_outlier_review.md``. Each test builds a small synthetic mask that
reproduces one defect and pins the repaired behaviour; the defect numbers are the evidence file's:

* #1 / #2 -- the CT lesser-trochanter search takes the *plausible* component and drops specks
  before counting (``femur.py::_find_trochanter_minor`` with ``ct=True``); the MRI path is unchanged;
* #8 -- ``split_ct_image`` refuses an empty or inverted crop with a named ``ValueError``;
* #9 (partial) -- a label absent from the distal reference layer raises ``ValueError``; the
  largest-component centroid was withdrawn (it measures the wrong leg when legs cross);
* #10 -- ``get_layer_with_largest_diameter`` is sized on the axial axis and refuses an empty mask;
* #11 -- a centre-edge angle whose edge is not above the head centre is ``NaN``, not >= 90 degrees;
* #12 -- the ``utils.py`` helpers raise ``ValueError`` (caught everywhere) on empty input instead of
  a raw ``IndexError`` (not caught by ``compute_torsion.py``).
"""
import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import center_of_mass

from morphometry.femur import _find_trochanter_minor
from morphometry.image_io import Segmentation, split_ct_image
from morphometry.measurements.hip import _center_edge_angle_or_nan
from morphometry.tibia import get_distal_reference_line, get_layer_with_largest_diameter
from morphometry.utils import find_notch, get_layer_with_biggest_convex_area, get_side_contour_points


def _segmentation(array: np.ndarray) -> Segmentation:
    """Wrap an array in a Segmentation with 1 mm isotropic voxels and an identity affine.

    :param array: The label array.
    :return: The Segmentation.
    """
    return Segmentation.from_nibabel(nib.Nifti1Image(array.astype(np.float64), np.eye(4)))


# --------------------------------------------------------------------------------------------
# #1 / #2: the CT lesser-trochanter search
# --------------------------------------------------------------------------------------------

# Head centre at x = 19 (the most medial slice the search starts on) and z = 0, so a component's
# distance in the (x, z) plane is simply its centroid's z in millimetres.
HEAD_CENTRE = np.array([19., 10., 0.])


def _hip_with_small_implausible_and_large_plausible() -> Segmentation:
    """One medial sagittal slice with two components: a small one ~21 mm from the head centre
    (implausible) and a larger one ~57 mm away (plausible, and holding the femur's most distal voxel
    so the shaft search ends on the same slice).

    :return: The hip Segmentation.
    """
    array = np.zeros((20, 20, 70))
    array[19, 8:12, 20:23] = 1   # 12 voxels, centroid z = 21  -> 21 mm, implausible
    array[19, 8:14, 55:61] = 1   # 36 voxels, centroid z = 57.5 -> plausible, most distal voxel z = 60
    return _segmentation(array)


def test_ct_search_takes_the_plausible_component_even_when_it_is_the_larger():
    """Defect #1: the CT path takes the component that passed the distance test."""
    _, superior = _find_trochanter_minor(_hip_with_small_implausible_and_large_plausible(), HEAD_CENTRE, ct=True)
    assert superior == 55


def test_mri_search_keeps_the_legacy_smaller_component():
    """The MRI paths (``ct=False``) are deliberately unchanged: the smaller component wins."""
    _, superior = _find_trochanter_minor(_hip_with_small_implausible_and_large_plausible(), HEAD_CENTRE, ct=False)
    assert superior == 20


def test_ct_search_drops_a_speck_before_counting_components():
    """Defect #2/#3 (speck part): head-side bone + trochanter + a 1-voxel speck is three raw
    components; the CT path drops the speck and finds the trochanter, the legacy path skips the slice."""
    array = np.zeros((20, 20, 70))
    array[19, 8:12, 10:15] = 1   # 20 voxels near the head, centroid z = 12 (implausible)
    array[19, 8:14, 55:61] = 1   # the trochanter, plausible, holds the most distal voxel
    array[19, 2, 35] = 1         # a 1-voxel speck
    hip = _segmentation(array)

    _, superior = _find_trochanter_minor(hip, HEAD_CENTRE, ct=True)
    assert superior == 55
    with pytest.raises(RuntimeError, match='no two components'):
        _find_trochanter_minor(hip, HEAD_CENTRE, ct=False)


# --------------------------------------------------------------------------------------------
# #8: inverted knee crop
# --------------------------------------------------------------------------------------------

def test_split_ct_image_refuses_a_fibula_above_the_patella():
    """Defect #8: a (mislabelled) fibula whose top lies above the patella's top inverts the knee crop."""
    array = np.zeros((20, 20, 200))
    array[8:12, 8:12, 20:60] = 7    # hip
    array[8:12, 8:12, 30:120] = 1   # femur
    array[8:12, 8:12, 100:104] = 5  # patella
    array[8:12, 8:12, 120:190] = 2  # tibia
    array[6:8, 8:12, 125:190] = 3   # fibula
    array[0:2, 0:2, 40:60] = 3      # an "arm" labelled fibula, above the patella
    with pytest.raises(ValueError, match='knee crop is empty or inverted'):
        split_ct_image(_segmentation(array), None)


# --------------------------------------------------------------------------------------------
# #9 / #10: the distal tibia-fibula line
# --------------------------------------------------------------------------------------------

def _ankle(extra_fragment: bool) -> np.ndarray:
    """A short ankle mask: tibia (2) and fibula (3), optionally with a tibia fragment of another leg.

    :param extra_fragment: Whether to add a small tibia-labelled fragment far from the tibia.
    :return: The label array (fewer sagittal columns than axial layers, to exercise defect #10).
    """
    array = np.zeros((12, 30, 40))
    array[2:8, 5:15, 5:35] = 2     # tibia
    array[9:11, 18:22, 5:35] = 3   # fibula
    if extra_fragment:
        array[0:2, 26:29, 5:35] = 2  # another leg's tibia fragment on the same layers
    return array


def test_largest_diameter_layer_is_sized_on_the_axial_axis():
    """Defect #10: shape[0] (12) < shape[2] (40) used to raise IndexError on layer 12."""
    tibia = (_ankle(False) == 2).astype(int)
    assert 5 <= get_layer_with_largest_diameter(tibia) < 35


def test_largest_diameter_layer_refuses_an_empty_mask():
    """Defect #10: an empty tibia used to return the last layer index as if it were the widest."""
    with pytest.raises(ValueError):
        get_layer_with_largest_diameter(np.zeros((12, 30, 40)))


def test_distal_centroids_still_average_every_component():
    """Defect #9 is deliberately NOT fixed by taking the largest component: on legs crossed at the
    ankle the other leg's bone is the larger piece (CT fixture case-105094 image-left). Pin that the
    centroid still averages every voxel of the label, so a future fix is a deliberate change."""
    array = _ankle(True)
    layer, com_tibia, com_fibula = get_distal_reference_line(array, 2, 3)
    assert tuple(com_tibia[:2]) == center_of_mass((array == 2)[:, :, layer].astype(int))
    assert tuple(com_fibula[:2]) == center_of_mass((array == 3)[:, :, layer].astype(int))


def test_distal_line_refuses_a_missing_fibula():
    """Defect #9: a fibula absent from the reference layer used to give a NaN centroid."""
    array = _ankle(False)
    array[array == 3] = 0
    with pytest.raises(ValueError, match='no fibula voxels'):
        get_distal_reference_line(array, 2, 3)


# --------------------------------------------------------------------------------------------
# #11: centre-edge angle with the edge not above the head
# --------------------------------------------------------------------------------------------

def test_center_edge_angle_above_the_head_is_unchanged():
    """An edge above the head centre composes exactly as before (unsigned angle, degrees)."""
    d_perp = np.array([0., 0., -100.])
    edge = np.array([-10., 0., -20.])
    assert _center_edge_angle_or_nan(d_perp, edge, 'left') == pytest.approx(np.degrees(np.arctan2(10, 20)))


@pytest.mark.parametrize('edge', [np.array([-10., 0., 0.]), np.array([-10., 0., 15.])])
def test_center_edge_angle_at_or_below_the_head_is_nan(edge):
    """Defect #11: an edge at (90 degrees) or below (> 90) the head centre is NaN, not an angle."""
    assert np.isnan(_center_edge_angle_or_nan(np.array([0., 0., -100.]), edge, 'right'))


# --------------------------------------------------------------------------------------------
# #12: utils helpers on empty input
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('most_ventral', [None, 5])
def test_find_notch_on_an_empty_section_raises_value_error(most_ventral):
    """Defect #12: both branches used to raise a raw IndexError."""
    with pytest.raises(ValueError):
        find_notch(np.zeros((10, 10)), most_ventral=most_ventral)


def test_biggest_convex_area_layer_refuses_an_empty_mask():
    """Defect #12: an empty mask used to return the last layer index."""
    with pytest.raises(ValueError):
        get_layer_with_biggest_convex_area(np.zeros((5, 5, 4)))


def test_side_contour_points_on_an_empty_column_raises_value_error():
    """Defect #12: an empty coronal column raises a named ValueError."""
    with pytest.raises(ValueError, match='no mask voxel'):
        get_side_contour_points(np.zeros((5, 5)), 2)


def test_ct_search_ignores_a_speck_that_is_the_second_of_two_components():
    """Defect #2, the main NMDID failure mode: bone + a speck is exactly two raw components. The
    legacy path takes the speck as the trochanter; the CT path drops it, sees one component, and
    moves on laterally to the slice that really holds head-side bone and the trochanter."""
    array = np.zeros((20, 20, 70))
    array[19, 8:14, 50:57] = 1   # bone, centroid z = 53 (plausible), max z 56
    array[19, 2, 21] = 1         # a 1-voxel speck 21 mm from the head
    array[18, 8:12, 10:15] = 1   # head-side bone (implausible)
    array[18, 8:14, 55:61] = 1   # the trochanter, holds the most distal voxel (z = 60)
    hip = _segmentation(array)

    _, superior_ct = _find_trochanter_minor(hip, HEAD_CENTRE, ct=True)
    _, superior_legacy = _find_trochanter_minor(hip, HEAD_CENTRE, ct=False)
    assert superior_ct == 55
    assert superior_legacy == 21


# --------------------------------------------------------------------------------------------
# #11 in the NMDID script: a one-side NaN CE is a failure of that side's row only
# --------------------------------------------------------------------------------------------

import importlib.util  # noqa: E402  (script import kept next to the tests that need it)
import pathlib  # noqa: E402

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'process_nmdid.py'
_spec = importlib.util.spec_from_file_location('script_process_nmdid_outlier_defects', _SCRIPT)
nmdid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nmdid)


def test_measure_reports_a_returned_nan_separately():
    """A returned NaN goes to ``nan_failures`` (attributed per side by the caller), not ``failures``."""
    results, failures, refused = {}, [], []
    nmdid._measure(results, failures, ('cea_left', 'cea_right'), lambda: (25.0, np.nan), nan_failures=refused)
    assert results['cea_left'] == 25.0 and np.isnan(results['cea_right'])
    assert failures == [] and refused == ['cea_right']


def test_measure_without_nan_failures_is_unchanged():
    """Every other measurement keeps the old contract: a returned NaN is stored, nothing recorded."""
    results, failures = {}, []
    nmdid._measure(results, failures, 'x', lambda: np.nan)
    assert np.isnan(results['x']) and failures == []


# --------------------------------------------------------------------------------------------
# #9: bones crossing the split line are assigned to the half holding most of them
# --------------------------------------------------------------------------------------------

from morphometry.geometry import remove_other_halfs_bones  # noqa: E402


def _two_legs() -> np.ndarray:
    """Two tibiae (label 2) in a 20-wide volume split at x = 10; the right one crosses into the left.

    :return: The label array.
    """
    array = np.zeros((20, 10, 30), dtype=np.uint8)
    array[2:6, 2:6, :] = 2              # left leg's tibia, wholly in the left half
    array[12:16, 2:6, :] = 2            # right leg's tibia, in the right half ...
    array[7:12, 2:6, 20:24] = 2         # ... crossing into the left half at the "ankle"
    return array


def test_a_crossing_bone_is_removed_from_the_minority_half():
    """The right tibia's voxels left of the split are removed; nothing else changes."""
    array = _two_legs()
    out = remove_other_halfs_bones(array, (2,))
    assert not out[7:10, 2:6, 20:24].any()                       # intruder gone from the left half
    np.testing.assert_array_equal(out[10:], array[10:])          # right half untouched
    np.testing.assert_array_equal(out[2:6], array[2:6])          # own tibia untouched


def test_legs_that_do_not_cross_are_bit_identical():
    """Components lying wholly in one half are never touched."""
    array = _two_legs()
    array[7:10] = 0
    np.testing.assert_array_equal(remove_other_halfs_bones(array, (2,)), array)


def test_touching_bones_forming_one_component_are_left_alone():
    """Two tibiae in contact are one component; the guard keeps each half's share (no leg loses its bone)."""
    array = _two_legs()
    array[5:12, 2:6, 20:24] = 2   # bridge the left tibia to the crossing part: one component
    np.testing.assert_array_equal(remove_other_halfs_bones(array, (2,)), array)


def test_labels_not_listed_are_untouched():
    """Only the listed labels are processed."""
    array = _two_legs()
    np.testing.assert_array_equal(remove_other_halfs_bones(array, (3,)), array)

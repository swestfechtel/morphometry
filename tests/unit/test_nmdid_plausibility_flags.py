"""The NMDID extent and plausibility QC flags of ``scripts/process_nmdid.py`` (no image data needed).

These flags only describe a leg; none of them may change a measured value. Each test builds a
small synthetic label array (LPI: x = sagittal axis split into halves, z grows inferior) and checks
one flag family, so a regression names the rule that broke.
"""
import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'process_nmdid.py'
_spec = importlib.util.spec_from_file_location('script_process_nmdid_plausibility', _SCRIPT)
nmdid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nmdid)

SPACING = (0.8, 0.8, 3.0)


def _leg(shape=(40, 30, 60)) -> np.ndarray:
    """An empty single-leg label array."""
    return np.zeros(shape, dtype=np.int16)


# --- tilt bound ------------------------------------------------------------------------------

def test_tilt_bound_is_zero_for_an_upright_bone_and_grows_with_tilt():
    assert nmdid.axial_angle_error_bound_deg(0.0) == pytest.approx(0.0, abs=1e-12)
    bounds = [nmdid.axial_angle_error_bound_deg(t) for t in (10, 20, 30, 45, 60)]
    assert bounds == sorted(bounds)


def test_tilt_threshold_is_where_the_bound_reaches_the_tolerance():
    """Derived, not chosen: the threshold puts the worst-case line error exactly at 4 degrees."""
    assert nmdid.axial_angle_error_bound_deg(nmdid.TILT_THRESHOLD_DEG) == pytest.approx(
        nmdid.TILT_ANGLE_TOLERANCE_DEG, abs=1e-6)
    assert nmdid.TILT_THRESHOLD_DEG == pytest.approx(29.59, abs=0.01)


def test_tilt_bound_matches_a_brute_force_maximum_over_line_directions():
    """The closed form is the maximum of |atan(tan(phi) cos(theta)) - phi| over phi."""
    theta = np.radians(35.0)
    phi = np.linspace(0, np.pi / 2, 200001)[1:-1]
    brute = np.degrees(np.max(np.abs(np.arctan(np.tan(phi) * np.cos(theta)) - phi)))
    assert nmdid.axial_angle_error_bound_deg(35.0) == pytest.approx(brute, abs=1e-6)


# --- bone tilt -------------------------------------------------------------------------------

def _rod(tilt_deg: float, shape=(80, 80, 80), spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """A thin rod through the volume centre, tilted by ``tilt_deg`` from z towards x (in mm)."""
    array = np.zeros(shape, dtype=np.int16)
    spacing = np.asarray(spacing, dtype=float)
    centre = (np.asarray(shape) - 1) / 2 * spacing
    direction = np.array([np.sin(np.radians(tilt_deg)), 0, np.cos(np.radians(tilt_deg))])
    for t in np.linspace(-30, 30, 2000):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                point = centre + t * direction + np.array([dx, dy, 0])
                index = np.rint(point / spacing).astype(int)
                if np.all(index >= 0) and np.all(index < shape):
                    array[tuple(index)] = nmdid.TIBIA_LABEL
    return array


@pytest.mark.parametrize('tilt', [0.0, 15.0, 40.0])
def test_bone_tilt_recovers_the_rod_tilt(tilt):
    assert nmdid.bone_tilt_deg(_rod(tilt), nmdid.TIBIA_LABEL, (1.0, 1.0, 1.0)) == pytest.approx(tilt, abs=2.0)


def test_bone_tilt_is_measured_in_millimetres_not_voxels():
    """On 1 x 1 x 3 mm voxels a 30-degree rod is still read as 30 degrees, not as its index-space tilt."""
    rod = _rod(30.0, shape=(80, 80, 40), spacing=(1.0, 1.0, 3.0))
    assert nmdid.bone_tilt_deg(rod, nmdid.TIBIA_LABEL, (1.0, 1.0, 3.0)) == pytest.approx(30.0, abs=2.5)


def test_bone_tilt_ignores_everything_but_the_largest_component():
    rod = _rod(0.0)
    rod[2:6, 2:6, 2:6] = nmdid.TIBIA_LABEL  # a far-away speck that would swing a whole-label axis
    rod[70:78, 2:6, 70:78] = nmdid.TIBIA_LABEL
    assert nmdid.bone_tilt_deg(rod, nmdid.TIBIA_LABEL, (1.0, 1.0, 1.0)) == pytest.approx(0.0, abs=2.0)


def test_bone_tilt_is_none_for_a_missing_bone():
    assert nmdid.bone_tilt_deg(_leg(), nmdid.FEMUR_LABEL, SPACING) is None


# --- extent flags ----------------------------------------------------------------------------

def test_a_bone_inside_the_volume_is_not_flagged():
    array = _leg()
    array[10:20, 10:20, 5:25] = nmdid.FEMUR_LABEL
    array[10:20, 10:20, 30:50] = nmdid.TIBIA_LABEL
    assert nmdid.extent_flags(array, 'left', SPACING) == ''


def test_a_femur_in_the_last_axial_slice_is_flagged_with_its_voxel_count():
    array = _leg()
    array[10:12, 10:13, 20:] = nmdid.FEMUR_LABEL  # 2 x 3 voxels in the most inferior slice
    assert nmdid.extent_flags(array, 'left', SPACING) == 'femur:at_fov_bottom:6'


def test_the_top_slice_is_checked_too():
    array = _leg()
    array[10, 10, 0:5] = nmdid.FIBULA_LABEL
    assert nmdid.extent_flags(array, 'right', SPACING) == 'fibula:at_fov_top:1'


@pytest.mark.parametrize('side, x, tag', [('left', -1, 'at_split'), ('left', 0, 'at_lateral_edge'),
                                          ('right', 0, 'at_split'), ('right', -1, 'at_lateral_edge')])
def test_the_split_line_is_the_inner_sagittal_border_of_each_half(side, x, tag):
    """Left half: split at its last x index; right half: split at its first."""
    array = _leg()
    array[x, 10, 10:20] = nmdid.TIBIA_LABEL
    assert nmdid.extent_flags(array, side, SPACING) == f'tibia:{tag}:10'


def test_the_pelvis_meeting_the_split_line_is_anatomy_and_not_flagged():
    array = _leg()
    array[-1, 10:20, 5:10] = nmdid.PELVIS_LABEL
    assert nmdid.extent_flags(array, 'left', SPACING) == ''


def test_a_tibia_starting_above_the_femur_midshaft_is_flagged_in_mm():
    array = _leg()
    array[10:20, 10:20, 10:50] = nmdid.FEMUR_LABEL  # midshaft at slice 29.5
    array[25:30, 10:20, 20:40] = nmdid.TIBIA_LABEL  # starts 9.5 slices above it
    assert nmdid.extent_flags(array, 'left', SPACING) == 'tibia:above_femur_midshaft:28'


def test_a_tibia_below_the_femur_midshaft_is_not_flagged_even_overlapping_the_femur():
    """The tibial eminence can reach above the femur's lowest slice; that is a knee, not a flag."""
    array = _leg()
    array[10:20, 10:20, 10:40] = nmdid.FEMUR_LABEL
    array[10:20, 10:20, 37:55] = nmdid.TIBIA_LABEL
    assert nmdid.extent_flags(array, 'left', SPACING) == ''


# --- plausibility flags ----------------------------------------------------------------------

def _upright_leg() -> SimpleNamespace:
    array = _leg((20, 20, 60))
    array[8:12, 8:12, 5:30] = nmdid.FEMUR_LABEL
    array[8:12, 8:12, 32:58] = nmdid.TIBIA_LABEL
    return SimpleNamespace(array=array, spacing=(1.0, 1.0, 1.0))


@pytest.mark.parametrize('radius, expected', [(23.0, ''), (9.0, ''), (52.0, ''),
                                              (2.9, 'femoral_head_radius_implausible:2.9'),
                                              (79.6, 'femoral_head_radius_implausible:79.6')])
def test_the_head_radius_gate_fires_only_outside_twice_the_adult_range(monkeypatch, radius, expected):
    monkeypatch.setattr(nmdid, 'get_femoral_head_center_ct', lambda *a, **k: (radius, np.zeros(3)))
    assert nmdid.plausibility_flags(_upright_leg(), 'left', {'center_edge_angle': 30.}) == expected


def test_a_failed_head_fit_is_reported_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise IndexError('no femur')

    monkeypatch.setattr(nmdid, 'get_femoral_head_center_ct', boom)
    assert nmdid.plausibility_flags(_upright_leg(), 'left', {}) == 'femoral_head_landmark_failed:IndexError'


@pytest.mark.parametrize('ce, expected', [(89.9, ''), (90.0, 'center_edge_not_above_head:90.0'),
                                          (151.3, 'center_edge_not_above_head:151.3'), (np.nan, '')])
def test_a_center_edge_angle_of_ninety_or_more_is_flagged(monkeypatch, ce, expected):
    monkeypatch.setattr(nmdid, 'get_femoral_head_center_ct', lambda *a, **k: (23.0, np.zeros(3)))
    assert nmdid.plausibility_flags(_upright_leg(), 'left', {'center_edge_angle': ce}) == expected


def test_a_tilted_tibia_is_flagged_after_the_hip_flags(monkeypatch):
    monkeypatch.setattr(nmdid, 'get_femoral_head_center_ct', lambda *a, **k: (80.0, np.zeros(3)))
    leg = SimpleNamespace(array=_rod(45.0), spacing=(1.0, 1.0, 1.0))
    flags = nmdid.plausibility_flags(leg, 'left', {'center_edge_angle': 120.})
    names = [f.split(':')[0] for f in flags.split(';')]
    assert names == ['femoral_head_radius_implausible', 'center_edge_not_above_head', 'tibia_tilt']
    assert float(flags.split(':')[-1]) == pytest.approx(45.0, abs=2.0)


def test_a_tilted_femur_is_flagged_too(monkeypatch):
    monkeypatch.setattr(nmdid, 'get_femoral_head_center_ct', lambda *a, **k: (23.0, np.zeros(3)))
    rod = _rod(40.0)
    rod[rod == nmdid.TIBIA_LABEL] = nmdid.FEMUR_LABEL
    flags = nmdid.plausibility_flags(SimpleNamespace(array=rod, spacing=(1.0, 1.0, 1.0)), 'left', {})
    assert flags.startswith('femur_tilt:') and float(flags.split(':')[1]) == pytest.approx(40.0, abs=2.0)


def test_a_failing_tilt_check_drops_only_the_tilt_flags(monkeypatch):
    monkeypatch.setattr(nmdid, 'get_femoral_head_center_ct', lambda *a, **k: (80.0, np.zeros(3)))

    def boom(*args, **kwargs):
        raise MemoryError('no room')

    monkeypatch.setattr(nmdid, 'bone_tilt_deg', boom)
    assert nmdid.plausibility_flags(_upright_leg(), 'left', {}) == 'femoral_head_radius_implausible:80.0'


def test_a_failing_extent_check_is_reported_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise IndexError('bad array')

    monkeypatch.setattr(nmdid, '_extent_flags', boom)
    assert nmdid.extent_flags(_leg(), 'left', SPACING) == 'extent_qc_failed:IndexError'


def test_the_plausibility_checks_do_not_mutate_the_leg(monkeypatch):
    """Flag, never repair: the leg's array is byte-equal after the QC ran."""
    monkeypatch.setattr(nmdid, 'get_femoral_head_center_ct', lambda *a, **k: (23.0, np.zeros(3)))
    leg = SimpleNamespace(array=_rod(45.0), spacing=(1.0, 1.0, 1.0))
    before = leg.array.copy()
    nmdid.plausibility_flags(leg, 'left', {'center_edge_angle': 120.})
    nmdid.extent_flags(leg.array, 'left', leg.spacing)
    assert np.array_equal(leg.array, before)


def test_split_legs_joins_the_extent_flags_after_the_integrity_flags():
    """A both-legs mask whose image-left femur reaches the last slice: pieces flag first, then extent."""
    import nibabel as nib
    from morphometry.image_io import Segmentation

    array = np.zeros((40, 20, 30), dtype=np.int16)
    array[:, 5:15, 0:4] = nmdid.PELVIS_LABEL
    array[17:20, 5:15, 0:4] = 0  # one hip bone per half
    array[20:23, 5:15, 0:4] = 0
    array[5:10, 5:10, 6:30] = nmdid.FEMUR_LABEL  # image-left femur runs into the last axial slice
    array[5:10, 5:10, 2:3] = 0
    array[30:35, 5:10, 6:20] = nmdid.FEMUR_LABEL
    array[30:35, 5:10, 22:28] = nmdid.TIBIA_LABEL
    seg = Segmentation.from_nibabel(nib.Nifti1Image(array, np.diag([0.8, 0.8, 3.0, 1.0])))
    _, _, flags = nmdid.split_legs(seg)
    left = flags['left'].split(';')
    assert left[-1] == 'femur:at_fov_bottom:25'
    assert 'tibia:missing' in left[:-1]
    assert 'femur:at_fov_bottom' not in flags['right']


def test_measure_case_puts_the_plausibility_flags_last(monkeypatch):
    leg = SimpleNamespace(name='fake leg')
    monkeypatch.setattr(nmdid, 'load_case', lambda path: leg)
    monkeypatch.setattr(nmdid, 'split_legs',
                        lambda seg: ({'left': leg, 'right': leg}, leg, {'left': 'femur:at_fov_bottom:5', 'right': ''}))
    monkeypatch.setattr(nmdid, 'split_ct_image', lambda *a, **k: (None, [None, leg, None]))
    monkeypatch.setattr(nmdid, '_measure',
                        lambda results, failures, key, fn, nan_failures=None: [results.__setitem__(k, np.nan)
                                                            for k in ((key,) if isinstance(key, str) else key)])
    monkeypatch.setattr(nmdid, 'get_murphy_neck_qc_ct', lambda *a, **k: 'murphy_axis_short:7.00')
    monkeypatch.setattr(nmdid, 'get_condylar_line_qc', lambda image, bone, **k:
                        'tibial_condylar_line_short:9.00' if bone == 'tibia' else '')
    monkeypatch.setattr(nmdid, 'plausibility_flags', lambda leg, side, row: 'femur_tilt:41.0')
    rows = {row['side']: row for row in nmdid.measure_case(pathlib.Path('case-1.nii.gz'))}
    assert rows['left']['qc_flags'] == ('femur:at_fov_bottom:5;murphy_axis_short:7.00;'
                                        'tibial_condylar_line_short:9.00;femur_tilt:41.0')
    assert rows['right']['qc_flags'] == 'murphy_axis_short:7.00;tibial_condylar_line_short:9.00;femur_tilt:41.0'

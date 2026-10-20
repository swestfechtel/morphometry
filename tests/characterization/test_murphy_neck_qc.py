"""Characterization of the Murphy neck-base QC flags on the real hips that motivate them.

``morphometry.femur.murphy_neck_base_qc`` flags the degenerate femoral-neck geometries that
produce a plausible-looking antetorsion anyway. The gates were chosen from the distributions over
every hip reachable here -- 954 Augsburg large-cohort cases, the 50 reader-study cases and the 60
NaKo sample cases, 1887 of whose 2128 image halves yield a Murphy landmark (1678 / 90 / 119); the
NMDID CT cohort could not be measured (its sshfs mount is down, ``data_paths.CT_WHOLE_LEG`` is
``None``), so no CT flag rate is known. Five of those 1887 are flagged:

* ``PA001127``, **both** halves -- head-to-base axis 6.97 and 12.77 mm (6.4 and 11.7 in-plane
  voxels) against a cohort median of 31.2 mm, i.e. 8.9 and 4.9 degrees of torsion per voxel of
  landmark error; their Murphy angles, -67.1 and -31.8 degrees, are the cohort's two most negative
  (``|z|`` 5.9 and 3.7 against mean 25.1, SD 15.6).
* ``PA000445`` image-right -- axis 14.95 mm (13.7 voxels), but a Murphy angle of 20.2 degrees
  (``|z|`` 0.3): the plausible false alarm of the axis gate at 4 degrees, kept because the gate is
  about conditioning, not about the value coming out wrong.
* ``127609_30_PD_FS_SPC_COR`` (NaKo), both halves -- the neck base lies 3.8 / 6.1 mm *inside* the
  fitted femoral head sphere (in 3-D; 11.3 / 13.0 mm by the superior-inferior drop alone), over
  1300 of the base slice's voxels are head, and that slice is two real pieces of bone
  (1773 + 484 voxels).

``PA000308`` image-right is the near miss: its base slice carries one stray voxel beside the neck,
which the speck rule in ``_significant_components`` keeps from being flagged as two pieces.

The clean hips must stay unflagged, and nothing here changes a measurement: these tests also check
that the flagged hips still return their angle.
"""
import warnings

import numpy as np
import pytest

from morphometry.femur import get_murphy_neck_qc
from morphometry.image_io import Segmentation, split_image_left_right
from morphometry.measurements.femur import calculate_femoral_torsion
from tests import data_paths


def _load(path) -> Segmentation:
    """Load one segmentation the way the batch scripts do (LPI, outliers removed).

    :param path: Path to the ``.nii.gz`` file; the test is skipped when it is absent.
    :return: The prepared Segmentation.
    """
    if not path.exists():
        pytest.skip(f'{path} not found')
    seg = Segmentation('nibabel')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seg.read_image(str(path))
        seg.transform_coordinate_system()
        seg.remove_outliers()
    return seg


def _augsburg_flags(case_dir, suffix: str = '') -> dict:
    """QC flags of both image halves of one Augsburg hip, as the cohort scripts compute them.

    :param case_dir: The patient directory holding ``hip_seg<suffix>.nii.gz``.
    :param suffix: ``'_test'`` for the reader-study segmentations, ``''`` for the cohort ones.
    :return: ``{'left': flags, 'right': flags}``.
    """
    hip = _load(case_dir / f'hip_seg{suffix}.nii.gz')
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])  # the scripts' x_ratio
    halves = dict(zip(('left', 'right'), split_image_left_right(hip)))
    return {side: get_murphy_neck_qc(half, side=side, x_ratio=x_ratio)
            for side, half in halves.items()}


@pytest.mark.needs_augsburg
def test_a_sound_augsburg_hip_is_not_flagged(augsburg_pa000001):
    """PA000001, the fixture every torsion golden is pinned on, is clean on both sides."""
    hip, knee = augsburg_pa000001['hip'], augsburg_pa000001['knee']
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    hips = dict(zip(('left', 'right'), split_image_left_right(hip)))
    knees = dict(zip(('left', 'right'), split_image_left_right(knee)))

    for side in ('left', 'right'):
        assert get_murphy_neck_qc(hips[side], side=side, x_ratio=x_ratio) == ''
        # and the measurement is still there, unchanged (the goldens pin its value)
        assert np.isfinite(calculate_femoral_torsion(hips[side], knees[side].array, side=side,
                                                     method='murphy', x_ratio=x_ratio))


@pytest.mark.needs_augsburg_large
def test_the_short_axis_hip_is_flagged_but_still_measured():
    """PA001127, both halves: 6.97 and 12.77 mm axial-plane neck axes, flagged and still reported."""
    case = data_paths.AUGSBURG_LARGE_DIR / 'PA001127'
    assert _augsburg_flags(case) == {'left': 'murphy_axis_short:6.97',
                                     'right': 'murphy_axis_short:12.77'}

    hip, knee = _load(case / 'hip_seg.nii.gz'), _load(case / 'knee_seg.nii.gz')
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    left_hip, left_knee = split_image_left_right(hip)[0], split_image_left_right(knee)[0]
    angle = calculate_femoral_torsion(left_hip, left_knee.array, side='left', method='murphy',
                                      x_ratio=x_ratio)
    assert np.isfinite(angle) and angle == pytest.approx(-67.08, abs=0.05)


@pytest.mark.needs_augsburg_large
def test_a_one_voxel_speck_on_the_base_slice_is_not_flagged():
    """PA000308's image-right half: the base slice is 719 voxels of neck plus one stray voxel.

    The only hip in the three cohorts whose base slice is not a single connected piece for a
    reason other than real damage, and its angle (20.7 degrees) is unremarkable -- hence the
    speck rule in ``_significant_components`` (a component must be worth a tenth of the largest).
    """
    assert _augsburg_flags(data_paths.AUGSBURG_LARGE_DIR / 'PA000308') == {'left': '', 'right': ''}


@pytest.mark.needs_augsburg_large
def test_the_axis_gate_s_false_alarm_is_named():
    """PA000445 image-right: a 14.95 mm axis with an unremarkable angle, flagged all the same.

    Pinned deliberately. The gate is on conditioning, so it must flag a short axis whether or not
    the angle that came out looks wrong; this is the one case in the three cohorts where it does.
    """
    flags = _augsburg_flags(data_paths.AUGSBURG_LARGE_DIR / 'PA000445')
    assert flags['right'] == 'murphy_axis_short:14.95'
    assert flags['left'] == 'murphy_landmark_failed:trochanter_implausible'


@pytest.mark.needs_augsburg_reader_study
def test_a_hip_whose_landmark_search_fails_is_reported_not_raised():
    """PA000886 (reader study): the trochanter search fails on both halves.

    The two halves give up at different steps, which is what the reason codes are for: the image-
    left one finds no plausible trochanter candidate at all, the image-right one finds one so far
    proximal that the existing ``MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`` gate rejects the level
    (its "neck base" lies 11.5 mm *above* the head centre). Either way the row keeps a flag rather
    than a silent NaN.
    """
    flags = _augsburg_flags(data_paths.AUGSBURG_READER_STUDY / 'PA000886', suffix='_test')
    assert flags == {'left': 'murphy_landmark_failed:trochanter_implausible',
                     'right': 'murphy_landmark_failed:base_at_head_level'}


@pytest.mark.needs_nako
@pytest.mark.parametrize('case, expected', [
    ('127609_30_PD_FS_SPC_COR', {'left': 'murphy_base_in_head:3.84;murphy_base_slice_multipart:2',
                                 'right': 'murphy_base_in_head:6.10;murphy_base_slice_multipart:2'}),
    ('100186_30_PD_FS_SPC_COR', {'left': '', 'right': ''}),
])
def test_the_nako_hip_whose_neck_base_sits_inside_the_head(case, expected):
    """The one NaKo case whose neck base is inside the head sphere, against a clean one."""
    mask = _load(data_paths.NAKO_SAMPLE_DIR / f'{case}.nii.gz')
    halves = dict(zip(('left', 'right'), split_image_left_right(mask)))
    assert {side: get_murphy_neck_qc(half, side=side, isotropic=True)
            for side, half in halves.items()} == expected

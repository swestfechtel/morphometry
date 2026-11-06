"""Characterization of the Lee neck-axis QC flag on the real hips that motivate it.

``morphometry.femur.lee_neck_axis_qc`` flags a Lee proximal neck axis whose ring was drawn closer
to the head axis than the fitted head's own radius -- ``a = |notch - head centre| / r`` below
``constants.LEE_MIN_NOTCH_DISTANCE_FACTOR`` -- leaving it far too small to follow the peripheral
neck contour the line is fitted to. It is deliberately **not** the stronger claim that the ring lies
inside the head: ``a`` is an in-plane distance in the selected layer and a containment test would
flag only one of the four halves below, for which see ``femur.lee_neck_axis_qc``.

The gate was placed from the distribution of ``a`` over the Augsburg large cohort: of its 1908
image halves (954 cases x 2), 1896 resolve a Lee axis -- the 12 that do not are 6 halves of 3
missing files, 3 ``ValueError`` and 3 ``RuntimeError`` -- with min 0.3089, median 1.6222, max
2.2406. **Four** fall below 1, with a gap in the distribution from 0.8434 to 1.0192 that makes
every threshold in [0.844, 1.019] equivalent on that cohort:

* ``PA000346`` image-left -- ``a = 0.768``. The case the gate exists for: its Lee torsion moved
  35.755 -> 3.168 degrees across the task #9 notch read-out change, the single largest mover of the
  100 reader-study halves, while its own neighbouring eligible layer reads ``a = 1.753`` and its
  contour fit kept 7 points against a cohort median of 53. It is also the only one of the four that
  is in the 50-patient reader study at all, and its reader-study segmentation gives the same
  ``a`` to every digit as its large-cohort one.
* ``PA000622`` image-right -- ``a = 0.843``, the highest of the four and the lower lip of the gap.
* ``PA000810`` image-right -- ``a = 0.326``.
* ``PA000932`` image-left -- ``a = 0.309``, the cohort minimum.

Nothing here changes a measurement: a flagged half keeps its angle, which these tests check, and
the clean halves of the same hips (and of PA000001, the fixture every torsion golden is pinned on)
must stay unflagged. ``a`` is a ratio of two distances that a reflection preserves, so the flag is
mirror-invariant; that is checked on the flagged half rather than on an empty flag string, where it
would be vacuous.
"""
import warnings

import numpy as np
import pytest

from morphometry.femur import get_lee_neck_qc, get_proximal_reference_line
from morphometry.image_io import Segmentation, mirror_image_left_right, split_image_left_right
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


def _lee_flags(case_dir, suffix: str = '') -> dict:
    """Lee neck-axis QC flags of both image halves of one Augsburg hip, as the scripts compute them.

    :param case_dir: The patient directory holding ``hip_seg<suffix>.nii.gz``.
    :param suffix: ``'_test'`` for the reader-study segmentations, ``''`` for the cohort ones.
    :return: ``{'left': flags, 'right': flags}``.
    """
    hip = _load(case_dir / f'hip_seg{suffix}.nii.gz')
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])  # the scripts' x_ratio
    halves = dict(zip(('left', 'right'), split_image_left_right(hip)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {side: get_lee_neck_qc(half, side=side, x_ratio=x_ratio)
                for side, half in halves.items()}


@pytest.mark.needs_augsburg
def test_a_sound_augsburg_hip_is_not_flagged(augsburg_pa000001):
    """PA000001, the fixture every torsion golden is pinned on, is clean on both sides."""
    hip, knee = augsburg_pa000001['hip'], augsburg_pa000001['knee']
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    hips = dict(zip(('left', 'right'), split_image_left_right(hip)))
    knees = dict(zip(('left', 'right'), split_image_left_right(knee)))

    for side in ('left', 'right'):
        assert get_lee_neck_qc(hips[side], side=side, x_ratio=x_ratio) == ''
        # and the measurement is still there, unchanged (the goldens pin its value)
        assert np.isfinite(calculate_femoral_torsion(hips[side], knees[side].array, side=side,
                                                     method='lee', x_ratio=x_ratio))


@pytest.mark.needs_augsburg_reader_study
def test_the_reader_study_half_the_gate_exists_for_is_flagged_but_still_measured():
    """PA000346 image-left: ``a = 0.768``, flagged, and its 3.168 degree angle still reported.

    The one flagged half that is in the reader study, and the half whose Lee torsion the task #9
    read-out change moved by 32.6 degrees. The gate does not repair it: the value this test pins is
    exactly the one the measurement produces, including for the large-cohort copy of the same hip
    (``test_the_large_cohort_halves_below_one_head_radius``).
    """
    case = data_paths.AUGSBURG_READER_STUDY / 'PA000346'
    assert _lee_flags(case, suffix='_test') == {'left': 'lee_ring_within_head_radius:0.768', 'right': ''}

    hip, knee = _load(case / 'hip_seg_test.nii.gz'), _load(case / 'knee_seg_test.nii.gz')
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    left_hip = split_image_left_right(hip)[0]
    left_knee = split_image_left_right(knee)[0]
    angle = calculate_femoral_torsion(left_hip, left_knee.array, side='left', method='lee',
                                      x_ratio=x_ratio)
    assert np.isfinite(angle) and angle == pytest.approx(3.1678, abs=0.01)


@pytest.mark.needs_augsburg_large
@pytest.mark.parametrize('case, expected', [
    ('PA000346', {'left': 'lee_ring_within_head_radius:0.768', 'right': ''}),
    ('PA000622', {'left': '', 'right': 'lee_ring_within_head_radius:0.843'}),
    ('PA000810', {'left': '', 'right': 'lee_ring_within_head_radius:0.326'}),
    ('PA000932', {'left': 'lee_ring_within_head_radius:0.309', 'right': ''}),
])
def test_the_large_cohort_halves_below_one_head_radius(case, expected):
    """The four halves of 1896 whose ring sits within the head radius, and each hip's clean half.

    Each of these hips has exactly one flagged half; pinning the other side with it is what keeps
    the gate from being read as "this patient is bad". 0.843 is the lower lip of the distribution's
    gap, so it is the value that would move first if the threshold were ever retuned.
    """
    assert _lee_flags(data_paths.AUGSBURG_LARGE_DIR / case) == expected


@pytest.mark.needs_augsburg_large
def test_a_flagged_half_still_carries_its_angle():
    """PA000932 image-left (``a = 0.309``, the cohort minimum) is measured, not turned into a NaN.

    The gate flags and never repairs, so the row keeps its number; that is the whole difference
    between this and the hard rejections inside ``get_femoral_neck_base``.
    """
    case = data_paths.AUGSBURG_LARGE_DIR / 'PA000932'
    hip, knee = _load(case / 'hip_seg.nii.gz'), _load(case / 'knee_seg.nii.gz')
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    left_hip = split_image_left_right(hip)[0]
    left_knee = split_image_left_right(knee)[0]
    assert get_lee_neck_qc(left_hip, side='left', x_ratio=x_ratio) == 'lee_ring_within_head_radius:0.309'
    assert np.isfinite(calculate_femoral_torsion(left_hip, left_knee.array, side='left',
                                                 method='lee', x_ratio=x_ratio))


@pytest.mark.needs_nako
@pytest.mark.parametrize('case, expected_a', [
    ('100186_30_PD_FS_SPC_COR', {'left': 1.4937, 'right': 1.4653}),
    ('127609_30_PD_FS_SPC_COR', {'left': 1.5619, 'right': 1.5071}),
])
def test_the_isotropic_path_is_covered_and_clean_on_nako(case, expected_a):
    """Two NaKo hips through ``isotropic=True``: unflagged, with ``a`` well clear of 1.

    The only real-data coverage of the isotropic path, and the cohort that shows the gate is
    specific rather than trigger-happy: over all 60 NaKo cases (120 halves, 1 mm isotropic) every
    half resolves and **none** is flagged, the minimum ``a`` being 1.1388 -- 14% clear of the
    threshold -- against a median of 1.5231. That matters because the two cohorts' distributions are
    comparable (NaKo median 1.5231, Augsburg 1.6222) despite 1 mm isotropic voxels against
    1.09375 x 1.09375 x 10 mm, which is a property of ``a`` being a dimensionless ratio rather than
    a length. ``127609_30_PD_FS_SPC_COR`` is deliberately included: it is the one NaKo case whose
    *Murphy* neck base is degenerate (``murphy_base_in_head`` on both halves, see
    ``test_murphy_neck_qc.py``), so it pins that the two gates are independent and a hip bad for one
    method is not automatically flagged for the other.
    """
    mask = _load(data_paths.NAKO_SAMPLE_DIR / f'{case}.nii.gz')
    halves = dict(zip(('left', 'right'), split_image_left_right(mask)))

    for side, half in halves.items():
        _, _, _, a = get_proximal_reference_line(half, side=side, method='lee',
                                                 segmentation_label=1, isotropic=True,
                                                 return_notch_distance=True)
        assert a == pytest.approx(expected_a[side], abs=5e-4)
        assert get_lee_neck_qc(half, side=side, isotropic=True) == ''


@pytest.mark.needs_augsburg_large
def test_the_ring_factor_and_the_flag_are_mirror_invariant_on_a_real_hip():
    """A half and its mirror measured as the other side give bit-identical ``a`` and flags.

    Checked on PA000346, whose image-left half is flagged and whose image-right half is not, so
    neither direction is a vacuous comparison of two empty strings. ``get_proximal_reference_line``
    measures a right half as the reflection of a left one and ``a`` divides one distance by another,
    both preserved by a reflection -- hence equality rather than a tolerance.
    """
    hip = _load(data_paths.AUGSBURG_LARGE_DIR / 'PA000346' / 'hip_seg.nii.gz')
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    halves = dict(zip(('left', 'right'), split_image_left_right(hip)))
    other = {'left': 'right', 'right': 'left'}

    for side, half in halves.items():
        _, _, _, a = get_proximal_reference_line(half, side=side, method='lee', x_ratio=x_ratio,
                                                 return_notch_distance=True)
        _, _, _, a_mirror = get_proximal_reference_line(mirror_image_left_right(half),
                                                        side=other[side], method='lee',
                                                        x_ratio=x_ratio,
                                                        return_notch_distance=True)
        assert a_mirror == a, f'the {side} half and its mirror disagree on a'
        assert (get_lee_neck_qc(mirror_image_left_right(half), side=other[side], x_ratio=x_ratio)
                == get_lee_neck_qc(half, side=side, x_ratio=x_ratio))

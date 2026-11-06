"""Characterization of the condylar-line QC flag on the real knees that motivate it.

``morphometry.knee.knee_condylar_line_qc`` flags a posterior condylar reference line too short to
carry a direction robustly -- below ``1 / tan(constants.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG)`` =
14.3007 in-plane voxels, i.e. one voxel of landmark error would tilt it by more than the tolerance.
Only the in-plane direction of that line enters the femoral torsion, the tibial torsion and the knee
rotation angle, so that tilt is the whole error.

Surveyed over the Augsburg large cohort -- 954 case directories, 952 with a ``knee_seg.nii.gz``,
i.e. 1904 image halves; 1904 femoral lines resolve and 1902 tibial ones, the two failures being
both halves of PA000120 -- **through the line the measurements actually use** -- i.e. with a
right half reflected, as all three consumers do -- the gate flags **1 femoral line (0.05%)** and
**3 tibial ones (0.16%)**:

* ``PA000877`` image-right **femur** -- 5.39 voxels (5.89 mm), 10.5 degrees of tilt per voxel. The
  case the gate exists for: its femoral search lands on reference layer 0, the edge of the field of
  view, takes the no-notch centroid fallback, and its knee rotation angle moved 69.16 -> 14.97
  degrees across the task #17 mirror-equivariance fix against a cohort mean of about 4 degrees,
  neither answer being trustworthy. It was the one leg that change could not account for, and it
  had no gate; the second-largest mover of that change was 7.01 degrees. Its image-**left** femur
  is on layer 0 too -- those two are the only layer-0 sections in the cohort -- and is **not**
  flagged, at 41.44 voxels: the gate marks the half whose line is degenerate and leaves the other
  alone, which is the verdict the evidence supports and the reason layer 0 is not itself a gate.
* ``PA000155`` image-right **tibia** -- 11.05 voxels (12.08 mm), the cohort's shortest tibial line.
* ``PA000571`` image-right **tibia** -- 13.00 voxels (14.22 mm).
* ``PA000435`` image-right **tibia** -- 13.04 voxels (21.39 mm, a 1.640625 mm in-plane spacing),
  and the only one of the four whose section resolves a notch, so it is the witness that the gate
  is not a proxy for the no-notch fallback.

``PA000001``, the fixture every torsion golden is pinned on, is clean on all four of its lines, and
``PA000120`` -- whose tibia label is empty on both halves -- comes back as
``tibial_condylar_landmark_failed:IndexError`` rather than raising.

**The reflection is not a formality, and it changes which legs are flagged.** A survey that searches
a right half directly, without the reflection its measurements apply, reproduces the cohort
distribution closely (femoral min 9.49 then 22.80, median 44.34; tibial min 12.04 then 13.04,
median 30.81) and arrives at the same *counts*, 1 femoral and 3 tibial -- but **two of the three
tibial legs are different ones**: it flags PA000591 and PA000671 right, which reflect to 18.11 and
18.25 voxels and are sound, and misses PA000435 and PA000571. It also reads the motivating femoral
leg as 9.49 voxels where the measured line is 5.39. Over the 1903 measurable right halves the two
readings differ by a median 2.27 voxels (p90 9.48, max 18.03) and agree exactly on only 276, so the
agreement in counts is a coincidence of this cohort. These tests pin the reflected reading, because
that is the line the three angles were composed from. A third figure exists for PA000877 and is
neither of these: CLAUDE.md's task #17 paragraph quotes 7.3 direct / 6.3 reflected, which predate
task #5's condylar read-out change.

**The flag reports voxels, unlike ``murphy_axis_short``'s millimetres, and this cohort is why.** It
carries four in-plane spacings (1.09375, 1.145833, 1.302083, 1.640625 mm), so a millimetre value
would not be monotone in the verdict: 51 unflagged lines are *shorter in millimetres* than
PA000435's flagged 21.39 mm (16.41 to 21.24 mm) and all 51 are longer in voxels. The cost is that
the two numbers in one ``qc_flags`` cell are in different units.

Nothing here changes a measurement: a flagged leg keeps its angles, which these tests check.
"""
import logging
import warnings

import numpy as np
import pytest

from morphometry.image_io import Segmentation, split_image_left_right
from morphometry.knee import get_condylar_line_qc, get_knee_reference_line, knee_condylar_line_qc
from morphometry.measurements.knee import calculate_knee_rotation_angle
from tests import data_paths

#: Label of each bone in the Augsburg knee segmentations, with the adjective its flag carries.
BONES = (('femur', 1, 'femoral'), ('tibia', 2, 'tibial'))


def _load(path) -> Segmentation:
    """Load one knee segmentation the way the batch scripts do (LPI, outliers removed).

    :param path: Path to the ``.nii.gz`` file; the test is skipped when it is absent.
    :return: The prepared Segmentation.
    """
    if not path.exists():
        pytest.skip(f'{path} not found')
    seg = Segmentation('nibabel')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        logging.getLogger('morphometry').setLevel(logging.CRITICAL)
        seg.read_image(str(path))
        seg.transform_coordinate_system()
        seg.remove_outliers()
    return seg


def _flags(knee: Segmentation) -> dict:
    """Condylar-line QC flags of both bones of both image halves, as the cohort scripts compute them.

    :param knee: The prepared knee Segmentation (both halves).
    :return: ``{(side, bone): flags}`` for the four lines of one knee.
    """
    halves = dict(zip(('left', 'right'), split_image_left_right(knee)))
    return {(side, bone): get_condylar_line_qc(half, bone, segmentation_label=label, side=side)
            for side, half in halves.items() for bone, label, _ in BONES}


def _cohort_knee(case: str) -> Segmentation:
    """Load one Augsburg large-cohort knee by case name.

    :param case: The patient directory name, e.g. ``'PA000877'``.
    :return: The prepared knee Segmentation.
    """
    return _load(data_paths.AUGSBURG_LARGE_DIR / case / 'knee_seg.nii.gz')


@pytest.mark.needs_augsburg
def test_a_sound_augsburg_knee_is_not_flagged(augsburg_pa000001):
    """PA000001, the fixture every torsion golden is pinned on, is clean on all four lines.

    Also checks the knee rotation angle is still there: this gate is purely additive, and PA000001's
    value is itself a committed golden.
    """
    knee = augsburg_pa000001['knee']
    assert set(_flags(knee).values()) == {''}

    halves = dict(zip(('left', 'right'), split_image_left_right(knee)))
    for side, half in halves.items():
        assert np.isfinite(calculate_knee_rotation_angle(half.array, 1, 2, side))


@pytest.mark.needs_augsburg_large
def test_the_layer_zero_femur_is_flagged_on_one_half_only():
    """PA000877: the motivating leg. Image-right femur flagged at 5.89 mm, image-left not.

    Both of its femoral searches land on reference layer 0 (the only two such sections in the
    cohort) and both take the no-notch centroid fallback, so neither the layer nor the fallback
    separates them -- the line length does: 5.39 voxels against 41.44. That is why this gate is on
    the length and layer 0 is not itself a gate.
    """
    flags = _flags(_cohort_knee('PA000877'))
    assert flags[('right', 'femur')] == 'femoral_condylar_line_short:5.39'
    assert flags[('left', 'femur')] == ''
    assert flags[('left', 'tibia')] == flags[('right', 'tibia')] == ''


@pytest.mark.needs_augsburg_large
def test_the_flagged_leg_keeps_its_knee_rotation_angle():
    """PA000877 image-right is flagged and still measured: the gate repairs and refuses nothing.

    The value is deliberately not pinned to a tolerance here. It rests on a 5.4-voxel line, which
    is the whole point of the flag -- pinning it would read as an endorsement.
    """
    halves = dict(zip(('left', 'right'), split_image_left_right(_cohort_knee('PA000877'))))
    assert np.isfinite(calculate_knee_rotation_angle(halves['right'].array, 1, 2, 'right'))


@pytest.mark.needs_augsburg_large
@pytest.mark.parametrize('case, expected, fallback', [
    ('PA000155', 'tibial_condylar_line_short:11.05', True),
    ('PA000571', 'tibial_condylar_line_short:13.00', True),
    ('PA000435', 'tibial_condylar_line_short:13.04', False),
])
def test_the_three_short_tibial_lines_are_flagged(case, expected, fallback):
    """The cohort's three short tibial lines, all image-right, all still measured.

    ``PA000435`` is pinned specifically because its section resolves a notch -- it is flagged
    **without** taking the no-notch centroid fallback, so the gate is not a proxy for that fallback
    (which fires on 15.77% of tibial sections and is measured to be nearly harmless on its own).
    It is also the case that decides the flag's units: its in-plane spacing is 1.640625 mm against
    the other two's 1.09375, so in millimetres it reads 21.39 against 12.08 -- larger than 51
    *unflagged* lines in this same cohort -- while in the voxels the flag reports, all three sit
    below the 14.30 threshold.
    """
    knee = _cohort_knee(case)
    flags = _flags(knee)
    assert flags[('right', 'tibia')] == expected
    assert flags[('left', 'tibia')] == flags[('left', 'femur')] == flags[('right', 'femur')] == ''

    halves = dict(zip(('left', 'right'), split_image_left_right(knee)))
    assert np.isfinite(calculate_knee_rotation_angle(halves['right'].array, 1, 2, 'right'))


@pytest.mark.needs_augsburg_large
def test_an_empty_bone_label_is_a_failure_flag_not_an_exception():
    """PA000120's tibia label is empty on both halves, and the wrapper reports rather than raises.

    The Augsburg scripts call the wrapper outside any try block, so this is the property that keeps
    one unusable label from costing the whole patient. The measurements that read this line do raise
    (``IndexError`` out of ``utils.find_notch``), which is why the flag is worth having.
    """
    flags = _flags(_cohort_knee('PA000120'))
    assert flags[('left', 'tibia')] == flags[('right', 'tibia')] == \
        'tibial_condylar_landmark_failed:IndexError'
    assert flags[('left', 'femur')] == flags[('right', 'femur')] == ''


@pytest.mark.needs_augsburg_large
def test_the_wrapper_reports_the_reflected_line_the_measurement_uses():
    """On PA000877 image-right the reflected and unreflected searches differ by 4.10 voxels.

    The three consumers measure a right half as the left-right reflection of a left one, so the
    reflected line is the one the angles were composed from. Pinned on a real half because the
    phantom version of this claim (a double reflection being bitwise identical) cannot show that the
    distinction has any size -- here it is 5.39 against 9.49 voxels, and over the cohort it changes
    which tibial legs are flagged at all.
    """
    half = split_image_left_right(_cohort_knee('PA000877'))[1]
    mask = np.where(half.array == 1, 1, 0)

    reflected = knee_condylar_line_qc(*get_knee_reference_line(mask[::-1], 'femur')[1:],
                                      spacing=half.spacing, bone='femur')
    direct = knee_condylar_line_qc(*get_knee_reference_line(mask, 'femur')[1:],
                                   spacing=half.spacing, bone='femur')

    assert reflected['line_length_voxels'] == pytest.approx(5.385, abs=0.001)
    assert direct['line_length_voxels'] == pytest.approx(9.487, abs=0.001)
    # the wrapper reports the first of the two, bitwise
    assert get_condylar_line_qc(half, 'femur', segmentation_label=1,
                                side='right') == reflected['flags']
    assert reflected['flags'] != direct['flags']


@pytest.mark.needs_augsburg_large
def test_the_flag_is_mirror_invariant_on_a_flagged_half():
    """A half and its mirror, measured as the other side, give a bitwise identical flag.

    Checked on the flagged half rather than on an empty flag string, where it would be vacuous. The
    wrapper reflects a right half, so a mirrored right half passed in as ``'left'`` is the array the
    right-side path would have produced -- the same computation on the same bits.
    """
    from morphometry.image_io import mirror_image_left_right

    half = split_image_left_right(_cohort_knee('PA000877'))[1]
    as_right = get_condylar_line_qc(half, 'femur', segmentation_label=1, side='right')
    as_left = get_condylar_line_qc(mirror_image_left_right(half), 'femur', segmentation_label=1,
                                   side='left')
    assert as_right == as_left == 'femoral_condylar_line_short:5.39'

"""Mirror equivariance of the knee rotation angle (``calculate_knee_rotation_angle``), on real data.

Backlog task #17. The knee rotation angle was the last measurement reading a posterior
condylar line straight out of ``knee.get_knee_reference_line`` without reflecting a
right image half first -- ``measurements.femur._mri_torsion`` and
``measurements.tibia._tibial_torsion_points`` both already did. It now measures a right
half as the reflection of a left one, so a knee and its mirror image give the same angle.

**Where the asymmetry was, and why these probes have content.** The *composition* was
already symmetric: ``side='right'`` negates the x components, which is algebraically the
same as mirroring both condylar lines, so it contributed exactly nothing to the
disagreement. The whole of it was in the landmark *search*, which is not invariant
under ``x -> n - 1 - x``. (Up to task #4 the search also rotated the section about the
image centre and rounded the result; task #5 made the read-out exact, which shrank the
condylar line's own mirror gap to a median 0.33 and max 2.06 degrees -- from 0.71 and 4.28
-- without closing it, because the layer choice, the notch search and the dorsal pair that
sets the frame are still not mirror-symmetric. The reflection below is what makes the
measurement equivariant, and it is still needed.)
Measured over the 100 Augsburg reader-study halves before the change, a half and its
mirror disagreed by a median 1.16 degrees, p90 3.06, max 6.21, and agreed exactly on only
5. After it, all 100 agree to 0.000e+00.

The three patients below are chosen because they carry that content rather than for being
typical:

* ``PA001055`` image-LEFT -- the largest gap in the cohort, 6.21 degrees, and it crosses
  zero (-5.476 against +0.730). Note what this case is: a left half's *value* does not
  move, because the left-side path is unchanged; what moved is the answer you get by
  mirroring the mask and asking for the other side. So it is an equivariance gap, not a
  movement, and quoting 6.21 as the size of the change would overstate it.
* ``PA000717`` image-RIGHT -- the largest real movement, 3.831 -> 7.969 degrees. Its two
  knees also went from 4.06 degrees apart to 0.08, which is suggestive but not evidence:
  over the 50 patients the left/right agreement does **not** improve significantly
  (mean |L-R| 3.39 -> 3.17, Wilcoxon p = 0.37, 28 patients better and 20 worse). The case
  for this change is equivariance and internal consistency, not accuracy.
* ``PA000699`` image-RIGHT -- 4.755 -> 0.846 degrees, a second independent mover.

**What these probes cannot catch.** After the change both sides reach one code path, so
the two calls run the same arithmetic on the same array and the equality holds by
construction. They are a guard against a right-side search being reintroduced, not an
independent derivation -- the same caveat ``test_femoral_head_equivariance.py`` carries.
They did fail against the pre-change code, which is the only reason they are worth having.
"""
import numpy as np
import pytest

from morphometry.image_io import split_image_left_right
from morphometry.measurements.knee import calculate_knee_rotation_angle

FEMUR_LABEL, TIBIA_LABEL = 1, 2
OTHER = {'left': 'right', 'right': 'left'}

#: (patient, image side) pairs whose mirror disagreement was non-zero before task #17,
#: with the pre-change gap in degrees for the record.
CASES = [('PA001055', 'left', 6.2057), ('PA000717', 'right', 4.1380),
         ('PA000699', 'right', 3.9092)]

pytestmark = pytest.mark.needs_augsburg_reader_study


def _half(knee, side: str) -> np.ndarray:
    """The requested image half of a knee segmentation, as a bare array.

    :param knee: The whole knee Segmentation (both legs).
    :param side: 'left' or 'right' image side.
    :return: That half's label array.
    """
    return split_image_left_right(knee)[0 if side == 'left' else 1].array


@pytest.mark.parametrize('patient,side,pre_change_gap', CASES,
                         ids=[f'{p}-{s}' for p, s, _ in CASES])
def test_knee_rotation_angle_is_mirror_equivariant(augsburg_reader_study, patient, side,
                                                   pre_change_gap):
    """A half and its mirror, measured as the two sides, give the same angle.

    Asserted exactly: after the change both calls reach the same left-side code with the
    same array, so any difference at all is a defect in the reflection rather than
    floating-point reassociation.

    :param pre_change_gap: The disagreement this case showed before task #17, in degrees;
        carried only so the case cannot be mistaken for an arbitrary pick.
    """
    mask = _half(augsburg_reader_study(patient)['knee'], side)

    direct = calculate_knee_rotation_angle(mask, FEMUR_LABEL, TIBIA_LABEL, side=side)
    mirrored = calculate_knee_rotation_angle(mask[::-1], FEMUR_LABEL, TIBIA_LABEL,
                                             side=OTHER[side])

    assert direct == mirrored, (f'{patient} image-{side} disagrees with its mirror by '
                                f'{abs(direct - mirrored):.4f} deg (was {pre_change_gap:.4f} '
                                f'before task #17)')


@pytest.mark.parametrize('patient,side,pre_change_gap', CASES,
                         ids=[f'{p}-{s}' for p, s, _ in CASES])
def test_knee_rotation_shares_the_torsions_condylar_lines(augsburg_reader_study, patient, side,
                                                          pre_change_gap):
    """The angle is composed from the lines the two torsions use on the same mask.

    This is the defect that mattered in production rather than between the sides: before
    task #17 a right-side cohort row's knee rotation angle rested on a different posterior
    condylar line from the femoral and tibial torsion in the *same* row, because those two
    reflected and this did not. Over the 50 right halves the tibial line's orientation
    differed by a median 1.23 degrees (max 4.28, equal on only 8) and the femoral line's by
    a median 0.62 (max 2.70, equal on 10); on left halves all 50 agreed exactly.

    Reproduced here through the measurement's own result: composing from the torsions'
    lines is, for a right half, exactly measuring the reversed mask as a left one.
    """
    mask = _half(augsburg_reader_study(patient)['knee'], side)

    reported = calculate_knee_rotation_angle(mask, FEMUR_LABEL, TIBIA_LABEL, side=side)
    # what the torsions' reflected search yields for this half
    as_torsions_see_it = calculate_knee_rotation_angle(
        mask[::-1] if side == 'right' else mask, FEMUR_LABEL, TIBIA_LABEL, side='left')

    assert reported == as_torsions_see_it

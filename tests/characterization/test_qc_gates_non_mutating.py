"""The three QC gate families are non-mutating at RUNTIME, not only by diff.

Each gate -- ``femur.get_murphy_neck_qc``, ``femur.get_lee_neck_qc`` and
``knee.get_condylar_line_qc`` -- is deliberately built to *re-resolve* the landmarks its
measurement already resolved, rather than thread an out-parameter through the ``calculate_*``
signatures. That design is what keeps a gate from drifting away from the measurement it describes,
but it also means a gate runs the same searches over the same arrays a published value depends on.

"Purely additive" is normally argued from ``git diff``: no measurement module touched, no golden
moved. That is necessary and not sufficient. The strong form, which these tests pin, is to measure,
call the gate, measure again, and compare **bitwise**, with the array checked byte-equal either
side -- so a gate that wrote into the mask it was handed, or left a value moved behind it, fails
here rather than in a cohort table six months later.

It holds by construction today: every gate binarises with ``np.where`` (a copy) and reflects a
right half with ``mask[::-1]`` (a read-only view). The point is that nothing enforced it. CLAUDE.md
recorded this check as having been run by hand for the condylar gate only, and noted that neither
the Murphy nor the Lee gate had it, making it "the one verification in this family that is not yet
uniform"; no test pinned it for any of the three. These tests make it uniform and durable.

The measurements chosen are exactly the ones that read the landmarks each gate re-resolves: the Lee
and Murphy femoral torsion and the anteversion for the two hip gates (the anteversion shares the
Murphy line), and the knee rotation angle for the condylar gate, which reads both condylar lines.
PA000001 is clean on all of its lines, so these also pin that a *sound* case stays sound -- the
flags are asserted to be empty, which is what makes a silent change of verdict visible too.
"""
import warnings

import numpy as np
import pytest

from morphometry.femur import get_lee_neck_qc, get_murphy_neck_qc
from morphometry.image_io import split_image_left_right
from morphometry.knee import get_condylar_line_qc
from morphometry.measurements.femur import calculate_femoral_torsion
from morphometry.measurements.hip import calculate_anteversion
from morphometry.measurements.knee import calculate_knee_rotation_angle

pytestmark = pytest.mark.needs_augsburg

SIDES = ('left', 'right')


def _halves(segmentation):
    """Split one segmentation into an image-side-keyed dict of halves.

    :param segmentation: The LPI Segmentation to split.
    :return: ``{'left': left_half, 'right': right_half}``.
    """
    left, right = split_image_left_right(segmentation)
    return {'left': left, 'right': right}


@pytest.mark.parametrize('side', SIDES)
def test_the_hip_gates_leave_the_femoral_torsion_bit_identical(augsburg_pa000001, side):
    """Murphy and Lee QC must not move the three measurements that share their landmarks.

    Discriminating in the way that matters: the comparison is ``np.array_equal`` on the returned
    angles, not an ``approx``, so a gate that perturbed a landmark by a single float ULP would
    fail. The hip array is checked byte-equal as well, since an in-place write is the failure mode
    a value comparison alone could hide if the mutation happened to be idempotent.
    """
    hip = _halves(augsburg_pa000001['hip'])[side]
    knee = _halves(augsburg_pa000001['knee'])[side]

    def measure():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return (calculate_femoral_torsion(hip, knee, side=side, method='lee'),
                    calculate_femoral_torsion(hip, knee, side=side, method='murphy'),
                    calculate_anteversion(hip, side=side))

    before = hip.array.tobytes()
    pre = measure()
    murphy_flags = get_murphy_neck_qc(hip, side=side)
    lee_flags = get_lee_neck_qc(hip, side=side)
    post = measure()

    assert murphy_flags == '', f'PA000001 {side} hip is sound; Murphy flagged {murphy_flags!r}'
    assert lee_flags == '', f'PA000001 {side} hip is sound; Lee flagged {lee_flags!r}'
    assert hip.array.tobytes() == before, 'a hip gate wrote into the mask it was handed'
    for name, a, b in zip(('lee', 'murphy', 'anteversion'), pre, post):
        assert np.array_equal(np.asarray(a, float), np.asarray(b, float)), \
            f'{name} moved across the hip QC gates on the {side} half'


@pytest.mark.parametrize('side', SIDES)
def test_the_condylar_gate_leaves_the_knee_rotation_angle_bit_identical(augsburg_pa000001, side):
    """The condylar gate must not move the measurement that reads both of its lines.

    Both bones are gated, because the gate is called once per bone and a single ``qc_flags`` cell
    carries both; the knee rotation angle is the measurement that reads the femoral *and* tibial
    condylar line, so it is the one value sensitive to either call.
    """
    knee = _halves(augsburg_pa000001['knee'])[side]

    def measure():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return calculate_knee_rotation_angle(knee, 1, 2, side)

    before = knee.array.tobytes()
    pre = measure()
    femur_flags = get_condylar_line_qc(knee, 'femur', segmentation_label=1, side=side)
    tibia_flags = get_condylar_line_qc(knee, 'tibia', segmentation_label=2, side=side)
    post = measure()

    assert femur_flags == '', f'PA000001 {side} femoral line is sound; flagged {femur_flags!r}'
    assert tibia_flags == '', f'PA000001 {side} tibial line is sound; flagged {tibia_flags!r}'
    assert knee.array.tobytes() == before, 'the condylar gate wrote into the mask it was handed'
    assert np.array_equal(np.asarray(pre, float), np.asarray(post, float)), \
        f'the knee rotation angle moved across the condylar gate on the {side} half'

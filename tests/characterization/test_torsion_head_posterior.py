"""Femoral torsion on real legs whose femoral head lies level with or posterior to the neck point.

The Augsburg PA000001 goldens never exercise this configuration: its head is anterior to the neck
point on all four legs, where the former fold-and-sign composition happened to agree with the signed
angle. The two legs here are ones where it did not: it returned the negated torsion. Both are Lee
measurements on the Augsburg reader study, one through the image-left path and one through the
reflection path of the image-right half, and both readers recorded clear antetorsion for them.

Besides the goldens, each case asserts the sign and a loose agreement with the mean of the two
readers, so a regression to the old composition fails with a readable message rather than only a
golden drift. The pipeline mirrors ``scripts/process_augsburg_reader_study.py``.
"""
import pytest

from morphometry.image_io import split_image_left_right
from morphometry.measurements.femur import calculate_femoral_torsion
from tests.conftest import assert_golden

pytestmark = pytest.mark.needs_augsburg_reader_study

# (patient, image side, method, reader values (felix, jonathan) for that leg, in degrees).
# image-left half = patient right ("rechts"), image-right half = patient left ("links").
CASES = [
    ('PA001023', 'left', 'lee', (25.4, 29.0)),   # former composition: -20.8; neck image angle -1.4 deg
    ('PA000147', 'right', 'lee', (25.7, 24.0)),  # former composition: -15.9; via the reflection path
]

# Lee under-reads these readers by ~9 degrees on average across the cohort, so allow a loose margin.
READER_TOLERANCE = 12.


@pytest.mark.parametrize('patient,side,method,readers', CASES)
def test_femoral_torsion_with_head_level_or_posterior(augsburg_reader_study, patient, side, method, readers):
    """The torsion is positive, near the readers, and matches its golden."""
    segs = augsburg_reader_study(patient)
    hip, knee = segs['hip'], segs['knee']
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    hips = dict(zip(('left', 'right'), split_image_left_right(hip)))
    knees = dict(zip(('left', 'right'), split_image_left_right(knee)))

    angle = calculate_femoral_torsion(hips[side], knees[side].array, side=side, method=method, x_ratio=x_ratio)

    reader_mean = sum(readers) / len(readers)
    assert angle > 0, f'{patient} {side} {method}: {angle:.2f} deg has the wrong sign (readers {readers})'
    assert abs(angle - reader_mean) < READER_TOLERANCE, \
        f'{patient} {side} {method}: {angle:.2f} deg is far from the reader mean {reader_mean:.2f}'
    assert_golden(f'femoral_torsion_{patient}_{side}_{method}', angle)

"""Characterization of femoral and tibial torsion on the Augsburg PA000001 case.

These mirror the per-side / per-method calls that the production torsion pipeline
(``morphometry/docker/compute_torsion.py``) performs, and lock the resulting
angles so the measurements refactor can be proven not to change them.

Note the image-side / patient-side inversion: ``side='left'`` (left image half)
corresponds to the patient's right leg, hence the ``_patient_right`` labels.
"""
import nibabel as nib
import pytest

from morphometry.image_io import Image
from morphometry.measurements.femur import calculate_femoral_torsion
from morphometry.tibia import calculate_tibial_torsion
from tests.conftest import assert_golden

pytestmark = pytest.mark.needs_augsburg


@pytest.fixture
def torsion_inputs(augsburg_pa000001):
    """Left/right image halves of the hip/knee/ankle masks plus the x_ratio."""
    hip, knee, ankle = (augsburg_pa000001[k] for k in ("hip", "knee", "ankle"))
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    half_h = hip.array.shape[0] // 2
    half_k = knee.array.shape[0] // 2
    half_a = ankle.array.shape[0] // 2
    return {
        "left_hip": Image.from_nibabel(nib.Nifti1Image(hip.array[:half_h], hip.affine, hip.header)),
        "right_hip": Image.from_nibabel(nib.Nifti1Image(hip.array[half_h:], hip.affine, hip.header)),
        "left_knee": knee.array[:half_k],
        "right_knee": knee.array[half_k:],
        "left_ankle": ankle.array[:half_a],
        "right_ankle": ankle.array[half_a:],
        "x_ratio": x_ratio,
    }


@pytest.mark.parametrize("side,method", [
    ("left", "lee"), ("left", "murphy"), ("right", "lee"), ("right", "murphy"),
])
def test_femoral_torsion(torsion_inputs, side, method):
    hip = torsion_inputs[f"{side}_hip"]
    knee = torsion_inputs[f"{side}_knee"]
    angle = calculate_femoral_torsion(hip, knee, side=side, method=method,
                                      x_ratio=torsion_inputs["x_ratio"])
    assert_golden(f"femoral_torsion_{side}_{method}", angle)


@pytest.mark.parametrize("side", ["left", "right"])
def test_tibial_torsion(torsion_inputs, side):
    knee = torsion_inputs[f"{side}_knee"]
    ankle = torsion_inputs[f"{side}_ankle"]
    angle = calculate_tibial_torsion(knee, ankle, tibia_label_knee=2, tibia_label_ankle=1,
                                     fibula_label=2, side=side)
    assert_golden(f"tibial_torsion_{side}", angle)

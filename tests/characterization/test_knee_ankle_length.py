"""Characterization of knee rotation, ankle PMA, and MRI bone length on Augsburg
PA000001.

Joint-line convergence angle is intentionally not characterized here: it needs a
knee mask with both condyles resolvable and raises "Could not find condyles!" on
this single-side case. It is marked xfail pending suitable data.
"""
import nibabel as nib
import pytest

from morphometry.image_io import Image, Segmentation
from morphometry.measurements import knee as K, ankle as A, whole_leg as W
from tests.conftest import assert_golden

pytestmark = pytest.mark.needs_augsburg


def test_knee_rotation_angle_left(augsburg_pa000001):
    knee = augsburg_pa000001["knee"]
    half = knee.array.shape[0] // 2
    assert_golden("knee_rotation_angle_left",
                  K.calculate_knee_rotation_angle(knee.array[:half], 1, 2, side="left"))


def test_pma_angle_left(augsburg_pa000001):
    ankle = augsburg_pa000001["ankle"]
    half = ankle.array.shape[0] // 2
    assert_golden("pma_angle_left", A.calculate_pma_angle(ankle.array[:half], 1, 2))


def test_bone_length_femur_left(augsburg_pa000001):
    hip, knee = augsburg_pa000001["hip"], augsburg_pa000001["knee"]
    half_h, half_k = hip.array.shape[0] // 2, knee.array.shape[0] // 2
    prox = Image.from_nibabel(nib.Nifti1Image(hip.array[:half_h], hip.affine, hip.header))
    dist = Image.from_nibabel(nib.Nifti1Image(knee.array[:half_k], knee.affine, knee.header))
    assert_golden("bone_length_femur_left",
                  W.calculate_bone_length(prox, dist, 1, 1, tibia=False))


@pytest.mark.xfail(reason="JLCA needs a knee mask with resolvable condyles; not available in PA000001",
                   raises=RuntimeError, strict=False)
def test_joint_line_convergence_angle_left(augsburg_pa000001):
    knee = augsburg_pa000001["knee"]
    half = knee.array.shape[0] // 2
    sub = Segmentation.from_nibabel(nib.Nifti1Image(knee.array[:half], knee.affine, knee.header))
    assert_golden("jlca_left", K.calculate_joint_line_convergence_angle(sub, 1, 2, side="left"))

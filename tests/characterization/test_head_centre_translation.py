"""Translation invariance of the MRI femoral-head centre and what builds on it, on Augsburg PA000001.

Moving the hip within the image -- here by padding a half with empty voxels at the low end and
moving the affine's origin back so every voxel keeps its physical position -- must move the head
centre by exactly the padding, keep its radius, and leave the measurements built on it unchanged.
The former head fit trimmed the slab at ``0.9`` times an absolute voxel index, so none of that held.

Scope: the head centre and radius (both image sides, x / y / z), and, with the knee left in place,
Murphy femoral torsion, femoral anteversion and CCD (x / y). Lee torsion, the knee condylar line and
the alpha angle still depend on the position through slice rotations about the image centre
(``rotate_mask_dorsal_points`` and friends); they are pinned as expected failures until that is
redesigned (task #9).
"""
import nibabel as nib
import numpy as np
import pytest

from morphometry.hip import get_femoral_head_center
from morphometry.image_io import Segmentation, split_image_left_right
from morphometry.measurements.femur import calculate_femoral_torsion, get_femoral_torsion_landmarks
from morphometry.measurements.hip import calculate_alpha_angle, calculate_anteversion, calculate_ccd

pytestmark = pytest.mark.needs_augsburg

IN_PLANE_SHIFTS = [(9, 0, 0), (31, 0, 0), (0, 7, 0)]
ALL_SHIFTS = IN_PLANE_SHIFTS + [(0, 0, 2)]


@pytest.fixture
def halves(augsburg_pa000001):
    """Hip and knee image halves of PA000001 per image side, plus the x_ratio of the pipeline."""
    hip, knee = augsburg_pa000001['hip'], augsburg_pa000001['knee']
    return {
        'hip': dict(zip(('left', 'right'), split_image_left_right(hip))),
        'knee': dict(zip(('left', 'right'), split_image_left_right(knee))),
        'x_ratio': abs(hip.spacing[2]) / 2 * abs(hip.spacing[0]),
    }


def _shifted(image: Segmentation, shift) -> Segmentation:
    """Pad an image at the low end of each axis and move its origin so voxels keep their physical position.

    :param image: A nibabel-backed Segmentation.
    :param shift: Non-negative voxel padding ``(kx, ky, kz)``.
    :return: The padded Segmentation.
    """
    affine = image.affine.copy()
    affine[:3, 3] -= affine[:3, :3] @ np.asarray(shift, dtype=float)
    array = np.pad(image.array, [(k, 0) for k in shift])
    return Segmentation.from_nibabel(nib.Nifti1Image(array.astype(np.float32), affine))


@pytest.mark.parametrize('shift', ALL_SHIFTS)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_head_centre_and_radius_follow_the_shift(halves, side, shift):
    hip = halves['hip'][side]
    kwargs = dict(side=side, segmentation_label=1, x_ratio=halves['x_ratio'], spacing=hip.spacing)
    radius, centre = get_femoral_head_center(hip.array, **kwargs)
    moved_radius, moved_centre = get_femoral_head_center(_shifted(hip, shift).array, **kwargs)
    np.testing.assert_allclose(np.asarray(moved_centre) - centre, shift, atol=1e-6)
    assert moved_radius == pytest.approx(radius, abs=1e-6)


@pytest.mark.parametrize('shift', IN_PLANE_SHIFTS)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_murphy_torsion_is_unchanged(halves, side, shift):
    hip, knee = halves['hip'][side], halves['knee'][side].array
    before = calculate_femoral_torsion(hip, knee, side, 'murphy', x_ratio=halves['x_ratio'])
    after = calculate_femoral_torsion(_shifted(hip, shift), knee, side, 'murphy', x_ratio=halves['x_ratio'])
    assert after == pytest.approx(before, abs=1e-6)


@pytest.mark.parametrize('shift', IN_PLANE_SHIFTS)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_anteversion_is_unchanged(halves, side, shift):
    hip = halves['hip'][side]
    assert calculate_anteversion(_shifted(hip, shift), side) == pytest.approx(calculate_anteversion(hip, side), abs=1e-6)


@pytest.mark.parametrize('shift', IN_PLANE_SHIFTS)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_ccd_is_unchanged(halves, side, shift):
    hip, knee = halves['hip'][side], halves['knee'][side]
    before = calculate_ccd(hip, knee, side, 1, isotropic=False, x_ratio=halves['x_ratio'])
    after = calculate_ccd(_shifted(hip, shift), knee, side, 1, isotropic=False, x_ratio=halves['x_ratio'])
    np.testing.assert_allclose(after[:2], before[:2], atol=1e-4)


# --- still position-dependent, independently of the head fit (task #9) ---------------------------

TASK_9 = 'slice rotations about the image centre make this position-dependent (task #9)'


@pytest.mark.xfail(reason=TASK_9, strict=True)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_lee_torsion_is_unchanged(halves, side):
    hip, knee = halves['hip'][side], halves['knee'][side].array
    before = calculate_femoral_torsion(hip, knee, side, 'lee', x_ratio=halves['x_ratio'])
    for shift in IN_PLANE_SHIFTS:
        after = calculate_femoral_torsion(_shifted(hip, shift), knee, side, 'lee', x_ratio=halves['x_ratio'])
        assert after == pytest.approx(before, abs=1e-6)


@pytest.mark.xfail(reason=TASK_9, strict=True)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_knee_condylar_line_follows_the_shift(halves, side):
    hip, knee = halves['hip'][side], halves['knee'][side]
    before = get_femoral_torsion_landmarks(hip, knee.array, side, 'murphy', x_ratio=halves['x_ratio'])
    for shift in IN_PLANE_SHIFTS:
        after = get_femoral_torsion_landmarks(hip, _shifted(knee, shift).array, side, 'murphy',
                                              x_ratio=halves['x_ratio'])
        for key in ('knee_start', 'knee_end'):
            np.testing.assert_allclose(np.asarray(after[key][:2], dtype=float) - before[key][:2], shift[:2])


@pytest.mark.xfail(reason=TASK_9, strict=True)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_alpha_angle_is_unchanged(halves, side):
    hip = halves['hip'][side]
    before = calculate_alpha_angle(hip.array, side, 1, isotropic=False, x_ratio=halves['x_ratio'])
    for shift in IN_PLANE_SHIFTS:
        after = calculate_alpha_angle(_shifted(hip, shift).array, side, 1, isotropic=False, x_ratio=halves['x_ratio'])
        np.testing.assert_allclose(after, before, atol=1e-6)

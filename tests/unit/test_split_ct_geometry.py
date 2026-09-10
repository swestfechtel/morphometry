"""Unit tests for the geometry of the whole-leg CT hip / knee / ankle split.

``split_ct_image`` crops the whole-leg volume along the superior-inferior axis. The
crops used to inherit the parent affine unchanged, so index 0 of the ankle crop
mapped to the *whole leg's* index 0 -- every landmark taken from a sub-volume and
converted with ``transform_index_to_physical_point`` was off by the crop's start
slice (hundreds of millimetres), which silently corrupted the mechanical axis and
hence the hip-knee-ankle angle and the mechanical axis deviation.
"""
import nibabel as nib
import numpy as np
import pytest

from morphometry.image_io import Segmentation, split_ct_image


def _whole_leg(spacing_z: float = 3.0) -> Segmentation:
    """Build a synthetic whole-leg CT mask with every label the split needs.

    :param spacing_z: The slice thickness, so the affine has a non-unit z scale.
    :return: A Segmentation in the LPI orientation with femur, tibia, fibula, patella and hip.
    """
    array = np.zeros((20, 20, 200), dtype=np.float64)
    array[8:12, 8:12, 20:60] = 7    # hip
    array[8:12, 8:12, 30:120] = 1   # femur
    array[8:12, 8:12, 100:104] = 5  # patella
    array[8:12, 8:12, 120:190] = 2  # tibia
    array[6:8, 8:12, 125:190] = 3   # fibula

    affine = np.diag([-1.0, -1.0, -spacing_z, 1.0])
    affine[:3, 3] = [10.0, 20.0, -30.0]
    return Segmentation.from_nibabel(nib.Nifti1Image(array, affine))


# Where split_ct_image cuts the mask built by _whole_leg: the hip starts at the most
# proximal hip slice (20) and runs 30% of the femur's 100 slices; the knee spans the
# most proximal patella slice (100) to the most proximal fibula slice (125); the ankle
# is the distal 30% of the fibula's 65 slices, ending at its most distal slice (189).
CROP_STARTS = {'hip': 20, 'knee': 100, 'ankle': 170}


@pytest.mark.parametrize('region,start', sorted(CROP_STARTS.items()))
def test_sub_volume_indices_map_to_the_same_physical_points_as_the_parent(region, start):
    """A voxel keeps its physical location whether addressed in the crop or the whole leg."""
    seg = _whole_leg()
    _, crops = split_ct_image(seg, None)
    crop = dict(zip(('hip', 'knee', 'ankle'), crops))[region]

    depth = crop.array.shape[2]
    assert np.array_equal(crop.array, seg.array[:, :, start:start + depth])

    for index in ([9, 9, 0], [9, 9, depth // 2], [9, 9, depth - 1]):
        index = np.array(index)
        assert crop.transform_index_to_physical_point(index) == pytest.approx(
            seg.transform_index_to_physical_point(index + np.array([0, 0, start])))


def test_sub_volume_keeps_the_parent_spacing_and_orientation():
    """Only the affine's translation moves; the direction / spacing block is untouched."""
    seg = _whole_leg(spacing_z=2.5)
    _, crops = split_ct_image(seg, None)

    for crop in crops:
        assert crop.affine[:3, :3] == pytest.approx(seg.affine[:3, :3])
        # the shift is purely along the cropped (superior-inferior) axis
        shift = crop.affine[:3, 3] - seg.affine[:3, 3]
        assert shift[:2] == pytest.approx([0, 0])


def test_mechanical_axis_landmarks_run_proximal_to_distal():
    """Femoral head, knee and ankle centres must be ordered along the leg.

    With the crops inheriting the parent affine the knee and the ankle landed *above*
    the femoral head, which is what made the hip-knee-ankle angle meaningless.
    """
    from morphometry.whole_leg import get_mechanical_axis

    seg = _whole_leg()
    femoral_head_center, ankle_center = get_mechanical_axis(seg, side='left')

    # LPI physical z decreases inferiorly, so the ankle sits below the femoral head
    assert ankle_center[2] < femoral_head_center[2]

"""Translation invariance of the MRI femoral-head centre and what builds on it, on Augsburg PA000001.

Moving the hip within the image -- here by padding a half with empty voxels at the low end and
moving the affine's origin back so every voxel keeps its physical position -- must move the head
centre by exactly the padding, keep its radius, and leave the measurements built on it unchanged.
The former head fit trimmed the slab at ``0.9`` times an absolute voxel index, so none of that held.

Scope: the head centre and radius (both image sides, x / y / z), and, with the knee left in place,
Murphy femoral torsion, femoral anteversion, CCD (x / y), the knee condylar line, the alpha angle
and Lee femoral torsion. **Nothing here is an expected failure any more**, which closes the task #9
family: the knee condylar line followed the shift as of task #5
(``knee.get_knee_reference_line`` finds its endpoints in coordinates instead of reading them off a
resampled rotated section), the alpha angle as of the oblique-reslice fix
(``measurements.hip._reslice_femoral_neck_section`` samples the neck section on a grid anchored to
the head centre instead of slicing a ``scipy.ndimage.rotate(..., reshape=True)`` of the whole
volume), and Lee torsion as of the neck-notch fix (``femur._dorsal_frame_angle`` accumulates the
dorsal frame in coordinates and ``femur._sample_section_in_frame`` samples the notch raster on a
grid anchored to the head centre's cell, where ``contour_femoral_neck`` used to rotate the layer
twice with ``skimage.transform.rotate(..., resize=True)`` and map the notch back through offsets
derived from the rotated array's shape).
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


@pytest.mark.parametrize('side', ['left', 'right'])
def test_lee_torsion_is_unchanged(halves, side):
    """Padding the hip leaves the Lee femoral torsion unchanged.

    A strict xfail until the femoral-neck notch frame was fixed (task #9's last member): the notch
    that sets the neck ring was found by rotating the 2-D layer twice with
    ``skimage.transform.rotate(..., resize=True)`` and mapped back through offsets derived from that
    array's *shape*, so the ring radii -- and with them the fitted neck axis -- depended on where the
    hip sat in the array. Measured over the 100 reader-study halves under four pads the pre-change
    code was exact on 11 of 100, off by a median 1.41 and up to 11.42 degrees; it is now exact on
    100 of 100. ``tests/unit/test_femoral_neck_notch.py`` carries the frame and the window on a
    phantom.
    """
    hip, knee = halves['hip'][side], halves['knee'][side].array
    before = calculate_femoral_torsion(hip, knee, side, 'lee', x_ratio=halves['x_ratio'])
    for shift in IN_PLANE_SHIFTS:
        after = calculate_femoral_torsion(_shifted(hip, shift), knee, side, 'lee', x_ratio=halves['x_ratio'])
        assert after == pytest.approx(before, abs=1e-6)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_knee_condylar_line_follows_the_shift(halves, side):
    """Padding the knee shifts the reported condylar endpoints by exactly the padding.

    A strict xfail until task #5: the endpoints were read off a resampled rotated section and
    mapped back through offsets derived from the rotated array's shape, so they followed the index
    grid rather than the anatomy. They are now mask points found in coordinates.

    Note what the right side exercises: ``_mri_torsion`` measures a right half as the reflection of
    a left one and maps the landmarks back with ``x -> n - 1 - x``, so the padding has to come out
    of the *reflection* as well, where it lands past the anatomy instead of in front of it.
    ``test_condylar_line_invariance.py`` probes the line itself (and a crop) on four reader-study
    halves chosen for having carried the defect; this one keeps it inside the measurement.
    """
    hip, knee = halves['hip'][side], halves['knee'][side]
    before = get_femoral_torsion_landmarks(hip, knee.array, side, 'murphy', x_ratio=halves['x_ratio'])
    for shift in IN_PLANE_SHIFTS:
        after = get_femoral_torsion_landmarks(hip, _shifted(knee, shift).array, side, 'murphy',
                                              x_ratio=halves['x_ratio'])
        for key in ('knee_start', 'knee_end'):
            np.testing.assert_allclose(np.asarray(after[key][:2], dtype=float) - before[key][:2], shift[:2])


@pytest.mark.parametrize('side', ['left', 'right'])
def test_alpha_angle_is_unchanged(halves, side):
    """Padding the hip leaves both alpha angles unchanged.

    A strict xfail until the oblique reslice was fixed: the section was a slice of a
    ``scipy.ndimage.rotate(..., reshape=True)`` of the whole volume, which rotates about the
    array's centre, and the head and neck landmarks were transported through it by painting and
    re-centroiding a 3x3x3 cube. It is now sampled on a grid anchored to the head centre.

    This is the **non-isotropic** path, which has no production caller and on
    1.09375 x 1.09375 x 10 mm voxels is not a rotation of the anatomy at all -- so what it pins is
    invariance, not validity (``calculate_alpha_angle`` now refuses such data when it is given the
    spacing; this call passes a bare array, so the guard stays silent). Measured over the 100
    reader-study halves the pre-change code was exact on 0 of 100 under the worst of these shifts,
    off by a median 37.1 and up to 129.7 degrees; it is now exact on 100 of 100.
    ``test_alpha_angle_invariance.py`` carries the isotropic NaKo probes and the guard.
    """
    hip = halves['hip'][side]
    before = calculate_alpha_angle(hip.array, side, 1, isotropic=False, x_ratio=halves['x_ratio'])
    for shift in IN_PLANE_SHIFTS:
        after = calculate_alpha_angle(_shifted(hip, shift).array, side, 1, isotropic=False, x_ratio=halves['x_ratio'])
        np.testing.assert_allclose(after, before, atol=1e-6)

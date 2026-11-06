"""The Lee reference-line endpoint is floored, not truncated toward zero.

``get_femoral_neck_center_lee`` reports a point ``_NECK_AXIS_REACH`` voxels back along the fitted
neck axis from the head centre. It is a synthetic far point rather than a landmark on bone, so it
routinely lands outside the array, and the rounding therefore has to be ``np.floor`` rather than
``int()``: ``int()`` truncates TOWARD ZERO, rounding a positive argument down and a negative one
up, so it does not commute with adding an integer across zero.

That was the last crop residual of the task #9 work on this function, and it had been mis-attributed
twice -- to ``utils.get_contour``'s ``binary_erosion(border_value=0)`` reporting the array border as
contour. It is not that. Traced stage by stage on the two halves below, the head centre, the
selected layer, the contour (4258 and 5151 voxels), ``r_1`` / ``r_2`` / ``a``, ``center_fn`` (an
int16 centre of mass, which DOES commute here), ``r_3`` and the surviving contour point set are all
identical between the uncropped and cropped arrays, and so is the fitted slope. Only the final
rounding differs: cropping drops ``center[1]`` below the axis reach, the coronal argument falls in
``(-1, 0]``, and ``int()`` rounds it UP to 0 where ``np.floor`` gives -1 and reproduces the
uncropped answer.

These two halves are the whole of it. Over the 100 reader-study halves the coronal argument is
non-negative on all 100 in the production framing -- so the change is zero-movement there, and the
Lee neck centre and proximal axis bearing are bit-identical to the pre-change code on all 100 --
and under a crop to the femur's own bounding box at margin 0 it falls in ``(-1, 0]`` on exactly
``PA000012`` image-right and ``PA000667`` image-left. With the floor in place the neck centre
follows that crop exactly on 100 of 100, where it was 98 of 100 before.
"""
import numpy as np
import pytest

from morphometry.femur import get_femoral_neck_center_lee
from morphometry.hip import get_femoral_head_center
from morphometry.image_io import split_image_left_right

#: The two halves whose coronal argument falls in ``(-1, 0]`` under a margin-0 femur-bbox crop,
#: i.e. the only two on which ``int()`` and ``np.floor`` disagree anywhere in the cohort.
CASES = [('PA000012', 'right'), ('PA000667', 'left')]


def _half(hip, side):
    """The hip half in the left-oriented frame the Lee search is written for.

    :param hip: The whole-hip ``Segmentation``, already LPI and de-outliered.
    :param side: ``'left'`` or ``'right'``; a right half is reversed along x.
    :return: The label array of that half, oriented as a left half.
    """
    halves = dict(zip(('left', 'right'), split_image_left_right(hip)))
    arr = halves[side].array
    return arr if side == 'left' else arr[::-1]


def _neck_centre(arr, x_ratio):
    """Resolve the Lee neck centre on a label array, as the production path does.

    :param arr: A hip-half label array.
    :param x_ratio: The pipelines' slice-thickness factor for this image.
    :return: The reported reference-line endpoint, as an integer index triple.
    """
    r, centre = get_femoral_head_center(arr, side='left', segmentation_label=1,
                                        isotropic=False, x_ratio=x_ratio)
    return np.asarray(get_femoral_neck_center_lee(
        np.where(arr == 1, 1, 0), np.asarray(centre).astype(np.int16), r), dtype=float)


@pytest.mark.needs_augsburg_reader_study
@pytest.mark.parametrize('patient,side', CASES)
def test_endpoint_follows_a_margin_zero_crop(augsburg_reader_study, patient, side):
    """The endpoint must shift by exactly the crop offset, not by the offset plus one.

    Cropping to the femur's own bounding box removes no bone voxel, so a correct landmark follows
    the crop exactly. Pre-change these two halves were off by one in the coronal coordinate.
    """
    hip = augsburg_reader_study(patient)['hip']
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    arr = _half(hip, side)

    base = _neck_centre(arr, x_ratio)

    femur = np.where(arr == 1, 1, 0)
    spans = [np.argwhere(femur.any(axis=tuple(j for j in range(3) if j != k)))[:, 0]
             for k in range(3)]
    low = np.array([s.min() for s in spans])
    high = [s.max() + 1 for s in spans]
    cropped = arr[low[0]:high[0], low[1]:high[1], low[2]:high[2]]

    # the crop is margin 0, so bone sits on the border: assert the probe is the sharp one
    assert femur.sum() == np.count_nonzero(cropped == 1), 'the crop dropped a bone voxel'

    got = _neck_centre(cropped, x_ratio) + low
    np.testing.assert_array_equal(
        got, base,
        err_msg=f'{patient} image-{side}: the endpoint did not follow the crop '
                f'({list(base)} -> {list(got)}); int() truncating toward zero is the usual cause')

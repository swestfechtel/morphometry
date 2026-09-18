"""Unit tests for the Murphy distal landmark (``femur.get_femoral_neck_base``) on a proximal-femur phantom.

Murphy et al. (1987) define the femoral neck axis from the centre of the femoral head to the centre of
the femoral neck base "at the level just above the lesser trochanter". The lesser trochanter (LT) is
found by ``get_trochanter_minor`` as the smaller component of the first sagittal slice, scanning from
medial to lateral, that splits the femur in two. The neck-base point must lie on the first axial slice
above the LT's superior extent -- not on the slice through the LT, whose section includes the trochanter
-- and be that slice's femur centroid as a continuous index, as must the head centre it is paired with (truncating either to an
integer index shifts the short head-to-base axis by up to a voxel).

The phantom is a left image half (medial = +x, posterior = +y, inferior = +z), defined in mm and sampled
on 1 x 1 x 10 mm unless a test says otherwise:
a head, a neck running laterally and inferiorly into a shaft that reaches the bottom of the volume, a
greater-trochanter block, and a posteromedial lesser-trochanter block on one or two slices.
"""
import nibabel as nib
import numpy as np
import pytest

from morphometry.femur import get_femoral_neck_base, get_proximal_reference_line, get_trochanter_minor
from morphometry.hip import get_femoral_head_center
from morphometry.image_io import Segmentation

SPACING = (1.0, 1.0, 10.0)
HEAD_CENTRE_MM = np.array([90.3, 50.6, 25.0])
#: Lesser-trochanter block extents (voxel index in x and y, inclusive).
LT_X = (63, 71)
LT_Y = (74, 82)
#: Superior-inferior extent of the phantom, in mm.
HEIGHT_MM = 160.0


def proximal_femur(lt_slices=None, spacing=SPACING, lt_mm=None) -> np.ndarray:
    """A proximal-femur phantom with the lesser trochanter at a given level.

    The anatomy is defined in millimetres, so the same femur can be sampled at any slice spacing.
    :param lt_slices: Axial slice indices carrying the lesser-trochanter block (at ``spacing``).
    :param spacing: Voxel spacing in mm; in-plane voxels are 1 mm.
    :param lt_mm: Alternatively, the ``(top, bottom)`` superior-inferior range of the block in mm.
    :return: A uint8 mask with the femur labelled 1.
    """
    shape = (130, 110, int(round(HEIGHT_MM / spacing[2])))
    xx, yy, zz = np.indices(shape).astype(float)
    z_mm = zz * spacing[2]
    head = (xx - HEAD_CENTRE_MM[0]) ** 2 + (yy - HEAD_CENTRE_MM[1]) ** 2 + (z_mm - HEAD_CENTRE_MM[2]) ** 2 <= 20 ** 2
    # neck: a cylinder from the head centre laterally and inferiorly into the shaft
    shaft_top = np.array([52.0, 56.0, 75.0])
    axis = (shaft_top - HEAD_CENTRE_MM) / np.linalg.norm(shaft_top - HEAD_CENTRE_MM)
    offset = np.stack([xx, yy, z_mm], -1) - HEAD_CENTRE_MM
    along = offset @ axis
    neck = ((np.linalg.norm(offset - along[..., None] * axis, axis=-1) <= 11)
            & (along >= 0) & (along <= np.linalg.norm(shaft_top - HEAD_CENTRE_MM)))
    shaft = ((xx - 52) ** 2 + (yy - 56) ** 2 <= 13 ** 2) & (z_mm >= 60)
    greater_trochanter = (xx >= 28) & (xx <= 45) & (np.abs(yy - 60) <= 12) & (z_mm >= 40) & (z_mm <= 80)
    in_lt_level = np.isin(zz, lt_slices) if lt_mm is None else (z_mm >= lt_mm[0]) & (z_mm <= lt_mm[1])
    lesser_trochanter = (xx >= LT_X[0]) & (xx <= LT_X[1]) & (yy >= LT_Y[0]) & (yy <= LT_Y[1]) & in_lt_level
    return (head | neck | shaft | greater_trochanter | lesser_trochanter).astype(np.uint8)


def _image(mask: np.ndarray, spacing=SPACING) -> Segmentation:
    """Wrap a phantom in a Segmentation with the given spacing (LPI index axes).

    :param mask: The phantom mask.
    :param spacing: Voxel spacing in mm.
    :return: The Segmentation.
    """
    return Segmentation.from_nibabel(nib.Nifti1Image(mask, np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])))


@pytest.mark.parametrize('lt_slices', [(9,), (8, 9)], ids=['LT on one slice', 'LT on two slices'])
def test_neck_base_is_the_centroid_of_the_slice_just_above_the_lesser_trochanter(lt_slices):
    mask = proximal_femur(lt_slices)
    head = HEAD_CENTRE_MM / SPACING
    base = get_femoral_neck_base(_image(mask), head)

    level = min(lt_slices) - 1
    assert base[2] == level
    expected = np.argwhere(mask[:, :, level] == 1).mean(axis=0)
    np.testing.assert_allclose(base[:2], expected, atol=1e-9)  # continuous, not truncated


def test_trochanter_minor_reports_its_own_anteroposterior_coordinate():
    """The returned point is ``(x, y, layer)``; ``y`` was formerly a copy of ``x``."""
    lesser_trochanter = get_trochanter_minor(_image(proximal_femur((9,))), HEAD_CENTRE_MM / SPACING)
    assert LT_X[0] - 15 <= lesser_trochanter[0] <= LT_X[1]
    assert LT_Y[0] - 20 <= lesser_trochanter[1] <= LT_Y[1]
    assert lesser_trochanter[1] != lesser_trochanter[0]
    assert lesser_trochanter[2] == 9


def test_murphy_reference_line_keeps_head_and_base_continuous():
    """The Murphy line starts at the unrounded head centre and ends at the unrounded neck-base centroid."""
    image = _image(proximal_femur((9,)))
    x_ratio = SPACING[2] / SPACING[0]
    start, end, _ = get_proximal_reference_line(image, side='left', method='murphy', x_ratio=x_ratio)
    _, head = get_femoral_head_center(image.array, side='left', x_ratio=x_ratio, spacing=image.spacing)
    np.testing.assert_allclose(start, head, atol=1e-9)
    np.testing.assert_allclose(end[:2], np.argwhere(image.array[:, :, 8] == 1).mean(axis=0), atol=1e-9)


def test_a_trochanter_minor_at_head_level_is_rejected():
    """A neck base less than 10 mm below the head centre is a mis-detected trochanter, not a femoral neck axis.

    On such a leg the slice above the "trochanter" lies level with the head centre, and the resulting
    axis is a few millimetres long; it must fail (and be recorded as NaN by the batch scripts) rather
    than return an arbitrary angle.
    """
    image = _image(proximal_femur((9,)))
    head_at_the_base_level = np.array([110.0, 50.6, 8.0])  # plausible 40-75 mm from the trochanter, but level with it
    with pytest.raises(RuntimeError, match='too close to'):
        get_femoral_neck_base(image, head_at_the_base_level)


@pytest.mark.parametrize('slice_mm', [10.0, 3.0, 1.0])
def test_neck_base_level_is_the_first_slice_above_the_lesser_trochanter_at_any_slice_spacing(slice_mm):
    """The level is the first slice above the trochanter's top, however thick the slices are.

    The same femur, with the trochanter's top at 90 mm, is sampled at 10, 3 and 1 mm slices; the level
    must be one slice above the top in each, i.e. as close above the trochanter as the acquisition
    resolves (Murphy's "just above"), not a fixed distance.
    """
    spacing = (1.0, 1.0, slice_mm)
    lt_top_mm = 90.0
    image = _image(proximal_femur(spacing=spacing, lt_mm=(lt_top_mm, lt_top_mm + 10.0)), spacing)
    base = get_femoral_neck_base(image, HEAD_CENTRE_MM / np.array(spacing))

    assert base[2] == lt_top_mm / slice_mm - 1
    np.testing.assert_allclose(base[:2], np.argwhere(image.array[:, :, int(base[2])] == 1).mean(axis=0), atol=1e-9)

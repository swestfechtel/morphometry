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

from morphometry import constants as C
from morphometry.femur import (_landmark_failure_flag, get_femoral_neck_base, get_murphy_neck_qc,
                               get_proximal_reference_line, get_trochanter_minor, murphy_neck_base_qc)
from morphometry.hip import get_femoral_head_center
from morphometry.image_io import Segmentation, mirror_image_left_right

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


def test_anteversion_derives_the_slice_aspect_from_the_image():
    """``calculate_anteversion`` passes the image's own slice aspect down, not a hard-coded 1.

    ``x_ratio`` bounds the femoral-head slab walk; hard-coding 1 told that bound the slices were one
    in-plane voxel apart, so on anisotropic data it was effectively no bound at all.
    """
    from unittest import mock

    from morphometry.measurements import hip as hip_measurements

    image = _image(proximal_femur((9,)))
    with mock.patch.object(hip_measurements, 'get_proximal_reference_line',
                           wraps=hip_measurements.get_proximal_reference_line) as spy:
        hip_measurements.calculate_anteversion(image, side='left')
    assert spy.call_args.kwargs['x_ratio'] == pytest.approx(SPACING[2] / SPACING[0])


# --- QC of a degenerate neck (flagged, not repaired) ---------------------------------------

def _murphy_landmarks(image: Segmentation):
    """The Murphy reference line of a phantom, as the measurements resolve it.

    :param image: The phantom Segmentation.
    :return: ``(head_centre, neck_base, head_radius)``.
    """
    start, end, r_fh = get_proximal_reference_line(image, side='left', method='murphy',
                                                   x_ratio=SPACING[2] / SPACING[0])
    return start, end, r_fh


def test_a_sound_neck_is_not_flagged():
    """The phantom's neck passes every gate, and the QC reports its geometry without touching it."""
    image = _image(proximal_femur((9,)))
    head, base, r_fh = _murphy_landmarks(image)

    qc = murphy_neck_base_qc(image, head, base, head_radius=r_fh)
    assert qc['flags'] == ''
    assert qc['base_in_bone'] and qc['base_slice_components'] == 1
    assert qc['head_clearance_mm'] > 0           # the base level is below the head sphere
    assert qc['axis_length_mm'] == pytest.approx(float(np.hypot(*(base[:2] - head[:2]))))
    # the landmark itself is untouched: still the centroid of the slice above the trochanter
    np.testing.assert_allclose(base[:2], np.argwhere(image.array[:, :, 8] == 1).mean(axis=0), atol=1e-9)


def test_a_short_in_plane_axis_is_flagged_at_the_angular_tolerance():
    """``murphy_axis_short`` fires exactly below ``1 / tan(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG)`` voxels.

    The torsion is the in-plane direction of the head-to-base axis, so the gate is the axis
    length at which one in-plane voxel of landmark error tilts that direction by the tolerance.
    """
    image = _image(proximal_femur((9,)))
    head, _, _ = _murphy_landmarks(image)
    limit = 1.0 / np.tan(np.radians(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG))

    def flags(voxels: float) -> str:
        base = np.array([head[0] - voxels, head[1], 8.0])  # in-plane offset only, in voxels
        return murphy_neck_base_qc(image, head, base)['flags']

    assert f'murphy_axis_short:{limit * 0.99:.2f}' in flags(limit * 0.99)
    assert 'murphy_axis_short' not in flags(limit * 1.01)
    # the reported sensitivity is the angle one voxel of error would cost
    short = murphy_neck_base_qc(image, head, np.array([head[0] - 6.0, head[1], 8.0]))
    assert short['angle_per_voxel_deg'] == pytest.approx(np.degrees(np.arctan(1 / 6.0)))


def test_a_base_level_inside_the_fitted_head_sphere_is_flagged():
    """``murphy_base_in_head``: no femoral neck base lies inside the head.

    Exercised with an inflated head radius, which is how this happens in practice -- a head fit
    that ran down the neck into the shaft (the CT radius of 66.9 mm instead of 28.3 mm that the
    2026-09-10 goldens were re-captured for) makes the sphere swallow the neck base.
    """
    image = _image(proximal_femur((9,)))
    head, base, r_fh = _murphy_landmarks(image)
    distance_mm = float(np.linalg.norm((base - head) * np.array(SPACING)))

    qc = murphy_neck_base_qc(image, head, base, head_radius=distance_mm + 5.0)
    assert qc['flags'] == 'murphy_base_in_head:5.00'
    assert qc['head_clearance_mm'] == pytest.approx(-5.0)
    # without a radius the check is skipped rather than guessed
    assert murphy_neck_base_qc(image, head, base)['flags'] == ''
    assert murphy_neck_base_qc(image, head, base)['head_clearance_mm'] is None


def test_a_sideways_neck_is_not_mistaken_for_one_inside_the_head():
    """The sphere test is the 3-D distance, not the superior-inferior drop.

    A horizontal neck -- coxa valga, a large in-plane offset and a small drop -- has its base well
    outside the head sphere even when the drop alone is less than the radius, and must not be
    flagged for it. This is the condition ``process_nako_large.py`` runs over 30,407 cases.
    """
    image = _image(proximal_femur((9,)))
    head = HEAD_CENTRE_MM / np.array(SPACING)
    base = np.array([head[0] - 40.0, head[1], head[2] + 1.0])   # 40 mm sideways, 10 mm down
    radius_mm = 20.0

    drop_mm = (base[2] - head[2]) * SPACING[2]
    distance_mm = float(np.linalg.norm((base - head) * np.array(SPACING)))
    assert drop_mm < radius_mm < distance_mm    # flagged by the drop alone, sound in 3-D

    qc = murphy_neck_base_qc(image, head, base, head_radius=radius_mm)
    assert 'murphy_base_in_head' not in qc['flags']
    assert qc['head_clearance_mm'] == pytest.approx(distance_mm - radius_mm)


def test_a_base_centroid_off_the_bone_is_flagged():
    """``murphy_base_outside_bone`` / ``murphy_base_slice_multipart``: the centroid is not a neck centre.

    A second, disconnected piece of "femur" on the base slice (a mislabelled fragment) drags the
    slice centroid into the gap between the pieces, where there is no bone at all.
    """
    mask = proximal_femur((9,))
    mask[10:30, 10:30, 8] = 1  # a detached block on the neck-base slice
    image = _image(mask)
    head = HEAD_CENTRE_MM / np.array(SPACING)
    base = np.array([*np.argwhere(mask[:, :, 8] == 1).mean(axis=0), 8.0])

    qc = murphy_neck_base_qc(image, head, base)
    assert not qc['base_in_bone'] and qc['base_slice_components'] == 2
    assert qc['flags'] == (f"murphy_base_outside_bone:{qc['base_distance_to_bone_mm']:.2f}"
                           ';murphy_base_slice_multipart:2')
    assert qc['base_distance_to_bone_mm'] > 1.0


def test_an_unresolved_in_plane_axis_raises():
    """A neck base within one in-plane voxel of the head centre leaves no angle to measure.

    The only new *raise*: the axis has no in-plane direction at the sampling resolution, so there
    is nothing to flag and report. Everything short of that is flagged and still measured.
    """
    image = _image(proximal_femur((9,)))
    base = np.argwhere(image.array[:, :, 8] == 1).mean(axis=0)
    head_above_the_base = np.array([base[0], base[1], 0.0])  # same column, well above the base level

    with pytest.raises(RuntimeError, match='one in-plane voxel'):
        get_femoral_neck_base(image, head_above_the_base)


def test_the_flags_are_mirror_invariant():
    """A right image half is measured as the reflection of a left one, so its flags must agree.

    ``mirror_image_left_right`` relabels the indices of the phantom (a left image half) into a
    right one; every quantity the QC reports is a mirror-invariant scalar, so both must read the
    same flag string -- including the empty one.
    """
    mask = proximal_femur((9,))
    left = _image(mask)
    right = mirror_image_left_right(left)
    x_ratio = SPACING[2] / SPACING[0]

    assert get_murphy_neck_qc(left, side='left', x_ratio=x_ratio) == ''
    assert get_murphy_neck_qc(right, side='right', x_ratio=x_ratio) == ''

    mask[10:30, 10:30, 8] = 1  # make the neck base degenerate on both
    left, right = _image(mask), mirror_image_left_right(_image(mask))
    flagged = get_murphy_neck_qc(left, side='left', x_ratio=x_ratio)
    assert 'murphy_base_slice_multipart:2' in flagged
    assert get_murphy_neck_qc(right, side='right', x_ratio=x_ratio) == flagged


def test_the_qc_wrapper_reports_a_failed_landmark_instead_of_raising():
    """A hip whose trochanter search fails gets ``murphy_landmark_failed``, not an exception.

    The flag column must be filled for every row of a cohort run, including the rows whose
    Murphy angle is NaN because the landmark search itself gave up.
    """
    empty = _image(np.zeros((130, 110, 16), dtype=np.uint8))
    assert get_murphy_neck_qc(empty, side='left').startswith('murphy_landmark_failed:')


def test_a_one_voxel_speck_does_not_count_as_a_second_piece():
    """The multipart count ignores specks, the way ``process_nmdid.py``'s bone flags do.

    A single stray voxel diagonally off the cortex moves no centroid worth mentioning (it is the
    one base slice in the Augsburg cohort that is not simply connected, PA000308 image-right), so
    only pieces worth a tenth of the largest count.
    """
    mask = proximal_femur((9,))
    speck = mask.copy()
    speck[20, 20, 8] = 1                       # one voxel, far from the 881-voxel neck section
    head = HEAD_CENTRE_MM / np.array(SPACING)
    base = np.array([*np.argwhere(mask[:, :, 8] == 1).mean(axis=0), 8.0])
    assert murphy_neck_base_qc(_image(speck), head, base)['base_slice_components'] == 1

    lump = mask.copy()
    lump[18:28, 18:28, 8] = 1                  # 100 voxels, over a tenth of the section
    assert murphy_neck_base_qc(_image(lump), head, base)['base_slice_components'] == 2


def test_the_qc_refuses_in_plane_anisotropic_spacing():
    """The axis length is in in-plane voxels, so unequal x/y spacing is refused, not sheared.

    The same precondition the angle composition itself now checks
    (``geometry.validate_in_plane_isotropy``, task #10): on such a dataset the torsion is not
    measurable, so neither is the quality of its neck axis. The wrapper turns it into a flag.
    """
    anisotropic = Segmentation.from_nibabel(
        nib.Nifti1Image(proximal_femur((9,)), np.diag([-1.0, -2.5, 10.0, 1.0])))
    head, base = HEAD_CENTRE_MM / np.array(SPACING), np.array([45.0, 58.0, 8.0])
    with pytest.raises(ValueError, match='anisotropic'):
        murphy_neck_base_qc(anisotropic, head, base)
    assert get_murphy_neck_qc(anisotropic, side='left') == 'murphy_landmark_failed:anisotropic_in_plane'


#: The exact messages the Murphy landmark search raises, with the code each must map to.
_FAILURE_MESSAGES = [
    ('Could not find a plausible trochanter minor: no two components found.', 'trochanter_no_components'),
    ('Could not find a plausible trochanter minor: components too small; isotropic=True.',
     'trochanter_components_too_small'),
    ('Could not find a plausible trochanter minor: no plausible candidates found. Distances: 1.0, 2.0',
     'trochanter_implausible'),
    ('Could not find the femoral shaft', 'shaft_not_found'),
    ('Trochanter minor found too close to (or above) the femoral head centre; no neck base below the head.',
     'base_at_head_level'),
    ('The femoral head centre and the neck base are less than one in-plane voxel apart in the axial '
     'plane; the neck axis has no direction.', 'axis_unresolved'),
    ('The neck-base level lies above the image.', 'level_above_image'),
    ('No femur on the neck-base slice above the trochanter minor.', 'no_femur_on_level'),
]


@pytest.mark.parametrize('message, expected', _FAILURE_MESSAGES)
def test_a_failed_landmark_reports_which_step_gave_up(message, expected):
    """``murphy_landmark_failed:<reason>`` names the step, since every cause is a RuntimeError.

    This is the flag with by far the highest rate (241 of the 2128 Augsburg / NaKo image halves
    have no Murphy landmark at all), so the bare exception type would waste the column. The
    messages here are copied from the raise sites; a code carries no ``,`` or ``;``, so it is safe
    in the CSV columns.
    """
    assert _landmark_failure_flag(RuntimeError(message)) == f'murphy_landmark_failed:{expected}'
    assert ',' not in expected and ';' not in expected


def test_the_wrapper_flags_a_hip_the_landmark_search_cannot_handle():
    """End to end: a femur with no lesser trochanter at all comes back as a flag, not an exception."""
    flag = get_murphy_neck_qc(_image(proximal_femur(())), side='left')
    assert flag.startswith('murphy_landmark_failed:')


def test_an_unrecognised_failure_still_reports_its_type():
    """A new failure mode falls back to the exception type rather than being swallowed."""
    assert _landmark_failure_flag(KeyError('something new')) == 'murphy_landmark_failed:KeyError'


def test_the_qc_refuses_a_base_slice_without_femur():
    """``neck_base`` cannot be the centroid of an empty slice; that is a precondition, not a flag.

    ``get_femoral_neck_base`` rejects it before returning, so this only guards a direct call --
    and it keeps the flag value from rendering as an infinite distance.
    """
    image = _image(proximal_femur((9,)))
    head = HEAD_CENTRE_MM / np.array(SPACING)
    empty_level = 0.0            # above the head: the phantom's topmost slice is empty
    assert not image.array[:, :, int(empty_level)].any()
    with pytest.raises(ValueError, match='holds no femur'):
        murphy_neck_base_qc(image, head, np.array([45.0, 58.0, empty_level]))

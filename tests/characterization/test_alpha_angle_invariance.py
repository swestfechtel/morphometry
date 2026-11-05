"""Index-frame invariance of the alpha angle (``measurements.hip.calculate_alpha_angle``) on real data.

Padding or cropping a mask changes voxel indices while leaving every physical position untouched,
so no measurement may move. The alpha angle did: it brought the femoral neck axis into a fixed-z
section by rotating the whole volume with ``scipy.ndimage.rotate(..., reshape=True, order=0)``,
which rotates about the *input array's centre* and sizes the output to fit, and it transported the
head and neck landmarks through that rotation by painting a 3x3x3 cube, rotating it, re-centroiding
and rounding to ``int16``.

**Where the defect was, measured over the 120 NaKo sample halves before the change.** The upstream
was clean -- the head radius and the rotation angle were exact on 120 of 120 under a (3, 0, 2) pad
-- and the cube transport was the smaller half of the problem (81 and 84 of 120 exact). What moved
was the resampled section itself, on 118 of 120, and with it the contour (13 of 120 exact) and the
contour's shape relative to the head centre (0 of 120). End to end the angle was exact on **0 of
120** under a (3, 0, 2) pad, **0 of 120** under a (1, 1, 1) pad, **0 of 120** under the (6, 4, 3)
high-end pad and 1 of 120 under a crop to the femur's bounding box, off by a median 0.75 / 0.95 /
0.80 / 0.90 degrees and up to 7.32 / 5.88 / 4.20 / 4.74. It is now exact on 120 of 120 under the
first three and under an in-plane bounding-box crop.

**The one probe that is not 120 of 120 is the full bounding-box crop, and it is attributed.** It
is exact on 12 of 120 -- precisely the 12 halves on which the crop leaves this measurement's own
landmarks unchanged, where the angle is bit-identical. A crop that drops empty *slices* moves the
landmarks themselves, upstream in ``get_femoral_head_center``; see
:func:`test_the_section_follows_a_full_bounding_box_crop` for the full attribution. So the
property this module pins is invariance under **translation**, plus exactness **given the
landmarks** under any crop.

**How strong the invariance is.** It is **bitwise** on everything measured -- 0.000e+00 on all 120
NaKo halves under every probe, and on all 100 Augsburg halves -- so these assertions are not
merely within a tolerance. But unlike task #5's condylar divider, which translated an *integer*
index and was therefore provably exact, this translates a float: the head centre moves up to
5.19e-11 voxels beyond the pad, and the margin is that the closest any sampled coordinate comes to
an ``order=0`` rounding boundary over the cohort is 1.248e-06, some 2.4e4 times that noise. Over
30,407 cases that is roughly 1e3 of headroom -- probabilistically safe, not provable. The
assertions therefore carry a 1e-9 degree tolerance rather than claiming bit-identity, although
bit-identity is what they measure. They are a guard against a resampling step being reintroduced,
not an independent derivation; the same caveat ``test_condylar_line_invariance.py`` carries. They
do fail against the pre-change code, which is the only reason they are worth having.

**Invariance is not validity.** The whole construction is still carried out on voxel *indices*: the
rotation mixes index axes 0 and 2 and the contour is then read in the (rotated x, y) plane, which
is a rotation of the anatomy only on isotropic voxels. The last tests here pin the guard that
refuses anisotropic data outright rather than returning an invariantly sheared number.
"""
import nibabel as nib
import numpy as np
import pytest

from morphometry.hip import get_femoral_head_center, get_femoral_neck_center
from morphometry.image_io import Segmentation, split_image_left_right
from morphometry.measurements.hip import _reslice_femoral_neck_section, calculate_alpha_angle
from morphometry.utils import calculate_angle_between_vectors

pytestmark = pytest.mark.needs_nako

#: Pads applied at the low end of every axis. Several, because a single pad can be survived by
#: accident, and the slice axis is exercised as well as the two in-plane ones.
PADS = [(1, 1, 1), (3, 0, 2), (9, 7, 0), (0, 4, 3)]

#: A pad applied at the HIGH end instead. It leaves every anatomy index untouched, so the correct
#: answer is that nothing moves at all -- which isolates dependence on the array's *shape*, the
#: actual defect class (``rotate(reshape=True)`` sized its output from the input's bounds).
#: Pre-change, this moved the angle on 120 of 120 NaKo halves, by a median 0.80 and up to 4.20
#: degrees.
HIGH_END_PAD = (6, 4, 3)

#: Degrees. The construction is float arithmetic, so exact agreement is measured rather than
#: guaranteed; this is a tolerance far below anything anatomically meaningful and far above the
#: float noise (~1e-12 of rotation angle).
TOL = 1e-9


def _half(nako_sample, side: str) -> np.ndarray:
    """One image half of the NaKo hip segmentation, as a bare array.

    A bare array on purpose: it carries no spacing, so the isotropy guard stays silent and these
    probes measure the reslice rather than the guard (NaKo is 1 mm isotropic either way).

    :param nako_sample: The whole NaKo Segmentation (both image sides).
    :param side: 'left' or 'right' image side.
    :return: That half's label array.
    """
    return split_image_left_right(nako_sample)[0 if side == 'left' else 1].array


def _angles(mask: np.ndarray, side: str) -> np.ndarray:
    """The anterior and posterior alpha angle of one image half.

    :param mask: The half's label array (femur = 1).
    :param side: 'left' or 'right' image side.
    :return: ``[anterior, posterior]`` in degrees.
    """
    return np.asarray(calculate_alpha_angle(mask, side, 1, isotropic=True), dtype=float)


@pytest.mark.parametrize('pad', PADS)
@pytest.mark.parametrize('side', ['left', 'right'])
def test_angle_follows_a_pad(nako_sample, side, pad):
    """Padding the mask at the low end of every axis leaves both angles unchanged.

    The anatomy has not moved, only its indices have. Pre-change this failed on every one of the
    120 cohort halves under each of these pads.
    """
    mask = _half(nako_sample, side)
    before = _angles(mask, side)
    after = _angles(np.pad(mask, [(k, 0) for k in pad]), side)
    np.testing.assert_allclose(after, before, atol=TOL,
                               err_msg=f'image-{side} moved under a pad of {pad}')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_angle_ignores_a_pad_at_the_high_end(nako_sample, side):
    """Growing the array past the anatomy must not move the angle at all.

    The sharper of the two translation probes: no anatomy index changes, so any movement is pure
    dependence on the array's shape. It also exercises the sampler's boundary mode -- scipy's
    ``mode='constant'`` returns ``cval`` for any coordinate outside ``[0, n - 1]``, dropping the
    outer half voxel of every axis, so a section whose plane passes through that band read
    background until the array was grown. On the Augsburg reader-study hips (12 slices) that cost
    one half 14.98 degrees under this exact pad, which is why the sampler uses ``'grid-constant'``.
    """
    mask = _half(nako_sample, side)
    before = _angles(mask, side)
    after = _angles(np.pad(mask, [(0, k) for k in HIGH_END_PAD]), side)
    np.testing.assert_allclose(after, before, atol=TOL,
                               err_msg=f'image-{side} moved when the array was grown past the anatomy')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_angle_follows_an_in_plane_crop_to_the_bounding_box(nako_sample, side):
    """Cropping the empty in-plane margin away leaves both angles unchanged.

    The operation the API actually performs (``api/ingest/dicom.py`` crops at detected gaps and
    changepoints through ``image_io.crop_image``), and harder than a pad because it also *shrinks*
    the array and puts the anatomy hard against the border -- which is where a boundary mode that
    drops half a voxel bites.

    The crop is in the two in-plane axes only; see
    :func:`test_the_section_follows_a_full_bounding_box_crop` for why the slice axis is probed
    separately.
    """
    mask = _half(nako_sample, side)
    low = [int(axis.min()) for axis in np.nonzero(mask == 1)]
    high = [int(axis.max()) + 1 for axis in np.nonzero(mask == 1)]
    before = _angles(mask, side)
    after = _angles(mask[low[0]:high[0], low[1]:high[1], :], side)
    np.testing.assert_allclose(after, before, atol=TOL,
                               err_msg=f'image-{side} moved under an in-plane bounding-box crop')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_the_section_follows_a_full_bounding_box_crop(nako_sample, side):
    """The reslice follows a crop in **all three** axes exactly, landmarks included.

    The measurement itself is not probed that way, and that is a finding rather than an omission:
    a crop that drops empty *slices* moves the femoral head centre, upstream of anything this
    change touches. ``get_femoral_head_center`` starts at the topmost slice holding a contour point
    and walks down while that slice's voxel count is more than two standard deviations from the
    mean *over all slices of the array*, so removing empty slices changes the mean, the z-scores
    and sometimes the chosen layer. Each NaKo half is 264 slices with the femur in 90-142 of them,
    so a femur-bbox crop trims 122-174 empty slices and reduces that vector from mostly zeros to
    all-nonzero; padding, by contrast, adds zeros to a vector already dominated by them, which is
    why every pad probe is exact.

    **The attribution is clean, which is why the claim elsewhere is "invariant under translation,
    and exact given its landmarks".** Over the 120 halves under the 3-axis crop, the four inputs
    the reslice consumes -- head centre, neck centre, rotation angle, head radius -- are unchanged
    on exactly 12, and on those 12 the alpha angle is bit-identical (12 of 12 at 0.000e+00); on the
    108 where a landmark moved it is exact on **none** (median 0.63, max 4.51 degrees). A z-only
    crop reproduces the 3-axis crop on every statistic to the digit, and an in-plane-only crop
    leaves all four inputs clean to ~1e-11 and the angle exact on 120 of 120. The landmark
    movement itself is a median 0.265 / max 1.960 voxels of head centre, 0.086 / 0.619 of radius
    and 0.278 / 1.201 degrees of rotation angle -- those, not the angle's 0.523 degrees, are the
    figures that belong to the slab walk.

    So this test hands the reslice the landmarks the crop should have given it and asserts that the
    section and both cells are then reproduced **exactly** -- which is the part of the chain this
    change owns, measured on real anatomy rather than on the phantom of
    ``tests/unit/test_alpha_angle_reslice.py``.
    """
    mask = _half(nako_sample, side)
    if side == 'right':  # as the measurement does: a right half is measured reversed
        mask = mask[::-1]
    low = np.array([int(axis.min()) for axis in np.nonzero(mask == 1)])
    high = np.array([int(axis.max()) + 1 for axis in np.nonzero(mask == 1)])

    radius, centre = get_femoral_head_center(mask, side='left', segmentation_label=1, isotropic=True)
    _, neck = get_femoral_neck_center(mask, (radius, centre), side='left', segmentation_label=1)
    angle = float(calculate_angle_between_vectors(np.asarray(neck, dtype=float)
                                                  - np.asarray(centre, dtype=float),
                                                  np.array([-1.0, 0.0, 0.0])))

    femur = np.where(mask == 1, 1, 0)
    before = _reslice_femoral_neck_section(femur, centre, neck, angle)
    cropped = femur[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    after = _reslice_femoral_neck_section(cropped, np.asarray(centre, dtype=float) - low,
                                          np.asarray(neck, dtype=float) - low, angle)

    assert np.array_equal(after[0], before[0]), 'the section moved under a bounding-box crop'
    assert np.array_equal(after[1], before[1]), 'the head cell moved under a bounding-box crop'
    np.testing.assert_allclose(after[2], before[2], atol=1e-12)


# --------------------------------------------------------------- the isotropy precondition

def test_an_isotropic_image_is_accepted_and_agrees_with_the_bare_array(nako_sample):
    """Passing the NaKo half as an ``Image`` changes nothing: the guard is a no-op on 1 mm voxels.

    This is what the production call sites now do (``scripts/process_nako.py`` and
    ``process_nako_large.py`` dropped their ``.array``), so it has to be bit-identical to the bare
    array they passed before.
    """
    half = split_image_left_right(nako_sample)[0]
    from_image = np.asarray(calculate_alpha_angle(half, 'left', 1, isotropic=True), dtype=float)
    from_array = np.asarray(calculate_alpha_angle(half.array, 'left', 1, isotropic=True), dtype=float)
    assert (from_image == from_array).all()


def _with_spacing(mask: np.ndarray, zooms) -> Segmentation:
    """A nibabel-backed Segmentation holding ``mask`` with the given voxel zooms.

    :param mask: The label array.
    :param zooms: The three voxel edge lengths in mm.
    :return: The Segmentation.
    """
    affine = np.diag([-zooms[0], -zooms[1], -zooms[2], 1.0])
    return Segmentation.from_nibabel(nib.Nifti1Image(mask.astype(np.float32), affine))


def test_an_anisotropic_image_is_refused(nako_sample):
    """An index-space rotation shears anisotropic voxels, so the measurement refuses to answer.

    ``ValueError`` specifically: the batch scripts and ``compute_torsion.py`` catch
    ``(RuntimeError, AssertionError, ValueError)`` per measurement and write a NaN, so one
    unsuitable dataset does not abort a cohort run.
    """
    mask = _half(nako_sample, 'left')
    with pytest.raises(ValueError, match='anisotropic'):
        calculate_alpha_angle(_with_spacing(mask, (1.0, 1.0, 3.0)), 'left', 1, isotropic=True)


def test_an_explicit_spacing_is_checked_and_wins_over_the_images_own(nako_sample):
    """``spacing=`` overrides the image's, in both directions.

    An anisotropic spacing passed against an isotropic image is refused, and an isotropic spacing
    passed against an anisotropic image is accepted and gives the image's own answer -- which is
    the escape hatch for a caller that knows its geometry better than its header does.
    """
    mask = _half(nako_sample, 'left')
    isotropic_image = split_image_left_right(nako_sample)[0]
    with pytest.raises(ValueError, match='anisotropic'):
        calculate_alpha_angle(isotropic_image, 'left', 1, isotropic=True, spacing=(1.0, 1.0, 3.0))

    expected = _angles(mask, 'left')
    got = np.asarray(calculate_alpha_angle(_with_spacing(mask, (1.0, 1.0, 3.0)), 'left', 1,
                                            isotropic=True, spacing=(1.0, 1.0, 1.0)), dtype=float)
    np.testing.assert_allclose(got, expected, atol=TOL)


def test_a_bare_array_is_not_checked_and_not_warned_about(nako_sample, recwarn):
    """An unknown spacing stays silent -- no exception and no warning.

    Task #10's deliberate choice, for the same reason: the alpha angle's production callers used to
    hand it a bare array over 30,407 isotropic NaKo cases, so a warning would fire on every one of
    them and be tuned out. The precondition rests on the caller, whose fix is to pass ``spacing=``.
    """
    mask = _half(nako_sample, 'left')
    _angles(mask, 'left')
    spacing_warnings = [w for w in recwarn
                        if 'spacing' in str(w.message).lower() or 'isotrop' in str(w.message).lower()]
    assert spacing_warnings == [], f'the unknown spacing was warned about: {spacing_warnings}'

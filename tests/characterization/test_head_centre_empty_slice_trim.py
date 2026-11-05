"""The femoral head centre must not depend on the array's empty-slice count, on real data.

``get_femoral_head_center`` starts its slab at the topmost slice holding a femur voxel. Until
2026-11-05 it then z-scored the per-slice voxel counts over the array's **full** z extent -- one
zero per empty slice -- and advanced that start while the first slice's ``|z| > 2``. The mean and
standard deviation of that vector, and hence every z-score, were therefore functions of how many
empty slices surrounded the anatomy, so the head centre moved when the array was cropped or padded
in z although not one labelled voxel changed. The guard has been removed (see
``get_femoral_head_center``); this module pins the property it broke.

In the **as-given** frames the guard fires on neither cohort (first-slice ``|z|`` at most 0.847 on
NaKo and 1.325 on Augsburg, against a threshold of 2), which is why the change is bit-identical on
all 220 halves. **Cropped, NaKo fires**: its ``|z|`` rises to a median 2.466 and the guard fires on
108 of 120 halves, so the NaKo cases below exercise the firing path and not merely the field-of-view
coupling. Augsburg's rises only to a median 0.904 and fires on 2 of 100 -- at 10 mm the apex slice
is already half the mean slice area, so the stack is too variable for the first slice to read as an
outlier. That is the whole reason the two cohorts behave so differently here.

The probe is a crop to the occupied z slices -- it discards only all-zero slices, so it removes no
labelled voxel and a correct landmark **must** reproduce the uncropped centre, to within float
noise, once the crop offset is added back. (Not bitwise; see "Why the tolerance is not zero" below.
That word belongs to the pre-change versus post-change comparison on a fixed array, which is exact
on all 220 halves.) It is not a hypothetical frame either: ``image_io.crop_image`` is how
the API's ``_detect_slabs`` / ``_split_volume`` cut an ingested series, so the same hip re-ingested
with a slab boundary a slice over went through this.

The cases are chosen for carrying the defect, as in ``test_femoral_head_equivariance.py``:

* **NaKo** (isotropic path, 1 mm, so a voxel is a millimetre). The strongest cohort: the pre-change
  code failed this probe at the 1e-9 tolerance asserted below on **108 of the 120** sample halves,
  by a median 0.265 and up to 1.960 mm of centre (radius up to 0.619 mm); not one of the 120 was
  bitwise exact. The fixture's own image-right half moved by 0.267 mm of centre and 0.132 of
  radius; its image-left half is one of the 12 that agreed to within 1e-9, so both halves are
  probed and only one of them is an independent check against the old code.
* **Augsburg reader study** (the non-isotropic consensus fit, 1.09 x 1.09 x 10 mm). On 10 mm slices
  the top-of-head slice is a large cross-section, no slice is ever a 2-sigma outlier of the padded
  vector, and the consensus refit absorbs a one-slice change of the slab start on nearly every
  half, so 99 of 100 agreed to within 1e-9 before the change. The two that did not are pinned
  here, one per calling convention: ``PA000110`` image-right moved by 0.226 voxel of centre (0.124
  of radius) with ``x_ratio`` alone -- and that displacement is along the **slice** axis, so on
  these 10 mm slices it is 2.26 mm, not the sub-voxel figure the number suggests -- while
  ``PA000077`` image-left moved by 0.0195 voxel in plane (0.021 mm, radius 0.00053) with the
  explicit ``spacing`` production passes. ``PA000001``, the usual Augsburg fixture, agreed to 6e-11
  before the change and so would pass either way; it is kept as the regression anchor and
  deliberately not relied on for discrimination.

Both calling conventions are exercised on every Augsburg half (``spacing=None``, which derives the
geometry from ``x_ratio``, and the explicit ``spacing`` the measurements pass), because the removed
guard sat above the ``if isotropic:`` branch and so reached every path through the function.

**Why the tolerance is not zero.** After the change the crop residual is not bitwise 0 but float
noise: over all 220 halves it is at most 1.1e-10 voxel of centre and 1.7e-10 of radius (7.4e-11 and
1.3e-10 on the halves pinned here). A crop shifts every z coordinate in the point cloud by a
constant, so ``sphere_fit``'s ``np.linalg.lstsq`` sees a different design matrix and its SVD is not
bit-exact under the shift -- the same floor this file's mirror-equivariance notes record for the
NaKo path. ``ATOL`` sits an order of magnitude above that and six below the smallest movement the
pre-change code produced (0.0195 voxel / 1e-8 = 1.95e6), so it discriminates without pinning
LAPACK noise.
"""
import numpy as np
import pytest

from morphometry.hip import get_femoral_head_center
from morphometry.image_io import crop_image, split_image_left_right

#: Augsburg reader-study patients: the two halves that carried the defect, plus the usual fixture.
AUGSBURG_PATIENTS = ['PA000001', 'PA000077', 'PA000110']
SIDES = ['left', 'right']
#: Tolerance on the crop residual; see "Why the tolerance is not zero" above.
ATOL = 1e-8


def _occupied_z(mask: np.ndarray, label: int = 1):
    """The first and one-past-last z index holding a voxel of ``label``.

    :param mask: A 3D label array.
    :param label: The label to look for.
    :return: ``(start, end)``, suitable for ``crop_image(..., axis=2, start, end)``.
    """
    occupied = np.argwhere((mask == label).sum(axis=(0, 1)) > 0)[:, 0]
    assert occupied.size, 'the half holds no voxel of this label'
    return int(occupied.min()), int(occupied.max() + 1)


def _assert_crop_is_free(half, side: str, **kwargs):
    """Cropping a half to its occupied z slices must move the centre by exactly the crop offset.

    :param half: The hip image half (an Image / Segmentation).
    :param side: Image side, 'left' or 'right', passed straight to the measurement.
    :param kwargs: Further arguments for ``get_femoral_head_center``.
    :return: None; asserts.
    """
    start, end = _occupied_z(half.array, kwargs.get('segmentation_label', 1))
    assert start > 0 or end < half.array.shape[2], 'nothing to crop -- the probe would be vacuous'
    radius, centre = get_femoral_head_center(half.array, side=side, **kwargs)
    cropped = crop_image(half, 2, start, end)
    crop_radius, crop_centre = get_femoral_head_center(cropped.array, side=side, **kwargs)
    shift = np.asarray(crop_centre, dtype=float) - np.asarray(centre, dtype=float)
    np.testing.assert_allclose(shift, [0.0, 0.0, -start], atol=ATOL,
                               err_msg=f'trimming {start} leading and '
                                       f'{half.array.shape[2] - end} trailing empty z slices '
                                       f'moved the centre by {shift}')
    assert crop_radius == pytest.approx(radius, abs=ATOL), \
        f'the empty-slice trim moved the radius by {crop_radius - radius}'


@pytest.mark.needs_nako
@pytest.mark.parametrize('side', SIDES)
def test_nako_head_centre_survives_an_empty_slice_trim(nako_sample, side):
    """Isotropic path: the NaKo fixture's centre and radius are unchanged by an empty-z trim."""
    half = dict(zip(SIDES, split_image_left_right(nako_sample)))[side]
    _assert_crop_is_free(half, side, segmentation_label=1, x_ratio=1.0, isotropic=True)


@pytest.mark.needs_augsburg_reader_study
@pytest.mark.parametrize('spacing_given', [False, True], ids=['x_ratio_only', 'explicit_spacing'])
@pytest.mark.parametrize('side', SIDES)
@pytest.mark.parametrize('patient', AUGSBURG_PATIENTS)
def test_augsburg_head_centre_survives_an_empty_slice_trim(augsburg_reader_study, patient, side,
                                                           spacing_given):
    """Consensus-fit path: an Augsburg reader-study hip's centre is unchanged by an empty-z trim."""
    hip = augsburg_reader_study(patient)['hip']
    half = dict(zip(SIDES, split_image_left_right(hip)))[side]
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])  # the pipelines' factor, see task #6
    _assert_crop_is_free(half, side, segmentation_label=1, x_ratio=x_ratio, isotropic=False,
                         spacing=half.spacing if spacing_given else None)

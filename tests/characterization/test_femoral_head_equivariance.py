"""Mirror equivariance of the femoral head centre (``get_femoral_head_center{,_ct}``), on real data.

These two were the last landmark functions in the library with genuine ``side == 'right'`` logic
inside them (backlog task #11). Both now measure a right image half as the reflection of a left one
and map the centre back, the way the torsion, CCD and tibial-torsion landmark code already did, so a
half and its mirror image give the same head centre and the same radius. Two probes, as in
``test_femoral_torsion_equivariance.py`` and ``test_ccd_tibial_equivariance.py``:

* ``left_via_right``: the LEFT half mirrored and measured with ``side='right'`` must give the mirror
  of the left half measured with ``side='left'``;
* ``right_via_left``: the RIGHT half mirrored and measured with ``side='left'`` must give the mirror
  of the right half measured with ``side='right'``.

**What these tests catch, and what they cannot.** Because the right side is now reached by
reflecting the half and running the left-side code, the two sides share one code path and the
probes hold by construction -- they are a guard against reintroducing a right-side branch, not an
independent derivation. They do have real content against the pre-change code, which is why the
cases below are the ones they are:

* Non-isotropic MRI (the seeded consensus fit): the old code seeded from the slab points medial of
  ``int(centroid[0])``, and truncation does not commute with ``x -> n - 1 - x``, so the seed's
  boundary column was off by one on essentially every half. The consensus refit absorbed that on
  most of them -- measured over the 100 halves of the 50 Augsburg reader-study patients, 91 agreed
  with their own mirror to within 1e-9 mm and 9 did not, 7 of those by under 0.08 mm. The two that
  differed materially are the two pinned here: ``PA000390`` (image-left, centre 3.01 mm off its own
  mirror, radius 0.43 in-plane voxels) and ``PA000397`` (image-right, 1.52 mm, 0.68). ``PA000001``,
  the usual Augsburg fixture, agreed to under 1e-9 mm before the change and so would pass either
  way, which is why it is not used here.
* CT: the lateral cutoff is applied to *physical* x, so the probe has to reflect the anatomy
  (``reflect_image_left_right``) rather than relabel indices, exactly as the measurement does. The
  defect there was the same truncation, in ``_largest_component_centroid``, and on ``case-105094``
  it was worth 0.483 mm of centre (radius 0.200 mm) on the image-left half and 0.469 mm (radius
  0.144 mm) on the image-right one -- about half an in-plane voxel, the size a centroid truncation
  predicts. Both are exactly 0 after the change.
* Isotropic (NaKo): this path was already equivariant before the change, but to **float noise and
  not bit-exactly** -- its half-voxel tie-break (``floor(x + 0.5)`` left / ``ceil(x - 0.5)`` right)
  and its per-side lateral trim were written so the direct right-side call would reproduce the
  mirror of the left one, and they did, to within 4.4e-11 voxels with the gap nonzero on 120 of the
  120 NaKo sample halves (``sphere_fit`` ends in ``np.linalg.lstsq``, whose SVD runs on a
  differently-ordered design matrix under the mirror). After the change the gap is exactly 0 on all
  120, because the two sides are now literally the same arithmetic on the same array. So these
  probes are not quite tautological before the change and are exactly so after it; they are kept so
  a future edit cannot quietly move the NaKo cohort, whose centre-edge angle, alpha angle,
  anteversion, CCD and femoral offset all run through this path.

The mapping back into the caller's frame is written out here from the image's own affine rather than
taken from ``hip._mirror_physical_point_x``, so the test does not simply re-apply the implementation
it is checking.
"""
import numpy as np
import pytest

from morphometry.hip import get_femoral_head_center, get_femoral_head_center_ct
from morphometry.image_io import reflect_image_left_right, split_image_left_right
from tests import data_paths
from tests.conftest import _load_seg

OTHER = {'left': 'right', 'right': 'left'}
PROBES = {'left_via_right': 'left', 'right_via_left': 'right'}  #: probe name -> side of the reference half

#: The two reader-study patients whose head centre disagreed with its own mirror before task #11.
MRI_PATIENTS = ('PA000390', 'PA000397')


def _mirror_index_x(point, width: int) -> np.ndarray:
    """Map a voxel index of a ``width``-wide image into its left-right mirror (``x -> width - 1 - x``).

    :param point: A length-3 (possibly fractional) voxel index.
    :param width: The size of the image along index axis 0.
    :return: The mirrored index as a float array of length 3.
    """
    point = np.asarray(point, dtype=float).copy()
    point[0] = width - 1 - point[0]
    return point


def _mirror_physical_x(image, point) -> np.ndarray:
    """Map a physical point through a left-right reflection of a nibabel-backed image's index grid.

    ``reflect_image_left_right`` moves the voxel at index ``i`` to index ``n - 1 - i`` and keeps the
    affine, so a physical point found on the reflected image belongs, in the original image, at the
    physical location of the mirrored index. Written out from the 4x4 affine so this test is
    independent of ``hip._mirror_physical_point_x`` -- including its width lookup, which reads
    ``Image.shape[0]`` there and comes off the affine's own array here.
    :param image: A nibabel-backed Image / Segmentation.
    :param point: The physical point to map, length 3.
    :return: The mapped physical point as a float array of length 3.
    """
    matrix, origin = image.affine[:3, :3], image.affine[:3, 3]
    index = np.linalg.solve(matrix, np.asarray(point, dtype=float) - origin)
    index[0] = (image.shape[0] - 1) - index[0]
    return matrix @ index + origin


@pytest.fixture(scope='module')
def ct_halves():
    """The whole-leg CT fixture split into its two image halves."""
    if data_paths.CT_WHOLE_LEG is None:
        pytest.skip('whole-leg CT sample not found (set MORPH_CT_SAMPLE)')
    return dict(zip(('left', 'right'), split_image_left_right(_load_seg(data_paths.CT_WHOLE_LEG))))


@pytest.fixture(scope='module')
def nako_halves():
    """The first NaKo hip segmentation (isotropic) split into its two image halves."""
    directory = data_paths.NAKO_SAMPLE_DIR
    if not directory.exists():
        pytest.skip(f'NaKo sample dir not found at {directory}')
    files = sorted(directory.glob('*.nii.gz'))
    if not files:
        pytest.skip(f'No NaKo segmentations in {directory}')
    return dict(zip(('left', 'right'), split_image_left_right(_load_seg(files[0]))))


@pytest.mark.needs_augsburg_reader_study
@pytest.mark.parametrize('patient', MRI_PATIENTS)
@pytest.mark.parametrize('probe', PROBES)
def test_mri_head_centre_is_mirror_equivariant(augsburg_reader_study, patient, probe):
    """The non-isotropic (consensus) head centre of a half equals the mirror of its mirror's.

    Run on the two reader-study patients the pre-change seed truncation got wrong, with the
    production ``spacing=`` argument, so the assertion has content against that code.
    """
    side = PROBES[probe]
    hip = augsburg_reader_study(patient)['hip']
    spacing = tuple(abs(float(value)) for value in hip.spacing[:3])
    half = split_image_left_right(hip)[0 if side == 'left' else 1]
    width = half.array.shape[0]

    radius, centre = get_femoral_head_center(half.array, side=side, segmentation_label=1,
                                             isotropic=False, spacing=spacing)
    mirrored_radius, mirrored_centre = get_femoral_head_center(half.array[::-1], side=OTHER[side],
                                                               segmentation_label=1, isotropic=False,
                                                               spacing=spacing)

    np.testing.assert_allclose(_mirror_index_x(mirrored_centre, width), np.asarray(centre, dtype=float),
                               rtol=0, atol=1e-9)
    assert mirrored_radius == pytest.approx(radius, rel=0, abs=1e-9)


@pytest.mark.needs_augsburg_reader_study
@pytest.mark.parametrize('patient', MRI_PATIENTS)
@pytest.mark.parametrize('probe', PROBES)
def test_mri_head_slab_layers_are_mirror_equivariant(augsburg_reader_study, patient, probe):
    """``return_layers=True`` gives the same two slab bounds on a half and on its mirror.

    The layers are z indices, so the left-right mirror must not touch them: an inequality here
    would mean the column walk that ends the slab ran down a different column.
    """
    side = PROBES[probe]
    hip = augsburg_reader_study(patient)['hip']
    spacing = tuple(abs(float(value)) for value in hip.spacing[:3])
    half = split_image_left_right(hip)[0 if side == 'left' else 1]

    *_, high, low = get_femoral_head_center(half.array, side=side, segmentation_label=1,
                                            return_layers=True, isotropic=False, spacing=spacing)
    *_, mirrored_high, mirrored_low = get_femoral_head_center(half.array[::-1], side=OTHER[side],
                                                              segmentation_label=1, return_layers=True,
                                                              isotropic=False, spacing=spacing)

    assert (mirrored_high, mirrored_low) == (high, low)


@pytest.mark.needs_ct
@pytest.mark.parametrize('probe', PROBES)
def test_ct_head_centre_is_mirror_equivariant(ct_halves, probe):
    """The CT head centre of a half equals the mirror of the reflected half's, radii equal.

    The CT landmark reads physical directions (which physical x is lateral), so the probe reflects
    the anatomy with ``reflect_image_left_right`` -- an index-relabelling mirror would leave a left
    leg a left leg in physical space and the probe would not be a mirror probe at all.
    """
    side = PROBES[probe]
    half = ct_halves[side]

    radius, centre = get_femoral_head_center_ct(half, 1, side)
    mirrored_radius, mirrored_centre = get_femoral_head_center_ct(reflect_image_left_right(half), 1, OTHER[side])

    np.testing.assert_allclose(_mirror_physical_x(half, mirrored_centre), np.asarray(centre, dtype=float),
                               rtol=0, atol=1e-6)
    assert mirrored_radius == pytest.approx(radius, rel=0, abs=1e-6)


@pytest.mark.needs_nako
@pytest.mark.parametrize('probe', PROBES)
def test_isotropic_head_centre_is_mirror_equivariant_on_real_data(nako_halves, probe):
    """The isotropic (NaKo) head centre of a half equals the mirror of its mirror's.

    This held before task #11 as well, but only to ~4e-11 voxels; it is now exact (see the module
    docstring). Kept because the whole NaKo hip family -- centre-edge angle, alpha angle,
    anteversion, CCD, projected femoral offset -- reaches its head centre through this path, and a
    future edit here would move all of it.
    """
    side = PROBES[probe]
    half = nako_halves[side]
    width = half.array.shape[0]

    radius, centre = get_femoral_head_center(half.array, side=side, segmentation_label=1, isotropic=True)
    mirrored_radius, mirrored_centre = get_femoral_head_center(half.array[::-1], side=OTHER[side],
                                                               segmentation_label=1, isotropic=True)

    np.testing.assert_allclose(_mirror_index_x(mirrored_centre, width), np.asarray(centre, dtype=float),
                               rtol=0, atol=1e-9)
    assert mirrored_radius == pytest.approx(radius, rel=0, abs=1e-9)

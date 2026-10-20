"""Mirror equivariance of the CCD angle and the tibial torsion (MRI and whole-leg CT).

The two image halves hold mirror-image anatomy, so the right-side code must behave like the left-side
code on the mirror image. As in ``test_femoral_torsion_equivariance.py``, two probes are checked:

* ``left_via_right``: the LEFT half mirrored and measured with ``side='right'`` must match the left
  half measured with ``side='left'``;
* ``right_via_left``: the RIGHT half mirrored and measured with ``side='left'`` must match the right
  half measured with ``side='right'``.

Both measurements reach the right side by reflecting the half and running the left-side code, so the
probes hold by construction; they guard against reintroducing a separate right-side branch. Before
that, the two sides disagreed through landmark code that is not mirror-symmetric -- the femoral-neck
shell sampling for the CCD, the knee condylar search (which rotates the slice about the image centre
and rounds) for the tibial torsion -- by a median 0.16-0.23 degrees on the Augsburg MRI CCD, 0.85-1.02
on the CT CCD, 0.8-1.1 on the Augsburg tibial torsion and 1.7 on the CT tibial torsion, the last with
a tail into the tens of degrees on the legs whose notch search fails (that tail belonged to the old
no-notch condylar fallback, since replaced -- see ``tests/characterization/test_condylar_fallback.py``).

The CCD and the tibial torsion are angles between two landmark-derived vectors, so a reflection of
the anatomy cannot change them; only the landmark search is mirrored.

What these tests can and cannot catch: because the right side is now measured by reflecting the half
and running the left-side code, the array handed to that code is bit-identical between reference and
probe for the tibial and the CT tests, so their *angle* assertions are tautological -- they can only
fail if a separate right-side branch is reintroduced or ``mirror_image_left_right`` and
``reflect_image_left_right`` are swapped. The landmark assertions are not: they compare mapped
indices, and they do fail against the pre-change code (see below). The MRI CCD probe has real content, because
``_reflected_about_world_x0`` gives it a different affine from the production mirror. They cannot
catch defects in the shared left-side code, which a mirrored probe reproduces identically on both
sides. Note also that the two assertions of the tibial test have different strength on PA000001: the
landmark assertion is the one that fails against the pre-change code (the condylar x is off by one
voxel, 67 vs 66), while the angle assertion passed there too, because that leg is one of the ~12%
whose condylar *vector* survives the mirror unchanged. ``tibial_torsion_ct_right`` is the golden that
records a right-side tibial value which actually moved.

The MRI CCD spans two images (hip and knee) whose affines have different origins, so the probe has to
reflect both about ONE common physical plane (``_reflected_about_world_x0``). Reflecting each about
its own array centre — what ``reflect_image_left_right`` does — moves the two by different amounts
and distorts the hip-to-knee shaft vector by ~1.2 degrees, which is a property of the probe, not of
the measurement. For the same reason ``calculate_ccd`` mirrors rather than reflects internally: a
mirror keeps every voxel's physical position. The single-image CT path uses
``reflect_image_left_right``, since its landmark code reads physical directions, and the plain knee
and ankle arrays are reversed with ``[::-1]``.
"""
import nibabel as nib
import numpy as np
import pytest

from morphometry.image_io import Segmentation, reflect_image_left_right, split_ct_image, split_image_left_right
from morphometry.measurements.hip import calculate_ccd, calculate_ccd_ct
from morphometry.measurements.tibia import calculate_tibial_torsion, get_tibial_torsion_landmarks
from tests import data_paths
from tests.conftest import _load_seg

OTHER = {'left': 'right', 'right': 'left'}
PROBES = {'left_via_right': 'left', 'right_via_left': 'right'}  # probe name -> side of the reference half


def _mirror_x(point, n: int) -> np.ndarray:
    """Map a voxel index of an ``n``-wide image to the mirrored image (``x -> n - 1 - x``).

    :param point: A length-3 index.
    :param n: Width of the image along x.
    :return: The mirrored index as a float array.
    """
    point = np.asarray(point, dtype=float).copy()
    point[0] = n - 1 - point[0]
    return point


def _same_pair(a, b, c, d, abs_tol: float) -> bool:
    """Whether the unordered point pairs ``{a, b}`` and ``{c, d}`` coincide within ``abs_tol``.

    :param a: First point of the first pair.
    :param b: Second point of the first pair.
    :param c: First point of the second pair.
    :param d: Second point of the second pair.
    :param abs_tol: Absolute tolerance per coordinate.
    :return: True when they match in either order.
    """
    close = lambda p, q: np.allclose(np.asarray(p, float), np.asarray(q, float), rtol=0, atol=abs_tol)
    return (close(a, c) and close(b, d)) or (close(a, d) and close(b, c))


def _reflected_about_world_x0(image):
    """Reflect an image's content about the world plane ``x = 0``: reverse the array, fold the affine.

    Unlike :func:`morphometry.image_io.reflect_image_left_right`, which keeps the affine and so
    reflects about each image's own centre, this puts every image through the *same* world-space
    reflection, which is what a two-image measurement needs.
    :param image: A nibabel-backed Segmentation.
    :return: The reflected Segmentation.
    """
    flip = np.eye(4)
    flip[0, 0], flip[0, 3] = -1, image.array.shape[0] - 1  # index i -> n - 1 - i
    plane = np.eye(4)
    plane[0, 0] = -1  # world x -> -x
    return Segmentation.from_nibabel(nib.Nifti1Image(image.array[::-1], plane @ image.affine @ flip, image.header))


@pytest.fixture(scope='module')
def mri():
    """The Augsburg PA000001 hip / knee / ankle halves and the pipeline's x_ratio."""
    base = data_paths.AUGSBURG_PA000001
    files = {name: base / f'{name}_seg.nii.gz' for name in ('hip', 'knee', 'ankle')}
    if not all(path.exists() for path in files.values()):
        pytest.skip(f'Augsburg PA000001 data not found under {base}')
    images = {name: _load_seg(path) for name, path in files.items()}
    halves = {name: dict(zip(('left', 'right'), split_image_left_right(image))) for name, image in images.items()}
    halves['x_ratio'] = abs(images['hip'].spacing[2]) / 2 * abs(images['hip'].spacing[0])
    return halves


@pytest.fixture(scope='module')
def ct():
    """The whole-leg CT halves."""
    if data_paths.CT_WHOLE_LEG is None:
        pytest.skip('whole-leg CT sample not found (set MORPH_CT_SAMPLE)')
    return dict(zip(('left', 'right'), split_image_left_right(_load_seg(data_paths.CT_WHOLE_LEG))))


@pytest.mark.needs_augsburg
@pytest.mark.parametrize('probe', PROBES)
def test_mri_ccd_is_mirror_equivariant(mri, probe):
    """The CCD of a half equals the CCD of its reflection measured with the other side's code."""
    side = PROBES[probe]
    hip, knee = mri['hip'][side], mri['knee'][side]
    reference = calculate_ccd(hip, knee, side, 1, isotropic=False, x_ratio=mri['x_ratio'])[:2]
    mirrored = calculate_ccd(_reflected_about_world_x0(hip), _reflected_about_world_x0(knee), OTHER[side], 1,
                             isotropic=False, x_ratio=mri['x_ratio'])[:2]
    np.testing.assert_allclose(mirrored, reference, atol=1e-9)


@pytest.mark.needs_augsburg
@pytest.mark.parametrize('probe', PROBES)
def test_mri_tibial_torsion_is_mirror_equivariant(mri, probe):
    """The tibial torsion and its landmarks mirror with the side."""
    side = PROBES[probe]
    knee, ankle = mri['knee'][side].array, mri['ankle'][side].array
    kwargs = dict(tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2)
    reference = calculate_tibial_torsion(knee, ankle, side=side, **kwargs)
    mirrored = calculate_tibial_torsion(knee[::-1], ankle[::-1], side=OTHER[side], **kwargs)
    assert mirrored == pytest.approx(reference, abs=1e-9)

    ref_landmarks = get_tibial_torsion_landmarks(knee, ankle, side=side, **kwargs)
    probe_landmarks = get_tibial_torsion_landmarks(knee[::-1], ankle[::-1], side=OTHER[side], **kwargs)
    widths = {'knee': knee.shape[0], 'ankle': ankle.shape[0]}
    mapped = {name: _mirror_x(point, widths[name.split('_')[0]]) for name, point in probe_landmarks.items()}
    # The condylar endpoints are ordered by descending x in each frame, so mirroring normally swaps
    # them -- but the ordering uses a strict ``<``, so on a leg whose two endpoints share an x
    # neither frame swaps and a correct implementation would report them in the same order. Compare
    # the pair unordered so that cannot fail the test; the line is undirected anyway.
    assert _same_pair(mapped['knee_start'], mapped['knee_end'],
                      ref_landmarks['knee_end'], ref_landmarks['knee_start'], 1e-7)
    for name in ('ankle_start', 'ankle_end'):
        np.testing.assert_allclose(mapped[name], np.asarray(ref_landmarks[name], dtype=float))


@pytest.mark.needs_ct
@pytest.mark.parametrize('probe', PROBES)
def test_ct_ccd_is_mirror_equivariant(ct, probe):
    """Whole-leg CT: the same probe for ``calculate_ccd_ct``."""
    side = PROBES[probe]
    leg = ct[side]
    reference = calculate_ccd_ct(leg, side=side, segmentation_label=1)[:2]
    mirrored = calculate_ccd_ct(reflect_image_left_right(leg), side=OTHER[side], segmentation_label=1)[:2]
    np.testing.assert_allclose(mirrored, reference, atol=1e-9)


@pytest.mark.needs_ct
@pytest.mark.parametrize('probe', PROBES)
def test_ct_tibial_torsion_is_mirror_equivariant(ct, probe):
    """Whole-leg CT: the tibial torsion on the knee / ankle crops of a leg and of its reflection."""
    side = PROBES[probe]

    def torsion(leg, leg_side):
        _, masks = split_ct_image(leg, None, femur_label=1, hip_label=7, patella_label=5, fibula_label=3)
        return calculate_tibial_torsion(masks[1].array, masks[2].array, tibia_label_knee=2, tibia_label_ankle=2,
                                        fibula_label=3, side=leg_side)

    reference = torsion(ct[side], side)
    mirrored = torsion(reflect_image_left_right(ct[side]), OTHER[side])
    assert mirrored == pytest.approx(reference, abs=1e-9)

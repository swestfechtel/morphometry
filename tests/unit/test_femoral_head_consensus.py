"""Unit tests for the MRI femoral-head sphere fit (``get_femoral_head_center``, non-isotropic path).

The head centre is fitted to the contour points of a slab of slices that starts at the top of the
femur and usually runs one to three slices past the head's equator, into the femoral neck and the
greater trochanter. Those points must be kept out of the sphere fit. The former fit removed them
with a lateral cutoff at ``0.9 * max_x`` -- 0.9 times an absolute voxel index -- so the cutoff, and
with it the centre, depended on where the hip sat in the image. It also could not tell lateral head
surface at the equator from neck surface below it.

The fit is now a seeded consensus fit in millimetres: a sphere fitted to the medial half of the slab
(which holds no neck), then refitted on every slab point within a tolerance of the current sphere
until the inlier set is stable. The phantom here is a sphere with a femoral neck running laterally
and inferiorly from it plus a trochanter block, on a 1 x 1 x 10 mm grid like the Augsburg hip MRI.
"""
import numpy as np
import pytest

from morphometry.hip import get_femoral_head_center

SPACING = (1.0, 1.0, 10.0)  #: voxel spacing in mm (in-plane voxels are 1 mm, so x_ratio == spacing_z)
X_RATIO = SPACING[2] / SPACING[0]
SHAPE = (200, 90, 16)
HEAD_RADIUS_MM = 22.0
NECK_RADIUS_MM = 16.0
NECK_ANGLE_DEG = 35.0  #: inclination of the neck axis below the horizontal, running laterally
#: Off-grid head centres (voxel index) at three medial-lateral positions in the image.
HEAD_CENTRES = [np.array([x, 45.4, 3.3]) for x in (70.3, 110.3, 150.3)]
#: Accuracy the consensus fit must reach on the phantom, in mm.
CENTRE_TOLERANCE_MM = 0.5


def head_and_neck(head: np.ndarray, shape=SHAPE, spacing=SPACING) -> np.ndarray:
    """A femoral head with a neck and a trochanter block, as a left image half (medial = +x).

    :param head: The head centre as a (possibly fractional) voxel index.
    :param shape: The volume shape in voxels.
    :param spacing: Voxel spacing in mm.
    :return: A uint8 mask with the femur labelled 1.
    """
    xx, yy, zz = np.indices(shape).astype(float)
    offset = np.stack([(xx - head[0]) * spacing[0], (yy - head[1]) * spacing[1], (zz - head[2]) * spacing[2]], -1)
    sphere = (offset ** 2).sum(-1) <= HEAD_RADIUS_MM ** 2
    # the neck axis runs laterally (-x) and inferiorly (+z) out of the head centre
    axis = np.array([-np.cos(np.radians(NECK_ANGLE_DEG)), 0.0, np.sin(np.radians(NECK_ANGLE_DEG))])
    along = offset @ axis
    radial = np.linalg.norm(offset - along[..., None] * axis, axis=-1)
    neck = (radial <= NECK_RADIUS_MM) & (along >= 0) & (along <= 60)
    dx, dy, dz = offset[..., 0], offset[..., 1], offset[..., 2]
    trochanter = (dx <= -40) & (dx >= -65) & (np.abs(dy) <= 20) & (dz >= 15) & (dz <= 80)
    return (sphere | neck | trochanter).astype(np.uint8)


def head_neck_and_calcar(head: np.ndarray, shape=SHAPE, spacing=SPACING) -> np.ndarray:
    """The head-and-neck phantom plus a medial calcar/shaft block running straight down from the head.

    The column dropped through the top-slice centroid stays inside this block, so the slab reaches far
    below the head, and the block's medial surface lies on the same side of the centroid as the medial
    head surface -- the configuration in which a seed drawn from every slab slice starts in the
    calcar instead of on the head.
    :param head: The head centre as a (possibly fractional) voxel index.
    :param shape: The volume shape in voxels.
    :param spacing: Voxel spacing in mm.
    :return: A uint8 mask with the femur labelled 1.
    """
    xx, yy, zz = np.indices(shape).astype(float)
    dx, dy, dz = (xx - head[0]) * spacing[0], (yy - head[1]) * spacing[1], (zz - head[2]) * spacing[2]
    calcar = (dz >= 0) & (dz <= 100) & (dx >= -6) & (dx <= 14) & (np.abs(dy) <= 12)
    return (head_and_neck(head, shape, spacing).astype(bool) | calcar).astype(np.uint8)


def _pad(mask: np.ndarray, shift) -> np.ndarray:
    """Translate the mask content by a non-negative integer voxel shift, padding at the low end.

    :param mask: A 3D mask.
    :param shift: ``(kx, ky, kz)``, each >= 0.
    :return: The padded mask.
    """
    return np.pad(mask, [(k, 0) for k in shift])


@pytest.mark.parametrize('head', HEAD_CENTRES, ids=lambda h: f'x{h[0]:.0f}')
@pytest.mark.parametrize('side', ['left', 'right'])
def test_head_centre_is_recovered_despite_the_neck(head, side):
    """The fitted centre lies within 0.5 mm of the true centre wherever the hip sits in the image.

    The right side is the mirror-image phantom measured with ``side='right'``.
    """
    mask = head_and_neck(head)
    truth = head.copy()
    if side == 'right':
        mask = mask[::-1]
        truth[0] = SHAPE[0] - 1 - truth[0]
    radius, centre = get_femoral_head_center(mask, side=side, segmentation_label=1, x_ratio=X_RATIO)
    error = np.linalg.norm((np.asarray(centre, dtype=float) - truth) * SPACING)
    assert error <= CENTRE_TOLERANCE_MM, f'centre {centre} is {error:.2f} mm from the true centre {truth}'
    assert abs(radius * SPACING[0] - HEAD_RADIUS_MM) <= 2.5, f'radius {radius * SPACING[0]:.2f} mm'


@pytest.mark.parametrize('shift', [(7, 0, 0), (33, 0, 0), (0, 9, 0), (0, 0, 2), (13, 5, 1)])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_head_centre_and_radius_follow_a_translation_exactly(shift, side):
    """Padding the image by integer voxels moves the centre by exactly that shift and keeps the radius."""
    mask = head_and_neck(HEAD_CENTRES[1])
    if side == 'right':
        mask = mask[::-1]
    radius, centre = get_femoral_head_center(mask, side=side, segmentation_label=1, x_ratio=X_RATIO)
    moved_radius, moved_centre = get_femoral_head_center(_pad(mask, shift), side=side, segmentation_label=1,
                                                         x_ratio=X_RATIO)
    np.testing.assert_allclose(np.asarray(moved_centre, dtype=float) - centre, shift, atol=1e-6)
    assert moved_radius == pytest.approx(radius, abs=1e-6)


def test_explicit_spacing_gives_the_same_fit_as_the_equivalent_x_ratio():
    """With 1 mm in-plane voxels, ``spacing`` and the ``x_ratio`` fallback describe the same geometry."""
    mask = head_and_neck(HEAD_CENTRES[0])
    r_fallback, c_fallback = get_femoral_head_center(mask, side='left', segmentation_label=1, x_ratio=X_RATIO)
    r_spacing, c_spacing = get_femoral_head_center(mask, side='left', segmentation_label=1, x_ratio=X_RATIO,
                                                   spacing=SPACING)
    np.testing.assert_allclose(c_spacing, c_fallback, atol=1e-6)
    assert r_spacing == pytest.approx(r_fallback, abs=1e-6)


def test_spacing_is_honoured_for_sub_millimetre_in_plane_voxels():
    """The same physical phantom on a 0.5 mm in-plane grid gives the same centre in mm and radius in voxels x 2."""
    fine = (0.5, 0.5, 10.0)
    head_mm = HEAD_CENTRES[0] * SPACING
    head_fine = head_mm / fine
    mask = head_and_neck(head_fine, shape=(400, 180, 16), spacing=fine)
    radius, centre = get_femoral_head_center(mask, side='left', segmentation_label=1,
                                             x_ratio=fine[2] / fine[0], spacing=fine)
    error = np.linalg.norm(np.asarray(centre, dtype=float) * fine - head_mm)
    assert error <= CENTRE_TOLERANCE_MM
    assert abs(radius * fine[0] - HEAD_RADIUS_MM) <= 2.5


@pytest.mark.parametrize('x_ratio', [X_RATIO, X_RATIO / 2], ids=['true_x_ratio', 'pipeline_like_x_ratio'])
def test_seed_is_taken_from_the_head_not_from_a_slab_that_runs_into_the_calcar(x_ratio):
    """A slab that runs far below the head must not pull the seed, and with it the fit, into the calcar.

    The pipelines' ``x_ratio`` (half the slice aspect) lets the slab bound admit even more slices; the
    fit itself uses ``spacing``, so both must recover the head. Where the calcar merges into the lower
    head a few of its surface points fall inside the inlier band, so this case is held to 1 mm rather
    than 0.5 mm; a seed started in the calcar misses by 5 to 20 mm.
    """
    head = HEAD_CENTRES[1]
    radius, centre = get_femoral_head_center(head_neck_and_calcar(head), side='left', segmentation_label=1,
                                             x_ratio=x_ratio, spacing=SPACING)
    error = np.linalg.norm((np.asarray(centre, dtype=float) - head) * SPACING)
    assert error <= 1.0, f'centre {centre} is {error:.2f} mm from the true centre {head}'
    assert abs(radius * SPACING[0] - HEAD_RADIUS_MM) <= 2.5, f'radius {radius * SPACING[0]:.2f} mm'

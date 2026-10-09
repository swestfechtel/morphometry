"""Mirror equivariance of the isotropic femoral-head centre (``get_femoral_head_center(isotropic=True)``).

The isotropic path (NaKo hip MRI) fits a sphere to the contour points of a slab of slices below the
top of the femur, after dropping everything lateral of a cutoff placed one "medial radius" on the far
side of the top-slice centroid. The left-side code puts the cutoff at ``2 * cx - max_x``, the
right-side code at ``2 * cx - min_x``; on a mirror image the two are the same plane only if ``cx``
mirrors too. It used to be ``int(cx)``, and floor does not commute with ``x -> n - 1 - x`` (the
mirror of ``floor(n - 1 - cx)`` is ``ceil(cx)``), so the right-side cutoff of a reflected hip sat two
voxels away from the left-side cutoff of the original, the column walk that ends the slab ran down a
neighbouring column, and the two image sides were measured with different head centres. On the 60
NaKo reference hips that made the left-side code read the centre-edge angle ~1.1 degrees lower than
the right-side code on the same hip.

The cutoff now uses the unrounded centroid, and the column walk the nearest column with half-voxel
ties broken towards medial (``floor(x + 0.5)`` on the left, ``ceil(x - 0.5)`` on the right), which
commutes with the mirror. Plain ``np.rint`` would not: it rounds ties to even, and the parity flips
under ``x -> n - 1 - x`` when ``n`` is even. The phantom is a femoral head with a neck running
laterally and inferiorly, on a 1 mm isotropic grid, with an off-grid head centre so the top-slice
centroid is not an integer; one centre sits on a half voxel, where the top slice is symmetric and
its centroid is an exact tie.
"""
import numpy as np
import pytest
from scipy.ndimage import center_of_mass

from morphometry.hip import get_femoral_head_center

SHAPE = (130, 100, 90)
HEAD_RADIUS = 22.0
NECK_RADIUS = 15.0
NECK_ANGLE_DEG = 35.0  #: inclination of the neck axis below the horizontal, running laterally
#: Off-grid head centres (voxel index) at several medial-lateral positions; x = 75.5 gives an exact tie.
HEAD_CENTRES = [np.array([x, 50.4, 30.6]) for x in (60.3, 75.5, 75.7, 90.2)]


def head_and_neck(head: np.ndarray, shape=SHAPE) -> np.ndarray:
    """A femoral head with a neck, as a left image half (medial = +x, inferior = +z), 1 mm isotropic.

    :param head: The head centre as a (possibly fractional) voxel index.
    :param shape: The volume shape in voxels.
    :return: A uint8 mask with the femur labelled 1.
    """
    offset = np.stack(np.indices(shape), -1).astype(float) - head
    sphere = (offset ** 2).sum(-1) <= HEAD_RADIUS ** 2
    # the neck axis runs laterally (-x) and inferiorly (+z) out of the head centre
    axis = np.array([-np.cos(np.radians(NECK_ANGLE_DEG)), 0.0, np.sin(np.radians(NECK_ANGLE_DEG))])
    along = offset @ axis
    radial = np.linalg.norm(offset - along[..., None] * axis, axis=-1)
    neck = (radial <= NECK_RADIUS) & (along >= 0) & (along <= 50)
    return (sphere | neck).astype(np.uint8)


@pytest.mark.parametrize('head', HEAD_CENTRES, ids=lambda h: f'x{h[0]:.1f}')
def test_isotropic_head_centre_is_mirror_equivariant(head):
    """The left half and its reflection measured with ``side='right'`` give mirrored centres, equal radii."""
    mask = head_and_neck(head)
    top = np.flatnonzero(mask.any(axis=(0, 1)))[0]
    cx = center_of_mass(mask[:, :, top])[0]
    # precondition: an off-grid centroid, i.e. the case floor got wrong
    assert 0.05 < cx % 1 < 0.95, f'top-slice centroid x {cx} is (nearly) an integer'

    r_left, c_left = get_femoral_head_center(mask, side='left', segmentation_label=1, isotropic=True)
    r_right, c_right = get_femoral_head_center(mask[::-1], side='right', segmentation_label=1, isotropic=True)

    mapped = np.asarray(c_right, dtype=float).copy()
    mapped[0] = SHAPE[0] - 1 - mapped[0]
    np.testing.assert_allclose(mapped, c_left, atol=1e-6)
    assert r_right == pytest.approx(r_left, abs=1e-6)

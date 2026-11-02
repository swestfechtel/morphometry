"""Mirror equivariance of the isotropic femoral-head centre (``get_femoral_head_center(isotropic=True)``).

**What this test pins now.** ``get_femoral_head_center`` measures a right image half as the
reflection of a left one (backlog task #11): ``mask[::-1]`` run through the left-side code, with the
centre mapped back by ``x -> n - 1 - x``. So the probe below -- a left half, reflected, measured
with ``side='right'`` -- reflects twice and reaches the left-side code with the *identical* array.
The arithmetic that produces the centre is therefore the same arithmetic in both calls, and the
assertion is **exact**: what is left to fail is the index mapping itself, which is what this test
guards. An ``n`` versus ``n - 1`` slip, a mirror about the wrong axis, or a ``floor`` reintroduced
into the map-back all fail it; a change inside the shared left-side code cannot, because the probe
reproduces it identically on both sides.

**What it used to pin, and why that content is gone.** The isotropic path once had a genuine
right-side branch: the lateral cutoff (one "medial radius" on the far side of the top-slice
centroid) was ``2 * cx - max_x`` on the left and ``2 * cx - min_x`` on the right, and the column
walk rounded with ``floor(x + 0.5)`` on the left against ``ceil(x - 0.5)`` on the right. Those two
had to be each other's mirror for the two independent code paths to agree, and before that they
were both ``int(cx)`` -- floor does not commute with ``x -> n - 1 - x`` (the mirror of
``floor(n - 1 - cx)`` is ``ceil(cx)``), so the right-side cutoff of a reflected hip sat two voxels
from the left-side cutoff of the original, the column walk ran down a neighbouring column, and the
two image sides were measured with different head centres -- worth ~1.1 degrees of centre-edge
angle on the same hip over the 60 NaKo reference hips.

With a single code path that requirement has disappeared rather than moved: there is no second
rounding rule left to agree with, so an exact-tie centroid is no longer a hazard at all. The
``floor(x + 0.5)`` rule survives only as bit-compatibility with the shipped left-side value (it is
what every NaKo golden was captured with), not as a symmetry constraint.

The phantom is a femoral head with a neck running laterally and inferiorly, on a 1 mm isotropic
grid, with an off-grid head centre so the top-slice centroid is not an integer. The ``x = 75.5``
case is kept deliberately: it is the exact tie that used to be the difficult input, and it is now
no more interesting than the others, which is the point.
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
    """The left half and its reflection measured with ``side='right'`` give mirrored centres, equal radii.

    Asserted as exact equality, not within a tolerance: the probe double-reflects to the identical
    array, so the two calls run the same arithmetic and any non-zero difference is a defect in the
    index map-back rather than floating-point reassociation.
    """
    mask = head_and_neck(head)
    top = np.flatnonzero(mask.any(axis=(0, 1)))[0]
    cx = center_of_mass(mask[:, :, top])[0]
    # precondition: an off-grid centroid, i.e. the case floor got wrong
    assert 0.05 < cx % 1 < 0.95, f'top-slice centroid x {cx} is (nearly) an integer'

    r_left, c_left = get_femoral_head_center(mask, side='left', segmentation_label=1, isotropic=True)
    r_right, c_right = get_femoral_head_center(mask[::-1], side='right', segmentation_label=1, isotropic=True)

    mapped = np.asarray(c_right, dtype=float).copy()
    mapped[0] = SHAPE[0] - 1 - mapped[0]
    np.testing.assert_array_equal(mapped, np.asarray(c_left, dtype=float))
    assert r_right == r_left

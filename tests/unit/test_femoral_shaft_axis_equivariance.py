"""Mirror equivariance of the MRI femoral shaft axis (``hip.py::get_femoral_shaft_axis``).

Both endpoints of the shaft axis are slice centroids: the distal one is the femur centroid a few
slices above the bottom of the hip mask, the proximal one the centroid of the greater trochanter on
the first slice where it separates from the head. They used to be truncated with ``int()``, and floor
does not commute with the mirror ``x -> n - 1 - x`` (the mirror of ``floor(n - 1 - x)`` is
``ceil(x)``), so the shaft axis of a reflected hip sat one voxel away from the mirror of the
original's. ``calculate_femoral_offset_projected`` measures the head centre's distance from that
axis, and on the 60 NaKo hips it read a hip ~1 mm differently depending on the image side it was on
(median 0.99, max 1.00 mm) -- the whole of the remaining mirror error once the head centre was fixed.

The endpoints are continuous coordinates that index no array, so they are no longer rounded. The
phantom is a femoral head plus a separate trochanter column that continues down as the shaft, on a
1 mm isotropic grid, with off-grid centres so both centroids are fractional.
"""
import numpy as np
import pytest
from scipy.ndimage import center_of_mass

from morphometry.hip import get_femoral_shaft_axis

SHAPE = (80, 60, 100)
HEAD_CENTRE = np.array([50.3, 30.4, 25.0])
HEAD_RADIUS = 15.0
#: In-plane centre and radius of the trochanter/shaft column, which starts at ``SHAFT_TOP`` and runs to the bottom.
SHAFT_CENTRES = [np.array([x, 30.6]) for x in (18.3, 21.7)]
SHAFT_RADIUS = 7.0
SHAFT_TOP = 18


def head_and_shaft(shaft: np.ndarray, shape=SHAPE) -> np.ndarray:
    """A femoral head and a separate trochanter/shaft column, as a left image half, 1 mm isotropic.

    :param shaft: The column's in-plane centre ``(x, y)`` as a (possibly fractional) voxel index.
    :param shape: The volume shape in voxels.
    :return: A uint8 mask with the femur labelled 1.
    """
    xx, yy, zz = np.indices(shape).astype(float)
    head = (xx - HEAD_CENTRE[0]) ** 2 + (yy - HEAD_CENTRE[1]) ** 2 + (zz - HEAD_CENTRE[2]) ** 2 <= HEAD_RADIUS ** 2
    column = ((xx - shaft[0]) ** 2 + (yy - shaft[1]) ** 2 <= SHAFT_RADIUS ** 2) & (zz >= SHAFT_TOP)
    return (head | column).astype(np.uint8)


@pytest.mark.parametrize('shaft', SHAFT_CENTRES, ids=lambda s: f'x{s[0]:.1f}')
def test_shaft_axis_is_mirror_equivariant(shaft):
    """The shaft axis of the reflected mask is the mirror of the original's, endpoint by endpoint."""
    mask = head_and_shaft(shaft)
    cx = center_of_mass(mask[:, :, SHAFT_TOP])[0]
    # precondition: a fractional centroid, i.e. the case int() got wrong
    assert 0.05 < cx % 1 < 0.95, f'column centroid x {cx} is (nearly) an integer'

    low, high = get_femoral_shaft_axis(mask, isotropic=True)
    mirrored_low, mirrored_high = get_femoral_shaft_axis(mask[::-1], isotropic=True)
    for point, mirrored in ((low, mirrored_low), (high, mirrored_high)):
        mapped = np.asarray(mirrored, dtype=float).copy()
        mapped[0] = SHAPE[0] - 1 - mapped[0]
        np.testing.assert_allclose(mapped, np.asarray(point, dtype=float), atol=1e-6)


def test_shaft_axis_endpoints_are_the_slice_centroids():
    """The endpoints are the unrounded centroids of the trochanter's top slice and the distal slice."""
    shaft = SHAFT_CENTRES[0]
    mask = head_and_shaft(shaft)
    low, high = get_femoral_shaft_axis(mask, isotropic=True)
    distal = SHAPE[2] - 1 - 5  # the function steps 5 slices up from the bottom on isotropic images
    np.testing.assert_allclose(low[:2], center_of_mass(mask[:, :, distal]), atol=1e-9)
    assert low[2] == distal and high[2] == SHAFT_TOP
    trochanter = mask[:, :, SHAFT_TOP].copy()
    trochanter[int(HEAD_CENTRE[0]) - 20:] = 0  # keep only the column; the head lies medial (+x) of it
    np.testing.assert_allclose(high[:2], center_of_mass(trochanter), atol=1e-9)

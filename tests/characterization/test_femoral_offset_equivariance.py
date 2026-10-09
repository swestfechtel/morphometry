"""Mirror equivariance of the NaKo projected femoral offset (isotropic hip MRI).

``calculate_femoral_offset_projected`` measures each image half with its own ``side`` code, so a hip
must give the same offset on either side of the image: the right half of a reflected scan is the
mirror of the original left half, and vice versa. Called as ``scripts/process_nako_large.py`` calls it
(per image half, ``knee_image=None``, ``isotropic=True``).

The offset combines two landmarks, the isotropic femoral-head centre (mirror-exact since its cutoff
stopped flooring the centroid, see ``test_cea_equivariance.py``) and the shaft axis, whose two slice
centroids were truncated with ``int()`` (see ``tests/unit/test_femoral_shaft_axis_equivariance.py``).
Over the 60 NaKo reference hips that left the reflected offset off by a median 0.99 mm (max 1.00, 0 of
118 measurable halves exact); with unrounded endpoints all 118 agree to floating point, so the
tolerance is exact.
"""
import numpy as np
import pytest

from morphometry.image_io import reflect_image_left_right, split_image_left_right
from morphometry.measurements.hip import calculate_femoral_offset_projected

pytestmark = pytest.mark.needs_nako


def _offsets(seg):
    """Projected femoral offset of the image-left and image-right halves, as the NaKo pipeline calls it.

    :param seg: A both-sides NaKo segmentation.
    :return: ``(left, right)`` offsets.
    """
    left, right = split_image_left_right(seg)
    return (calculate_femoral_offset_projected(left, None, 'left', 1, isotropic=True),
            calculate_femoral_offset_projected(right, None, 'right', 1, isotropic=True))


def test_projected_femoral_offset_is_mirror_equivariant(nako_sample):
    """The offsets of the reflected hip are the original offsets with the two image sides swapped."""
    original = _offsets(nako_sample)
    reflected = _offsets(reflect_image_left_right(nako_sample))
    np.testing.assert_allclose(reflected[::-1], original, atol=1e-6)

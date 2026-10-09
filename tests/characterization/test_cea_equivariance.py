"""Mirror equivariance of the NaKo centre-edge angle (isotropic hip MRI).

A reflected hip swaps the two image halves, so its CEA must be the original's with the two values
swapped. Two things used to break that, and together made the NaKo cohort read the patient-left CEA
2.4 degrees higher than the patient-right one (about 1.9 from the first, 1.1 from the second):

* the lateral-edge search band above the femoral head was ``1.5`` head radii on the image-left half
  and ``1.1`` on the image-right half (``CEA_UPPER_MARGIN_FACTOR_LEFT/_RIGHT``); it is now one
  ``CEA_UPPER_MARGIN_FACTOR`` for both;
* the isotropic femoral-head centre built its lateral cutoff from the floor of the top-slice
  centroid, which does not commute with the mirror (see
  ``tests/unit/test_femoral_head_isotropic_equivariance.py``), so the image-left half's centre sat
  ~0.4 voxel more lateral than the mirror of the image-right result and its CEA read ~1.1 degrees
  lower on the same hip.

Measured on the 60 NaKo reference segmentations after the fix, the reflected CEA reproduces the
swapped original on all 60 to within 3e-11 degrees, including the three whose top-slice centroid sits
exactly on a half voxel (where ``np.rint``'s round-half-to-even would not mirror). Before, it matched
on none: median 1.39 degrees, max 12.8 with the old per-side factors.
"""
import numpy as np
import pytest

from morphometry.image_io import reflect_image_left_right
from morphometry.measurements.hip import calculate_center_edge_angle

pytestmark = pytest.mark.needs_nako


def test_center_edge_angle_is_mirror_equivariant(nako_sample):
    """The CEA of the reflected hip is the original CEA with the two image sides swapped."""
    cea = calculate_center_edge_angle(nako_sample, 1, 3, isotropic=True, project=True)
    reflected = calculate_center_edge_angle(reflect_image_left_right(nako_sample), 1, 3, isotropic=True,
                                            project=True)
    np.testing.assert_allclose(reflected[::-1], cea, atol=1e-6)

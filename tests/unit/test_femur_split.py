"""Unit tests for the femoral cartilage medial/lateral split.

The ``Femur`` class splits the femoral cartilage into a left (image-side) and right
condyle along the intercondylar notch. These tests use small synthetic "horseshoe"
phantoms (two condyle bars joined anteriorly by a trochlea bridge, separated
posteriorly by a notch) so the correct split is known analytically, without needing
private patient data.
"""
import numpy as np
import nibabel as nib

from morphometry.image_io import Segmentation
from morphometry.cartilage.knee import Femur, _femoral_condyle_dividing_line

LABEL = 3


def _horseshoe(offset: int = 0) -> Segmentation:
    """
    Build a synthetic femoral-cartilage segmentation in LPI index convention.

    Axis 0 is left-right, axis 1 is anterior(low)-posterior(high), axis 2 is
    superior-inferior. Two condyle bars sit in the posterior region separated by an
    intercondylar notch; an anterior bridge (trochlea) joins them. With ``offset=0``
    the notch centre is at L-R index ~65; ``offset`` shifts the lateral (high L-R)
    condyle to move the notch.

    :param offset: Left-right shift (voxels) applied to the lateral condyle.
    :return: A ``Segmentation`` wrapping the phantom volume.
    """
    a = np.zeros((140, 140, 40), np.uint8)
    a[30:55, 40:120, 12:28] = LABEL                    # right condyle (low L-R)
    a[75 + offset:100 + offset, 40:120, 12:28] = LABEL  # left condyle (high L-R)
    a[30:100 + offset, 20:42, 12:28] = LABEL            # anterior trochlea bridge
    return Segmentation.from_nibabel(nib.Nifti1Image(a, np.eye(4)))


def test_dividing_line_is_near_vertical_and_centered():
    """The dividing axis should run along A-P (near-zero slope) through the notch."""
    cartilage = np.where(_horseshoe().array == LABEL, 1, 0)
    slope, intercept = _femoral_condyle_dividing_line(cartilage)
    assert abs(slope) < 0.05, f'expected near-vertical axis, got slope={slope}'
    assert 60 <= intercept <= 70, f'notch centre off, intercept={intercept}'


def test_condyles_split_cleanly():
    """A symmetric phantom splits into two clean, roughly balanced condyles."""
    fem = Femur(_horseshoe(), LABEL)
    left, right = fem.left_part, fem.right_part
    assert len(left) > 0 and len(right) > 0
    # No left-compartment voxel sits clearly on the right side and vice versa.
    assert (left[:, 0] < 55).sum() == 0
    assert (right[:, 0] > 75).sum() == 0
    balance = min(len(left), len(right)) / max(len(left), len(right))
    assert balance > 0.8, f'unbalanced split, balance={balance}'


def test_offset_condyle_tracks_notch():
    """When the lateral condyle is shifted, the split follows the moved notch."""
    fem = Femur(_horseshoe(offset=20), LABEL)
    left, right = fem.left_part, fem.right_part
    assert len(left) > 0 and len(right) > 0
    # left_part is the high-L-R (image-left) condyle.
    assert left[:, 0].mean() > right[:, 0].mean()

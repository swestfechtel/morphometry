"""Unit tests for cartilage-mask outlier removal in the knee cartilage classes.

Segmentations occasionally contain small mislabelled blobs or detached voxel clusters
that corrupt the reconstructed meshes. ``Tibia``/``Femur`` clean their extracted
cartilage mask (small connected components removed) before further processing.
"""
import numpy as np
import nibabel as nib

from morphometry.image_io import Segmentation
from morphometry.cartilage.knee import Tibia, Femur, _remove_mask_outliers

LABEL = 3


def test_remove_mask_outliers_keeps_large_components_drops_small():
    """Two large components are kept; a small detached blob is dropped."""
    mask = np.zeros((60, 60, 20), np.uint8)
    mask[5:25, 5:25, 5:15] = 1     # large component A
    mask[35:55, 35:55, 5:15] = 1   # large component B (comparable size)
    mask[0:2, 0:2, 0:2] = 1        # tiny detached outlier blob
    cleaned = _remove_mask_outliers(mask, threshold_ratio=0.1)
    assert cleaned[5, 5, 5] == 1 and cleaned[35, 35, 5] == 1   # both plates survive
    assert cleaned[0, 0, 0] == 0                                # outlier removed
    # exactly the two large components remain
    assert cleaned.sum() == 2 * (20 * 20 * 10)


def test_remove_mask_outliers_disabled():
    """A threshold of 0 leaves the mask untouched."""
    mask = np.zeros((10, 10, 10), np.uint8)
    mask[0, 0, 0] = 1
    mask[5:8, 5:8, 5:8] = 1
    cleaned = _remove_mask_outliers(mask, threshold_ratio=0.0)
    assert np.array_equal(cleaned, mask)


def _horseshoe_with_outlier():
    """A femoral-cartilage phantom (one horseshoe) plus a small detached blob."""
    a = np.zeros((140, 140, 40), np.uint8)
    a[30:55, 40:120, 12:28] = LABEL                 # right condyle
    a[75:100, 40:120, 12:28] = LABEL                # left condyle
    a[30:100, 20:42, 12:28] = LABEL                 # anterior trochlea bridge
    a[10:16, 10:16, 12:18] = LABEL                  # detached outlier blob (~1% of volume)
    return Segmentation.from_nibabel(nib.Nifti1Image(a, np.eye(4)))


def test_femur_excludes_outlier_blob_from_point_cloud():
    """The detached blob is absent from the femoral point cloud after construction."""
    fem = Femur(_horseshoe_with_outlier(), LABEL)
    pc = fem.point_cloud
    in_blob = (pc[:, 0] < 20) & (pc[:, 1] < 20)
    assert in_blob.sum() == 0, 'outlier blob should have been removed before processing'
    assert len(pc) > 0


def test_tibia_excludes_outlier_blob_from_point_cloud():
    """The tibia's mask is cleaned in get_surface_points before the point cloud is built."""
    a = np.zeros((60, 60, 20), np.uint8)
    a[5:25, 5:25, 5:15] = 4        # tibial plateau A
    a[35:55, 5:25, 5:15] = 4       # tibial plateau B
    a[0:2, 50:52, 0:2] = 4         # tiny detached outlier blob
    seg = Segmentation.from_nibabel(nib.Nifti1Image(a, np.eye(4)))
    tibia = Tibia(seg, 4)
    tibia.get_surface_points()
    pc = tibia.point_cloud
    assert ((pc[:, 0] < 3) & (pc[:, 1] > 49)).sum() == 0   # outlier gone
    assert len(pc) == 2 * (20 * 20 * 10)                    # both plateaus retained

"""Pure numpy mask combination (no matplotlib).

Kept separate from :mod:`api.domain.encode` so the API request path can build a
combined label mask without importing matplotlib (which ``encode`` pulls in for
PNG rendering). The worker uses this when persisting the combined mask; the API
uses it as a lazy fallback for examinations processed before the combined mask
existed.
"""
import numpy as np

from morphometry.image_io import Segmentation


def combine_region_masks(hip_mask: Segmentation, knee_mask: Segmentation,
                         ankle_mask: Segmentation) -> np.ndarray:
    """Concatenate the three region segmentations into one labelled volume.

    The regions are stacked along the inferior (z / axis 2) axis in the same
    order the ``transformed`` source volume is built (hip → knee → ankle), so the
    result shares that volume's voxel grid. The ankle mask is relabelled
    (tibia 1→2, fibula 2→3) so each region's labels stay distinct in the combined
    volume; hip and knee masks are concatenated as-is. This mirrors the overlay
    construction in :func:`api.domain.encode.encode_torsion_images`.

    :param hip_mask: Hip region segmentation (LPI, same in-plane shape as the others).
    :param knee_mask: Knee region segmentation.
    :param ankle_mask: Ankle region segmentation (tibia/fibula relabelled here).
    :return: The combined integer label volume as a numpy array.
    """
    relabelled_ankle = ankle_mask.array.copy()
    relabelled_ankle = np.where(relabelled_ankle == 2, 3, relabelled_ankle)
    relabelled_ankle = np.where(relabelled_ankle == 1, 2, relabelled_ankle)
    return np.concatenate((hip_mask.array, knee_mask.array, relabelled_ankle), axis=2)

"""Small pure-geometry helpers shared across the measurement code.

These factor out patterns that were previously copy-pasted across the region
modules: folding an unsigned vector angle into the acute or obtuse range, the
image-side left/right split at ``shape[0] // 2``, the sagittal-coordinate mirror
used when processing the right image side, and wrapping a 2D slice centroid into
a 3D point.

Orientation reminder (LPI): axis 0 = medial-lateral (toward patient Left),
axis 1 = anterior-posterior (toward Posterior), axis 2 = superior-inferior
(toward Inferior). "left"/"right" everywhere refers to the *image* side, not the
patient side.
"""
from typing import Literal, Tuple

import numpy as np
from scipy.ndimage import center_of_mass

Side = Literal["left", "right"]


def validate_side(side: str) -> None:
    """Assert that ``side`` is one of the allowed image-side values."""
    assert side in ("left", "right"), 'Side must be either "left" or "right"'


def fold_to_acute(angle: float) -> float:
    """Fold an angle in [0, 180] degrees into the acute range [0, 90].

    Replaces the repeated ``if angle > 90: angle = 180 - angle`` idiom used by the
    torsion, knee-rotation and joint-line-convergence calculations.
    :param angle: An unsigned angle in degrees.
    :return: The equivalent acute angle (``min(angle, 180 - angle)``).
    """
    return 180.0 - angle if angle > 90.0 else angle


def fold_to_obtuse(angle: float) -> float:
    """Fold an angle in [0, 180] degrees into the obtuse range [90, 180].

    Replaces the repeated ``angle = 180 - angle if angle < 90 else angle`` idiom
    used by the CCD calculation.
    :param angle: An unsigned angle in degrees.
    :return: The equivalent obtuse angle (``max(angle, 180 - angle)``).
    """
    return 180.0 - angle if angle < 90.0 else angle


def split_left_right(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a both-sides mask into (left, right) halves along the sagittal axis.

    The split point is ``array.shape[0] // 2``. "left"/"right" is the image side.
    :param array: A 3D mask whose first axis is medial-lateral.
    :return: The left and right image halves (views into ``array``).
    """
    half = array.shape[0] // 2
    return array[:half], array[half:]


def mirror_sagittal_coordinate(coord: float, half_extent: int) -> float:
    """Mirror a sagittal coordinate about the image midline.

    Reproduces the exact arithmetic ``coord + (half_extent - coord) * 2`` used when
    flipping the right image side so it can be processed with left-side logic.
    :param coord: The sagittal (axis-0) coordinate to mirror.
    :param half_extent: ``array.shape[0] // 2`` of the array being mirrored.
    :return: The mirrored coordinate.
    """
    return coord + (half_extent - coord) * 2


def slice_centroid_to_point(mask_2d: np.ndarray, layer: int) -> np.ndarray:
    """Compute the centre of mass of a 2D slice and embed it as a 3D point.

    :param mask_2d: A binary 2D slice.
    :param layer: The axis-2 (slice) index to attach as the third coordinate.
    :return: ``np.array([centroid_x, centroid_y, layer])``.
    """
    cx, cy = center_of_mass(mask_2d)
    return np.array([cx, cy, layer])

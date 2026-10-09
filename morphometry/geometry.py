"""Small pure-geometry helpers shared across the measurement code.

These factor out patterns that were previously copy-pasted across the region
modules: folding an unsigned vector angle into the acute or obtuse range, the
image-side left/right split at ``shape[0] // 2``, the sagittal-coordinate mirror
used when processing the right image side, and wrapping a 2D slice centroid into
a 3D point. They also hold the two helpers the mirror-equivariant measurements
share: the index mirror ``x -> n - 1 - x`` that maps a landmark found on a
reflected image half back into the caller's frame, and the posterior-condylar
endpoint ordering that keeps such a landmark pair reported in the target frame's
order.

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


def signed_axial_angle(reference_line: np.ndarray, direction: np.ndarray, side: str) -> float:
    """Signed axial-plane angle from an undirected reference line to a direction, for torsion measurements.

    The angle is measured in the (x, y) index plane from ``reference_line`` to ``direction`` and folded
    into (-90, 90]. On a left image half (a patient-right limb) it is positive for a rotation from
    medial (+x) towards anterior (-y), which is the same sense as from lateral (-x) towards posterior
    (+y): ``atan2(-(r_x d_y - r_y d_x), r . d)``. ``side='right'`` (the mirror-image limb) negates the x
    components first, so mirrored inputs give the same angle.

    This is the sign of femoral antetorsion (neck turned medial -> anterior relative to the posterior
    condylar line) and of external tibial torsion (distal tibia -> fibula line turned lateral ->
    posterior relative to the posterior tibial condylar line). Only the directions of the two lines
    enter, so the angle is invariant to an in-plane rotation of the whole limb. The reference line is
    undirected (the fold makes its endpoint order irrelevant). The angle is computed from voxel
    indices, so it assumes equal in-plane x/y spacing.
    :param reference_line: A vector along the reference line (only x and y are used).
    :param direction: The measured direction vector (only x and y are used).
    :param side: Image side, 'left' or 'right'.
    :return: The signed angle in degrees, in (-90, 90].
    """
    reference = np.asarray(reference_line, dtype=float)[:2].copy()
    measured = np.asarray(direction, dtype=float)[:2].copy()
    if side == 'right':
        reference[0], measured[0] = -reference[0], -measured[0]

    cross = -(reference[0] * measured[1] - reference[1] * measured[0])
    dot = reference[0] * measured[0] + reference[1] * measured[1]
    angle = float(np.degrees(np.arctan2(cross, dot)))
    # the reference line is undirected: fold into (-90, 90]
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle


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


def mirror_index_x(point: np.ndarray, width: int) -> np.ndarray:
    """
    Map a voxel index between an image and its left-right reflection: ``x -> width - 1 - x``.

    This is the correct index mirror, and the one the equivariant torsion / CCD paths use to map
    landmarks found on a reflected half back into the caller's frame. Exact for integer and
    continuous indices alike, and its own inverse. Unlike
    :func:`mirror_sagittal_coordinate`, which reproduces the legacy off-by-one arithmetic, it
    mirrors about the array's true centre.
    :param point: A length-3 index (integer or float); it is not modified.
    :param width: The size of the image along the first index axis.
    :return: A copy of ``point`` with the first coordinate mirrored (same dtype).
    """
    mirrored = np.array(point, copy=True)
    mirrored[0] = width - 1 - mirrored[0]
    return mirrored


def order_condylar_points(start: np.ndarray, end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Order the two endpoints of a posterior-condylar line so the start has the larger x index.

    In LPI, x increases towards the patient's left, so the start is the MEDIAL point on a left
    image half (a patient-right limb, medial = +x) and the lateral point on a right image half.
    The condylar line is undirected -- the order only fixes how the landmarks are reported, not
    the femoral or tibial torsion angle composed from them. Shared by both so that a line found
    on a reflected half and mirrored back is reported in the target frame's order.
    :param start: One posterior-condyle reference point.
    :param end: The other posterior-condyle reference point.
    :return: ``(start, end)`` with ``start[0] >= end[0]``.
    """
    if start[0] < end[0]:
        return end, start
    return start, end


def slice_centroid_to_point(mask_2d: np.ndarray, layer: int) -> np.ndarray:
    """Compute the centre of mass of a 2D slice and embed it as a 3D point.

    :param mask_2d: A binary 2D slice.
    :param layer: The axis-2 (slice) index to attach as the third coordinate.
    :return: ``np.array([centroid_x, centroid_y, layer])``.
    """
    cx, cy = center_of_mass(mask_2d)
    return np.array([cx, cy, layer])

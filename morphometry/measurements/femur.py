"""Femur measurement functions: femoral torsion (MRI and whole-leg CT).

The proximal reference line (femoral head centre -> neck axis, via Lee / Murphy /
Tomczak) and the distal posterior-condylar line are computed by ``get_*`` helpers
in ``morphometry.femur`` and ``morphometry.knee``; this module only combines them
into the torsion angle. The angle logic is shared between the MRI and CT variants
through :func:`_femoral_torsion_angle`.

Landmarks are exposed separately via :func:`get_femoral_torsion_landmarks` so the
``calculate_*`` functions always return just the angle (no flag-dependent arity).

"left"/"right" refers to the image side, not the patient side. The landmark searches are written
for a left image half; a right half is measured as the left-right reflection of a left one and its
landmarks are mapped back (``x -> n - 1 - x``), so both sides are mirror-equivariant by construction
(see :func:`_mri_torsion`).
"""
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from morphometry.image_io import Image, Segmentation, reflect_image_left_right
from morphometry.femur import _mirror_index_x, get_proximal_reference_line, get_proximal_reference_line_ct
from morphometry.knee import get_knee_reference_line
from morphometry.utils import calculate_angle_between_vectors
from morphometry import geometry as G


def _order_knee_points(knee_start: np.ndarray, knee_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Order the posterior-condylar endpoints so the start has the larger x index (start[0] >= end[0]).

    In LPI, x increases towards the patient's left, so the start is the MEDIAL point on a left image
    half (a patient-right femur, medial = +x) and the lateral point on a right image half. The order
    only fixes how the landmarks are reported; the torsion angle does not depend on it.
    :param knee_start: One posterior-condyle reference point.
    :param knee_end: The other posterior-condyle reference point.
    :return: ``(start, end)`` with ``start[0] >= end[0]``.
    """
    if knee_start[0] < knee_end[0]:
        return knee_end, knee_start
    return knee_start, knee_end


def _proximal_angle(hip_start: np.ndarray, hip_end: np.ndarray, side: str) -> float:
    """Acute angle between the proximal (neck) reference line and the medial-lateral axis."""
    proximal_line = hip_end - hip_start
    # the 'right' axis is still used by calculate_anteversion (landmarks mapped back to the right
    # half); femoral torsion itself only composes with 'left', in the reflected frame
    x = np.array([-1, 0, 0]) if side == 'left' else np.array([1, 0, 0])
    return G.fold_to_acute(calculate_angle_between_vectors(proximal_line, x))


def _femoral_torsion_angle(hip_start: np.ndarray, hip_end: np.ndarray,
                           knee_start: np.ndarray, knee_end: np.ndarray, side: str) -> float:
    """Compose the femoral torsion from the neck axis and the posterior condylar line.

    The torsion is the signed angle, in the axial (x, y) plane, from the posterior condylar line to
    the femoral neck axis, positive for antetorsion (neck turned anteriorly relative to the
    condyles) and negative for retrotorsion. Only the in-plane directions of the two lines enter, so
    the result is invariant to an in-plane rotation of the whole leg in the scanner.

    Sign derivation (LPI index frame: x increases towards the patient's left, y posteriorly). A left
    image half holds a patient-right femur, so medial is +x and anterior is -y. With the neck
    directed neck point -> head centre (``n``, pointing medially) and ``c`` along the condylar line,
    the angle measured from medial towards anterior in the (x, anterior) plane is
    ``atan2(c_x * (-n_y) - (-c_y) * n_x, c . n) = atan2(-(c_x n_y - c_y n_x), c . n)``. The
    condylar line has no direction, so the angle is folded into (-90, 90] (equivalently, ``c`` is
    taken pointing the same way as ``n``); the endpoint order of the condylar points is irrelevant.
    A right image half holds the mirror-image femur (medial = -x), so ``side='right'`` negates the
    x components first.

    This replaces a composition that folded both lines to acute angles against the image x-axis
    and combined them by the signs of their y-offsets. That equalled the signed angle only while the
    femoral head lay anterior to the neck point in the image; otherwise it returned the negated
    torsion (a jump of ``2 * |torsion|`` as the neck crossed the image x-axis) and, past 90 degrees,
    a wrong magnitude.

    ``hip_start`` is expected to already be aligned to ``hip_end`` in the axial (z) coordinate by
    the caller; z is ignored here. The angle is computed from voxel indices, so it assumes equal
    in-plane x/y spacing (as did the former composition; true for the Augsburg MRI and typical CT).
    :param hip_start: Proximal reference line start (femoral head centre).
    :param hip_end: Proximal reference line end (neck axis point).
    :param knee_start: One posterior-condyle reference point.
    :param knee_end: The other posterior-condyle reference point.
    :param side: Image side, 'left' or 'right'. Production composes with 'left' only (a right half
        is reflected and measured in that frame, see :func:`_mri_torsion` / :func:`_ct_torsion`);
        'right' is the mirror of that convention, pinned by
        ``tests/unit/test_torsion_angle_equivariance.py``.
    :return: The femoral torsion angle in degrees, in (-90, 90]; negative means retrotorsion.
    """
    neck = np.asarray(hip_start, dtype=float)[:2] - np.asarray(hip_end, dtype=float)[:2]
    condyles = np.asarray(knee_start, dtype=float)[:2] - np.asarray(knee_end, dtype=float)[:2]
    if side == 'right':
        neck[0], condyles[0] = -neck[0], -condyles[0]

    cross = -(condyles[0] * neck[1] - condyles[1] * neck[0])  # positive: neck anterior of the condylar line
    dot = condyles[0] * neck[0] + condyles[1] * neck[1]
    angle = float(np.degrees(np.arctan2(cross, dot)))
    # the condylar line is undirected: fold into (-90, 90]
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle


def _mirror_landmarks(hip_start, hip_end, knee_start, knee_end, hip_width: int, knee_width: int):
    """Map torsion landmarks between a left-right mirrored pair of images and re-order the knee points.

    ``x -> width - 1 - x`` with the hip image's width for the ``hip_*`` points and the knee image's
    width for the ``knee_*`` points; the condylar endpoints are then re-ordered with
    :func:`_order_knee_points` in the target frame.
    :return: ``(hip_start, hip_end, knee_start, knee_end)`` in the mirrored frame.
    """
    hip_start, hip_end = _mirror_index_x(hip_start, hip_width), _mirror_index_x(hip_end, hip_width)
    knee_start, knee_end = _order_knee_points(_mirror_index_x(knee_start, knee_width),
                                              _mirror_index_x(knee_end, knee_width))
    return hip_start, hip_end, knee_start, knee_end


def _mri_torsion(hip_image: Image, knee_mask: np.ndarray, side: str, method: str,
                 segmentation_label: int, x_ratio: float, isotropic: bool):
    """Femoral torsion (MRI) and its landmarks in the frame of the given image half.

    A right image half is measured as the left-right reflection of a left half: the landmarks
    are found by the left-side code on the reflected hip image and the reversed knee mask, the
    angle is composed there with the left-side conventions, and only the landmarks are mapped
    back (``x -> n - 1 - x``). The result is mirror-equivariant by construction, including the
    knee condylar line (whose search is not itself mirror-symmetric) and ties in the reported
    condylar ordering (the angle itself no longer depends on that order).
    :return: ``(angle, (hip_start, hip_end, knee_start, knee_end))``.
    """
    if side == 'right':
        points = _mri_torsion_reference_points(reflect_image_left_right(hip_image), knee_mask[::-1], 'left', method,
                                               segmentation_label, x_ratio, isotropic)
        angle = _femoral_torsion_angle(*points, 'left')
        return angle, _mirror_landmarks(*points, hip_image.array.shape[0], knee_mask.shape[0])
    points = _mri_torsion_reference_points(hip_image, knee_mask, side, method, segmentation_label, x_ratio, isotropic)
    return _femoral_torsion_angle(*points, side), points


def _mri_torsion_reference_points(hip_image: Image, knee_mask: np.ndarray, side: str, method: str,
                                  segmentation_label: int, x_ratio: float, isotropic: bool):
    """Resolve the MRI proximal and distal reference points (axially aligned, ordered)."""
    proximal = get_proximal_reference_line(hip_image, side=side, method=method,
                                           segmentation_label=segmentation_label, x_ratio=x_ratio, isotropic=isotropic)
    hip_start, hip_end = proximal[0], proximal[1]
    hip_start[2] = hip_end[2]  # align axial coordinate

    knee_bin = np.where(knee_mask == segmentation_label, 1, 0)
    _, knee_start, knee_end = get_knee_reference_line(knee_bin, bone='femur', segmentation_label=segmentation_label)
    knee_start, knee_end = _order_knee_points(knee_start, knee_end)
    return hip_start, hip_end, knee_start, knee_end


def calculate_femoral_torsion(hip_image: Image, knee_mask: np.ndarray, side: str = 'left', method: str = 'lee',
                              segmentation_label: int = 1, x_ratio: float = 1., isotropic: bool = False,
                              plot: bool | plt.Axes = False) -> float:
    """
    Calculate the femoral torsion from an MRI hip + distal-femur segmentation.
    :param hip_image: An Image object of the proximal-femur segmentation mask.
    :param knee_mask: A segmentation mask of the distal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param method: Proximal reference line method ('lee', 'murphy' or 'tomczak').
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param x_ratio: Correction factor for slice thickness.
    :param isotropic: Whether the image has isotropic voxels.
    :param plot: A sequence of (at least two) matplotlib Axes to draw the reference lines on, or False.
    :return: The femoral torsion in degrees.
    """
    G.validate_side(side)
    assert method in ['lee', 'murphy', 'tomczak'], 'method must be "lee", "murphy" or "tomczak"'

    angle, (hip_start, hip_end, knee_start, knee_end) = _mri_torsion(
        hip_image, knee_mask, side, method, segmentation_label, x_ratio, isotropic)

    if plot is not False:
        knee_bin = np.where(knee_mask == segmentation_label, 1, 0)
        _plot_torsion(plot, hip_image.array, knee_bin, hip_start, hip_end, knee_start, knee_end, angle)
    return angle


def calculate_femoral_torsion_ct(femur_image: Segmentation, knee_image: Segmentation, side: str = 'left',
                                 segmentation_label: int = 1, plot: bool | plt.Axes = False) -> float:
    """
    Calculate the femoral torsion from a whole-leg CT segmentation.
    :param femur_image: A Segmentation of the femur (whole-leg) mask.
    :param knee_image: A Segmentation of the distal-femur (knee) mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param plot: A sequence of matplotlib Axes to draw the reference lines on, or False.
    :return: The femoral torsion in degrees.
    """
    G.validate_side(side)

    angle, (hip_start, hip_end, knee_start, knee_end) = _ct_torsion(femur_image, knee_image, side, segmentation_label)

    if plot is not False:
        knee_bin = np.where(knee_image.array == segmentation_label, 1, 0)
        _plot_torsion(plot, femur_image.array, knee_bin, hip_start, hip_end, knee_start, knee_end, angle)
    return angle


def _ct_torsion(femur_image: Segmentation, knee_image: Segmentation, side: str, segmentation_label: int):
    """Femoral torsion (whole-leg CT) and its landmarks in the frame of the given image half.

    Right image halves are measured as the left-right reflection of a left half, exactly as in
    :func:`_mri_torsion`; the reflection (not an index mirror) matters here because the CT femoral
    head centre is located in physical space.
    :return: ``(angle, (hip_start, hip_end, knee_start, knee_end))``.
    """
    if side == 'right':
        points = _ct_torsion_reference_points(reflect_image_left_right(femur_image),
                                              reflect_image_left_right(knee_image), 'left', segmentation_label)
        angle = _femoral_torsion_angle(*points, 'left')
        return angle, _mirror_landmarks(*points, femur_image.array.shape[0], knee_image.array.shape[0])
    points = _ct_torsion_reference_points(femur_image, knee_image, side, segmentation_label)
    return _femoral_torsion_angle(*points, side), points


def _ct_torsion_reference_points(femur_image: Segmentation, knee_image: Segmentation, side: str,
                                 segmentation_label: int):
    """Resolve the CT proximal and distal reference points (axially aligned, ordered)."""
    hip_start, hip_end = get_proximal_reference_line_ct(femur_image, side=side, segmentation_label=segmentation_label)
    hip_start[2] = hip_end[2]  # align axial coordinate

    knee_bin = np.where(knee_image.array == segmentation_label, 1, 0)
    _, knee_start, knee_end = get_knee_reference_line(knee_bin, bone='femur', segmentation_label=segmentation_label)
    knee_start, knee_end = _order_knee_points(knee_start, knee_end)
    return hip_start, hip_end, knee_start, knee_end


def get_femoral_torsion_landmarks_ct(femur_image: Segmentation, knee_image: Segmentation, side: str = 'left',
                                     segmentation_label: int = 1) -> dict:
    """
    Return the femoral-torsion reference landmarks (whole-leg CT) as a dict.

    The CT counterpart of :func:`get_femoral_torsion_landmarks`: the points are voxel indices
    (``hip_*`` in ``femur_image``, ``knee_*`` in ``knee_image``), axially aligned and ordered
    exactly as used by :func:`calculate_femoral_torsion_ct`.
    :param femur_image: A Segmentation of the femur (whole-leg) mask.
    :param knee_image: A Segmentation of the distal-femur (knee) mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: ``{'hip_start', 'hip_end', 'knee_start', 'knee_end'}`` as numpy arrays.
    """
    G.validate_side(side)
    _, (hip_start, hip_end, knee_start, knee_end) = _ct_torsion(femur_image, knee_image, side, segmentation_label)
    return {'hip_start': hip_start, 'hip_end': hip_end, 'knee_start': knee_start, 'knee_end': knee_end}


def get_femoral_torsion_landmarks(hip_image: Image, knee_mask: np.ndarray, side: str = 'left', method: str = 'lee',
                                  segmentation_label: int = 1, x_ratio: float = 1., isotropic: bool = False) -> dict:
    """
    Return the femoral-torsion reference landmarks (MRI) as a dict.

    Provided separately from :func:`calculate_femoral_torsion` so the calculation
    keeps a single return type. The points are axially aligned and ordered exactly
    as used in the angle computation.
    :return: ``{'hip_start', 'hip_end', 'knee_start', 'knee_end'}`` as numpy arrays.
    """
    G.validate_side(side)
    _, (hip_start, hip_end, knee_start, knee_end) = _mri_torsion(
        hip_image, knee_mask, side, method, segmentation_label, x_ratio, isotropic)
    return {'hip_start': hip_start, 'hip_end': hip_end, 'knee_start': knee_start, 'knee_end': knee_end}


def _plot_torsion(axes, hip_array: np.ndarray, knee_bin: np.ndarray, hip_start, hip_end,
                  knee_start, knee_end, angle: float) -> None:
    """Draw the proximal and distal reference lines onto a provided sequence of Axes."""
    hip_layer = int(hip_end[2])
    knee_layer = int(knee_start[2])
    axes[0].imshow(np.where(hip_array[:, :, hip_layer] == 0, np.nan, hip_array[:, :, hip_layer]).T)
    axes[0].plot([hip_start[0], hip_end[0]], [hip_start[1], hip_end[1]], 'r')
    axes[0].set_title(f'Torsion: {angle:.2f}°')
    axes[1].imshow(knee_bin[:, :, knee_layer].T)
    axes[1].plot([knee_start[0], knee_end[0]], [knee_start[1], knee_end[1]], 'r')

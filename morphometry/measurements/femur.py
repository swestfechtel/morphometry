"""Femur measurement functions: femoral torsion (MRI and whole-leg CT).

The proximal reference line (femoral head centre -> neck axis, via Lee / Murphy /
Tomczak) and the distal posterior-condylar line are computed by ``get_*`` helpers
in ``morphometry.femur`` and ``morphometry.knee``; this module only combines them
into the torsion angle. The angle logic is shared between the MRI and CT variants
through :func:`_femoral_torsion_angle`.

Landmarks are exposed separately via :func:`get_femoral_torsion_landmarks` so the
``calculate_*`` functions always return just the angle (no flag-dependent arity).

"left"/"right" refers to the image side, not the patient side.
"""
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from morphometry.image_io import Image, Segmentation
from morphometry.femur import get_proximal_reference_line, get_proximal_reference_line_ct
from morphometry.knee import get_knee_reference_line
from morphometry.utils import calculate_angle_between_vectors
from morphometry import geometry as G


def _order_knee_points(knee_start: np.ndarray, knee_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Order the posterior-condylar endpoints so the start is lateral to the end (start[0] >= end[0])."""
    if knee_start[0] < knee_end[0]:
        return knee_end, knee_start
    return knee_start, knee_end


def _proximal_angle(hip_start: np.ndarray, hip_end: np.ndarray, side: str) -> float:
    """Acute angle between the proximal (neck) reference line and the medial-lateral axis."""
    proximal_line = hip_end - hip_start
    x = np.array([-1, 0, 0]) if side == 'left' else np.array([1, 0, 0])
    return G.fold_to_acute(calculate_angle_between_vectors(proximal_line, x))


def _femoral_torsion_angle(hip_start: np.ndarray, hip_end: np.ndarray,
                           knee_start: np.ndarray, knee_end: np.ndarray, side: str) -> float:
    """Combine the proximal neck angle and distal condylar angle into the femoral torsion.

    The proximal and distal angles are each folded into [0, 90]; they are added when
    the neck and condylar lines point to opposite anterior-posterior sides and
    subtracted otherwise. ``hip_start`` is expected to already be aligned to
    ``hip_end`` in the axial (z) coordinate by the caller.
    :param hip_start: Proximal reference line start (femoral head centre).
    :param hip_end: Proximal reference line end (neck axis point).
    :param knee_start: One posterior-condyle reference point.
    :param knee_end: The other posterior-condyle reference point.
    :param side: Image side, 'left' or 'right'.
    :return: The femoral torsion angle in degrees.
    """
    knee_start, knee_end = _order_knee_points(knee_start, knee_end)

    proximal_angle = _proximal_angle(hip_start, hip_end, side)
    proximal_orientation = hip_end[1] - hip_start[1]  # positive if hip_end is posterior to hip_start

    distal_line = knee_end - knee_start
    distal_angle = G.fold_to_acute(calculate_angle_between_vectors(distal_line, np.array([-1, 0, 0])))
    distal_orientation = (knee_end[1] - knee_start[1]) if side == 'left' else (knee_start[1] - knee_end[1])

    if np.sign(proximal_orientation) != np.sign(distal_orientation):
        return proximal_angle + distal_angle
    return proximal_angle - distal_angle


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

    hip_start, hip_end, knee_start, knee_end = _mri_torsion_reference_points(
        hip_image, knee_mask, side, method, segmentation_label, x_ratio, isotropic)
    angle = _femoral_torsion_angle(hip_start, hip_end, knee_start, knee_end, side)

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

    hip_start, hip_end = get_proximal_reference_line_ct(femur_image, side=side, segmentation_label=segmentation_label)
    hip_start[2] = hip_end[2]  # align axial coordinate

    knee_bin = np.where(knee_image.array == segmentation_label, 1, 0)
    _, knee_start, knee_end = get_knee_reference_line(knee_bin, bone='femur', segmentation_label=segmentation_label)
    knee_start, knee_end = _order_knee_points(knee_start, knee_end)

    angle = _femoral_torsion_angle(hip_start, hip_end, knee_start, knee_end, side)

    if plot is not False:
        _plot_torsion(plot, femur_image.array, knee_bin, hip_start, hip_end, knee_start, knee_end, angle)
    return angle


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
    hip_start, hip_end, knee_start, knee_end = _mri_torsion_reference_points(
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

"""Tibia measurement functions: tibial torsion.

Combines the proximal posterior-condylar line (``get_knee_reference_line``) and the
distal tibia-fibula line (``get_distal_reference_line``) into the tibial torsion
angle. Landmarks are exposed separately via :func:`get_tibial_torsion_landmarks`.

"left"/"right" refers to the image side, not the patient side. As for the femoral torsion, a right
image half is measured as the left-right reflection of a left one: the landmark searches run on the
reversed masks with the left-side conventions and the landmarks are mapped back (``x -> n - 1 - x``),
so the two sides are mirror-equivariant by construction (see :func:`_tibial_torsion_points`). Do not
reintroduce a ``side == 'right'`` branch into the landmark search -- add left-side logic and let the
reflection handle the right.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from morphometry.image_io import Image, mask_and_spacing
from morphometry.knee import get_knee_reference_line
from morphometry.tibia import get_distal_reference_line
from morphometry import geometry as G


def _tibial_torsion_points(knee_mask: np.ndarray, ankle_mask: np.ndarray, tibia_label_knee: int,
                           tibia_label_ankle: int, fibula_label: int, side: str,
                           knee_spacing: Optional[Sequence[float]] = None,
                           ankle_spacing: Optional[Sequence[float]] = None):
    """Tibial torsion and its landmarks in the frame of the given image half.

    A right image half is measured as the left-right reflection of a left half: the landmarks are
    found by the left-side code on the reversed knee and ankle masks, the angle is composed there
    with the left-side conventions, and only the landmarks are mapped back
    (:func:`morphometry.geometry.mirror_index_x`, each point with its own mask's width, condylar
    endpoints re-ordered in the target frame). The result is mirror-equivariant by construction,
    which the landmark search is not: the knee condylar search rotates the slice about the image
    centre and rounds. Measuring the two sides directly disagreed by a median 0.8-1.1 degrees on
    the Augsburg hip/knee/ankle MRI and 1.7 degrees on the NMDID whole-leg CT, with a tail into the
    tens of degrees on the legs whose notch search fails -- that tail belonged to the old no-notch
    fallback, which chose its frame from the section's longest contour chord and could pick a
    different one on a half and on its reflection (see ``knee.get_condylar_divider``).
    :param knee_mask: A 3D segmentation mask of the knee.
    :param ankle_mask: A 3D segmentation mask of the ankle.
    :param tibia_label_knee: The label of the tibia at knee level.
    :param tibia_label_ankle: The label of the tibia at ankle level.
    :param fibula_label: The label of the fibula.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param knee_spacing: Voxel spacing of ``knee_mask``, or ``None`` to leave it unchecked.
    :param ankle_spacing: Voxel spacing of ``ankle_mask``, or ``None`` to leave it unchecked.
        Both masks are bare arrays, so the equal-in-plane-spacing precondition of the index-space
        composition (:func:`_tibial_torsion_angle`) can only be checked when the caller passes
        the spacings.
    :return: ``(angle, (knee_start, knee_end, ankle_start, ankle_end))``, the landmarks as voxel
    indices of the masks as given.
    """
    G.validate_in_plane_isotropy(knee_spacing, 'tibial torsion knee mask')
    G.validate_in_plane_isotropy(ankle_spacing, 'tibial torsion ankle mask')
    if side == 'right':
        knee_start, knee_end, ankle_start, ankle_end = _tibial_torsion_reference_points(
            knee_mask[::-1], ankle_mask[::-1], tibia_label_knee, tibia_label_ankle, fibula_label)
        angle = _tibial_torsion_angle(knee_start, knee_end, ankle_start, ankle_end, 'left')
        knee_width, ankle_width = knee_mask.shape[0], ankle_mask.shape[0]
        # the condylar line is undirected, so only its reported order has to be restored; the
        # ankle points are labelled (tibia centroid, fibula centroid) and keep their roles
        knee_start, knee_end = G.order_condylar_points(G.mirror_index_x(knee_start, knee_width),
                                                       G.mirror_index_x(knee_end, knee_width))
        return angle, (knee_start, knee_end,
                       G.mirror_index_x(ankle_start, ankle_width), G.mirror_index_x(ankle_end, ankle_width))

    points = _tibial_torsion_reference_points(knee_mask, ankle_mask, tibia_label_knee, tibia_label_ankle, fibula_label)
    return _tibial_torsion_angle(*points, side), points


def _tibial_torsion_reference_points(knee_mask: np.ndarray, ankle_mask: np.ndarray, tibia_label_knee: int,
                                     tibia_label_ankle: int, fibula_label: int):
    """Resolve the proximal (knee) and distal (ankle) tibial reference points, ordered.

    Written for a left image half; a right half reaches it through the reflection in
    :func:`_tibial_torsion_points`.
    :param knee_mask: A 3D segmentation mask of the knee.
    :param ankle_mask: A 3D segmentation mask of the ankle.
    :param tibia_label_knee: The label of the tibia at knee level.
    :param tibia_label_ankle: The label of the tibia at ankle level.
    :param fibula_label: The label of the fibula.
    :return: ``(knee_start, knee_end, ankle_start, ankle_end)``; the condylar endpoints ordered by
    descending x, the ankle points as ``(tibia centroid, fibula centroid)``.
    """
    knee_bin = np.where(knee_mask == tibia_label_knee, 1, 0)
    _, knee_start, knee_end = get_knee_reference_line(knee_bin, 'tibia')
    knee_start, knee_end = G.order_condylar_points(knee_start, knee_end)
    _, ankle_start, ankle_end = get_distal_reference_line(ankle_mask, tibia_label_ankle, fibula_label)
    return knee_start, knee_end, ankle_start, ankle_end


def _resolve_masks(knee_mask, knee_spacing, ankle_mask, ankle_spacing):
    """Accept ``Image``/``Segmentation`` or bare arrays for the two masks and resolve the spacings.

    The landmark searches work on arrays, so an ``Image`` is unwrapped here, at the public entry
    point, before anything else sees it; its voxel spacing is kept so the equal-in-plane-spacing
    precondition of the index-space composition can be checked without the caller passing it
    separately. An explicitly given spacing takes precedence over the image's.
    :param knee_mask: The knee mask, as an ``Image`` / ``Segmentation`` or an ``np.ndarray``.
    :param knee_spacing: An explicitly supplied knee spacing, or ``None``.
    :param ankle_mask: The ankle mask, same two forms.
    :param ankle_spacing: An explicitly supplied ankle spacing, or ``None``.
    :return: ``(knee_array, knee_spacing, ankle_array, ankle_spacing)``, each spacing resolved to
        the explicit value, else the image's, else ``None``.
    """
    knee_array, knee_from_image = mask_and_spacing(knee_mask)
    ankle_array, ankle_from_image = mask_and_spacing(ankle_mask)
    return (knee_array, knee_spacing if knee_spacing is not None else knee_from_image,
            ankle_array, ankle_spacing if ankle_spacing is not None else ankle_from_image)


def _tibial_torsion_angle(knee_start: np.ndarray, knee_end: np.ndarray,
                          ankle_start: np.ndarray, ankle_end: np.ndarray, side: str) -> float:
    """Compose the tibial torsion from the posterior condylar line and the tibia -> fibula line.

    Sign derivation (LPI index frame: x increases towards the patient's left, y posteriorly). A left
    image half holds a patient-right leg, so lateral is -x and posterior is +y. With ``p`` along the
    (undirected) condylar line and ``d`` directed tibia centroid -> fibula centroid (pointing
    laterally), external torsion turns ``d`` from lateral towards posterior relative to ``p``:
    ``atan2(-(p_x d_y - p_y d_x), p . d)``, folded into (-90, 90]. This is the same rotation sense as
    the femoral antetorsion, so both use :func:`morphometry.geometry.signed_axial_angle`; a right
    image half (mirror-image leg, lateral = +x) negates the x components. The result is invariant to
    an in-plane rotation of the whole leg and does not depend on the condylar endpoint order.

    This replaces a composition that folded both lines to acute angles against the image x-axis and
    combined them by the signs of their y-offsets. That equalled the signed angle only while the
    fibula centroid lay posterior to the tibia centroid in the image; otherwise it returned the
    negated torsion and, past 90 degrees, a wrong magnitude. The angle is computed from voxel indices
    (the knee and ankle masks are assumed to share the in-plane frame and have equal x/y spacing);
    the latter half of that precondition is checked by
    :func:`morphometry.geometry.validate_in_plane_isotropy` whenever the caller supplies the masks'
    spacings (``knee_spacing`` / ``ankle_spacing`` of :func:`calculate_tibial_torsion`).
    :param knee_start: One posterior tibial condyle reference point.
    :param knee_end: The other posterior tibial condyle reference point.
    :param ankle_start: The tibia centroid at ankle level.
    :param ankle_end: The fibula centroid at ankle level.
    :param side: Image side, 'left' or 'right'.
    :return: The tibial torsion in degrees, in (-90, 90]; negative means internal torsion.
    """
    condyles = np.asarray(knee_start, dtype=float) - np.asarray(knee_end, dtype=float)
    tibia_to_fibula = np.asarray(ankle_end, dtype=float) - np.asarray(ankle_start, dtype=float)
    return G.signed_axial_angle(condyles, tibia_to_fibula, side)


def calculate_tibial_torsion(knee_mask: Image | np.ndarray, ankle_mask: Image | np.ndarray,
                             tibia_label_knee: int, tibia_label_ankle: int, fibula_label: int,
                             side: str = 'left', plot: bool | plt.Axes = False,
                             knee_spacing: Optional[Sequence[float]] = None,
                             ankle_spacing: Optional[Sequence[float]] = None) -> float:
    """
    Calculate the tibial torsion angle.

    The signed axial-plane angle from the posterior tibial condylar line (knee level) to the
    line from the tibia centroid to the fibula centroid at the level of the tibia's largest
    cross-section, positive for external torsion (the distal line turned from lateral towards
    posterior relative to the condylar line) and negative for internal torsion, relative to
    the landmarks. See :func:`_tibial_torsion_angle`.
    :param knee_mask: A 3D segmentation mask of the knee, either as an ``Image`` / ``Segmentation``
        (preferred: its voxel spacing is then checked against the index-space angle's
        equal-in-plane-spacing precondition without any further argument) or as a bare
        ``np.ndarray`` (then see ``knee_spacing``).
    :param ankle_mask: A 3D segmentation mask of the ankle, same contract (see ``ankle_spacing``).
    :param tibia_label_knee: The label of the tibia at knee level.
    :param tibia_label_ankle: The label of the tibia at ankle level.
    :param fibula_label: The label of the fibula.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param plot: A sequence of two matplotlib Axes to draw the reference lines on, or False.
    :param knee_spacing: Voxel spacing of ``knee_mask``. Only needed when ``knee_mask`` is a bare
        array: pass the ``Segmentation`` instead and the spacing is taken from it. Given either
        way, it is checked for the equal in-plane x/y spacing that the index-space angle
        composition assumes; a bare array with no spacing leaves it unchecked and the precondition
        then rests on the caller. An explicit spacing wins over an ``Image``'s own.
    :param ankle_spacing: Voxel spacing of ``ankle_mask``, same contract.
    :raises ValueError: If a resolved spacing is in-plane anisotropic or malformed (see
        :func:`morphometry.geometry.validate_in_plane_isotropy`).
    :return: The tibial torsion angle in degrees.
    """
    G.validate_side(side)
    knee_mask, knee_spacing, ankle_mask, ankle_spacing = _resolve_masks(
        knee_mask, knee_spacing, ankle_mask, ankle_spacing)

    angle, (knee_start, knee_end, ankle_start, ankle_end) = _tibial_torsion_points(
        knee_mask, ankle_mask, tibia_label_knee, tibia_label_ankle, fibula_label, side,
        knee_spacing, ankle_spacing)

    if plot is not False:
        knee_bin = np.where(knee_mask == tibia_label_knee, 1, 0)
        plot[0].imshow(knee_bin[:, :, int(knee_start[2])].T)
        plot[0].plot([knee_start[0], knee_end[0]], [knee_start[1], knee_end[1]], color='red')
        plot[1].imshow(ankle_mask[:, :, int(ankle_start[2])].T)
        plot[1].plot([ankle_start[0], ankle_end[0]], [ankle_start[1], ankle_end[1]], color='red')
    return angle


def get_tibial_torsion_landmarks(knee_mask: Image | np.ndarray, ankle_mask: Image | np.ndarray,
                                 tibia_label_knee: int, tibia_label_ankle: int, fibula_label: int,
                                 side: str = 'left',
                                 knee_spacing: Optional[Sequence[float]] = None,
                                 ankle_spacing: Optional[Sequence[float]] = None) -> dict:
    """
    Return the tibial-torsion reference landmarks as a dict.

    Provided separately from :func:`calculate_tibial_torsion` so the calculation keeps
    a single return type. The points are voxel indices of the given masks, found as in
    :func:`calculate_tibial_torsion` -- for a right image half on the reflection, mapped back --
    so unlike before they DO depend on ``side``: a right half's condylar line is now the mirror of
    what the left-side search finds on the reflected mask.
    :param knee_mask: A 3D segmentation mask of the knee, either as an ``Image`` / ``Segmentation``
        (preferred: its voxel spacing is then checked against the index-space angle's
        equal-in-plane-spacing precondition without any further argument) or as a bare
        ``np.ndarray`` (then see ``knee_spacing``).
    :param ankle_mask: A 3D segmentation mask of the ankle, same contract (see ``ankle_spacing``).
    :param tibia_label_knee: The label of the tibia at knee level.
    :param tibia_label_ankle: The label of the tibia at ankle level.
    :param fibula_label: The label of the fibula.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param knee_spacing: Voxel spacing of ``knee_mask``, needed only when it is a bare array; see
        :func:`calculate_tibial_torsion`.
    :param ankle_spacing: Voxel spacing of ``ankle_mask``, same contract.
    :raises ValueError: If a resolved spacing is in-plane anisotropic or malformed.
    :return: ``{'knee_start', 'knee_end', 'ankle_start', 'ankle_end'}`` as numpy arrays (voxel
    indices of the masks as given).
    """
    G.validate_side(side)
    knee_mask, knee_spacing, ankle_mask, ankle_spacing = _resolve_masks(
        knee_mask, knee_spacing, ankle_mask, ankle_spacing)
    _, (knee_start, knee_end, ankle_start, ankle_end) = _tibial_torsion_points(
        knee_mask, ankle_mask, tibia_label_knee, tibia_label_ankle, fibula_label, side,
        knee_spacing, ankle_spacing)
    return {'knee_start': knee_start, 'knee_end': knee_end, 'ankle_start': ankle_start, 'ankle_end': ankle_end}

import logging

import numpy as np
import pyvista as pv
import pandas as pd
import nibabel as nib

from morphometry.utils import sphere_fit, get_contour_points, calculate_angle_between_vectors, \
    calculate_min_distance_between_point_clouds, get_vector_through_point_perpendicular_to_line, \
    get_minimum_distance_between_line_and_point, get_contour_points, num_connected_components, \
    extract_connected_components_2d, circumference_points, intersect_ndarrays, \
    sort_points_clockwise, sort_points_by_x, fit_circle_to_points
from morphometry.image_io import Image, Segmentation, reflect_image_left_right
from morphometry.bresenham import bresenhamline
from morphometry import constants as C
from morphometry import geometry as G

from scipy.ndimage import center_of_mass, label
from scipy.spatial import KDTree
from typing import Tuple, Optional
from skimage.measure import find_contours
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Module logger for non-fatal fit diagnostics; see the note in ``morphometry/measurements/hip.py``.
logger = logging.getLogger(__name__)


def _largest_component_centroid(slice_mask: np.ndarray) -> Tuple[int, int]:
    """Centroid of the largest connected component of a single mask slice.

    The topmost slice of a proximal-femur mask frequently holds more than one island: the
    femoral head and the greater trochanter reach the same superior level and appear as two
    separate blobs. The centroid of their *union* then lands in the gap between them, on
    background, and a column dropped there is outside the bone from the start. Restricting
    to the largest island first keeps the centroid on the anatomy, which is what
    :func:`get_femoral_head_center` has always done for MRI.

    Selecting the island is not sufficient on its own, because a centroid is not a
    guaranteed interior point: a concave island — an arc of cortex around a marrow void, a
    crescent of partial-volume voxels — places its own centroid in its concavity, on
    background, and the column starts outside the bone exactly as it did before. So the
    centroid is used where it already lands inside the island, and the nearest island voxel
    substitutes where it does not. Keeping the centroid in the ordinary case matters: the
    truncated centroid is rarely the *closest* voxel to the true centroid, so snapping
    unconditionally would move the column by up to a voxel on almost every hip.
    :param slice_mask: A 2D binary mask; must contain at least one foreground voxel.
    :return: The ``(x, y)`` index of a voxel of the largest connected component.
    """
    labelled, num_features = label(slice_mask)
    if num_features > 1:
        sizes = [np.count_nonzero(labelled == i) for i in range(1, num_features + 1)]
        slice_mask = labelled == (int(np.argmax(sizes)) + 1)

    centroid = center_of_mass(slice_mask)
    column = (int(centroid[0]), int(centroid[1]))
    if slice_mask[column]:
        return column

    voxels = np.argwhere(slice_mask)  # concave island: fall back to its nearest voxel
    nearest = voxels[np.argmin(np.sum((voxels - np.asarray(centroid)) ** 2, axis=1))]
    return int(nearest[0]), int(nearest[1])


def _femoral_head_slab_bound(mask: np.ndarray, column: Tuple[int, int], layer_high: int,
                             spacing: Tuple[float, float, float]) -> int:
    """Find the last slice a femoral-head slab may plausibly extend to.

    Both femoral-head-centre functions delimit the head by dropping a single-voxel
    column through the centroid of the topmost head slice and walking it inferiorly
    until it leaves the bone. On a straight, vertically aligned proximal femur the
    column never leaves the bone — it runs head into neck into shaft — and the sphere
    is then fitted to the whole proximal femur, silently returning a radius two to
    three times anatomical, centred far inferior of the real head.

    This bounds that walk by the head's own size, measured from the same data. Descending
    from ``layer_high``, the connected component containing the column is taken on each
    slice and its equivalent radius ``sqrt(area / pi)`` recorded. That radius grows to the
    head's equator and then shrinks; when the neck or greater trochanter merges into the
    component it jumps back up. The descent stops at that post-equator jump (or as soon as
    the column leaves the bone), and the largest radius seen bounds the slab at one head
    diameter below ``layer_high``.

    That reasoning holds only if the equator was actually in view, i.e. if the radius was
    seen to rise and then fall away from its peak by more than
    ``C.FEMORAL_HEAD_RADIUS_JUMP``. It is not in view when the acquisition is cropped
    through the head (the radius is still rising when the bone ends, so the largest radius
    seen is a chord and not the head's own) or when ``layer_high``'s centroid falls between
    two blobs and the column starts outside the bone. In both cases there is no measurement
    of the head to bound anything with, and guessing one truncates a sound slab — so no
    bound is applied and the caller's walk runs as it always did. The fall has to clear the
    same tolerance as the merge jump because a single voxel of rasterisation noise is
    otherwise enough to pass for an equator: on one NMDID hip a 713 -> 708 voxel flicker
    did exactly that, and the resulting cap cut a sound slab by a quarter.

    The bound is deliberately generous — a full diameter measured from the head's own
    equator, plus ``C.FEMORAL_HEAD_SLAB_MARGIN_SLICES`` — so that it is inert wherever the
    walk already terminates sensibly and only truncates the runaway case.
    :param mask: A binary 3D mask of the femur, index-ordered ``(x, y, z)`` with z
        increasing inferiorly (LPI).
    :param column: The ``(x, y)`` index of the column dropped through the head.
    :param layer_high: Index of the topmost femoral-head slice; the descent starts here.
    :param spacing: Voxel spacing ``(sx, sy, sz)``. Any consistent length unit works —
        millimetres for the CT path, in-plane voxels (so ``(1, 1, x_ratio)``) for the
        MRI path — as long as all three components share it.
    :return: The deepest slice index the column walk may stop at, clipped to the volume;
        the last slice of the volume (i.e. no effective bound) when the head's equator was
        never observed.
    """
    sx, sy, sz = spacing
    previous_radius = None
    max_radius = 0.0
    max_drop = 0.0  # how far the radius has fallen below its running peak
    shrinking = False
    layer = layer_high

    while layer < mask.shape[2]:
        labelled, _ = label(mask[:, :, layer])
        component = labelled[column[0], column[1]]
        if component == 0:
            break  # the column has left the bone; nothing below can belong to the head
        radius = float(np.sqrt(np.count_nonzero(labelled == component) * sx * sy / np.pi))
        if previous_radius is not None:
            if radius < previous_radius:
                shrinking = True  # past the head's equator
            elif shrinking and radius - previous_radius > C.FEMORAL_HEAD_RADIUS_JUMP:
                break  # neck / trochanter merged in: this slice is no longer head
        max_radius = max(max_radius, radius)
        max_drop = max(max_drop, max_radius - radius)
        previous_radius = radius
        layer += 1

    if max_drop <= C.FEMORAL_HEAD_RADIUS_JUMP:
        return mask.shape[2] - 1  # equator never seen: max_radius is not a head radius

    # The literal 2.0 this replaced is one head DIAMETER, which reaches a full radius past
    # the equator and into the neck; see C.FEMORAL_HEAD_SLAB_RADII for the measurement and
    # for why it is left at 2.0 rather than corrected. Named so the decision is one edit.
    slab_layers = (int(round(C.FEMORAL_HEAD_SLAB_RADII * max_radius / sz))
                   + C.FEMORAL_HEAD_SLAB_MARGIN_SLICES)
    return min(layer_high + slab_layers, mask.shape[2] - 1)


def _fit_femoral_head_consensus(points: np.ndarray, com_high: Tuple[int, int],
                                voxel_size: np.ndarray,
                                tolerance_mm: float = C.FEMORAL_HEAD_CONSENSUS_TOLERANCE_MM,
                                max_iterations: int = C.FEMORAL_HEAD_CONSENSUS_MAX_ITERATIONS,
                                seed_depth_mm: float = C.FEMORAL_HEAD_SEED_DEPTH_MM) -> Tuple[float, np.ndarray]:
    """Fit the femoral-head sphere to slab contour points by a seeded consensus fit, in millimetres.

    The slab handed over by :func:`get_femoral_head_center` starts at the top of the femur and
    usually reaches one to three slices past the head's equator, so it holds femoral-neck and
    greater-trochanter surface besides the head. That surface lies lateral of the head and well
    inside or outside the head sphere, so it is excluded by its distance from the sphere, not by a
    medial-lateral cutoff (a cutoff cannot tell lateral head surface at the equator from neck
    surface below it):

    1. Seed: fit a sphere to the points medial of the top-slice centroid (``x >=`` centroid; this is
       a LEFT image half, where medial is ``+x``) that lie within ``seed_depth_mm`` of the top slab
       slice, trimmed anteroposteriorly to the centroid's symmetric range. The neck and
       trochanter are lateral, and the depth limit keeps the seed above the neck basin and the calcar,
       which the slab can reach when the column walk stays in bone; so the seed is head surface.
    2. Consensus: refit on every slab point whose distance from the current sphere's surface is at
       most ``tolerance_mm``; repeat until the inlier set no longer changes (or ``max_iterations``).

    Every step is defined relative to the head's own points and centroid, so the fit follows a
    translation of the image exactly. There is no right-side branch: a right image half is measured
    by :func:`get_femoral_head_center` as the reflection of a left one, so this helper only ever
    sees left-sided geometry. The former ``x <=`` switch was keyed on ``int(centroid[0])``, and
    truncation does not commute with ``x -> n - 1 - x``, so the seed's boundary column was off by
    one on essentially every half. The consensus refit absorbed that on most of them: over the 100
    halves of the 50 Augsburg reader-study patients, 91 agreed with their own mirror to within
    1e-9 mm and 9 did not -- 7 of those by under 0.08 mm, and two materially, ``PA000390``
    image-left by 3.01 mm (radius 0.43 in-plane voxels) and ``PA000397`` image-right by 1.52 mm
    (radius 0.68).
    :param points: Slab contour points as ``(x, y, z)`` voxel indices, shape ``(n, 3)``.
    :param com_high: The ``(x, y)`` voxel index of the top-slice centroid.
    :param voxel_size: Voxel size in mm along ``(x, y, z)``.
    :param tolerance_mm: Inlier band around the sphere surface, in mm.
    :param max_iterations: Maximum number of consensus refits.
    :param seed_depth_mm: How far below the top slab slice, in mm, seed points may lie.
    :return: The radius in mm and the centre in mm (index times ``voxel_size``).
    :raises RuntimeError: If no seed points exist or the consensus collapses.
    """
    points = np.asarray(points, dtype=float)
    points_mm = points * voxel_size

    medial = points[:, 0] >= com_high[0]  # left image half: medial is +x
    near_top = (points[:, 2] - np.min(points[:, 2])) * voxel_size[2] <= seed_depth_mm
    seed = points[medial & near_top]
    if len(seed) == 0:
        raise RuntimeError('Femoral head consensus fit: no slab points medial of the top-slice centroid to seed from.')
    # anteroposterior trim of the seed, symmetric about the centroid (as the former trim did)
    seed = seed[seed[:, 1] <= 2 * com_high[1] - np.min(seed[:, 1])]
    radius, center = sphere_fit(seed * voxel_size)

    inliers = None
    for _ in range(max_iterations):
        candidate = np.abs(np.linalg.norm(points_mm - center, axis=1) - radius) <= tolerance_mm
        if inliers is not None and np.array_equal(candidate, inliers):
            break
        if candidate.sum() < 4:
            raise RuntimeError('Femoral head consensus fit collapsed: fewer than 4 points near the sphere.')
        inliers = candidate
        radius, center = sphere_fit(points_mm[inliers])
    else:
        logger.warning('Femoral head consensus fit did not converge within %d refits; using the last fit '
                       '(radius %.1f mm).', max_iterations, radius)
    return radius, center


def get_femoral_head_center(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1, return_layers: bool = False, x_ratio: float = 1., isotropic: bool = False, spacing: Optional[Tuple[float, float, float]] = None) -> Tuple[float, np.ndarray] | Tuple[float, np.ndarray, int, int]:
    """
    Get the center of the femoral head from a segmentation mask.

    The head is delimited by dropping a column through the centroid of its topmost slice
    and walking it inferiorly until it leaves the bone; that walk is capped at one head
    diameter by :func:`_femoral_head_slab_bound` so a column that stays inside bone past
    the head cannot drag neck and shaft into the sphere fit.

    **The slab starts at the topmost slice holding a femur voxel, with nothing skipped.** It used
    to try to skip a top-of-head slice with "too few mask points" by z-scoring the per-slice voxel
    counts over the array's **full** z extent -- one zero per empty slice -- and advancing while the
    first slice's ``|z| > 2``. That guard was removed (2026-11-05) rather than corrected, and the
    reason is a property of the statistic, not of our cohorts.

    **Its trigger is anti-correlated with its stated purpose.** Write ``f`` for the fraction of the
    array's z extent that holds femur and ``c^2 = 1 + CV^2`` for the normalised second moment of
    the non-empty slice areas. For a near-empty first slice the z-score is
    ``|z| = sqrt(f) / sqrt(c^2 - f)``, so the guard fires iff ``f > 0.8 * c^2``.

    That form holds only while the apex slice is small against the mean slice area, and **slice
    thickness is what decides whether it does**. At 1 mm (NaKo) the top-of-head slice is a thin cap
    at a median 4.1% of the mean area, the assumption holds, and the form predicts the true
    ``|z|`` to within a median +4.2% and gets the fire / no-fire verdict right on 112 of 120 halves.
    At 10 mm (Augsburg) that first slice already cuts a median 52% of the mean area, the assumption
    fails outright, and the form over-predicts by a median +108% (verdict right on 78 of 100). The
    phantom in ``tests/unit/test_femoral_head_slab_start.py`` sits in between, at 95 voxels, where
    it over-predicts by 8% and misjudges one padding. So use it on thin-sliced data and compute the
    statistic directly on thick-sliced data.

    The guard fires most readily when the slice areas are **uniform** (``c^2`` near 1) -- when no
    slice is an outlier and there is nothing to reject -- and is suppressed exactly when the areas
    genuinely vary, which is when an outlier top slice might really exist.

    That anti-correlation is measured, not just algebra. Holding ``f = 1`` (the most favourable case
    for firing) over 56 slices with the first one at 6% of the mean of the rest -- a genuine
    top-of-head cap, exactly what the guard was written to reject -- the first slice's ``|z|`` falls
    monotonically as the remaining areas are made more variable: 7.42 / 6.01 / 4.24 / 3.15 / 2.49 /
    2.06 / 1.56 / 1.18 at a slice-area CV of 0.00 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50 / 0.70 / 1.00
    (lognormal areas; a linear ramp gives 7.42 / 5.76 / 3.89 / 2.82 / 2.19 / 1.78 / 1.55 / 1.55).
    So the apex cap is flagged when the stack is featureless and goes unflagged once the stack has
    any real spread -- the opposite of an outlier test.

    And whichever side of the threshold a scan falls on is decided partly by ``f``, i.e. **by the
    field of view rather than by the anatomy**: empty slices drag the mean down and inflate the
    standard deviation, so padding an array switches the guard off and cropping it switches the
    guard on. That half of the argument needs no statistics at all.

    **What that cost in practice.** Trimming the empty leading and trailing z slices -- which
    removes no labelled voxel and must therefore change nothing -- moved the centre by more than
    1e-9 mm on 108 of the 120 NaKo halves (none of the 120 was bitwise exact), by a median 0.26 and
    up to 1.96 mm. Removing the guard is **bit-identical on every half measured**: the centre and
    radius match the pre-change code exactly on all 220 -- the 120 NaKo sample halves (isotropic
    path) and the 100 Augsburg reader-study halves (consensus path) -- so no golden and no cohort
    row moves.

    That bit-identity is a measured result on those 220, **scoped to the frames production uses and
    not a structural guarantee**: the guard fires on 0 of them, but it is not inert by construction.
    The direct evidence is the first femur slice's ``|z|`` against the threshold of 2: in the
    as-given frames it maxes at 0.847 on NaKo and 1.325 on Augsburg, so neither fires. **Trim the
    empty slices and NaKo does**: its ``|z|`` rises to a median 2.466 (max 3.307) and the guard
    fires on 108 of 120 halves, where Augsburg's rises only to a median 0.904 and fires on 2 of 100.
    Removal moves the NaKo centre there by a median 0.265 and up to 1.960 voxel. So "zero movement"
    is a property of how much empty z these cohorts carry, not of the code -- which is the argument
    *for* removing it.

    Note what that rules out: a tight crop **alone** is sufficient on 1 mm isotropic data, because
    NaKo's ``c^2`` is already a median 1.155 and 116 of its 120 halves sit inside the firing regime
    once trimmed. It is not sufficient at 10 mm, where the slice areas vary enough to hold ``c^2``
    up. Whether cropping is enough is therefore a question about the cohort's slice thickness, and
    on thin-sliced data the answer is yes.

    **Reachability of the firing configuration is a mechanism claim, not a measurement.**
    ``api/ingest/dicom.py``'s ``_detect_slabs`` / ``_split_volume`` cut an ingested series at
    position gaps and changepoints, so a slab the femur nearly fills -- the firing configuration --
    is producible there; no such volume exists locally to test, so this is reasoning about the
    mechanism and is labelled as such. It is also **bounded**: that path reaches the non-isotropic
    consensus fit, where the slab start only seeds a refit that re-converges, and removal there
    moves the centre by at most 0.0195 voxel even on the halves where the guard does fire. The
    large numbers belong to the isotropic path, where ``layer_high`` delimits the point cloud handed
    to ``sphere_fit`` directly. The phantom in ``tests/unit/test_femoral_head_slab_start.py`` is in
    the firing regime, which is why it fails against the old code.

    Whether a tiny top-of-head slice *should* be skipped is a separate, open question. Correcting
    the statistic instead (z-scoring over the non-empty slices only) is a different measurement,
    not a better one -- it moves 108 of the 120 NaKo halves by a median 0.26 mm, with no evidence
    that the resulting centre is more accurate -- so it would need its own accuracy case, the
    natural check being the top-slice full-contour circle fits that validated the consensus centre.
    Do not reinstate a guard whose only support is that the old one existed.

    For non-isotropic images the sphere is fitted with :func:`_fit_femoral_head_consensus`, in
    millimetres from ``spacing``. Without ``spacing`` the geometry is derived from ``x_ratio``
    alone, i.e. voxels of ``(1, 1, x_ratio)`` with the in-plane voxel taken as 1 mm (exact for 1 mm
    in-plane voxels; otherwise only the inlier band is scaled). This replaced a lateral cutoff at
    ``0.9`` times an absolute voxel index, which made the centre depend on where the hip sat in the
    image and let femoral-neck surface into the fit.

    The isotropic path keeps the lateral cutoff (one "medial radius" from the top-slice centroid
    on the lateral side), built from the unrounded centroid, and walks down the nearest column
    (half-voxel ties broken towards medial). It used to floor the centroid for both, which does
    not commute with the left/right mirror, so the image-left half came out ~0.4 voxel more
    lateral than the mirror of the image-right one (and its centre-edge angle ~1.1 degrees lower
    on the same hip).

    **Mirror-equivariant on both paths**: a RIGHT image half is measured as the reflection of a
    left one (``mask[::-1]`` measured with ``side='left'``, the centre mapped back with
    :func:`morphometry.geometry.mirror_index_x`), so a half and its mirror image give the same head
    centre and the same radius. The body below is therefore left-side-only -- the rule being the one
    the torsion, CCD and tibial-torsion landmark code already follows: add left-side logic and let
    the reflection handle the right, never a ``side == 'right'`` branch. The former right-side
    branches were the ``ceil(x - 0.5)`` half of the isotropic column tie-break and the mirrored
    lateral trim, both written so the direct right-side call would reproduce the mirror of the left
    one; the reflection reproduces them by construction. On the non-isotropic path the truncated
    seed boundary of :func:`_fit_femoral_head_consensus` did not mirror, which made 9 of the 100
    Augsburg reader-study halves disagree with their own mirror, two of them by 1.52 and 3.01 mm
    (see that function).

    One residual exception, pre-existing and deliberately not fixed: the top-slice island selection
    below -- and :func:`_largest_component_centroid` on the CT path -- resolves a tie with
    ``np.argmax``, which takes the *first* maximum in ``scipy.ndimage.label``'s raster order, and
    reflection reverses that order along x. Two islands of exactly equal size therefore select
    different islands in the direct and in the reflected run. It needs a true tie to bite, so it
    has never been observed on real data; a code change here would move every hip with more than
    one island on its top slice, which is not worth it for a tie-break.
    :param segmentation_mask: A segmentation mask of the proximal femur where the femur is 1 and everything else 0.
    :param side: Side of the image (not patient!), either 'left' or 'right'. A right half is
        measured as the reflection of a left one (see above), so the two sides cannot diverge.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param return_layers: Whether to return layer_high and layer_low.
    :param x_ratio: Slice-axis scale factor, in in-plane voxels per slice. On the non-isotropic
        path it reaches only the femoral-head slab bound (the consensus fit itself uses
        ``spacing``), and that bound is inert on every dataset measured -- identical results at 1,
        5.47 and 9.14. Where it still decides a result is ``get_femoral_neck_center``; see its
        docstring and backlog task #6 before changing how callers compute it. On the isotropic
        path, and in the ``spacing``-less fallback of the consensus fit, it does scale the
        geometry, so an array-only caller on anisotropic data gets a centre up to ~14 mm out in z
        -- latent only, since the array-only callers (alpha angle, acetabular depth) run on
        isotropic NaKo.
    :param isotropic: Whether the image has isotropic voxels.
    :param spacing: Voxel spacing in mm along the three index axes (e.g. ``Image.spacing``). Used by
        the non-isotropic sphere fit; ``None`` derives the geometry from ``x_ratio``.
    :return: The radius (in in-plane voxels) and the location (voxel index) of the femoral head center.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    if side == 'right':
        # measure the right half as the reflection of a left one and mirror the centre back, so
        # the two sides cannot be different measurements. The radius is a scalar and the two layer
        # bounds are z indices, so the x-mirror leaves all three alone.
        width = segmentation_mask.shape[0]
        radius, center, *layers = get_femoral_head_center(segmentation_mask[::-1], side='left',
                                                          segmentation_label=segmentation_label,
                                                          return_layers=return_layers, x_ratio=x_ratio,
                                                          isotropic=isotropic, spacing=spacing)
        return (radius, G.mirror_index_x(center, width), *layers)

    segmentation_mask = np.where(segmentation_mask == segmentation_label, 1, 0)

    contour_pts = get_contour_points(segmentation_mask)

    # get highest layer with a mask point, its centroid
    # and lowest layer with mask point on this centroid. Nothing is skipped here: the former
    # z-score guard z-scored the per-slice voxel counts over the array's full z extent, so whether
    # it fired depended on the number of EMPTY slices, and it fired most readily when the slice
    # areas were uniform -- when there was no outlier to reject. See the docstring before adding
    # any top-slice rejection back.
    layer_high = np.amin(contour_pts[:, 2])

    # if layer_high has two connected components, choose the larger one
    label_mask, num_features = label(segmentation_mask[:, :, layer_high])
    if num_features > 1:
        sizes = [len(np.argwhere(label_mask == i)) for i in range(1, num_features + 1)]
        largest_component = np.argmax(sizes) + 1
        segmentation_mask[:, :, layer_high] = np.where(label_mask == largest_component, 1, 0)

    centroid = center_of_mass(segmentation_mask[:, :, layer_high])
    com_high = (int(centroid[0]), int(centroid[1]))
    if isotropic:
        # The column walk indexes the mask, so it needs an integer x, and this is nearest-with-
        # ties-up rather than the plain int() above for one reason only: it is what the shipped
        # left-side value was computed with, so keeping it is what holds every NaKo golden still.
        # Its ORIGINAL reason is gone -- it was chosen because floor does not commute with the
        # mirror x -> n-1-x while this does, and with a right half now measured through the
        # reflection there are no longer two code paths that have to agree, so any rounding rule
        # would mirror. Do not preserve it as a symmetry constraint; it is a bit-compatibility one.
        # The lateral cutoff below uses the unrounded centroid; the consensus path keeps int().
        com_high = (int(np.floor(centroid[0] + .5)), com_high[1])
    # the walk works in index space, so the length unit is the in-plane voxel and x_ratio is how
    # many of those the caller says a slice is worth. The pipelines' x_ratio is not the true
    # aspect (task #6), but this bound is inert either way: identical at 1, 5.47 and 9.14.
    layer_max = _femoral_head_slab_bound(segmentation_mask, com_high, layer_high, (1., 1., x_ratio))
    layer_low = layer_high + 1
    while layer_low < layer_max and segmentation_mask[com_high[0], com_high[1], layer_low] != 0:
        layer_low += 1

    point_cloud = get_contour_points(segmentation_mask)
    point_cloud = point_cloud[point_cloud[:, 2] >= layer_high]
    point_cloud = point_cloud[point_cloud[:, 2] <= layer_low]  # exclude everything that is not between layer_high and layer_low

    if not isotropic:
        voxel_size = np.abs(np.asarray(spacing, dtype=float)) if spacing is not None else np.array([1., 1., x_ratio])
        r_mm, center_mm = _fit_femoral_head_consensus(point_cloud, com_high, voxel_size)
        # back to the units the callers expect: in-plane voxels for the radius, voxel index for the centre
        r, center = r_mm / voxel_size[0], center_mm / voxel_size
        return (r, center) if not return_layers else (r, center, layer_high, layer_low)

    point_cloud = point_cloud.astype(np.float32) * np.array([1, 1, x_ratio])  # adjust for slice thickness

    # need to exclude lateral parts of the mask: compute distance between com and max medial point of femoral head,
    # then exclude everything that is farther away than this distance in the lateral direction.
    # This is a left image half, so medial is +x and the cutoff goes on the -x side; the mirrored
    # form the right side used to carry is now reached through the reflection at the top.
    max_s = np.max(point_cloud[:, 0])
    radius = max_s - centroid[0]
    min_s = centroid[0] - radius  # the most lateral point of the femoral head
    point_cloud = point_cloud[point_cloud[:, 0] >= min_s]

    min_c = np.min(point_cloud[:, 1])
    radius = com_high[1] - min_c
    max_c = com_high[1] + radius
    point_cloud = point_cloud[point_cloud[:, 1] <= max_c]

    # get center coordinates of the fitting sphere
    r, center = sphere_fit(point_cloud)

    # compensate pixel mm ratio between x, y and z axis
    center = np.array([center[0], center[1], center[2] / x_ratio])

    return (r, center) if not return_layers else (r, center, layer_high, layer_low)


def _mirror_physical_point_x(image: Image, point: np.ndarray) -> np.ndarray:
    """Map a physical point through a left-right reflection of ``image``'s index grid.

    :func:`morphometry.image_io.reflect_image_left_right` moves the voxel at index ``i`` to index
    ``n - 1 - i`` and leaves the index-to-physical mapping alone, so a physical point found on the
    reflected image belongs, in the original image, at the physical location of the mirrored index.
    That is ``p -> A ((n - 1) - t, u, v) + o`` where ``(t, u, v)`` is the continuous index of ``p``.

    The index is recovered from the affine assembled out of four
    :meth:`Image.transform_index_to_physical_point` calls rather than from
    :meth:`Image.transform_physical_point_to_index`, which on the SimpleITK backend is
    ``TransformPhysicalPointToIndex`` and returns an *integer* index -- it would snap a sphere-fit
    centre to a whole voxel (0.8 mm in plane, 3.0 mm in z on whole-leg CT), against this module's
    rule that physical coordinates stay floats. Four integer index calls are exact on both backends
    and recover the full index-to-physical matrix, so the round trip is continuous. The width comes
    from ``image.shape`` and not from ``image.array.shape`` for the same reason of backend
    independence: ``Image.array`` is ``(x, y, z)`` on nibabel but ``(z, y, x)`` on SimpleITK, where
    ``array.shape[0]`` would be the slice count, while ``Image.shape`` is the width on both (and
    needs no float64 copy of the volume to read).
    :param image: The image (or Segmentation) whose index grid is reflected; not modified.
    :param point: The physical point to map, length 3.
    :return: The mapped physical point as a float array of length 3.
    """
    origin = np.asarray(image.transform_index_to_physical_point((0, 0, 0)), dtype=float)
    # columns of the index-to-physical matrix: where each unit index step lands in physical space
    matrix = np.stack([np.asarray(image.transform_index_to_physical_point(step), dtype=float) - origin
                       for step in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], axis=1)
    index = np.linalg.solve(matrix, np.asarray(point, dtype=float) - origin)
    index[0] = (image.shape[0] - 1) - index[0]
    return matrix @ index + origin


def get_femoral_head_center_ct(femur_image: Segmentation, segmentation_label: int = 1, side: str = 'left') -> Tuple[float, np.ndarray]:
    """
    Get the center of the femoral head from a whole-leg CT segmentation mask.

    As in the MRI variant, the column that delimits the head starts at the centroid of the
    largest island of the topmost femur slice (:func:`_largest_component_centroid`, so a
    slice holding both head and trochanter does not put the column in the gap between
    them), and the slab handed to the sphere fit is capped at one head diameter by
    :func:`_femoral_head_slab_bound` (so on a straight femur the walk cannot run head into
    neck into shaft and return a radius far above anatomical).

    The slab is then trimmed in plane by a **two-sided box about the top-slice centroid, both of
    whose half-widths come from the head-dominated superior half of the slab** -- on each in-plane
    axis, one "head radius" from the centroid out to that half's own smallest coordinate (in x that
    is the head's medial surface), applied symmetrically on either side of the centroid.

    **The operative element is the two-sidedness, and within it the near-side bound in y.** Each
    axis used to be trimmed on its far side only, so the excursion that *set* a half-width was
    never itself trimmed: a slab reaching far anteriorly sized the coronal half-width on its own
    anterior reach, was then granted an equally large posterior allowance it never came near, and
    kept every one of those anterior points in the cloud the sphere was fitted to. Closing that
    near side in y is where essentially all of the movement comes from -- over the 46 NMDID
    verification cases (92 image halves) the full box moves the fitted radius on 54 halves, by up
    to 6.034 mm.

    Two things that are **not** the mechanism, recorded so the attribution is not misread. The
    lateral (+x) cut is the primary cutoff and does real work, unchanged in substance: on this
    module's CT fixture it removes 15.4% and 19.0% of the slab's points. The medial (-x) bound
    added beside it is near-vacuous -- 0 points removed on either of those halves -- and *by
    construction* it can never clip a point of ``superior_half`` at all, since ``half_x`` is that
    half's own distance to its own minimum x. Second, taking the coronal half-width from
    ``superior_half`` rather than from the already x-cut cloud is a tidy-up and not the fix: the
    old half-width was genuinely inflated by the neck and shaft points it was measured on (by a
    factor 1.35 and 1.91 on the two fixture halves), but making only that change, still one-sided,
    moves the radius on 2 of 92 halves by at most 0.053 mm -- a median 0.00% and at most 9.77% of
    the total movement. It is done anyway, because a bound sized by the contamination it exists to
    exclude has no defensible reading, not because it is what moved the numbers.

    **Mirror-equivariant, as in the MRI variant**: a RIGHT image half is measured as the
    reflection of a left one and the centre mapped back with :func:`_mirror_physical_point_x`, so
    the body below is left-side-only and the two sides cannot be different measurements. The
    reflection (:func:`morphometry.image_io.reflect_image_left_right`), not the index-relabelling
    :func:`morphometry.image_io.mirror_image_left_right`, is the operation needed here, for the
    same reason as in :func:`morphometry.femur.get_proximal_reference_line_ct` and
    ``measurements.hip.calculate_ccd_ct``: the lateral cutoff below is applied to *physical* x, so
    the landmark code only sees a consistent left-sided leg after the anatomy itself has been
    reflected. The deleted right-side branch was the mirror image of the kept one, written so the
    direct right-side call would reproduce the left-side answer; it did not, and unlike the MRI path
    it failed *systematically* rather than occasionally -- on the NMDID verification sample only 2
    of 46 right halves agreed with their mirror to 1e-9 mm, median gap 0.78 mm, max 3.26, against
    4 of 100 beyond 0.01 mm on the Augsburg MRI. So the ``int()`` truncation of the seed boundary
    that explains the MRI path is **not** a sufficient account of the CT one.
    :func:`_largest_component_centroid`, which starts this column, has three steps that do not
    commute with ``x -> n - 1 - x``: ``np.argmax`` over island sizes (first maximum in raster
    order), ``int()`` on the centroid, and the nearest-voxel fallback for a concave island
    (``np.argmin`` over ``np.argwhere`` order). The last can move the column by more than a voxel
    and with it the slab bound, which is the likeliest reason the CT gap is systematic. None of
    that is diagnosed here and none of it needs to be: the reflection makes the question moot.
    :param femur_image: A segmentation mask of the whole leg.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'. A right half is
        measured as the reflection of a left one (see above), so the two sides cannot diverge.
    :return: The radius and location (physical mm) of the femoral head center.
    """
    G.validate_side(side)

    if side == 'right':
        radius, center = get_femoral_head_center_ct(reflect_image_left_right(femur_image),
                                                    segmentation_label=segmentation_label, side='left')
        # a reflection is an isometry, so only the centre moves; the radius carries over unchanged
        return radius, _mirror_physical_point_x(femur_image, center)

    mask = np.where(femur_image.array == segmentation_label, 1, 0)
    contour_points = get_contour_points(mask)

    layer_high = np.amin(contour_points[:, 2])
    # restrict to the largest island first: a top slice holding both the head and the
    # greater trochanter would otherwise put the column in the gap between them
    com_high = _largest_component_centroid(mask[:, :, layer_high])
    layer_max = _femoral_head_slab_bound(mask, com_high, layer_high,
                                         tuple(abs(float(s)) for s in femur_image.spacing[:3]))
    layer_low = layer_high + 1

    while layer_low < layer_max and mask[com_high[0], com_high[1], layer_low] != 0:
        layer_low += 1

    contour_points = contour_points[contour_points[:, 2] >= layer_high]
    contour_points = contour_points[contour_points[:, 2] <= layer_low]

    # transform everything into physical coordinates
    contour_points = np.array([femur_image.transform_index_to_physical_point(x) for x in contour_points])
    layer_high = femur_image.transform_index_to_physical_point((0, 0, layer_high))[2]
    layer_low = femur_image.transform_index_to_physical_point((0, 0, layer_low))[2]
    com_high = femur_image.transform_index_to_physical_point((com_high[0], com_high[1], 0))[:2]

    middle_slice = (layer_high + (layer_low - layer_high) // 2)
    superior_half = contour_points[contour_points[:, 2] >= middle_slice]

    # Left image half: a two-sided box about the top-slice centroid, with BOTH half-widths taken
    # from ``superior_half`` -- the head-dominated top of the slab. In this physical frame the
    # head's medial surface is at the smallest x, so each half-width is one "head radius" measured
    # from the centroid out to the superior half's own smallest coordinate on that axis, and each
    # axis is then trimmed on *both* sides of the centroid.
    # What does the work is the two-sidedness in y. The old code trimmed each axis on its far side
    # only, so the anterior excursion that SET the coronal half-width was never trimmed, and the
    # posterior allowance it bought was one the cloud never reached; closing that near side is
    # where essentially all of the movement comes from (54 of 92 verification halves, up to
    # 6.034 mm of radius). The lateral (+x) cut is unchanged in substance and still removes
    # 15.4% / 19.0% of the slab on the CT fixture's two halves, while the medial (-x) bound added
    # beside it is near-vacuous and cannot clip a ``superior_half`` point at all, half_x being that
    # half's own distance to its own minimum x. Sourcing half_y from ``superior_half`` rather than
    # from the already x-cut cloud is a tidy-up -- the old half-width really was inflated by the
    # neck and shaft points it was measured on (x1.35 and x1.91 here) -- but on its own it is worth
    # under 10% of the movement on any half. The mirrored form the right side used to carry is now
    # reached through the reflection at the top.
    half_x = com_high[0] - np.min(superior_half[:, 0])
    half_y = com_high[1] - np.min(superior_half[:, 1])
    contour_points = contour_points[
        (np.abs(contour_points[:, 0] - com_high[0]) <= half_x)
        & (np.abs(contour_points[:, 1] - com_high[1]) <= half_y)]

    return sphere_fit(contour_points)

def _femoral_neck_points_and_center(point_cloud: np.ndarray, r: float, center: np.ndarray, *,
                                    distal_greater_z: bool, lateral_greater_x: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Extract femoral-neck points and their centroid from a femur point cloud.

    Samples a hollow sphere shell (radius ``r`` to ``FEMORAL_NECK_SPHERE_OUTER_FACTOR
    * r``) around the femoral head centre, keeps the distal/lateral quadrant,
    removes outliers, intersects the shell with the femur point cloud (KDTree
    radius 2), and returns the resulting neck points and their centroid. Works in
    whatever coordinate space the caller supplies (index space for MRI, physical mm
    for CT); the only modality difference is which half-space is distal/lateral.
    :param point_cloud: Femur coordinates in the working space (N x 3).
    :param r: Femoral head radius in the working space.
    :param center: Femoral head centre in the working space.
    :param distal_greater_z: True if distal is larger z (MRI), False if smaller (CT).
    :param lateral_greater_x: True if lateral is larger x for this side, else False.
    :return: The femoral neck points (M x 3) and their centroid (3,).
    """
    sphere = pv.SolidSphere(inner_radius=r, outer_radius=C.FEMORAL_NECK_SPHERE_OUTER_FACTOR * r, center=center)
    shell = np.array(sphere.points)
    shell = shell[shell[:, 2] > center[2]] if distal_greater_z else shell[shell[:, 2] < center[2]]
    shell = shell[shell[:, 0] > center[0]] if lateral_greater_x else shell[shell[:, 0] < center[0]]

    com = KMeans(n_clusters=1, random_state=0).fit(shell).cluster_centers_[0]
    distances = np.linalg.norm(shell - com, axis=1)
    shell = shell[distances <= distances.mean() + C.FEMORAL_NECK_OUTLIER_STD * distances.std()]

    pc_tree = KDTree(point_cloud)
    shell_tree = KDTree(shell)
    pairs = shell_tree.query_ball_tree(pc_tree, 2)
    neck_points = np.array([point_cloud[index] for pair in pairs for index in pair])

    com = KMeans(n_clusters=1, random_state=0).fit(neck_points).cluster_centers_[0]
    return neck_points, np.array(com)


def get_femoral_neck_center(segmentation_mask: np.ndarray, femoral_head_center: Tuple[float, np.ndarray], side: str = 'left', segmentation_label: int = 1, x_ratio: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the endpoint of the femoral neck axis from an MRI proximal-femur segmentation.
    :param segmentation_mask: A segmentation mask of the proximal femur where the femur is 1 and everything else 0.
    :param femoral_head_center: The radius and center of the femoral head.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param x_ratio: The factor the point cloud's slice axis is scaled by before the neck shell is
        sampled. **It is not a slice aspect ratio, despite reading like one** (backlog task #6):
        it sets how far the sphere shell reaches in *slices*, and on thick-slice data that reach
        is what decides the landmark. The torsion pipelines pass
        ``spacing_z / 2 * spacing_x``, which is dimensionally ``mm / 2 x mm`` and is **not** the
        true aspect ``spacing_z / spacing_x``; correcting it to the true aspect measurably
        degrades the CCD, because on the 10 mm Augsburg hips the correctly-scaled shell leaves 30
        of 100 image halves with a neck point cloud spanning two slices or fewer, so the
        centroid's z is quantised to one or two sample positions. See ``docs/ccd_x_ratio.md`` and
        ``scripts/analyse_ccd_x_ratio.py``; treat the value as an empirical conditioning
        parameter of this landmark and do not "fix" it without replacing the landmark.
    :return: The points constituting the femoral neck and the endpoint of the femoral neck axis.
    """
    G.validate_side(side)
    segmentation_mask = np.where(segmentation_mask == segmentation_label, 1, 0)

    r, center = femoral_head_center
    scale = np.array([1, 1, x_ratio])  # adjust for slice thickness
    point_cloud = np.argwhere(segmentation_mask) * scale
    center = center * scale

    neck_points, com = _femoral_neck_points_and_center(
        point_cloud, r, center, distal_greater_z=True, lateral_greater_x=(side == 'right'))

    return neck_points / scale, com / scale


def get_femoral_neck_center_ct(femur_image: Segmentation, femoral_head_center: Tuple[float, np.ndarray], segmentation_label: int = 1, side: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the endpoint of the femoral neck axis from a whole-leg CT segmentation mask.

    Identical to :func:`get_femoral_neck_center` but works in physical (mm) space and
    with the CT distal/lateral half-space directions (z increases superiorly and the
    sagittal sign is inverted relative to the MRI image-side convention).
    :param femur_image: A segmentation mask of the whole leg.
    :param femoral_head_center: The radius and center of the femoral head (physical mm).
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :return: The points constituting the femoral neck and the endpoint of the femoral neck axis.
    """
    G.validate_side(side)
    mask = np.where(femur_image.array == segmentation_label, 1, 0)
    point_cloud = np.array([femur_image.transform_index_to_physical_point(x) for x in np.argwhere(mask)])
    r, center = femoral_head_center

    neck_points, com = _femoral_neck_points_and_center(
        point_cloud, r, center, distal_greater_z=False, lateral_greater_x=(side == 'left'))

    return neck_points, com

def get_femoral_shaft_axis(hip_mask: np.ndarray, knee_mask: np.ndarray = None, femur_label: int = 1, isotropic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the femoral shaft axis from a segmentation mask.

    Both endpoints are slice centroids and are returned unrounded (continuous voxel indices; only
    the slice index is an integer). They index no array, and the former ``int()`` truncation did
    not commute with the left/right mirror ``x -> n-1-x``: the axis of a reflected hip sat one voxel
    from the mirror of the original's, so the femoral offset read a hip ~1 mm differently
    depending on its image side.
    :param hip_mask: A segmentation mask of the hip.
    :param knee_mask: A segmentation mask of the knee.
    :param femur_label: The label of the femur in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: Start and end point of the vector representing the femoral shaft axis (float voxel indices; the
        start lies in the knee mask's index space when ``knee_mask`` is given, otherwise both lie in the hip's).
    """
    hip_mask = np.where(hip_mask == femur_label, 1, 0)

    point_cloud_hip = np.argwhere(hip_mask)

    offset = 5  # TODO derive this dynamically
    layer_low_hip = np.max(point_cloud_hip[:, 2]) - (offset if isotropic else 1)  # get the most distal layer with a mask point

    com_low_hip = center_of_mass(hip_mask[:, :, layer_low_hip])
    com_low_hip = (com_low_hip[0], com_low_hip[1], layer_low_hip)

    if knee_mask is not None:
        knee_mask = np.where(knee_mask == femur_label, 1, 0)
        point_cloud_knee = np.argwhere(knee_mask)
        layer_high_knee = np.min(point_cloud_knee[:, 2]) + (
            offset if isotropic else 1)  # get the most proximal layer with a mask point
        com_high_knee = center_of_mass(knee_mask[:, :, layer_high_knee])
        com_high_knee = (com_high_knee[0], com_high_knee[1], layer_high_knee)

        return np.array(com_high_knee), np.array(com_low_hip)

    # find the most proximal layer of the greater trochanter, which will serve as the high point of the femoral shaft axis
    connected_components = list()

    two_components_found = False
    components_large_enough = False
    for layer in range(hip_mask.shape[2]):
        connected_components = extract_connected_components_2d(hip_mask[:, :, layer])
        if len(connected_components) == 2:
            two_components_found = True

            if isotropic and (np.count_nonzero(connected_components[0]) < 10 or np.count_nonzero(connected_components[1]) < 10):  # avoid small components that belong to the femoral head
                continue

            components_large_enough = True

            break
    else:
        if not two_components_found:
            raise RuntimeError('No two connected components found in segmentation mask')
        elif not components_large_enough:
            raise RuntimeError('Connected components are too small in segmentation mask')
        else:
            raise RuntimeError('Tip of greater trochanter not found in segmentation mask')

    layer_high_hip = layer  # the layer with two connected components is the one we want
    # get the center of mass of the smaller component
    smaller_component = min(connected_components, key=lambda x: np.count_nonzero(x))
    com_high_hip = center_of_mass(smaller_component)

    com_high_hip = (com_high_hip[0], com_high_hip[1], layer_high_hip)

    return np.array(com_low_hip), np.array(com_high_hip)  # return the two points as the femoral shaft axis

def get_femoral_shaft_axis_ct(femur_image: Segmentation, segmentation_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the femoral shaft axis from a whole-leg CT segmentation mask using PCA.
    :param femur_image: A segmentation mask of the whole leg.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: Start and end point of the vector representing the femoral shaft axis, as float
    physical coordinates in mm. (They are deliberately not cast to integers: truncating
    millimetre coordinates toward zero shifts the axis by up to 1 mm depending on the sign
    of each coordinate, i.e. on where the image sits in the scanner frame.)
    """

    cleaned_mask = np.where(femur_image.array == segmentation_label, 1, 0)

    # Extract shaft region (exclude proximal femur and distal condyles)
    point_cloud = np.argwhere(cleaned_mask)
    point_cloud = np.array([femur_image.transform_index_to_physical_point(x) for x in point_cloud])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
    z_range = z_max - z_min

    # Keep middle 60-70% to avoid complex geometry at ends
    shaft_mask = (point_cloud[:, 2] > z_min + 0.15 * z_range) & \
                 (point_cloud[:, 2] < z_max - 0.15 * z_range)
    shaft_points = point_cloud[shaft_mask]

    # Fit PCA to find principal axis
    pca = PCA(n_components=3)
    pca.fit(shaft_points)

    # First principal component is the shaft axis direction
    centroid = pca.mean_
    direction = pca.components_[0]

    # Project extremes onto the axis to get start/end points
    projections = np.dot(shaft_points - centroid, direction)
    t_min, t_max = np.min(projections), np.max(projections)

    start_point = centroid + t_min * direction
    end_point = centroid + t_max * direction

    return start_point, end_point

def get_femoral_neck_transition(neck_points: np.ndarray, side: str = 'left') -> np.ndarray:
    """
    Get the point where the femoral neck transitions into the femoral head.
    :param neck_points: The points constituting the femoral neck.
    :param side: Side of the image (not patient!), either 'left' or 'right'
    :return: The point where the femoral neck transitions into the femoral head.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    most_proximal_points = neck_points[neck_points[:, 2] == np.min(neck_points[:, 2])]  # get the most proximal points

    if side == 'left':
        most_proximal_medial_point = most_proximal_points[most_proximal_points[:, 0] == np.max(most_proximal_points[:, 0])]  # of the most proximal points, get the most medial one
    else:
        most_proximal_medial_point = most_proximal_points[most_proximal_points[:, 0] == np.min(most_proximal_points[:, 0])]  # of the most proximal points, get the most medial one

    return most_proximal_medial_point[0]  # this point is one possible transition point

def get_acetabular_surface(hip_mask: np.ndarray, femoral_head_center: np.ndarray, radius: float,
                           spacing: np.ndarray, radius_factor: float = 1.4,
                           angular_resolution: float = None) -> np.ndarray:
    """
    Extract the acetabular articular surface from a whole hip bone mask.

    Whole-leg CT label maps carry one label for the entire hip bone, so the rim searches
    that expect an acetabulum-only mask latch onto the iliac wing instead. The acetabulum
    is the part of the hip bone that *faces* the femoral head: casting rays outward from
    the femoral head centre, the first bone voxel each ray meets is on the articular
    surface, and everything behind it (the ilium above the roof, the ischium behind the
    wall) is shadowed. The cast is vectorised -- directions are binned on a spherical grid
    and the nearest voxel per bin is that bin's first hit -- rather than marched voxel by
    voxel.

    Rays that leave the cup through its opening find no bone within ``radius_factor``
    head radii and contribute nothing, so the extracted patch ends at the acetabular rim.
    :param hip_mask: A binary 3D mask of the hip bone (single image side).
    :param femoral_head_center: The femoral head centre in index coordinates.
    :param radius: The femoral head radius in millimetres.
    :param spacing: The voxel size in millimetres along each axis.
    :param radius_factor: How many head radii outward to look for the articular surface.
    :param angular_resolution: The angular bin width in degrees; one first hit is kept per bin.
    Defaults to one voxel of arc at the femoral head surface. Do not set this much finer:
    bins narrower than the surface's own voxel spacing fall between voxel centres and come
    up empty, and an empty bin lets the bone behind the acetabulum through.
    :return: A binary mask of the acetabular articular surface.
    """
    surface = np.zeros_like(hip_mask)
    voxels = np.argwhere(hip_mask)
    if len(voxels) == 0:
        return surface

    offsets = (voxels - np.asarray(femoral_head_center, dtype=float)) * np.asarray(spacing, dtype=float)
    distances = np.linalg.norm(offsets, axis=1)

    within = (distances > 0) & (distances <= radius_factor * radius)
    if not within.any():
        return surface
    voxels, offsets, distances = voxels[within], offsets[within], distances[within]

    # bin the ray directions on a (polar, azimuth) grid; one bin is one ray
    directions = offsets / distances[:, None]
    if angular_resolution is None:
        resolution = float(np.min(spacing)) / radius  # one voxel of arc at the head surface
    else:
        resolution = np.radians(angular_resolution)

    theta = np.arccos(np.clip(directions[:, 2], -1, 1))
    phi = np.arctan2(directions[:, 1], directions[:, 0]) + np.pi
    polar = np.floor(theta / resolution).astype(np.int64)
    # azimuthal circles shrink towards the poles, so widen the bins by 1/sin(theta) to keep
    # every bin about one voxel of arc across -- otherwise polar bins are empty and leak
    azimuth = np.floor(phi * np.maximum(np.sin(theta), resolution) / resolution).astype(np.int64)
    rays = polar * (int(2 * np.pi / resolution) + 2) + azimuth

    # within each ray keep the nearest voxel: sort by (ray, distance) and take each group's first
    order = np.lexsort((distances, rays))
    rays, voxels = rays[order], voxels[order]
    first_hit = np.empty(len(rays), dtype=bool)
    first_hit[0] = True
    np.not_equal(rays[1:], rays[:-1], out=first_hit[1:])

    surface[tuple(voxels[first_hit].T)] = 1
    return surface


def get_p1(acetabulum_array: np.ndarray, side: str = 'left', segmentation_label: int = 3, fhc_y_index: int = -1) -> np.ndarray:
    """
    Get the posterior acetabulum rim.
    :param acetabulum_array: A 2D segmentation mask of the acetabulum
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the acetabulum in the segmentation mask.
    :param fhc_y_index: The y index of the femoral head center, used to determine anterior and posterior.
    :return: The posterior acetabulum rim.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    acetabulum_array = np.where(acetabulum_array == segmentation_label, 1, 0)

    if fhc_y_index == -1:
        fhc_y_index = acetabulum_array.shape[1] // 3  # fallback
    acetabulum_array[:, :fhc_y_index] = 0  # zero out the anterior one-third of the acetabulum

    # p1 = np.argwhere(acetabulum_array[:, acetabulum_array.shape[1] // 3:])  # only look at the posterior two-thirds of the acetabulum
    p1 = np.argwhere(acetabulum_array)
    p1 = p1[p1[:, 0].argmin()] if side == 'left' else p1[p1[:, 0].argmax()]

    return p1

def get_p2(acetabulum_array: np.ndarray, side: str = 'left', segmentation_label: int = 3, fhc_y_index: int = -1) -> np.ndarray:
    """
    Get the anterior acetabulum rim.
    :param acetabulum_array: A 2D segmentation mask of the acetabulum
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the acetabulum in the segmentation mask.
    :param fhc_y_index: The y index of the femoral head center, used to determine anterior and posterior.
    :return: The anterior acetabulum rim.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    acetabulum_array = np.where(acetabulum_array == segmentation_label, 1, 0)

    if fhc_y_index == -1:
        fhc_y_index = acetabulum_array.shape[1] // 3  # fallback
    acetabulum_array[:, fhc_y_index:] = 0

    # p2 = np.argwhere(acetabulum_array[:, :acetabulum_array.shape[1] // 3])  # only look at the anterior one-third of the acetabulum
    p2 = np.argwhere(acetabulum_array)
    p2 = p2[p2[:, 0].argmin()] if side == 'left' else p2[p2[:, 0].argmax()]

    return p2

def get_cartilage_inner_and_outer_surface_points(segmentation_mask: np.ndarray, cartilage_label: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the inner and outer surface points of the cartilage.
    :param segmentation_mask: A 3D segmentation mask of the hip.
    :param cartilage_label: The segmentation label of the cartilage.
    :return: The inner and outer surface points.
    """
    cartilage = np.argwhere(segmentation_mask == cartilage_label)
    radius, center = sphere_fit(cartilage)  # fit a sphere to the cartilage

    # define one sphere that is "inside" of the cartilage and one that is "outside" of it
    resolution = int(np.sqrt(len(cartilage) * 2))  # resolution of the sphere; assuming ~half of the sphere is inside the cartilage, it needs to have twice the number of cartilage points
    inner_sphere = pv.Sphere(radius=0.8 * radius, center=center, theta_resolution=resolution, phi_resolution=resolution)
    outer_sphere = pv.Sphere(radius=1.2 * radius, center=center, theta_resolution=resolution, phi_resolution=resolution)

    # extract the "inner" and "outer" surface points of the cartilage
    inner_surface, outer_surface = dict(), dict()
    cartilage_tree = KDTree(cartilage)

    for point in inner_sphere.points:  # for every sphere point, find the closest cartilage point
        distance, index = cartilage_tree.query(point)
        if index in inner_surface.keys():
            if distance > inner_surface[index]:
                continue  # skip if the cartilage point has already been added with a smaller distance
        inner_surface[index] = distance  # save index and distance for the corresponding cartilage point

    for point in outer_sphere.points:
        distance, index = cartilage_tree.query(point)
        if index in outer_surface.keys():
            if distance > outer_surface[index]:
                continue
        outer_surface[index] = distance

    # need to filter out points that are too far away from the sphere points
    # this is necessary because ~half the sphere is not covered by the cartilage and thus these sphere points
    # should have no corresponding cartilage point
    inner_distances = pd.Series(inner_surface.values())
    outer_distances = pd.Series(outer_surface.values())

    # get the indices of the points that are within the 75th percentile of the distances
    inner_surface_indices = [k for k, v in inner_surface.items() if v < inner_distances.quantile(0.75)]
    outer_surface_indices = [k for k, v in outer_surface.items() if v < outer_distances.quantile(0.75)]

    inner_surface = cartilage[inner_surface_indices]
    outer_surface = cartilage[outer_surface_indices]

    return inner_surface, outer_surface

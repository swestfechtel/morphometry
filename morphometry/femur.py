import math
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass, label, map_coordinates
from scipy.optimize import curve_fit
from skimage.measure import find_contours
from morphometry.hip import get_femoral_head_center, get_femoral_head_center_ct
from morphometry.utils import find_notch, points_on_circle, \
    get_contour, circle_fit, draw_circle, draw_line, extract_connected_components_2d, \
    vec_parallel_rotation_angle
from morphometry.image_io import Image, Segmentation, reflect_image_left_right, split_ct_image
from morphometry import constants as C
from morphometry.geometry import mirror_index_x, rotate_indices, validate_in_plane_isotropy
from typing import Tuple

#: Refinement passes of the dorsal frame the femoral-neck notch is searched in (see
#: :func:`_dorsal_frame_angle`). Two, because the replaced implementation rotated the section twice,
#: re-measuring the frame on the result of the first rotation; it is an iterative refinement of one
#: frame and not two different frames, so the count is part of the published construction rather
#: than a tunable.
_NECK_FRAME_PASSES = 2

#: Voxels either side of the section's own sagittal centre of mass that are excluded when the second
#: dorsal point is picked, so that the two points cannot come from the same side. The coordinate
#: reading of the ``mask[thresh + 3:]`` / ``mask[:thresh - 3]`` slicing of the replaced
#: ``utils.dorsal_points_rotation_angle`` call, including its one-voxel asymmetry between the two
#: sides (see :func:`_dorsal_frame_angle`, which reproduces it deliberately).
_NECK_FRAME_HALF_GAP = 3

#: Voxels of zero margin left around the anatomy when the notch section is sampled
#: (:func:`_sample_section_in_frame`). One is enough and is load-bearing: ``utils.get_contour`` is a
#: ``binary_erosion`` with the default ``border_value=0``, so a mask voxel on the window border
#: would be reported as contour and invent a straight edge for the notch search. With
#: nearest-neighbour sampling a cell returns a mask voxel only if that voxel is within half a voxel
#: of it on every input axis, i.e. within ``0.5 * (|cos a| + |sin a|) <= 0.708`` of it along either
#: axis of the rotated frame, which is less than one cell.
_NECK_SECTION_MARGIN = 1

#: Voxels of zero border the neck-layer tests of :func:`get_femoral_neck_center_lee` are evaluated
#: on. One is enough and is load-bearing for the same kind of reason as
#: :data:`_NECK_SECTION_MARGIN`: ``skimage.measure.find_contours`` does not close a region that
#: reaches the array edge -- it returns one *open* polyline per disjoint interval of border contact,
#: so a femur sitting on two sides of the layer reads as two contours and fails the single-outline
#: test (one side still reads as one polyline and passes, which is why the defect needs a bone on
#: two or more sides). One zero voxel all round puts the whole region strictly inside the evaluated
#: array, which is all the closure needs; a wider border cannot change any contour count, because
#: the border is zero everywhere.
_NECK_LAYER_BORDER = 1

#: How far back along the fitted neck axis the reported reference-line endpoint is placed, in
#: voxels. Historically an inline ``-80``. It is a synthetic far point rather than a landmark on
#: bone, so it often falls outside the array; see the rounding note at the end of
#: :func:`get_femoral_neck_center_lee`.
#: Whether the Lee reference line's reported far endpoint is floored to an integer voxel index.
#:
#: **Shipped `True`, which is the current behaviour, and the repo's own "floats stay floats" rule
#: argues against it**: that point is a *synthetic* far point 80 voxels back along the fitted neck
#: axis, nothing indexes an array with it, and only its direction from the head centre is ever
#: used -- so by the convention recorded in CLAUDE.md it should not be rounded at all.
#:
#: **The cost of changing it is bounded analytically at under three quarters of a degree.** The
#: endpoint is `floor(-80 + cx), floor(-80 * m + cy)`, so its x component is *exactly*
#: :data:`_NECK_AXIS_REACH` from the head centre and only the y component is perturbed, by an
#: amount in `[0, 1)` voxels. With `dx = -80` fixed that tilts the reported bearing -- the only
#: thing used -- by at most `atan(1/80)` = **0.716 deg**, with an expected **0.358**. Against this
#: method's cohort SD of 14.24 deg, and against the 1.430 deg median that task #9's own change
#: moved it by, that is small. It is a bound from the construction and not a cohort measurement,
#: which is the reliable form here: an empirical probe that monkeypatched `np.floor` is *not*
#: trustworthy (it perturbs every floor in the call stack; the tell was that only 24 of 100 halves
#: still resolved) and its numbers are deliberately unrecorded.
#:
#: So this is a decision and not a maintenance task: flipping it to `False` moves every Lee value
#: ever produced, by under 0.72 deg each, and carries a do-not-pool burden for that column.
#: Flipping it is one edit; the `.astype(np.int64)` below goes with it.
_NECK_ENDPOINT_ROUNDED = True

_NECK_AXIS_REACH = 80


def _most_dorsal_coordinate(points: np.ndarray) -> np.ndarray:
    """
    Return the most dorsal (posterior) of a set of 2D coordinates.

    The coordinate-space reading of :func:`morphometry.utils.get_dorsal_mask_point` with
    ``knee=False``: the point with the greatest coronal coordinate, and where several points tie for
    it, the *mean* of their sagittal coordinates floored to an integer. (``get_dorsal_mask_point``
    takes the contour points, but every mask point at the section's greatest coronal coordinate is a
    contour point by definition, so the two see the same set.)

    **The floor is the published behaviour, preserved on purpose.** It coarsens a landmark that
    indexes no array, which this repo would normally keep continuous, and once the frame has been
    rotated the tie is a single point whose sagittal coordinate it simply truncates. It is kept so
    that the first pass of :func:`_dorsal_frame_angle` reproduces the replaced implementation
    bitwise, which is what makes the movement of this change attributable to the one mechanism it
    is about. ``floor`` rather than the original's ``int()``: the two agree on the positive index
    coordinates the original saw, and unlike truncation towards zero ``floor`` commutes exactly with
    an integer translation, so the invariance is provable rather than merely observed. The caller
    passes coordinates *relative* to an integer origin, where this floor is the same quantity as the
    absolute one the original took, by ``floor(x + m) == floor(x) + m``.

    :param points: An Nx2 array of (sagittal, coronal) coordinates; must not be empty.
    :return: The most dorsal point as a float ``(sagittal, coronal)`` pair, its sagittal coordinate
        integral.
    :raises RuntimeError: If ``points`` is empty, which means the section has no mask point on that
        side of its own centre of mass.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        raise RuntimeError('No mask points to take the dorsal femoral-neck landmark from; the '
                           'section does not reach past its own centre of mass.')
    most_dorsal = points[:, 1].max()
    tied = points[points[:, 1] == most_dorsal]
    return np.array([np.floor(tied[:, 0].mean()), most_dorsal])


def _dorsal_frame_angle(points: np.ndarray, origin: np.ndarray) -> float:
    """
    Return the rotation that levels the two dorsal points of a proximal-femur section.

    The frame the femoral-neck notch is searched in: the section is turned until the line between
    its most dorsal (posterior) point and the most dorsal point on the other side of its sagittal
    centre of mass is parallel to the sagittal axis. The frame is then re-measured in that frame and
    the rotations accumulate (:data:`_NECK_FRAME_PASSES` passes), which is the refinement the
    replaced implementation performed by rotating the *image* twice.

    **It is carried out in coordinates, which is what makes it a function of the anatomy.** The
    replaced implementation obtained the second pass's input by resampling the section with
    ``skimage.transform.rotate(mask, -angle, resize=True) > 0``, so the second angle was measured on
    a re-binarised raster whose grid was sized and placed by the *array's* bounds. Measured over the
    100 Augsburg reader-study halves under four pads of the section (which shift every index and
    move no anatomy), the first pass was already exactly invariant on 100 of 100, while the second
    reproduced itself on only 9 of 100, by a median 1.40 and up to 21.19 degrees. Rotating
    coordinates is exact instead: with an integer ``origin``, ``points - origin`` is exact integer
    arithmetic, so a translated section reproduces every rotated coordinate bitwise, and every
    quantity below is either such a coordinate or a floor of one, which commutes with an integer
    translation. The accumulated angle is therefore invariant by construction, not by measurement
    (observed: exact on 400 of 400 probes).

    **The pass is a faithful replica of ``utils.dorsal_points_rotation_angle``, including two
    quirks, so that the first pass stays bitwise identical** (verified on all 100 reader-study
    halves) and the whole movement of this change belongs to the resampling it removes:

    * the exclusion band around the centre of mass is asymmetric by one voxel
      (``>= centre + 3`` against ``<= centre - 4``), because it came from the array slices
      ``mask[centre + 3:]`` and ``mask[:centre - 3]``;
    * the far-side point is reported **three voxels medial of where it is**. The replaced code
      sliced the mask at ``centre + 3`` and then added back ``centre`` rather than ``centre + 3``,
      so on the branch that takes a far-side point (and only on that branch) the dorsal line is
      sheared by three voxels. That is a real defect of the published construction, but it is an
      index-frame-*invariant* one -- a constant shift, whatever the array bounds -- so it is out of
      scope here and preserved rather than silently fixed. Measured: removing it moves the Lee
      torsion on 13 of the 100 Augsburg reader-study halves, by a median 1.15 and up to 10.34
      degrees (p90 over all 100: 0.62). It wants its own backlog item.

    The passes iterate rather than reach a fixed point -- the three-voxel shear above is what denies
    them one -- which is why their count is part of the construction and not a convergence
    criterion. Measured on the 100 halves: the first pass turns the frame by a median 23.47 degrees,
    the second by a median 3.46 (p90 16.68, max 42.89), the third by 0.03 (p90 0.62, max 16.64) and
    the fourth by 0.0008 (max 1.50), so two passes land a median 0.03, p90 0.63 and at most 14.93
    degrees from the frame six passes converge to.

    :param points: An Nx2 array of the section's mask-point ``(sagittal, coronal)`` indices.
    :param origin: The rotation origin, as an integer ``(sagittal, coronal)`` index pair. It does
        not enter the angle (an angle between two points is translation-invariant), but an integer
        origin makes the arithmetic exact; see :func:`morphometry.geometry.rotate_indices`.
    :return: The accumulated rotation angle in degrees, in
        :func:`morphometry.utils.rotate_point`'s sense.
    :raises RuntimeError: If the section holds no mask point past the exclusion band on the side the
        second dorsal point has to come from.
    """
    angle = 0.0
    for _ in range(_NECK_FRAME_PASSES):
        # the coordinates stay RELATIVE to the integer origin, which is what makes the floors below
        # exactly translation-equivariant: `floor(x + m) == floor(x) + m` for an integer `m`, so a
        # relative floor is the same quantity as the absolute one the replaced code took -- but
        # computed on a bitwise-invariant value. Flooring `origin + relative` instead costs one
        # float addition whose rounding need not commute with the translation, which was measured to
        # move the frame by up to 0.32 degrees under a bounding-box crop (and the divider by a
        # voxel) on 3 of 100 Augsburg halves.
        rotated = rotate_indices(points, origin, angle)
        divider = np.floor(rotated[:, 0].mean())
        start = _most_dorsal_coordinate(rotated)
        if start[0] < divider:
            end = _most_dorsal_coordinate(rotated[rotated[:, 0] >= divider + _NECK_FRAME_HALF_GAP])
            end = end - np.array([_NECK_FRAME_HALF_GAP, 0.0])  # the replaced slice-offset arithmetic
        else:
            end = start
            start = _most_dorsal_coordinate(rotated[rotated[:, 0] <= divider - _NECK_FRAME_HALF_GAP - 1])
        angle += vec_parallel_rotation_angle(end - start, np.array([1.0, 0.0]))

    return angle


def _sample_section_in_frame(mask_layer: np.ndarray, origin: np.ndarray, angle: float) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Sample a 2D section on a grid rotated about ``origin``, sized from the anatomy's own extent.

    :func:`morphometry.utils.find_notch` scans an array, so one resampling of the section is
    unavoidable; what matters is that the grid is anchored to a landmark and not to the array. The
    window is placed on ``origin`` and sized to cover every mask voxel's rotated position with
    :data:`_NECK_SECTION_MARGIN` voxels of zero margin, so translating the array translates
    ``origin`` by the same shift and every cell reads the same anatomy at the same ``(i, j)``.

    What it replaced is ``utils.rotate_mask_by_angle``, i.e.
    ``skimage.transform.rotate(mask, -angle, resize=True, preserve_range=True) > 0``, which rotates
    about the *image* centre and sizes the output to hold the rotated array: both the grid's origin
    and its extent moved with the array bounds (over the 100 Augsburg reader-study halves, padding
    the section changed the rotated array's shape on 100 of 100, by a median 12 and up to 53
    voxels). Sampling is nearest-neighbour, as the binarised interpolation it replaces effectively
    was, and ``mode='grid-constant'`` rather than ``'constant'``: scipy's ``'constant'`` returns
    ``cval`` for any coordinate outside the *closed* ``[0, n - 1]``, i.e. it discards the outer half
    voxel of every axis, which is itself a dependence on where the array ends (it was measured to be
    worth 15 degrees on a shallow-FOV case in the alpha-angle reslice).

    **How strong the invariance is, stated precisely.** The window's bounds and the cell grid are
    derived from ``rotate_indices`` about an integer origin and are therefore bitwise invariant under
    a translation. The sampled *coordinates* are not provably so: they add the (translated) integer
    origin to those cells, and that float addition need not commute with the translation, so a
    coordinate sitting within rounding noise of an ``order=0`` boundary could in principle take a
    different voxel. Measured over the 100 Augsburg reader-study halves (501,220 sampled
    coordinates), the closest any coordinate comes to such a boundary is 2.6e-6, about eight orders
    of magnitude above the noise, and every probe -- four pads and a bounding-box crop -- reproduces
    the window and the notch exactly. So: bitwise on everything measured and probabilistically safe,
    not provable. (The same caveat the alpha angle's reslice carries, for the same reason.)

    :param mask_layer: A 2D binary section (non-zero inside the bone).
    :param origin: The rotation origin and window anchor, as an integer ``(sagittal, coronal)``
        index pair.
    :param angle: The rotation angle in degrees, in
        :func:`morphometry.utils.rotate_point`'s sense (the frame
        :func:`morphometry.geometry.rotate_indices` maps into).
    :return: ``(section, origin_cell)`` -- the sampled 2D section as ``uint8``, and the integer cell
        of the window that ``origin`` itself landed on. A point found in the section at cell ``q``
        lies at ``q - origin_cell`` in the rotated frame, so its distance from ``origin`` is
        ``|q - origin_cell|`` with no inverse transform involved.
    :raises RuntimeError: If ``mask_layer`` holds no mask voxel.
    """
    points = np.argwhere(np.asarray(mask_layer) > 0)
    if points.size == 0:
        raise RuntimeError('The femoral neck section holds no mask points.')

    rotated = rotate_indices(points, origin, angle)
    # the window covers the anatomy and the origin itself, with a zero margin all round
    low = np.minimum(np.floor(rotated.min(axis=0)), 0.0).astype(int) - _NECK_SECTION_MARGIN
    high = np.maximum(np.ceil(rotated.max(axis=0)), 0.0).astype(int) + _NECK_SECTION_MARGIN
    shape = tuple(int(n) for n in (high - low + 1))
    origin_cell = -low

    cells = np.stack(np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1),
                                 indexing='ij'), axis=-1).reshape(-1, 2)
    # rotating the cells back by -angle about a zero origin inverts rotate_indices exactly
    coords = np.asarray(origin, dtype=float) + rotate_indices(cells, np.zeros(2), -angle)
    section = map_coordinates(np.asarray(mask_layer), coords.T, order=0, mode='grid-constant',
                              cval=0).reshape(shape)
    section = (section > 0).astype(np.uint8)

    # the margin is load-bearing (see _NECK_SECTION_MARGIN), so it is checked rather than argued
    assert not (section[0].any() or section[-1].any() or section[:, 0].any() or section[:, -1].any()), (
        'The sampled femoral neck section reaches the window border, so its contour would be '
        'invented by the erosion; the window margin is wrong.')
    return section, origin_cell


def contour_femoral_neck(mask: np.ndarray, contour: np.ndarray, layer_selected: int, center: np.ndarray, r: float) -> \
        Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Get the contour of the femoral neck.

    The notch of the selected layer is used for one thing only: the distance ``a`` from the femoral
    head centre to it, in units of the head radius, which sets the two radii ``r_1 = (a - 0.1) * r``
    and ``r_2 = (a + 0.1) * r`` of the ring the neck contour is kept in. Everything downstream of
    the notch search depends on that one scalar.

    **The notch is found in coordinates and in a window anchored to the head centre** (backlog task
    #9), so ``a`` is invariant under translation of the anatomy within the array, where it used to
    depend on where in the array the anatomy sat. It is **not** a pure function of the anatomy: the
    window is anchored on the head centre's *rounded* cell, so the sampling grid's phase is set by
    which integer cell the centre falls in -- see the residual at the end of this docstring. The replaced
    implementation rotated the 2D layer twice with ``skimage.transform.rotate(..., resize=True)``,
    found the notch in the second resampling, and mapped it back through offsets derived from that
    array's *shape* (both ``int()``-truncated) — every one of which depends on where the bone sits
    in the array. Decomposed over the 100 Augsburg reader-study halves under four pads of the
    section (which shift every index while leaving every voxel's physical position untouched): the
    first rotation angle was exactly invariant on 100 of 100, the second on 9 of 100 (median 1.40,
    max 21.19 degrees), the rotated array's shape on 0 of 100 (median 12, max 53 voxels), the
    mapped-back notch on 0 of 100 (median 1.21, max 10.56 voxels) and ``a`` on 0 of 100 (median
    0.046, max 0.527; the notch figures are Euclidean norms of the error, not per-axis maxima). So
    both mechanisms leaked and fixing only the map-back would not have been enough. End to end on the
    production path, over eight pads (the four above at the low end of the axes and the same four at
    the **high** end, which change no anatomy index at all and so must leave the landmark
    *bit-identical* rather than merely shifted), the Lee neck centre was reproduced on **5 of 100**
    halves, median 3, max **30** voxels, and the proximal axis angle on the same 5, by a median 2.06,
    p90 6.42 and up to **20.27** degrees — against a cohort SD of 14.58 degrees for the Lee torsion
    itself. The high-end pads are the harsher half (exact 12 of 100, max 30 voxels / 20.27 degrees,
    against 11 of 100 and 16 / 11.42 at the low end). Now: **100 of 100 on all three, exactly**, at
    both ends.

    There is no inverse transform left, which is what removes the truncations: ``a`` is a distance,
    distances are invariant under rotation, and the head centre is the window's own anchor, so it is
    measured in the sampled frame as ``|notch_cell - head_cell|``.

    **The construction is preserved where it could be, and two things about it did change.** Kept:
    the two-pass dorsal frame (:func:`_dorsal_frame_angle`, whose first pass is bitwise the published
    one), the ``percentage=0.8`` / ``thresh=5`` notch search, the ``0.9`` / ``1.1`` factors on ``a``
    and the contour fit. Changed, and both move values:

    * the **raster the notch is counted on is no longer dilated**. The replaced resampling was
      ``skimage.transform.rotate(..., order=1, preserve_range=True) > 0``, i.e. a *linear*
      interpolation thresholded above zero, which grows the mask by a one-voxel fringe: on the real
      selected layers it carried a mean 1.079 times the mask area (max 1.124), where the
      nearest-neighbour sampling here carries 1.0003 (max 1.004). ``find_notch`` thresholds the
      *number* of empty cells across a row, so the same ``thresh=5`` is a materially easier test on a
      dilated raster than on this one. The constant is unchanged; what it tests is not.
    * the **second frame pass** is measured on coordinates instead of on that resampled raster. The
      frame it lands on differs from the old one by a median 0.46, p90 1.19 and at most 18.2 degrees.

    **Attributed, because the two could each have carried the movement and only one does.** Holding
    the frame at the *old* angle and changing only the window (anchored, undilated, no map-back)
    moves ``a`` on 100 of 100 halves by a median 0.0274 (p90 0.0897, max 0.5224); changing the angle
    on top of that moves it on 55 of 100 by a median 0.0060 (p90 0.0551, max 0.6186), against a
    combined median of 0.0269. So the window is what moves values at the median, and the angle only
    in the tail -- note this is the *contribution*, not the *sensitivity*: ``a`` responds to the
    sampling angle at roughly 0.025 per degree, so the small contribution is because the new frame
    lands near the old one, not because the angle does not matter.

    **It buys invariance, not accuracy, and one half is made worse.** There is no second cue
    favouring the new numbers: scored against the ``a`` the *same implementation* reads on the
    neighbouring layers of the same leg -- the within-leg corroboration that decided the condylar
    line's case -- it is a dead heat at best. Over the 99 halves with such a neighbour, |selected
    minus median of own neighbours| is 0.076 (median) / 0.108 (mean) for the replaced code against
    0.089 / 0.127 here, with the new answer closer on 41 halves and further on 58. (An independent
    measurement with a wider notion of "neighbouring eligible layer" split 49 to 50; either way
    there is no support in the leg for calling the movement a correction.) And ``PA000346``
    image-left is a **degradation**: its Lee
    torsion goes 35.755 -> 3.168 degrees, the only mover above 10 degrees and the whole tail of the
    distribution, because its ``a`` becomes 0.768 -- so ``r_1 = 0.67 r``, and the ring is then drawn
    closer to the head axis than the head's own radius rather than peripheral to it, far too small
    to follow the neck contour the line is fitted to: on that half the notch sits 6.2-16.1 mm from
    the head centre, against 32.2 and 33.1 mm on PA000001's two sound halves.

    (This sentence used to read "the whole 'neck' ring sits *inside* the fitted femoral head ... and
    the contour it keeps is head cortex rather than neck". **That stronger claim is false and was
    corrected in 2026-11 once it was measured.** ``a`` is an in-plane distance in the *selected*
    axial layer while the head is a sphere, so the head's cross-section there has radius
    ``sqrt(r**2 - dz**2)``; the Lee layer is the most proximal qualifying one and sits well off the
    equator. On the four Augsburg-large halves the shipped gate flags, that cross-section is
    0.516 / 0.496 / 0.000 / 0.332 of ``r``, so the notch is *outside* it on three of the four and on
    PA000810 the layer lies below the fitted sphere entirely -- a true containment test would flag
    1 of the 4. The defect is "the ring is too small to lie on the neck", which is why the shipped
    flag is :func:`lee_neck_axis_qc`'s ``lee_ring_within_head_radius`` and not ``lee_ring_in_head``.
    Nothing about the threshold or this leg's numbers changes; only the mechanism is weaker than it
    first looked.) Its own neighbouring layer says 1.753 under both
    implementations, so the new value is the one that disagrees with the leg. (That layer was poor
    before this change too: its old ``a`` of 1.190 was the cohort minimum and left only 7 contour
    points in the fit against a cohort median of 53.) An ``a < 1`` condition catches it, by the
    same argument as ``murphy_base_in_head``. It was deliberately **not** implemented in this commit
    -- 1 of 100 is not the prevalence evidence that gate earned its thresholds from -- and **is now
    implemented**, in :func:`lee_neck_axis_qc` / :func:`get_lee_neck_qc`, once that evidence existed
    on three cohorts. It flags and does not repair, so no returned value becomes a NaN row; see
    :data:`morphometry.constants.LEE_MIN_NOTCH_DISTANCE_FACTOR` for the distribution, the tie-break
    among the three geometrically available thresholds, and the disclosure.

    **Residual: the grid's phase.** The window is anchored on ``np.rint`` of the head centre, so the
    sampling grid's phase relative to the anatomy is set by which integer cell the centre rounds
    into -- the result is a function of the anatomy *and* of that rounded cell, not of the anatomy
    alone. Measured over the 100 halves, moving the anchor by one voxel moves ``a`` by a median
    0.012-0.018, p90 0.038-0.047 and at most 0.366 (per direction), with the notch still resolving on
    100 of 100 in all four: roughly half the magnitude at the median of the dependence this change
    removes (0.027), though both tails are of the same order. It is not a
    regression -- the replaced code had the same sensitivity with the *array* centre in place of the
    head centre, which is exactly what made it position-dependent -- and the rounding is kept
    deliberately, because an unrounded anchor would trade the exact integer-offset argument for
    empirical invariance and move every Lee value a third time for a different measurement with no
    evidence it is better. The forward-looking consequence is the useful half: **any future change to
    the MRI femoral-head fit will move every Lee value again, at about half the scale of this one.**

    :param mask: 3D segmentation mask of the proximal femur.
    :param contour: contour of segmentation mask
    :param layer_selected: index of layer with femoral neck
    :param center: center coordinates of femur head. Its in-plane part anchors the sampling window;
        the Lee path casts it to ``int16``, and an integer anchor is what makes the rotated
        coordinates exact (a continuous one is handled correctly, but is then invariant only to
        floating-point noise).
    :param r: radius of femur head
    :return: adjusted mask, contour of the femoral neck and radii of two circles around the femoral
    head center with 0.9 and 1.1 times the radius of the femoral head
    :raises RuntimeError: If the layer holds no mask voxel, if its dorsal frame cannot be built, or
        if no notch is found.
    """
    mask_new = mask[:, :, layer_selected].copy()
    center_2d = np.array([center[0], center[1]], dtype=float)  # (sagittal, coronal)

    # the sampling grid is anchored to the head centre's cell, which keeps the window's offset
    # arithmetic exact; the head centre's own sub-voxel position is carried separately below
    anchor = np.rint(center_2d).astype(np.int64)
    points = np.argwhere(mask_new > 0)
    if points.size == 0:
        raise RuntimeError('The femoral neck layer holds no mask points.')

    angle = _dorsal_frame_angle(points, anchor)
    section, anchor_cell = _sample_section_in_frame(mask_new, anchor, angle)

    notch = find_notch(section,
                       percentage=0.8,
                       thresh=5,
                       break_after_first=True)
    if notch is None:
        raise RuntimeError('No notch found.')

    # get 2 new radii a little closer and further to the center than the notch. The notch is in the
    # rotated frame and so is the head centre (exactly at `anchor_cell` whenever the caller's centre
    # is an integer index, as the Lee path's is); a rotation preserves distances, so `a` needs no
    # transform back into the caller's index frame.
    head_cell = anchor_cell + rotate_indices(center_2d[None, :], anchor, angle)[0]
    a = np.linalg.norm(np.asarray(notch, dtype=float) - head_cell) / r
    r_1 = (-0.1 + a) * r
    r_2 = (0.1 + a) * r

    # mask_new = mask[layer_selected].copy()

    # get contour only between the two circles on the selected layer. The slice bounds are CLAMPED
    # to the array rather than guarded: the replaced code skipped a nulling outright when the ring
    # reached past the array edge (`if center[1] + c_a < mask.shape[1]`, `if center[1] - c_a > 0`),
    # so on an array cropped close around the hip part of the contour that the ring is supposed to
    # cut survived into the fit, and a negative `center[1] - c_a` even wrapped the anterior slice
    # round to null almost the whole row. Clamping is what the guards were reaching for and is a
    # no-op wherever the ring fits: on all 100 Augsburg reader-study halves no guard fires in the
    # production framing, so no value moves (verified end to end, torsion and endpoint identical on
    # 100 of 100), while an in-plane crop to the mask's bounding box fires them on every half. The
    # loop also covers the whole sagittal extent now; it used to stop at `shape[0] - 1`, leaving
    # whatever lay in the last row uncut, which is the same kind of array-bounds dependence (it too
    # is a no-op on all 100 halves -- the ring never reaches that row in the production framing).
    coronal_extent = mask.shape[1]
    for s in range(mask.shape[0]):  # all sagittal coordinates
        if r_2 ** 2 - (s - center[0]) ** 2 >= 0:
            c_a = int(round(math.sqrt(r_2 ** 2 - (s - center[0]) ** 2)))
            posterior = min(center[1] + c_a, coronal_extent)
            anterior = max(center[1] - c_a, 0)
            contour[s, posterior:, layer_selected] = 0
            mask_new[s, posterior:] = 0  # mask_new is 2D
            contour[s, :anterior, layer_selected] = 0
            mask_new[s, :anterior] = 0
        else:
            contour[s, :, layer_selected] = 0
            mask_new[s] = 0

        if r_1 ** 2 - (s - center[0]) ** 2 >= 0:
            c_b = int(round(math.sqrt(r_1 ** 2 - (s - center[0]) ** 2)))
            inner_low, inner_high = max(center[1] - c_b, 0), min(center[1] + c_b, coronal_extent)
            contour[s, inner_low:inner_high, layer_selected] = 0
            mask_new[s, inner_low:inner_high] = 0

    return mask_new, contour, r_1, r_2

def get_femoral_neck_center_lee(segmentation_mask: np.ndarray, center: np.ndarray, r: float,
                                return_notch_distance: bool = False) -> np.ndarray | Tuple[np.ndarray, float]:
    """
    Get the proximal femoral neck axis, the method this project labels "Lee".

    **The label is a shorthand and the citation matters** (backlog task #2). What is implemented
    here is the automatic method of Schock et al., *Artificial intelligence-based automatic
    assessment of lower limb torsion on MRI*, Sci Rep 11:23244 (2021) -- the paper this pipeline
    derives from -- which the authors describe as "closest to the method of Lee", not as Lee's
    method: "While our method identified the femoral head to locate the peripheral femoral neck,
    Lee used it as part of the reference line. Also, our method defined the femoral neck *axis* as
    the reference line, whereas Lee estimated the femoral neck *centre* as part of the reference
    line." Lee's own method is in Lee et al., *3D femoral neck anteversion measurements based on
    the posterior femoral plane in ORTHODOC*, Med Biol Eng Comput 44:895-906 (2006). So the
    contour-fitted neck **axis** below is the published design and not a misreading of a
    point-based definition.

    The method takes the most proximal axial slice on which the femur is a single outline and a
    circle of radius ``2 * r`` about the head centre still meets bone, keeps the neck contour in a
    ring around the head centre and near the neck's own centre of mass, and fits a straight line
    through the head centre and those points (two fits averaged where the contour is split). Only
    the fitted direction reaches the angle: the returned point is the head centre displaced 80
    voxels along it, and the torsion composition aligns the axial coordinate, so the 80 and the
    returned ``layer_selected`` do not enter the result.

    **Index-frame invariant as of task #9.** Padding the hip half (which shifts every index and
    moves no anatomy) left the reported axis unchanged on only 5 of the 100 Augsburg reader-study
    halves while the notch was read off a rotated resampling of the layer; with
    :func:`contour_femoral_neck`'s read-out fixed it is unchanged on 100 of 100, at both ends of
    every axis. An in-plane crop to the femur's own bounding box is the harder probe, and the clamped
    ring bounds (see :func:`contour_femoral_neck`) took it from **3 of 96** measurable halves to
    **86 of 96**; the layer border below closes the remainder.

    **The layer selection was the whole crop residual, and it was ``find_contours``, not the
    circle.** Measured over the same 100 halves with the femur cropped to its own bounding box at
    margin 0 -- bone sitting *on* the array border, the only framing in which this fires -- the
    selected layer survived on 88, the ``find_contours`` count flipped on 12 (1 -> 2 on nine halves,
    1 -> 3 on three) and the ``2 * r`` circle test flipped on **0**, with the head centre and radius
    bit-identical on all 100. ``skimage.measure.find_contours`` does not close a region that reaches
    the array edge: it returns one *open* polyline per disjoint interval of border contact, so bone
    on two or three sides of the layer reads as two or three contours, the single-outline test
    fails, and the walk runs on to a later layer (7 -> 8, 8 -> 10, 18 -> 21 twice,
    18 -> 19, 8 -> 9, 6 -> 7 twice, 6 -> 8, 7 -> 9 twice, 16 -> 17). Evaluating it on a
    :data:`_NECK_LAYER_BORDER`-voxel zero border makes the layer invariant on **100 of 100** at
    margin 0, and it is **zero-movement in the production framing**: the same layer is selected on
    all 100 uncropped halves, so the Lee neck centre, the proximal axis and the torsion are
    bit-identical and no golden moved.

    ``points_on_circle`` is deliberately left on the raw section, because it cannot be responsible:
    it keeps a sample only if that sample is in bounds **and** on bone, so an out-of-bounds sample is
    already equivalent to a not-bone sample, and a crop that preserves every bone voxel cannot change
    its verdict -- which is why its 0 of 100 above is a property of the construction and not of this
    cohort. (Noted while in that loop and repaired separately in ``d980792``:
    ``utils.points_on_circle`` used to end ``return np.array(points) > 0``, returning a boolean
    array rather than the points it had collected. Only ``len(...) > 0`` is ever read and this is
    its only caller, so the length -- and therefore the layer chosen here -- was the same either
    way, which is why that repair moved nothing.)

    Two halves of the earlier crop residual were never the layer and are unaffected by the border:
    ``PA000012`` image-right and ``PA000667`` image-left (0.68 and 0.67 degrees, one quantisation
    step of the reported endpoint) are the bone touching the array border, where
    ``utils.get_contour``'s ``binary_erosion`` (``border_value=0``) reports the border row as contour
    and adds points to the fit below. Both are exact with a one-voxel margin round the bounding box,
    which is what identifies the cause.

    None of this is realised in production: the guards that the clamp replaced fired on 0 of the 100
    halves in the production framing, and no production caller crops in plane.

    **Being a proximal landmark, this reads systematically LOWER than Murphy, and that is
    anatomy rather than error.** Schock et al.'s own radiologists measured 16.1 deg by the Lee
    method against 32.2 by Murphy, a 16.1 deg gap, and explain it: "Because the distal femoral
    neck is more posterior, femoral torsion was higher" for the more distal landmarks. Their
    algorithm and their Lee reader did not differ significantly (p = 0.859). Lee et al. quantify
    the same effect as a law, the local neck-axis anteversion rising as
    ``theta = 25.3 * (L / D) - 5.4`` for ``L / D`` in 0.1-0.6, where ``L`` is the distal distance
    from the proximal-most neck confluence and ``D`` the femoral head diameter. On the 50 Augsburg
    reader-study hips this slice sits a median 12.8 mm below the head centre and Murphy's 44.9 mm,
    a 32.1 mm separation, for which that law predicts about 19 deg -- against the 18.97 deg
    measured. The ~21% of legs this method calls retrotorted is likewise not a defect: it is
    21.0% observed against 21.3% expected from its own mean and SD (11.59 +/- 14.58 deg).

    What does *not* reconcile is the Augsburg readers' Lee-minus-Murphy gap of 4.3-5.8 deg, which
    is too narrow for landmarks this far apart by both the pipeline and the literature; see
    ``docs/augsburg_reader_bias.md``.
    :param segmentation_mask: A 3D segmentation mask of the femur, binarised to the femur label.
    :param center: The centre of the femoral head, as a voxel index (cast to ``int16`` by the
        caller, because the slice search below indexes the mask with it).
    :param r: The radius of the femoral head, in in-plane voxels.
    :param return_notch_distance: Also return ``a``, the head-centre-to-notch distance in units of
        ``r`` that :func:`contour_femoral_neck` built the ring from, for
        :func:`lee_neck_axis_qc`. The flag widens the return tuple and changes nothing else (the
        precedent is ``return_layers`` on
        :func:`morphometry.hip.get_femoral_head_center`); the default path is untouched, so no
        measured value depends on it.
    :return: A point on the fitted neck axis, 80 voxels from the head centre, as a voxel index.
        Only its direction from the head centre is meaningful. With ``return_notch_distance``, the
        pair ``(point, a)``.
    :raises ValueError: If no slice satisfies the single-outline and ``2 * r`` circle conditions.
    """
    layer_selected = None
    for i in range(segmentation_mask.shape[2]):
        section = segmentation_mask[:, :, i]
        # need to use find_contours here because it can detect disconnected contours -- and it is
        # evaluated on a one-voxel zero border, because it does not close a region that touches the
        # array edge (see the array-border note above). ``points_on_circle`` needs no border: it is
        # unaffected by construction, so it is left on the raw section.
        contour_length = len(find_contours(np.pad(section, _NECK_LAYER_BORDER), 0.8))
        on_circle = points_on_circle(section, r * 2, center[:2])

        if contour_length == 1 and len(on_circle) > 0:
            layer_selected = i
            break

    if layer_selected is None:
        raise ValueError('Could not find a suitable layer for the femoral neck.')

    contour = get_contour(segmentation_mask)

    femoral_neck, contour, r_1, r_2 = contour_femoral_neck(segmentation_mask, contour,
                                                           layer_selected, center,
                                                           r)

    # The ring `contour_femoral_neck` kept the neck contour in is centred on `a * r`, where `a` is
    # its head-centre-to-notch distance in head radii: `r_1 = (a - 0.1) * r` and
    # `r_2 = (a + 0.1) * r` straddle it symmetrically, so their midpoint over `r` recovers `a`
    # without widening that function's documented 4-tuple. It is a reconstruction rather than the
    # value itself, so it carries one rounding of the two radii. Measured against both exact forms
    # (`r_1 / r + 0.1` and `r_2 / r - 0.1`) over 10 real Augsburg halves, the worst spread is
    # 1.110e-16 and 8 of the 10 agree to the bit -- sixteen orders below the
    # `C.LEE_MIN_NOTCH_DISTANCE_FACTOR` threshold it is compared against. It is also only ever read
    # when the caller asks for it, so the default path cannot see it at all.
    notch_distance = (r_1 + r_2) / (2.0 * r)

    # center of femoral neck
    center_fn = np.array(center_of_mass(femoral_neck)).astype(np.int16)

    # get coordinates of the selected contour points
    contour_pts_l = np.nonzero(contour[:, :, layer_selected])

    distance_center = (contour_pts_l[0] -
                       center_fn[0]) ** 2 + (contour_pts_l[1] - center_fn[1]) ** 2
    r_3 = math.sqrt(np.median(distance_center)) * 1.5

    # get contour also only with small distance to center_fn on the selected layer
    # basically what this does is draw a circle with a certain radius around center_fn and exclude all contour points
    # that are outside of that circle. As in ``contour_femoral_neck``, the slice bounds are clamped
    # to the array instead of the nulling being skipped when the circle reaches past its edge.
    coronal_extent = segmentation_mask.shape[1]
    for s in range(segmentation_mask.shape[0]):  # all sagittal coordinates
        if r_3 ** 2 - (s - center_fn[0]) ** 2 >= 0:  # if less than r_3 from center_fn
            c_a = int(round(math.sqrt(r_3 ** 2 - (s - center_fn[0]) ** 2)))
            # null the contour points posterior and anterior of the circle on this row
            contour[s, min(center_fn[1] + c_a, coronal_extent):, layer_selected] = 0
            contour[s, :max(center_fn[1] - c_a, 0), layer_selected] = 0
        else:
            contour[s, :, layer_selected] = 0

    contour_pts_l = np.nonzero(contour[:, :, layer_selected])

    # get index of the contour point after the largest gap
    # -> first contour point on the other side of the segmentation mask
    diff = np.ediff1d(contour_pts_l[1])
    ind_gap = np.argsort(diff)[-1] + 1

    def g(x, m):
        return m * x

    if diff.max() == 1:
        # lsf through the center and the contour points to get its slope
        popt, _ = curve_fit(g, contour_pts_l[0] - center[0],
                            contour_pts_l[1] - center[1])
        m_new = popt[0]
    else:
        # separated lsf through the center and the contour points on both sides
        # to get their slopes and calculate the mean
        popt1, _ = curve_fit(g, contour_pts_l[0][:ind_gap] - center[0],
                             contour_pts_l[1][:ind_gap] - center[1])
        popt2, _ = curve_fit(g, contour_pts_l[0][ind_gap:] - center[0],
                             contour_pts_l[1][ind_gap:] - center[1])
        m_new = np.mean([popt1[0], popt2[0]])

    # Find the endpoint of the reference line: a point `_NECK_AXIS_REACH` voxels back along the
    # fitted neck axis from the head centre. It is a synthetic far point, not a bone voxel, so it
    # routinely lands outside the array -- which is why the rounding has to be `np.floor` and not
    # `int()`. `int()` truncates TOWARD ZERO, so it rounds a positive argument down and a negative
    # one up, and therefore does not commute with adding an integer across zero: under a crop that
    # shifts `center[1]` below the axis reach, the same anatomy was reported one voxel away.
    # Measured over the 100 Augsburg reader-study halves, cropping the femur to its own bounding
    # box at margin 0: the coronal argument fell in (-1, 0] on exactly two halves (PA000012
    # image-right and PA000667 image-left) and `int()` rounded both UP to 0 where `np.floor` gives
    # -1, reproducing the uncropped answer. `np.floor` rounds down for either sign, so
    # floor(x + n) == floor(x) + n for integer n. It is ZERO-MOVEMENT in the production framing:
    # both arguments are non-negative on all 100 uncropped halves, where the two agree exactly.
    # (The sagittal argument is `-80 + center[0]` with `center` already int16, i.e. an exact
    # integer, so its sign never mattered; it is floored too so the pair reads consistently.)
    # `np.floor` and not `int()`: int truncates toward zero, so it does not commute with adding an
    # integer across zero, and a crop that drives `center[1]` below the axis reach made the
    # endpoint off by one (see the docstring). Floored at all only because
    # `_NECK_ENDPOINT_ROUNDED` is True -- that constant carries why, and the <0.72 deg bound on
    # changing it.
    end = np.array([np.floor(-_NECK_AXIS_REACH + center[0]),
                    np.floor((-_NECK_AXIS_REACH) * m_new + center[1]),
                    layer_selected]).astype(np.int64)
    return (end, float(notch_distance)) if return_notch_distance else end

def get_trochanter_major(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1) -> np.ndarray:
    """
    Find the trochanter major.

    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head.
    :param segmentation_label: The label of the segmentation mask.
    :return: The coordinates of the trochanter major.
    """
    # `Image.shape` is the (x, y, z) property -- `nibabel`'s `.shape` or SimpleITK's `GetSize()`.
    # This read `hip_image.get_shape()`, which no `Image` has ever defined, so `method='tomczak'`
    # raised `AttributeError` for its whole advertised life: it is named in
    # `calculate_femoral_torsion`'s docstring and accepted by its assertion, and no caller in the
    # repo exercises it, so nothing caught it. `AttributeError` is also outside the
    # `(RuntimeError, AssertionError, ValueError)` tuple the batch scripts catch, so a cohort run
    # that asked for tomczak would have aborted rather than written a NaN row.
    lateral_extents = np.zeros(hip_image.shape[2])
    array = hip_image.array.copy()
    array = np.where(array == segmentation_label, 1, 0)  # convert to binary mask

    # iterate from superior to inferior and get the maximum lateral extent of each layer
    for i in range(hip_image.shape[2]):
        if np.count_nonzero(array[:, :, i]):
            lateral_extents[i] = np.argwhere(array[:, :, i])[:, 0].min()
        else:
            lateral_extents[i] = 1000

    def check_plausibility() -> bool:
        """
        Check if the trochanter major on the given layer is plausible.

        :return: Whether the trochanter major is plausible.
        """
        return True
        tm = (int(trochanter_major_layer), int(trochanter_major[0]), int(trochanter_major[1]))
        fh = (int(femoral_head_centre[0]), int(femoral_head_centre[1]), int(femoral_head_centre[2]))
        tm_world = hip_image.transform_index_to_physical_point(tm)
        fh_world = hip_image.transform_index_to_physical_point(fh)
        distance = np.linalg.norm(np.array(tm_world) - np.array(fh_world))

        return True if (425 < distance < 515) else False

    # get the layer with the maximum lateral extent
    trochanter_major_layer = np.argmin(lateral_extents)
    while True:
        # on that layer, get the most lateral point, which should be the trochanter major
        try:
            trochanter_major_s = np.argwhere(array[:, :, trochanter_major_layer])[:,
                                 0].min()  # get the minimum sagittal coordinate, i.e. most lateral extent of this slice
            trochanter_major_c = np.median(np.argwhere(array[trochanter_major_s, :,
                                                       trochanter_major_layer]))  # select all points with the same sagittal coordinate and get the median coronal coordinate
            trochanter_major = np.array([trochanter_major_s, trochanter_major_c])
        except IndexError:
            raise RuntimeError('Could not find a plausible trochanter major.')

        if check_plausibility():
            break

        # if the trochanter major is not plausible, continue with the next layer
        trochanter_major_layer += 1

    return np.array([trochanter_major[0], trochanter_major[1], trochanter_major_layer])

def get_trochanter_minor(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1, isotropic: bool = False) -> np.ndarray:
    """
    Find the trochanter minor.

    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head.
    :param segmentation_label: The label of the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: ``(x, y, layer)``: the axial slice through the trochanter minor's centroid, and on it the
        centroid of the bone medial of the femoral shaft's medial edge (voxel indices).
    """
    point, _ = _find_trochanter_minor(hip_image, femoral_head_centre, segmentation_label, isotropic)
    return point


def _find_trochanter_minor(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1,
                           isotropic: bool = False) -> Tuple[np.ndarray, int]:
    """
    Find the trochanter minor and the axial slice of its superior extent.

    The trochanter minor is the smaller component of the first sagittal slice, scanning from medial to
    lateral, that splits the femur into two components one of which lies 40-75 mm from the femoral
    head centre in the medial-lateral / superior-inferior plane.
    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head (voxel index).
    :param segmentation_label: The label of the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The point returned by :func:`get_trochanter_minor`, and the most superior axial slice
        (smallest index; LPI) holding that sagittal component.
    """

    def check_plausibility(trochanter_minor) -> bool | float:
        """
        Check if the trochanter minor on the given layer is plausible.

        :return: Whether the trochanter minor is plausible.
        """
        tm = trochanter_minor.astype(np.int16)
        fh = femoral_head_centre.astype(np.int32)

        tm_world = hip_image.transform_index_to_physical_point(tm)
        fh_world = hip_image.transform_index_to_physical_point(fh)
        tm_world = np.array([tm_world[0], tm_world[2]])  # exclude coronal axis?
        fh_world = np.array([fh_world[0], fh_world[2]])

        distance_world = np.linalg.norm(np.array(tm_world) - np.array(fh_world))

        return True if 75 > distance_world > 40 else distance_world

    hip_mask = hip_image.array.copy()
    hip_mask = np.where(hip_mask == segmentation_label, 1, 0)  # convert to binary mask

    # start, stop, step = (hip_mask.shape[0], 0, -1) if side == 'right' else (0, hip_mask.shape[0], 1)
    start, stop, step = (hip_mask.shape[0] - 1, 0, -1)
    two_components_found = False
    candidate_1, candidate_2 = False, False
    components_large_enough = False
    for layer in range(start, stop, step):
        connected_components = extract_connected_components_2d(hip_mask[layer])
        if len(connected_components) == 2:
            two_components_found = True

            if isotropic and (np.count_nonzero(connected_components[0]) < 10 or np.count_nonzero(connected_components[1]) < 10):
                continue

            components_large_enough = True

            com_1 = center_of_mass(connected_components[0])
            com_2 = center_of_mass(connected_components[1])

            candidate_1 = np.array([layer, com_1[0], com_1[1]])
            candidate_2 = np.array([layer, com_2[0], com_2[1]])

            candidate_1 = check_plausibility(candidate_1)
            candidate_2 = check_plausibility(candidate_2)

            if (candidate_1 is not True) and (candidate_2 is not True):
                continue

            break
    else:
        if not two_components_found:
            raise RuntimeError('Could not find a plausible trochanter minor: no two components found.')
        elif not components_large_enough:
            raise RuntimeError(f'Could not find a plausible trochanter minor: components too small; isotropic={isotropic}.')
        elif (candidate_1 is not True) and (candidate_2 is not True):
            raise RuntimeError(f'Could not find a plausible trochanter minor: no plausible candidates found. Distances: {candidate_1}, {candidate_2}')
        else:
            raise RuntimeError('Could not find a plausible trochanter minor.')

    smaller_component = min(connected_components, key=lambda x: np.count_nonzero(x))

    start = layer
    most_distal_segmented_point_global = np.max(np.argwhere(hip_mask == segmentation_label)[:, 2])

    for layer in range(start, stop, step):
        sagittal_layer = hip_mask[layer]

        try:
            most_distal_segmented_point = np.max(np.argwhere(sagittal_layer == 1)[:, 1])
        except ValueError:
            # print(f'Something happened that should not happen - debug info. most_distal_segmented_point={most_distal_segmented_point}, most_distal_segmented_point_global={most_distal_segmented_point_global}, start={start}, stop={stop}, step={step}, layer={layer}')
            # plt.imshow(sagittal_layer)
            # plt.show()
            continue

        offset = 1 if not isotropic else 10
        if most_distal_segmented_point >= (most_distal_segmented_point_global - offset):  # if the point is at the bottom of the image, we have found the shaft
            break
    else:
        raise RuntimeError('Could not find the femoral shaft')

    com = center_of_mass(smaller_component)
    axial_layer = int(com[1])
    femur_shaft = hip_mask[:, :, axial_layer].copy()
    femur_shaft[:layer] = 0

    # axial_layer_com = center_of_mass(hip_mask[:, :, axial_layer])
    axial_layer_com = center_of_mass(femur_shaft)

    superior_layer = int(np.min(np.nonzero(smaller_component)[1]))  # sagittal slices are (y, z)
    return np.array([axial_layer_com[0], axial_layer_com[1], axial_layer]), superior_layer

def get_femoral_neck_base(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1, isotropic: bool = False) -> np.ndarray:
    """
    Get the centre of the base of the femoral neck as described by Murphy et al.

    Murphy et al. (1987) take the centre of the femoral neck base "at the level just above the lesser
    trochanter": the level is the first axial slice superior of the trochanter minor's superior extent
    (see :func:`_find_trochanter_minor`), i.e. as close above it as the acquisition resolves -- 10 mm on
    the Augsburg hip MRI, 3 mm on whole-leg CT, 1 mm on the isotropic NaKo hips. The centre is the femur
    centroid on that slice, as a continuous index.

    The slice through the trochanter itself, used formerly, includes the trochanter and the proximal
    shaft and read 1-2 degrees more antetorsion on the Augsburg hips. A larger fixed offset was tried
    and rejected: against the NaKo raters' signed manual Murphy values (1 mm data, 120 hips) a 10 mm
    level reads +5.6 degrees high with ICC2 0.85, against 0-5 mm levels (bias -1.2 to +2.2, ICC2 up to
    0.94) -- the raters place the point just above the trochanter, as described. The Augsburg hips cannot
    tell offsets up to 10 mm apart, since they all select the same slice there.

    Two kinds of broken neck geometry are rejected here outright, because they leave no angle to
    measure: a level at or above the femoral head centre (the existing
    ``C.MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`` gate) and a head-to-base axis whose axial-plane
    projection is shorter than one in-plane voxel, i.e. the two landmarks are not even resolved
    apart in the plane whose direction *is* the measurement. Everything else that merely looks
    wrong -- a short (ill-conditioned) axis, a base level inside the fitted head sphere, a base
    centroid outside the bone, a base slice made of several pieces -- is left to
    :func:`murphy_neck_base_qc`, which flags it without changing the number.
    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head (voxel index).
    :param segmentation_label: The label of the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The centre of the base of the femoral neck, ``(x, y, layer)`` with ``x`` and ``y`` continuous.
    :raises RuntimeError: If the level is less than ``C.MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`` below the
        head centre (a mis-detected trochanter), holds no femur, or lies less than one in-plane voxel
        from the head centre in the axial plane (no in-plane direction is resolved).
    """
    _, superior_layer = _find_trochanter_minor(hip_image=hip_image, femoral_head_centre=femoral_head_centre,
                                               segmentation_label=segmentation_label, isotropic=isotropic)
    slice_mm = abs(float(hip_image.spacing[2]))
    level = superior_layer - 1
    if (level - femoral_head_centre[2]) * slice_mm < C.MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM:
        # the neck base lies well distal to the head centre; a level within the head is a mis-detected
        # trochanter minor, and the resulting axis would be a few millimetres long
        raise RuntimeError('Trochanter minor found too close to (or above) the femoral head centre; '
                           'no neck base below the head.')
    if level < 0:
        raise RuntimeError('The neck-base level lies above the image.')
    array = np.where(hip_image.array == segmentation_label, 1, 0)
    if not array[:, :, level].any():
        raise RuntimeError('No femur on the neck-base slice above the trochanter minor.')

    center = center_of_mass(array[:, :, level])
    base = np.array([center[0], center[1], float(level)])

    # The torsion is the in-plane *direction* of head centre -> neck base, so a projection shorter
    # than one voxel carries no direction at all: the two landmarks are the same point at the
    # sampling resolution. This is a distance in the index grid, so it needs no isotropy
    # assumption, and it is independent of the gate above, which is on the z drop -- a perfectly
    # vertical neck clears that gate and lands here. On the 1887 measurable Augsburg / NaKo image
    # halves it never fires (shortest axis 6.4 voxels); it is untested on CT (no data available).
    if float(np.hypot(base[0] - femoral_head_centre[0], base[1] - femoral_head_centre[1])) < 1.0:
        raise RuntimeError('The femoral head centre and the neck base are less than one in-plane '
                           'voxel apart in the axial plane; the neck axis has no direction.')
    return base


def _significant_components(section: np.ndarray, ratio: float = 0.1) -> int:
    """Count the connected pieces of a binary slice, ignoring specks.

    Full (8-)connectivity, and only components worth at least ``ratio`` of the largest count --
    the rule ``scripts/process_nmdid.py::qc_flags`` uses on whole bones: a single stray voxel
    diagonally off the cortex is a labelling speck and moves no centroid, whereas a piece worth a
    tenth of the section is a second lump of bone.
    :param section: A binary 2D slice.
    :param ratio: Minimum size of a counted component as a fraction of the largest one.
    :return: The number of components at or above that size (0 for an empty slice).
    """
    labelled, count = label(section, np.ones((3, 3), dtype=bool))
    if count <= 1:
        return int(count)
    sizes = np.bincount(labelled.ravel())[1:]
    return int(np.count_nonzero(sizes >= ratio * sizes.max()))


def murphy_neck_base_qc(hip_image: Image, femoral_head_centre: np.ndarray, neck_base: np.ndarray,
                        head_radius: float | None = None, segmentation_label: int = 1) -> dict:
    """
    Describe how trustworthy one Murphy neck-base landmark is, without changing it.

    Murphy's femoral-torsion axis is short (~30 mm in the axial plane on the Augsburg hip MRI,
    ~29 mm on the NaKo hips), so a neck base a few millimetres out of place is a large angular
    error, and the measurement looks perfectly plausible when it happens.
    :func:`get_femoral_neck_base` rejects only the two geometries that leave no angle to measure
    at all (a level at or above the head centre, and an axis shorter than one in-plane voxel);
    this function reports the degenerate geometries that still produce a number, so a cohort run
    can mark the row instead of silently trusting it. Nothing here is repaired: a flagged neck is
    still measured, and the returned landmark is the one the measurement used.

    The flags, in the ``name:value`` / ``;``-joined format of ``scripts/process_nmdid.py``'s
    ``qc_flags`` column:

    * ``murphy_axis_short:<mm>`` -- the axial-plane head-to-base axis is shorter than
      ``1 / tan(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG)`` in-plane voxels, i.e. one voxel of landmark
      error would tilt the axis by more than that tolerance (see the constant for the derivation
      and its disclosure). ``<mm>`` is the axis length.
    * ``murphy_base_in_head:<mm>`` -- the neck base lies inside the fitted femoral head sphere,
      which no femoral neck base does: the trochanter was found far too proximal. The test is the
      full 3-D distance from the head centre (in mm, all three spacings) against ``head_radius``,
      not the superior-inferior drop alone: a horizontal neck (coxa valga, a large in-plane offset
      and a small drop) is well outside the sphere and must not be flagged for it. ``<mm>`` is how
      far inside the sphere the base sits. Skipped when no ``head_radius`` is given.
    * ``murphy_base_outside_bone:<mm>`` -- the base centroid does not fall on a femur voxel, so
      the base slice's section is not one compact cross-section the centroid can represent.
      ``<mm>`` is the distance to the nearest femur voxel centre on that slice.
    * ``murphy_base_slice_multipart:<n>`` -- the femur on the base slice is ``n`` disconnected
      pieces (:func:`_significant_components`, so a one-voxel speck does not count), and the
      centroid is then an average over disjoint bone rather than a neck centre. A real neck in two
      pieces (a hole in the segmentation) can still have its centroid near the neck centre, so
      this flags rather than raises.

    :param hip_image: The Image the landmark was found in (the Murphy ``hip_image``; for CT the
        hip crop :func:`get_proximal_reference_line_ct` builds, which shares the femur image's
        geometry). Only its spacing and the base slice of its array are read.
    :param femoral_head_centre: The femoral head centre as a (continuous) voxel index.
    :param neck_base: The neck base as returned by :func:`get_femoral_neck_base`.
    :param head_radius: The fitted femoral head radius in in-plane voxels (the ``r_fh`` of
        :func:`get_proximal_reference_line`), or ``None`` to skip ``murphy_base_in_head``.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: ``{'axis_length_mm', 'axis_length_voxels', 'angle_per_voxel_deg', 'head_clearance_mm',
        'base_distance_to_bone_mm', 'base_in_bone', 'base_slice_components', 'flags'}``, where
        ``head_clearance_mm`` is the base's distance outside the head sphere (negative = inside it,
        ``None`` without a ``head_radius``) and ``flags`` is the ``;``-joined flag string (``''``
        when the neck looks sound). Every entry is a scalar invariant under the left-right mirror,
        except that the ``base_in_bone`` lookup breaks an exactly half-integer centroid towards
        ``+x`` on both sides (``floor(x + 0.5)``; the mirror of that rule would be
        ``ceil(x - 0.5)``), so a centroid landing exactly on a voxel boundary *at the edge of the
        mask* could in principle read differently in the mirrored frame. Otherwise the dict may be
        computed in the reflected frame a right image half is measured in.
    :raises ValueError: If the image's two in-plane voxel edges differ -- the axis length and the
        flag threshold are in in-plane voxels, which is only a length if those agree (and on such
        a dataset the torsion itself is refused, see
        :func:`morphometry.geometry.validate_in_plane_isotropy`) -- or if the base slice holds no
        femur at all (``neck_base`` cannot be a centroid of it; :func:`get_femoral_neck_base`
        rejects that case before returning).
    """
    validate_in_plane_isotropy(hip_image.spacing, 'hip image (Murphy neck-base QC)')
    spacing = np.abs(np.asarray(hip_image.spacing, dtype=float))
    head = np.asarray(femoral_head_centre, dtype=float)
    base = np.asarray(neck_base, dtype=float)
    level = int(round(float(base[2])))

    # in-plane voxels: equal x/y spacing is guaranteed above, so this is a length in voxel edges
    axis_voxels = float(np.hypot(base[0] - head[0], base[1] - head[1]))
    metrics = {
        'axis_length_mm': axis_voxels * float(spacing[0]),
        'axis_length_voxels': axis_voxels,
        # how far one in-plane voxel of landmark error turns the axis
        'angle_per_voxel_deg': float(np.degrees(np.arctan2(1.0, axis_voxels))),
        'head_clearance_mm': None,
    }
    if head_radius is not None:
        # the full 3-D distance in mm, so a neck that runs sideways rather than down is not
        # mistaken for one that starts inside the head
        offset_mm = (base - head) * spacing
        metrics['head_clearance_mm'] = float(np.linalg.norm(offset_mm)) - float(head_radius) * float(spacing[0])

    array = np.where(hip_image.array == segmentation_label, 1, 0)
    section = array[:, :, level]
    points = np.argwhere(section)
    if not len(points):
        raise ValueError('The Murphy neck-base slice holds no femur, so the given neck base is '
                         'not its centroid; the QC has nothing to describe.')
    index = (int(np.floor(base[0] + .5)), int(np.floor(base[1] + .5)))
    metrics['base_in_bone'] = bool(0 <= index[0] < section.shape[0] and 0 <= index[1] < section.shape[1]
                                   and section[index])
    metrics['base_distance_to_bone_mm'] = float(np.min(np.hypot(
        (points[:, 0] - base[0]) * spacing[0], (points[:, 1] - base[1]) * spacing[1])))
    metrics['base_slice_components'] = _significant_components(section)

    flags = []
    if axis_voxels < 1.0 / math.tan(math.radians(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG)):
        # VOXELS, not millimetres. The comparison above is in in-plane voxels ("one voxel of
        # landmark error must not outweigh the tolerance"), and printing a different quantity means
        # the flag's value is not the one the threshold tested -- which on a cohort with several
        # in-plane spacings makes the printed value non-monotone in the verdict. Measured on
        # Augsburg-large: its 951 hip files carry three in-plane spacings (949 at 1.09375, one at
        # 1.145833, one at 1.25 mm), so [15.64, 17.88] mm is an ambiguous band in which an unflagged
        # half prints shorter than a flagged one. The same construction was measured end to end on
        # the condylar line, where it IS realised -- 51 unflagged lines shorter in mm than the
        # longest flagged one -- so this was changed rather than documented. `axis_length_mm` stays
        # in the returned dict, where nothing compares it against the threshold.
        flags.append(f"murphy_axis_short:{metrics['axis_length_voxels']:.2f}")
    if metrics['head_clearance_mm'] is not None and metrics['head_clearance_mm'] < 0:
        flags.append(f"murphy_base_in_head:{-metrics['head_clearance_mm']:.2f}")
    if not metrics['base_in_bone']:
        flags.append(f"murphy_base_outside_bone:{metrics['base_distance_to_bone_mm']:.2f}")
    if metrics['base_slice_components'] > 1:
        flags.append(f"murphy_base_slice_multipart:{metrics['base_slice_components']}")
    metrics['flags'] = ';'.join(flags)
    return metrics


#: Reason codes for ``<method>_landmark_failed``, matched against the exception message in order.
#: All of the Murphy landmark's own failures are ``RuntimeError``, so the type alone says nothing,
#: and this is the flag family with the highest rate by far: 241 of the 2128 Augsburg / NaKo image
#: halves have no Murphy landmark at all, as ``trochanter_implausible`` (109),
#: ``trochanter_no_components`` (69), ``base_at_head_level`` (36, the
#: ``MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`` gate), ``shaft_not_found`` (21) and 6 missing files.
#: The codes carry no ``,`` or ``;``, so they are safe in the CSV columns.
#:
#: The last five are the Lee path's own failures (:func:`get_femoral_neck_center_lee` and
#: :func:`contour_femoral_neck`), added for :func:`get_lee_neck_qc`. The table is shared because
#: the two paths share the femoral-head fit and therefore its failure modes; no Lee marker occurs
#: in any Murphy message or the reverse, so the Murphy codes are unchanged -- the markers are
#: *appended*, and the table is scanned in order, which is what makes that provable rather than
#: merely observed. The Lee axis is a much more reliable landmark than the Murphy one: it resolves
#: on 1896 of the 1908 Augsburg large-cohort image halves (954 cases x 2), against the Murphy
#: landmark's 1678 of 1908 on the same cases. The 12 Lee failures are 6 halves of 3 missing files,
#: 3 ``ValueError`` and 3 ``RuntimeError``.
#:
#: One asymmetry a reader may trip over: a failure raised by the femoral-head fit the two methods
#: *share* comes back unnamed, as ``<method>_landmark_failed:<ExceptionType>``, for Lee exactly as
#: it already did for Murphy. That is parity rather than a gap -- the named codes describe the
#: method-specific searches, and nothing in the shared fit distinguishes the two callers.

_LANDMARK_FAILURE_REASONS = (
    ('no two components found', 'trochanter_no_components'),
    ('components too small', 'trochanter_components_too_small'),
    ('no plausible candidates', 'trochanter_implausible'),
    ('plausible trochanter minor', 'trochanter_not_found'),
    ('Could not find the femoral shaft', 'shaft_not_found'),
    ('too close to', 'base_at_head_level'),
    ('one in-plane voxel', 'axis_unresolved'),
    ('lies above the image', 'level_above_image'),
    ('No femur on the neck-base slice', 'no_femur_on_level'),
    ('holds no femur', 'no_femur_on_level'),
    ('in-plane voxel spacing', 'anisotropic_in_plane'),
    ('suitable layer for the femoral neck', 'neck_layer_not_found'),
    ('No notch found', 'notch_not_found'),
    ('femoral neck layer holds no mask points', 'neck_layer_empty'),
    ('femoral neck section holds no mask points', 'neck_section_empty'),
    ('dorsal femoral-neck landmark', 'dorsal_point_not_found'),
)


def _landmark_failure_flag(exc: BaseException, method: str = 'murphy') -> str:
    """Turn a failed proximal-landmark search into one ``<method>_landmark_failed`` flag.

    :param exc: The exception the landmark search raised.
    :param method: The femoral-torsion method the flag belongs to, ``'murphy'`` or ``'lee'``; it
        only names the flag, and the reason table is shared (see
        :data:`_LANDMARK_FAILURE_REASONS`).
    :return: ``<method>_landmark_failed:<reason>``, where ``<reason>`` is the matching code from
        :data:`_LANDMARK_FAILURE_REASONS` or, for anything unrecognised, the exception's type name
        (so a new failure mode is still visible, just unnamed).
    """
    message = str(exc)
    for marker, reason in _LANDMARK_FAILURE_REASONS:
        if marker in message:
            return f'{method}_landmark_failed:{reason}'
    return f'{method}_landmark_failed:{type(exc).__name__}'


def get_murphy_neck_qc(hip_image: Image, side: str = 'left', segmentation_label: int = 1,
                       x_ratio: float = 1., isotropic: bool = False) -> str:
    """
    QC flags for the Murphy femoral-neck landmarks of one hip MRI image half.

    The entry point the batch scripts use: it resolves the Murphy reference line exactly as
    :func:`morphometry.measurements.femur.calculate_femoral_torsion` and
    :func:`morphometry.measurements.hip.calculate_anteversion` do (so the flags describe the neck
    those measurements actually used) and hands the landmarks to :func:`murphy_neck_base_qc`. The
    reference line is resolved a second time rather than threaded through the measurement
    functions, which keeps their signatures and return types untouched; the cost is one extra
    femoral-head fit and trochanter search per image half.

    A right image half is measured as the left-right reflection of a left one, and every quantity
    :func:`murphy_neck_base_qc` reports is mirror-invariant, so the flags are the same whichever
    frame they are computed in.
    :param hip_image: An Image object of the proximal-femur segmentation mask (one image half).
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param x_ratio: Slice-axis scale factor, as passed to the measurement (inert on this
        path -- see ``measurements.femur.calculate_femoral_torsion`` and backlog task #6).
    :param isotropic: Whether the image has isotropic voxels.
    :return: The ``;``-joined flag string, ``''`` when the neck looks sound, or
        ``murphy_landmark_failed:<reason>`` when the landmark search itself failed (the measurement
        is then NaN too; see :func:`_landmark_failure_flag` for the reason codes).
    """
    try:
        start, end, r_fh = get_proximal_reference_line(hip_image, side=side, method='murphy',
                                                       segmentation_label=segmentation_label,
                                                       x_ratio=x_ratio, isotropic=isotropic)
        return murphy_neck_base_qc(hip_image, start, end, head_radius=r_fh,
                                   segmentation_label=segmentation_label)['flags']
    except Exception as exc:
        # a QC helper must never cost its caller a measurement, and two call sites
        # (the Augsburg scripts, compute_torsion.py) run it outside any try block
        return _landmark_failure_flag(exc)


def get_murphy_neck_qc_ct(femur_image: Segmentation, side: str = 'left', segmentation_label: int = 1) -> str:
    """
    QC flags for the Murphy femoral-neck landmarks of one whole-leg CT image half.

    The CT counterpart of :func:`get_murphy_neck_qc`: it repeats the proximal-femur crop and
    landmark search of :func:`get_proximal_reference_line_ct` (through the shared
    :func:`_ct_proximal_landmarks`) so the flags describe the neck
    :func:`morphometry.measurements.femur.calculate_femoral_torsion_ct` measured.
    :param femur_image: A Segmentation object of the femur (whole-leg) mask, one image half.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The ``;``-joined flag string, ``''`` when the neck looks sound, or
        ``murphy_landmark_failed:<reason>`` (see :func:`_landmark_failure_flag`). Never raises.
    """
    try:
        image = reflect_image_left_right(femur_image) if side == 'right' else femur_image
        hip_image, fhc, r_fh, base = _ct_proximal_landmarks(image, segmentation_label)
        return murphy_neck_base_qc(hip_image, fhc, base, head_radius=r_fh,
                                   segmentation_label=segmentation_label)['flags']
    except Exception as exc:
        return _landmark_failure_flag(exc)


def lee_neck_axis_qc(notch_distance_factor: float, head_radius: float | None = None) -> dict:
    """
    Describe how trustworthy one Lee proximal neck axis is, without changing it.

    The Lee neck axis (:func:`get_femoral_neck_center_lee`) is a straight line fitted to the femoral
    contour inside a ring about the femoral head centre. The ring is placed entirely by one scalar
    that :func:`contour_femoral_neck` derives from the trochlear notch of the selected layer,

        ``a = |notch - head centre| / r``,

    the ring then spanning ``(a - 0.1) * r`` to ``(a + 0.1) * r``, where ``r`` is the radius of the
    *same* femoral-head fit. ``a`` is therefore the ring's centre radius expressed in head radii,
    and the ring is supposed to sit peripheral to the head, on the neck contour. When ``a`` drops
    below 1 it is drawn **closer to the head axis than the head's own radius**, far too small to
    follow that peripheral contour: on the four flagged Augsburg halves the notch sits 6.2 to
    16.1 mm from the head centre, where the two sound halves of PA000001 put it at 32.2 and
    33.1 mm. That is anatomically implausible rather than merely unusual, so the threshold is the
    head radius *itself* rather than a value fitted to a cohort -- like ``murphy_base_in_head``,
    and for the same reason, that the radius it is compared against comes from the same fit.
    **It is not cohort-independent, though**: the ring crosses the head radius at three places
    (outer edge at ``a = 0.9``, centre at 1.0, inner edge at 1.1) and it was the measured
    distribution that chose the centre among them. Read
    :data:`morphometry.constants.LEE_MIN_NOTCH_DISTANCE_FACTOR` before touching the value; the
    short version is "geometric family, data-broken tie".

    **What the flag does NOT say, measured rather than assumed.** It is tempting to describe this as
    "the ring lies inside the femoral head, so the line is fitted to head cortex". On these data that
    is false on three of the four flagged halves, and the distinction is why the flag is named for
    the head *radius* and not for the head. ``a`` is an **in-plane** distance in the selected axial
    layer, while the head is a sphere whose cross-section at that layer has radius
    ``sqrt(r^2 - dz^2)``, smaller than ``r`` wherever the layer is off the equator -- and the Lee
    layer is the most proximal one that qualifies, so it is routinely far off it. Recomputed with the
    *unrounded* head centre (the ``int16`` cast on the Lee branch inflates ``dz`` by up to a whole
    10 mm slice, which is a trap worth naming): the layer's own cross-section is 0.516, 0.496, 0.000
    and 0.332 of ``r`` on PA000346 left, PA000622 right, PA000810 right and PA000932 left, so the
    notch lies **outside** it on three of them, and on PA000810 the layer is below the fitted sphere
    altogether. A containment test would flag **1 of the 4**. ``a < 1`` is therefore deliberately the
    more inclusive reading and not a harmless simplification of a containment test; what bone the
    undersized ring actually keeps depends on the layer. (``contour_femoral_neck``'s own docstring
    carries the stronger "the contour it keeps is head cortex rather than neck" wording from task
    #9, written before this was measured.) Nothing about the gate or about the value 1.0 depends on
    which reading is right -- only the claimed mechanism does.

    Nothing here is repaired: a flagged axis is still measured and the torsion it produced is still
    reported, so a cohort run marks the row instead of silently trusting it. The one flag, in the
    ``name:value`` / ``;``-joined format of ``scripts/process_nmdid.py``'s ``qc_flags`` column:

    * ``lee_ring_within_head_radius:<a>`` -- ``a`` is below :data:`morphometry.constants.LEE_MIN_NOTCH_DISTANCE_FACTOR`
      (see that constant for the derivation, the prevalence survey and the disclosure that its
      threshold was placed using one cohort's own distribution). ``<a>`` is the tested quantity
      itself, as ``murphy_axis_short`` reports the axis length it tested rather than a derived
      excess. **The printed value is rounded to three decimals and the comparison is not**, so an
      ``a`` just under the threshold prints as ``1.000`` and two values either side of 0.9 both
      print as ``0.900``; read the string as a label, not as the number the gate saw. (The same is
      true of ``murphy_axis_short``'s two decimals. It is unreachable on real data here -- the
      closest of the 1896 surveyed halves comes within 0.0192 of 1.0 -- and no finite precision
      removes it, which is why this is documented rather than widened.)

    **The name states the test and nothing more.** ``lee_ring_within_head_radius`` is what ``a < 1``
    establishes: the ring's centre radius is under the fitted head radius. Two shorter names were
    rejected. ``lee_ring_in_head`` (and the parallel ``lee_notch_in_head``) would borrow
    ``murphy_base_in_head``'s phrasing for a weaker test -- that flag really does mean inside the
    sphere, because it compares the full 3-D distance in mm -- and the paragraph above is why that
    difference matters here. ``lee_ring_too_small`` describes the defect but names no reference, and
    the reference is the whole content of the test.

    At the threshold the ring's *centre* crosses the head radius; its outer edge is inside that
    radius only below ``a = 0.9`` (``ring_wholly_within_head_radius``). Both readings flag the same
    four Augsburg halves, and all four are below 0.844, so on the cohort measured the distinction is
    academic -- it is recorded in the returned dict rather than in a second flag.

    :param notch_distance_factor: ``a``, as returned by :func:`get_femoral_neck_center_lee` with
        ``return_notch_distance=True``. It is dimensionless, which is **not** the same as
        shear-free: it is a ratio of two index-space quantities (a voxel-index norm over a
        single-valued voxel radius) and never reads a spacing, so it is the geometric ratio this
        flag names only where the in-plane voxels are square. On in-plane anisotropic data the same
        anatomy can land either side of the threshold -- sampled onto a 1.4 : 1 grid, a phantom's
        ``a`` goes 0.771 -> 1.167 and the verdict flips, where an *isotropic* refinement to 0.5 mm
        moves it only 0.771 -> 0.714 and does not. That is task #10's shear on the very quantity
        the gate tests, so the precondition is enforced one level up, on the image, by
        :func:`get_lee_neck_qc` -- which is also where the measurement enforces it. This function
        takes the scalar on trust.
    :param head_radius: The fitted femoral head radius in in-plane voxels (the ``r_fh`` of
        :func:`get_proximal_reference_line`), or ``None`` to leave the voxel-valued entries
        ``None``. It is reported only; the flag does not depend on it, because ``a`` is already in
        units of it.
    :return: ``{'notch_distance_factor', 'ring_inner_factor', 'ring_outer_factor',
        'ring_wholly_within_head_radius', 'head_radius_voxels', 'notch_distance_voxels',
        'head_clearance_voxels', 'flags'}``, where the three ``factor`` entries are in head radii,
        ``head_clearance_voxels`` is how far outside the head radius the ring centre sits (negative
        = inside, ``None`` without a ``head_radius``, the sign convention of
        :func:`murphy_neck_base_qc`'s ``head_clearance_mm``) and ``flags`` is the ``;``-joined flag
        string, ``''`` when the axis looks sound.
    :raises ValueError: If ``notch_distance_factor`` is not a finite, positive number -- a
        non-finite ``a`` would make the ring itself meaningless and compare false against any
        threshold, so it must not pass silently as "sound".
    """
    a = float(notch_distance_factor)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError(f'The Lee notch distance factor must be finite and positive, got {a!r}.')

    # The two geometric entries below are measured against the head radius itself -- a literal 1,
    # since `a` is already in units of `r` -- and NOT against
    # `C.LEE_MIN_NOTCH_DISTANCE_FACTOR`. They are descriptions of where the ring sits, so they must
    # stay true if that threshold is ever moved off the head surface; only the flag reads the
    # constant. The 0.1 is `contour_femoral_neck`'s published ring half-width.
    metrics = {
        'notch_distance_factor': a,
        'ring_inner_factor': a - 0.1,
        'ring_outer_factor': a + 0.1,
        # the stricter reading of the same geometry: the ring's outer edge is inside the head too
        'ring_wholly_within_head_radius': bool(a + 0.1 < 1.0),
        'head_radius_voxels': None if head_radius is None else float(head_radius),
        'notch_distance_voxels': None,
        'head_clearance_voxels': None,
    }
    if head_radius is not None:
        metrics['notch_distance_voxels'] = a * float(head_radius)
        metrics['head_clearance_voxels'] = (a - 1.0) * float(head_radius)

    flags = []
    if a < C.LEE_MIN_NOTCH_DISTANCE_FACTOR:
        flags.append(f'lee_ring_within_head_radius:{a:.3f}')
    metrics['flags'] = ';'.join(flags)
    return metrics


def get_lee_neck_qc(hip_image: Image, side: str = 'left', segmentation_label: int = 1,
                    x_ratio: float = 1., isotropic: bool = False) -> str:
    """
    QC flags for the Lee proximal neck axis of one hip MRI image half.

    The entry point the batch scripts use, parallel to :func:`get_murphy_neck_qc`: it resolves the
    Lee reference line exactly as
    :func:`morphometry.measurements.femur.calculate_femoral_torsion` does -- the same
    :func:`get_proximal_reference_line` call with the same arguments, only asking it for the ring
    factor as well -- and hands that factor to :func:`lee_neck_axis_qc`. So the flags describe the
    neck axis the measurement actually used, and the layer walk and notch search are not
    reimplemented here, which is what would let the gate and the measurement drift apart. The
    reference line is resolved a second time rather than threaded out of the measurement functions,
    which keeps their signatures and return types untouched; the cost is one extra femoral-head fit,
    layer walk and notch search per image half.

    **The flag is mirror-invariant, and for a stronger reason than the Murphy flags are.** Those
    are mirror-invariant because every scalar :func:`murphy_neck_base_qc` reports happens to be
    preserved by a reflection (bar one half-integer tie-break). Here it is forced by the control
    flow: :func:`get_proximal_reference_line` measures a right image half by *reflecting* it and
    recursing into the left-side path, so a reflected half passed in as the other side **double**
    reflects to a bitwise identical array and ``a`` cannot differ at all -- it is the same
    computation on the same bits, not two computations that agree. (That ``a`` is anyway a ratio of
    two distances a reflection preserves is true and beside the point.) Checked on real data as
    well: on PA000346, whose image-left half is flagged and whose image-right half is not, a half
    and its mirror measured as the other side give bit-identical ``a``
    (``tests/characterization/test_lee_neck_qc.py``).

    **There is deliberately no CT counterpart**, where :func:`get_murphy_neck_qc_ct` has one:
    :func:`get_proximal_reference_line_ct` implements only the Murphy line, so whole-leg CT has no
    Lee neck ring to describe. The asymmetry is in the measurements, not in the QC.

    :param hip_image: An Image object of the proximal-femur segmentation mask (one image half).
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param x_ratio: Slice-axis scale factor, as passed to the measurement (inert on this path --
        see ``measurements.femur.calculate_femoral_torsion`` and backlog task #6).
    :param isotropic: Whether the image has isotropic voxels.
    :return: The ``;``-joined flag string, ``''`` when the axis looks sound, or
        ``lee_landmark_failed:<reason>`` when the landmark search itself failed (the measurement is
        then NaN too; see :func:`_landmark_failure_flag` for the reason codes). Never raises, for
        the same reason :func:`get_murphy_neck_qc` does not: two call sites (the Augsburg scripts
        and ``morphometry/docker/compute_torsion.py``) run it outside any try block.
    """
    try:
        # Refuse in-plane anisotropic voxels, exactly as the measurement does
        # (`measurements/femur.py::calculate_femoral_torsion` validates this same hip image's
        # spacing), so the two cannot disagree about whether the data is measurable at all.
        # Without it this returned '' -- a positive claim that the axis looks sound -- beside a NaN
        # angle, while the Murphy family in the *same* `qc_flags` cell reported
        # `murphy_landmark_failed:anisotropic_in_plane`. It belongs here and not in
        # `lee_neck_axis_qc`, because the scalar genuinely has no spacing; the precondition is the
        # image's.
        validate_in_plane_isotropy(hip_image.spacing, 'hip image (Lee neck-axis QC)')
        _, _, r_fh, notch_distance = get_proximal_reference_line(
            hip_image, side=side, method='lee', segmentation_label=segmentation_label,
            x_ratio=x_ratio, isotropic=isotropic, return_notch_distance=True)
        return lee_neck_axis_qc(notch_distance, head_radius=r_fh)['flags']
    except Exception as exc:
        return _landmark_failure_flag(exc, method='lee')


def get_trochanter_major_center(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1) -> Tuple[np.ndarray, float]:
    """
    Get the centre of the trochanter major at neck base level as described by Tomczak et al.
    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head.
    :param segmentation_label: The label of the segmentation mask.
    :return: The centre and radius of the trochanter major at neck base level.
    """
    trochanter_major = get_trochanter_major(hip_image, femoral_head_centre, segmentation_label)
    trochanter_major_layer = int(trochanter_major[2])

    array = hip_image.array.copy()
    array = np.where(array == segmentation_label, 1, 0)  # convert to binary mask

    lateral_mask_point = np.argwhere(array[:, :, trochanter_major_layer])[:, 0].min()
    medial_mask_point = np.argwhere(array[:, :, trochanter_major_layer])[:, 0].max()

    split_point = (lateral_mask_point + medial_mask_point) // 2
    lateral_mask = array[:, :, trochanter_major_layer].copy()
    lateral_mask[split_point:] = 0  # null everything medial to the split point
    center, radius = circle_fit(lateral_mask)
    center = np.array([center[0], center[1], trochanter_major_layer]).astype(np.int16)
    return center, radius


def get_proximal_reference_line(hip_image: Image, side: str = 'left',
                                method: str = 'lee', segmentation_label: int = 1,
                                x_ratio: float = 1., isotropic: bool = False,
                                return_notch_distance: bool = False) -> Tuple[
                                                                                     np.ndarray, np.ndarray, float, float] | \
                                                                                 Tuple[np.ndarray, np.ndarray, float]:
    """
    Get the proximal reference line of a segmentation mask for calculating the femoral torsion.

    The landmark search is written for a left image half. A right image half holds the mirror
    image of that anatomy, so it is measured by reflecting it left-right
    (:func:`morphometry.image_io.reflect_image_left_right`), running the left-side search on the
    reflection, and mapping the landmarks back with ``x -> n - 1 - x``. This makes the right side
    mirror-equivariant by construction: a mirrored left half measured with ``side='right'`` yields
    exactly the mirrored left-side landmarks. (Patching a separate right-side branch had left it
    systematically different from the left side: a flip off by one voxel, a head-trim that was not
    the mirror of the left one, and rounding by truncation, which is not mirror-consistent.)
    :param hip_image: An Image object of the hip segmentation mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param method: The method to use for determining the reference line.
    :param segmentation_label: The label of the segmentation mask.
    :param x_ratio: Slice-axis scale factor for the femoral-head landmark (inert on this path;
        see ``measurements.femur.calculate_femoral_torsion`` and backlog task #6).
    :param isotropic: Whether the image has isotropic voxels.
    :param return_notch_distance: ``method='lee'`` only: append ``a``, the ring factor
        :func:`contour_femoral_neck` placed the neck ring with, to the returned tuple, for
        :func:`get_lee_neck_qc`. It is a scalar and rides through the right-side mirror below
        untouched, as ``r_fh`` does. The default path is byte-identical to not having the flag, so
        no measured value depends on it; the precedent is ``return_layers`` on
        :func:`morphometry.hip.get_femoral_head_center`.
    :return: The start and end points of the reference line (voxel indices of ``hip_image``), the radius of the
    femoral head centre and the radius of the trochanter major if method is tomczak. With
    ``return_notch_distance``, ``(start, end, r_fh, a)``.
    :raises AssertionError: If ``side`` or ``method`` is not one of the supported values, or if
        ``return_notch_distance`` is asked for a method that has no neck ring.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    assert method in ['lee', 'murphy',
                      'tomczak'], 'Currently, only "lee", "murphy" and "tomczak" are supported as methods.'
    assert not (return_notch_distance and method != 'lee'), \
        'return_notch_distance is a property of the Lee neck ring and is only defined for method="lee".'

    if side == 'right':
        width = hip_image.array.shape[0]
        start, end, *radii = get_proximal_reference_line(reflect_image_left_right(hip_image), side='left',
                                                         method=method, segmentation_label=segmentation_label,
                                                         x_ratio=x_ratio, isotropic=isotropic,
                                                         return_notch_distance=return_notch_distance)
        return (mirror_index_x(start, width), mirror_index_x(end, width), *radii)

    r_fh, center, layer_high, layer_low = get_femoral_head_center(hip_image.array, side=side,
                                                                  segmentation_label=segmentation_label,
                                                                  return_layers=True, x_ratio=x_ratio,
                                                                  isotropic=isotropic, spacing=hip_image.spacing)

    if method == 'lee':
        center = center.astype(np.int16)  # the Lee neck search indexes the mask with the centre
        array = hip_image.array.copy()
        array = np.where(array == segmentation_label, 1, 0)
        if return_notch_distance:
            end, notch_distance = get_femoral_neck_center_lee(array, center, r_fh,
                                                              return_notch_distance=True)
        else:
            end = get_femoral_neck_center_lee(array, center, r_fh)
    elif method == 'murphy':
        # continuous head centre and neck-base centroid: the axis is short (~25 mm in-plane on the
        # Augsburg hips), so truncating either end to an index would rotate it by up to ~2 degrees
        end = get_femoral_neck_base(hip_image=hip_image, femoral_head_centre=center, segmentation_label=segmentation_label, isotropic=isotropic)
    elif method == 'tomczak':
        center = center.astype(np.int16)
        end, r_tm = get_trochanter_major_center(hip_image, center, segmentation_label)

    start: np.ndarray = np.array([center[0], center[1], center[2]])
    if method == 'tomczak':
        return start, end, r_fh, r_tm
    if return_notch_distance:
        return start, end, r_fh, notch_distance
    return start, end, r_fh

def _ct_proximal_landmarks(femur_image: Segmentation, segmentation_label: int = 1) -> Tuple[Segmentation, np.ndarray, float, np.ndarray]:
    """
    Resolve the proximal-femur crop and the Murphy landmarks of a left CT image half.

    Shared by :func:`get_proximal_reference_line_ct` and :func:`get_murphy_neck_qc_ct` so the QC
    flags describe exactly the landmark the measurement used. Written for a left image half, like
    everything the femoral torsion builds on; callers reflect a right half first.
    :param femur_image: A Segmentation object of the femur (whole-leg) mask, one image half.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: ``(hip_image, femoral_head_centre, head_radius, neck_base)`` -- the proximal 30% of
        the femur as its own Segmentation (same geometry as ``femur_image``), the head centre as a
        continuous voxel index, the head radius in in-plane voxels, and the Murphy neck base.
    """
    r_fh, fhc = get_femoral_head_center_ct(femur_image, segmentation_label, 'left')
    tmp = fhc + np.array([r_fh, 0, 0])  # rationale: point on the surface of the femoral head in arbitrary direction

    fhc = femur_image.transform_physical_point_to_index(fhc)
    tmp = femur_image.transform_physical_point_to_index(tmp)  # transform that point to index space
    r_fh = np.linalg.norm(fhc - tmp)  # recalculate radius in index space

    hip_start = np.min(np.argwhere(femur_image.array == segmentation_label)[:, 2])
    femur_end = np.max(np.argwhere(femur_image.array == segmentation_label)[:, 2])
    num_femur_slices = femur_end - hip_start + 1
    hip_end = hip_start + int(num_femur_slices * 0.30)  # take 30% of the femur length to find the femoral neck base

    mask = np.where(femur_image.array == segmentation_label, 1, 0)
    mask[:, :, :hip_start] = 0
    mask[:, :, hip_end:] = 0

    hip_image = Segmentation.from_nibabel(nib.Nifti1Image(mask, femur_image.affine, femur_image.header))
    return hip_image, fhc, float(r_fh), get_femoral_neck_base(hip_image, fhc, segmentation_label, False)


def get_proximal_reference_line_ct(femur_image: Segmentation, side: str = 'left', segmentation_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the proximal reference line of a CT femur segmentation for calculating the femoral torsion.

    As in :func:`get_proximal_reference_line`, a right image half is measured as the left-right
    reflection of a left one (left-side search on
    :func:`morphometry.image_io.reflect_image_left_right`, landmarks mapped back with
    ``x -> n - 1 - x``), so the two sides are mirror-equivariant by construction. The reflection,
    not an index-relabelling mirror, is what is needed here: the femoral head centre is found in
    physical space, where the lateral direction is a physical direction.
    :param femur_image: A Segmentation object of the femur segmentation mask.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The start and end points of the reference line (voxel indices of ``femur_image``; the start is a
    continuous index).
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    if side == 'right':
        width = femur_image.array.shape[0]
        start, end = get_proximal_reference_line_ct(reflect_image_left_right(femur_image), side='left',
                                                    segmentation_label=segmentation_label)
        return mirror_index_x(start, width), mirror_index_x(end, width)

    _, fhc, _, end = _ct_proximal_landmarks(femur_image, segmentation_label)

    start: np.ndarray = np.array([fhc[0], fhc[1], fhc[2]])
    return start, end

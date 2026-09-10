"""Unit tests for the femoral-head slab bound in ``get_femoral_head_center*``.

Both femoral-head-centre functions locate the superior end of the head, drop a
single-voxel column through the centroid of that slice and walk it inferiorly until
it leaves the bone. The layer where it leaves bounds the point cloud handed to
``sphere_fit``. That walk has no upper bound, so whenever the column happens to stay
inside bone past the head — head into neck into shaft, which is exactly what a
straight, vertically-aligned proximal femur does — the sphere is fitted to head plus
neck plus shaft and comes back with an implausible radius centred far inferior of the
real head. It is a silent failure: no exception, just a wrong landmark that every
downstream measurement (CCD, neck axis, torsion, offset, CEA) inherits.

The phantom here is that failure mode made explicit: a 24 mm spherical head sitting on
a straight 9 mm shaft that runs all the way down the volume directly under the head, so
the column never exits. On an anisotropic 1 x 1 x 3 mm grid, mirroring the slice
thickness of the real acquisitions, because the bound has to be computed in physical
length and converted back to a slice count — a bound that silently assumed isotropy
would be 3x wrong here and the test would still pass on a 1 mm grid.

The second phantom is the control: the same head, but with the neck running off
laterally so the column *does* leave the bone at the head's inferior pole. There the
walk already terminates correctly and the bound must not shorten it — the fix is only
allowed to remove the runaway, never to trim a sound slab.
"""
import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import center_of_mass, label

from morphometry import constants as C
from morphometry.hip import (_femoral_head_slab_bound, _largest_component_centroid,
                             get_femoral_head_center,
                             get_femoral_head_center_ct)
from morphometry.image_io import Segmentation

# axcodes -> the affine that maps index space to nibabel's RAS+ world space; each column
# is the world direction of one index axis ('L' = index axis 0 walks toward Left = -x).
AXIS_TO_RAS = {"R": (0, 1.0), "L": (0, -1.0), "A": (1, 1.0), "P": (1, -1.0),
               "S": (2, 1.0), "I": (2, -1.0)}

SHAPE = (80, 80, 60)                  #: phantom volume shape, in voxels
SPACING = (1.0, 1.0, 3.0)             #: anisotropic voxel spacing, in mm
HEAD_CENTER_INDEX = (40.0, 40.0, 8.0)  #: femoral-head centre of the phantom, in voxels
HEAD_RADIUS_MM = 24.0                 #: femoral-head radius of the phantom, in mm
SHAFT_RADIUS_MM = 9.0                 #: shaft / neck radius of the phantom, in mm
#: Index of the phantom's inferior head pole: centre + radius, converted to slices.
HEAD_BOTTOM_INDEX = int(HEAD_CENTER_INDEX[2] + HEAD_RADIUS_MM / SPACING[2])
BONE_END_INDEX = 55                   #: last slice carrying any bone
TROCHANTER_START_INDEX = 16           #: first slice carrying the lateral trochanter block

#: Anatomically plausible adult femoral-head radius, in mm. Cadaveric and CT series put
#: it at roughly 20-28 mm; the band is widened a little so the assertion is about
#: "a femoral head" rather than about population percentiles.
PLAUSIBLE_RADIUS_MM = (18.0, 32.0)


def _affine(axcodes: str = "LPI", spacing: tuple[float, float, float] = SPACING) -> np.ndarray:
    """Build a RAS+ affine whose index axes run along the given anatomical directions.

    :param axcodes: Three axis codes (e.g. ``'LPI'``) giving the direction each index
        axis increases toward.
    :param spacing: Voxel spacing along each index axis, in mm.
    :return: A (4, 4) index->world affine.
    """
    affine = np.zeros((4, 4))
    affine[3, 3] = 1.0
    for index_axis, code in enumerate(axcodes):
        world_axis, sign = AXIS_TO_RAS[code]
        affine[world_axis, index_axis] = sign * spacing[index_axis]
    return affine


def _offsets_mm() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-voxel offsets from the phantom's head centre, in mm, plus the raw z index.

    :return: ``(dx, dy, dz, z_index)`` arrays of shape ``SHAPE``; the first three are
        millimetre offsets from the head centre, the last the voxel index along axis 2.
    """
    xx, yy, zz = np.indices(SHAPE).astype(float)
    return ((xx - HEAD_CENTER_INDEX[0]) * SPACING[0],
            (yy - HEAD_CENTER_INDEX[1]) * SPACING[1],
            (zz - HEAD_CENTER_INDEX[2]) * SPACING[2],
            zz)


def straight_femur() -> np.ndarray:
    """A femoral head on a straight shaft running directly beneath it — the bug case.

    The head is a 24 mm sphere; a 9 mm shaft descends from the head's equator along the
    same (x, y) axis to the bottom of the bone, so the single-voxel column dropped
    through the head's superior centroid stays inside bone for 55 slices (165 mm)
    instead of the 48 mm the head actually occupies. A lateral block starting at the
    head's inferior pole plays the greater trochanter: it merges into the column's
    connected component there, which is the signal that the slab has left the head.
    :return: A ``SHAPE`` uint8 mask with the femur labelled 1.
    """
    dx, dy, dz, z_index = _offsets_mm()
    head = (dx ** 2 + dy ** 2 + dz ** 2) <= HEAD_RADIUS_MM ** 2
    in_bone = (z_index >= HEAD_CENTER_INDEX[2]) & (z_index < BONE_END_INDEX)
    shaft = ((dx ** 2 + dy ** 2) <= SHAFT_RADIUS_MM ** 2) & in_bone
    trochanter = ((np.indices(SHAPE)[0] >= 46) & (np.indices(SHAPE)[0] <= 70)
                  & (np.indices(SHAPE)[1] >= 30) & (np.indices(SHAPE)[1] <= 50)
                  & (z_index >= TROCHANTER_START_INDEX) & in_bone)
    return (head | shaft | trochanter).astype(np.uint8)


def offset_neck_femur() -> np.ndarray:
    """The same head, with the neck running off laterally — the control case.

    The neck's (x, y) axis migrates laterally with depth, so the column through the
    head's superior centroid leaves the bone at the head's inferior pole and the
    unbounded walk already terminates at the right place. Any bound applied here must
    be inert.
    :return: A ``SHAPE`` uint8 mask with the femur labelled 1.
    """
    dx, dy, dz, z_index = _offsets_mm()
    head = (dx ** 2 + dy ** 2 + dz ** 2) <= HEAD_RADIUS_MM ** 2
    # neck axis slides 2.5 mm laterally per slice from the head centre downward
    neck_dx = dx - np.clip(z_index - HEAD_CENTER_INDEX[2], 0.0, None) * 2.5
    neck = (((neck_dx ** 2 + dy ** 2) <= SHAFT_RADIUS_MM ** 2)
            & (z_index >= HEAD_CENTER_INDEX[2]) & (z_index < BONE_END_INDEX))
    return (head | neck).astype(np.uint8)


def _segmentation(mask: np.ndarray) -> Segmentation:
    """Wrap a phantom mask in a Segmentation on the LPI grid the CT path expects.

    :param mask: A ``SHAPE`` uint8 femur mask.
    :return: A nibabel-backed Segmentation carrying the anisotropic LPI affine.
    """
    return Segmentation.from_nibabel(nib.Nifti1Image(mask, _affine()))


def _column_exit_index(mask: np.ndarray, column: tuple[int, int], layer_high: int) -> int:
    """The slice at which a single-voxel column first leaves the bone.

    Reproduces the unbounded walk under test so the control case can assert against the
    walk's own natural stopping point rather than a hard-coded number.
    :param mask: A binary 3D mask.
    :param column: The ``(x, y)`` index of the column.
    :param layer_high: The slice the walk starts one below.
    :return: The first index at or below ``layer_high + 1`` where the column is background.
    """
    layer = layer_high + 1
    while layer < mask.shape[2] and mask[column[0], column[1], layer] != 0:
        layer += 1
    return layer


@pytest.mark.parametrize("side", ["left", "right"])
def test_ct_head_radius_is_plausible_on_a_straight_femur(side):
    """CT: the fitted head must be a head, not head + neck + shaft.

    Without a slab bound the column runs the full length of the bone and ``sphere_fit``
    returns a radius more than twice anatomical, centred well inferior of the real head.
    """
    radius, center = get_femoral_head_center_ct(_segmentation(straight_femur()),
                                                segmentation_label=1, side=side)
    assert PLAUSIBLE_RADIUS_MM[0] <= radius <= PLAUSIBLE_RADIUS_MM[1], (
        f"fitted radius {radius:.1f} mm outside {PLAUSIBLE_RADIUS_MM} mm — "
        "the slab almost certainly ran past the femoral head")


@pytest.mark.parametrize("side", ["left", "right"])
def test_ct_head_center_stays_inside_the_head(side):
    """CT: the fitted centre must land inside the phantom's actual head.

    A radius check alone can be satisfied by a badly placed sphere, so pin the centre
    too: it has to be within one head radius of the truth, in physical millimetres.
    """
    image = _segmentation(straight_femur())
    truth = np.asarray(image.transform_index_to_physical_point(HEAD_CENTER_INDEX), dtype=float)
    _, center = get_femoral_head_center_ct(image, segmentation_label=1, side=side)
    error = float(np.linalg.norm(np.asarray(center, dtype=float) - truth))
    assert error <= HEAD_RADIUS_MM, (
        f"fitted centre is {error:.1f} mm from the true head centre {truth} — "
        "the slab almost certainly ran past the femoral head")


@pytest.mark.parametrize("side", ["left", "right"])
def test_mri_head_radius_is_plausible_on_a_straight_femur(side):
    """MRI: the identical unbounded walk, so the identical failure.

    ``x_ratio`` is the slice-thickness correction, i.e. slice spacing expressed in
    in-plane voxels — 3.0 for this 1 x 1 x 3 mm phantom — so the returned radius is in
    in-plane voxels, which are 1 mm here.
    """
    radius, _ = get_femoral_head_center(straight_femur(), side=side, segmentation_label=1,
                                        x_ratio=SPACING[2] / SPACING[0])
    assert PLAUSIBLE_RADIUS_MM[0] <= radius <= PLAUSIBLE_RADIUS_MM[1], (
        f"fitted radius {radius:.1f} mm outside {PLAUSIBLE_RADIUS_MM} mm — "
        "the slab almost certainly ran past the femoral head")


@pytest.mark.parametrize("side", ["left", "right"])
def test_slab_is_not_shortened_when_the_column_already_exits(side):
    """The control: where the walk terminates on its own, the result must not move.

    This is the property that keeps the fix inert on data it was never meant to touch —
    every MRI golden depends on it. The expected layer is derived from the walk itself,
    so the assertion stays valid if the phantom is retuned.
    """
    mask = offset_neck_femur()
    _, _, layer_high, layer_low = get_femoral_head_center(
        mask, side=side, segmentation_label=1, x_ratio=SPACING[2] / SPACING[0],
        return_layers=True)

    natural_exit = _column_exit_index(mask, (int(HEAD_CENTER_INDEX[0]), int(HEAD_CENTER_INDEX[1])),
                                      layer_high)
    assert natural_exit <= HEAD_BOTTOM_INDEX + 1, "phantom no longer exits at the head pole"
    assert layer_low == natural_exit


# --- the bound itself ------------------------------------------------------------------

COLUMN = (int(HEAD_CENTER_INDEX[0]), int(HEAD_CENTER_INDEX[1]))


def test_bound_measures_one_head_diameter():
    """The bound is the head's own diameter, in slices, not a fixed guess.

    The phantom head reaches its equator at ``HEAD_CENTER_INDEX[2]`` and its inferior
    pole one radius further down, so the walk must be allowed to reach that pole, plus
    the fencepost and tolerance slices, and no further.
    """
    bound = _femoral_head_slab_bound(straight_femur(), COLUMN, 0, SPACING)
    assert HEAD_BOTTOM_INDEX <= bound <= HEAD_BOTTOM_INDEX + C.FEMORAL_HEAD_SLAB_MARGIN_SLICES


def test_bound_is_computed_in_physical_length_not_slices():
    """Halving the slice thickness must double the bound in slices, not leave it alone.

    The failure this pins is an implementation that counts slices instead of millimetres:
    it would return the same number here and on the 3 mm grid, and so be roughly three
    times too permissive or too tight depending on the protocol.
    """
    mask = straight_femur()
    coarse = _femoral_head_slab_bound(mask, COLUMN, 0, SPACING)
    fine = _femoral_head_slab_bound(mask, COLUMN, 0, (SPACING[0], SPACING[1], SPACING[2] / 2))
    assert fine == pytest.approx(2 * coarse, abs=3)


# --- refusing to bound what was never measured -----------------------------------------
#
# Both cases below were found on NMDID whole-leg CT, not reasoned up: a cropped
# acquisition (case-101742) and a starting centroid that lands between two blobs
# (case-127214). In both the largest radius the descent sees is not the head's, and a
# diameter derived from it truncates a slab that was terminating correctly. The bound has
# to recognise that it never saw the equator and stand down.


def truncated_head() -> np.ndarray:
    """A head whose equator is never reached, so no diameter can be derived from it.

    Keeps only the straight femur down to its equator and drops everything below, so the
    column's slice radius rises the whole way and the bone then simply ends. That rising-
    to-the-last-slice profile is the signature the bound has to recognise: the largest
    radius it saw is a chord of the head, not the head's own radius.
    :return: A ``SHAPE`` uint8 mask with the femur labelled 1.
    """
    mask = np.zeros(SHAPE, dtype=np.uint8)
    equator = int(HEAD_CENTER_INDEX[2])
    mask[:, :, :equator + 1] = straight_femur()[:, :, :equator + 1]
    return mask


def test_no_bound_when_the_head_is_cropped_through():
    """A still-rising radius means the equator was missed, so no diameter can be derived.

    Reproduces NMDID case-101742 left, where bounding on the largest radius *seen*
    (14.3 mm, not a head radius) cut a sound 48 mm slab to 33 mm and dropped the fitted
    radius from 19.9 mm to 15.7 mm — a regression the bound must not cause.
    """
    mask = truncated_head()
    assert _femoral_head_slab_bound(mask, COLUMN, 0, SPACING) == SHAPE[2] - 1


def _stack_with_areas(areas: list[int], width: int = 40) -> np.ndarray:
    """Build a mask whose column component has exactly the given area on each slice.

    Lets a test state a slice-area profile literally, which is how the real cases present
    themselves. Each slice is a raster block filled forward from ``COLUMN``, so it is one
    connected component and always contains the column.
    :param areas: Component area, in voxels, for slices 0, 1, 2, ... .
    :param width: Row width of the raster block, in voxels.
    :return: A ``SHAPE`` uint8 mask.
    """
    mask = np.zeros(SHAPE, dtype=np.uint8)
    for z, area in enumerate(areas):
        rows, remainder = divmod(area, width)
        mask[COLUMN[0]:COLUMN[0] + rows, COLUMN[1]:COLUMN[1] + width, z] = 1
        mask[COLUMN[0] + rows, COLUMN[1]:COLUMN[1] + remainder, z] = 1
    return mask


def test_a_single_voxel_flicker_is_not_an_equator():
    """One voxel of rasterisation noise must not pass for the head's widest point.

    These are the literal slice areas of NMDID case-101742 left, whose profile rises the
    whole way except for a 713 -> 708 voxel flicker. Testing merely "did the radius ever
    fall" accepts that flicker as an equator, derives a diameter from a chord and cuts the
    48 mm slab to 36 mm; the fall has to clear the same tolerance as the merge jump.
    """
    areas = [334, 372, 419, 460, 517, 570, 587, 618, 642, 654, 666, 689, 692, 713, 708, 728]
    mask = _stack_with_areas(areas)
    assert _femoral_head_slab_bound(mask, COLUMN, 0, (0.9375, 0.9375, 3.0)) == SHAPE[2] - 1


def test_no_bound_when_the_column_starts_outside_the_bone():
    """Nothing was measured at all, so there is nothing to bound.

    Reproduces NMDID case-127214 right, where the topmost femur slice holds two blobs and
    their centroid falls in the gap between them. Collapsing the slab to a single slice
    there took the fitted radius from 11.8 mm to 5.4 mm.
    """
    empty = np.zeros(SHAPE, dtype=np.uint8)
    assert _femoral_head_slab_bound(empty, COLUMN, 0, SPACING) == SHAPE[2] - 1


def test_bound_never_leaves_the_volume():
    """A head whose diameter overruns the array is clipped, not an out-of-range index.

    The unbounded walk indexed past the end of the volume whenever the bone reached the
    last slice; the bound has to stay in range for every input.
    """
    mask = np.zeros((40, 40, 10), dtype=np.uint8)
    radii = [15, 15, 15, 15, 15, 15, 15, 15, 14, 14]  # peaks, then shrinks: equator seen
    yy, xx = np.mgrid[0:40, 0:40]
    for z, radius in enumerate(radii):
        mask[:, :, z] = ((xx - 20) ** 2 + (yy - 20) ** 2 <= radius ** 2).astype(np.uint8)
    assert _femoral_head_slab_bound(mask, (20, 20), 0, (1.0, 1.0, 1.0)) == 9


# --- where the column starts: two islands on the top slice --------------------------------
#
# A second, independent defect in the same walk, and the one that produces the *short*
# slabs rather than the runaway. `get_femoral_head_center_ct` takes the centroid of the
# ENTIRE topmost femur slice. When the femoral head and the greater trochanter both reach
# that level the slice holds two islands, and the centroid of their union lands in the gap
# between them — on background. The walk's first test is then already false, layer_low
# never advances, and sphere_fit is handed a near-planar ring.
#
# `get_femoral_head_center` (MRI) has guarded this since it was written by keeping only the
# largest connected component of that slice. Confirmed on NMDID case-158269 right, whose
# top slice holds components of 13 and 3 voxels with the union centroid on background;
# taking the largest component's centroid instead reveals a perfectly normal head.

TWO_BLOB_SHAPE = (120, 80, 60)          #: wider grid, so the trochanter island clears the head
TWO_BLOB_HEAD_INDEX = (40.0, 40.0, 7.823)  #: head centre; the z offset puts a real disc on slice 0
TROCHANTER_COLUMN_XY = (95.0, 40.0)     #: centre of the separate trochanter island
TROCHANTER_COLUMN_RADIUS_MM = 3.6       #: its radius; smaller than the head's cap, as anatomy has it
TROCHANTER_MERGE_INDEX = 12             #: slice at which the trochanter joins the shaft


def two_blob_femur() -> np.ndarray:
    """A femoral head and a detached trochanter island reaching the same superior slice.

    The head's centre is offset by a fraction of a slice so that slice 0 carries a real
    disc (81 voxels) rather than a single pole voxel, and the trochanter island (37 voxels)
    sits far enough lateral that the centroid of the union falls between them, on
    background — and stays on background on the next slice down, which is what collapses
    the slab rather than merely misplacing the column. The two join at
    ``TROCHANTER_MERGE_INDEX`` via a bridge, as the real trochanter joins the shaft.
    :return: A ``TWO_BLOB_SHAPE`` uint8 mask with the femur labelled 1.
    """
    xx, yy, zz = np.indices(TWO_BLOB_SHAPE).astype(float)
    dx = (xx - TWO_BLOB_HEAD_INDEX[0]) * SPACING[0]
    dy = (yy - TWO_BLOB_HEAD_INDEX[1]) * SPACING[1]
    dz = (zz - TWO_BLOB_HEAD_INDEX[2]) * SPACING[2]
    head = (dx ** 2 + dy ** 2 + dz ** 2) <= HEAD_RADIUS_MM ** 2
    shaft = ((dx ** 2 + dy ** 2) <= SHAFT_RADIUS_MM ** 2) & (zz >= TWO_BLOB_HEAD_INDEX[2])
    tx = (xx - TROCHANTER_COLUMN_XY[0]) * SPACING[0]
    ty = (yy - TROCHANTER_COLUMN_XY[1]) * SPACING[1]
    trochanter = (tx ** 2 + ty ** 2) <= TROCHANTER_COLUMN_RADIUS_MM ** 2
    bridge = ((xx >= 46) & (xx <= 99) & (yy >= 34) & (yy <= 46)
              & (zz >= TROCHANTER_MERGE_INDEX))
    return (head | shaft | trochanter | bridge).astype(np.uint8)


def test_two_blob_phantom_actually_has_the_defect():
    """Guard the phantom itself: two islands, union centroid on background, head largest.

    Without this the measurement tests below could pass for the wrong reason — a phantom
    that quietly stopped reproducing the defect would look like a fix.
    """
    top = two_blob_femur()[:, :, 0]
    labelled, count = label(top)
    sizes = [int(np.count_nonzero(labelled == i)) for i in range(1, count + 1)]
    assert count == 2, f"expected two islands on the top slice, got {count}"

    centroid = center_of_mass(top)
    column = (int(centroid[0]), int(centroid[1]))
    assert top[column] == 0, "the union centroid must land on background for the defect to bite"

    largest = int(np.argmax(sizes)) + 1
    assert largest == labelled[int(TWO_BLOB_HEAD_INDEX[0]), int(TWO_BLOB_HEAD_INDEX[1])], (
        "the head must be the largest island, else the guard would select the trochanter")


@pytest.mark.parametrize("side", ["left", "right"])
def test_ct_two_blob_top_slice_still_measures_the_head(side):
    """The measurement must survive a two-island top slice.

    Both halves of the assertion carry their own side: with the defect present the left
    side returns a plausible-looking 27.1 mm radius centred 37 mm away from the head,
    while the right side returns a 15.1 mm radius. Only checking both catches both.
    """
    image = _segmentation(two_blob_femur())
    truth = np.asarray(image.transform_index_to_physical_point(TWO_BLOB_HEAD_INDEX), dtype=float)

    radius, center = get_femoral_head_center_ct(image, segmentation_label=1, side=side)
    error = float(np.linalg.norm(np.asarray(center, dtype=float) - truth))

    assert PLAUSIBLE_RADIUS_MM[0] <= radius <= PLAUSIBLE_RADIUS_MM[1], (
        f"fitted radius {radius:.1f} mm outside {PLAUSIBLE_RADIUS_MM} mm — "
        "the column probably started on background between the two islands")
    assert error <= HEAD_RADIUS_MM, (
        f"fitted centre is {error:.1f} mm from the true head centre {truth} — "
        "the column probably started on background between the two islands")


# --- the column must land *inside* the island, not merely near it ------------------------
#
# Picking the largest island is not by itself enough: a centroid is not a guaranteed
# interior point. NMDID case-127214 right has a top slice of islands [48, 1] whose 48-voxel
# component is concave enough that its own centroid falls in its concavity, on background,
# so the column starts outside the bone exactly as it did before the guard. The centroid is
# kept where it already lands inside the component — that keeps the guard inert on the
# ordinary convex case — and otherwise falls back to the nearest component voxel.


def crescent_slice(shape: tuple[int, int] = (80, 80)) -> np.ndarray:
    """A C-shaped component whose own centroid lies in its concavity, plus a speck.

    A 4-voxel-thick arc opening toward +x. Its centroid sits inside the arc's hole, on
    background, which is the geometry that defeats a plain centroid. The single-voxel speck
    reproduces the second island of the real case, so the largest-island selection runs too.
    :param shape: Slice shape in voxels.
    :return: A 2D uint8 mask.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    radius = np.sqrt((xx - 40.0) ** 2 + (yy - 40.0) ** 2)
    angle = np.degrees(np.arctan2(yy - 40.0, xx - 40.0))
    arc = (radius >= 10) & (radius <= 14) & (np.abs(angle) > 50)
    mask = arc.astype(np.uint8)
    mask[5, 5] = 1
    return mask


def test_crescent_phantom_actually_has_the_defect():
    """Guard the phantom: two islands, and the larger one's centroid is on background."""
    mask = crescent_slice()
    labelled, count = label(mask)
    sizes = [int(np.count_nonzero(labelled == i)) for i in range(1, count + 1)]
    assert count == 2 and max(sizes) > 100

    largest = int(np.argmax(sizes)) + 1
    centroid = center_of_mass(labelled == largest)
    assert (labelled == largest)[int(centroid[0]), int(centroid[1])] == 0, (
        "the largest island's centroid must fall outside it for this phantom to bite")


def test_column_lands_inside_a_concave_component():
    """The returned column must be a voxel of the largest island, not just close to it.

    Reproduces NMDID case-127214 right: islands [48, 1] where the larger one is concave, so
    selecting it is not enough — its centroid is still background and the slab still
    collapses.
    """
    mask = crescent_slice()
    labelled, _ = label(mask)
    sizes = [int(np.count_nonzero(labelled == i)) for i in range(1, labelled.max() + 1)]
    largest = int(np.argmax(sizes)) + 1

    column = _largest_component_centroid(mask)
    assert (labelled == largest)[column] == 1, (
        f"column {column} is not inside the largest island — the slab will collapse")


@pytest.mark.parametrize("centre", [(40.0, 40.0), (33.7, 51.2), (20.4, 19.6)])
def test_convex_component_keeps_the_truncated_centroid(centre):
    """On an ordinary convex island the answer must not move by even one voxel.

    This is the inertness pin for the fallback: snapping to the nearest component voxel
    unconditionally would shift the column by up to a voxel on essentially every hip, since
    the truncated centroid is rarely the closest voxel to the true centroid. The fallback
    must engage only when the truncated centroid is actually outside the component.
    """
    yy, xx = np.mgrid[0:80, 0:80]
    disc = (((xx - centre[1]) ** 2 + (yy - centre[0]) ** 2) <= 9.0 ** 2).astype(np.uint8)
    expected_centroid = center_of_mass(disc)
    expected = (int(expected_centroid[0]), int(expected_centroid[1]))
    assert disc[expected] == 1, "phantom disc must contain its own truncated centroid"
    assert _largest_component_centroid(disc) == expected

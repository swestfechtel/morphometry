"""The femoral-neck notch frame and its sampling window: invariance, basis and preconditions.

``femur.contour_femoral_neck`` uses the notch of the selected axial layer for one thing only, the
distance ``a`` from the femoral head centre to it in head radii, which sets the ring the neck
contour is kept in. It used to get there by rotating the 2-D layer twice with
``skimage.transform.rotate(mask, -angle, resize=True) > 0``, finding the notch in that second
resampling, and mapping it back through offsets derived from the rotated array's *shape*. Every one
of those steps depends on where the bone sits in the array, so ``a`` was partly a property of the
index frame: on the 100 Augsburg reader-study halves, padding the layer (which shifts every index
and moves no anatomy) reproduced the second rotation angle on 9 of 100, the rotated array's shape on
0 of 100 and ``a`` on 0 of 100, and the Lee proximal axis end to end on 11 of 100 -- by up to 11.42
degrees, against a cohort SD of 14.58 for the Lee torsion itself.

:func:`morphometry.femur._dorsal_frame_angle` now accumulates the frame in **coordinates** and
:func:`morphometry.femur._sample_section_in_frame` samples the notch raster on a grid anchored to
the head centre's cell, so there is no inverse transform left and ``a`` is read as a distance in the
sampled frame. These tests pin the three things that make that correct rather than merely different:

1. **the frame**, which must still be the published one. The first pass is checked to be bitwise
   identical to ``utils.dorsal_points_rotation_angle``, the raster rule it replicates (it is, on all
   100 reader-study halves as well);
2. **the window and the frame under translation**, which must come from the anatomy alone -- exact
   invariance under a low-end pad, a pad at the *high* end (which changes no anatomy index at all)
   and a crop to the mask's bounding box, plus the one-voxel zero margin that keeps
   ``utils.get_contour``'s ``binary_erosion`` from reading the border as contour;
3. **the sampling basis**, by round-tripping every mask point through the window and back, which
   pins the rotation's sense without the ``skimage`` call that is gone.

The phantom below breaks the replaced implementation: its four pads moved ``a`` by up to 0.17, the
mapped-back notch by up to 6.4 voxels and the accumulated frame angle by 2.4 degrees.
``tests/characterization/test_head_centre_translation.py`` carries the same property inside the
measurement, on the real PA000001 hip.
"""
import numpy as np
import pytest

from morphometry import femur as F
from morphometry.femur import (_NECK_SECTION_MARGIN, _dorsal_frame_angle,
                               _sample_section_in_frame, contour_femoral_neck)
from morphometry.geometry import rotate_indices
from morphometry.utils import dorsal_points_rotation_angle, find_notch, get_contour

from scipy.ndimage import center_of_mass

#: The head centre of :func:`section`, as the integer voxel index the Lee path hands down.
PHANTOM_CENTRE = np.array([46, 21])
#: A femoral head radius for the phantom, in voxels (its head lobe is ~26 voxels across).
PHANTOM_RADIUS = 13.0
#: Index shifts that must leave every reported quantity unchanged: two one-axis pads, a two-axis
#: pad, and a one-voxel pad (the boundary the API's own slab / changepoint splits can land on).
PADS = [(3, 0), (0, 4), (5, 7), (1, 1)]


def section() -> np.ndarray:
    """A 2-D proximal-femur-like section with a resolvable notch.

    Three parts, placed so that the section is a single outline with a genuine dorsal
    discontinuity between the trochanteric lobe and the head: a trochanteric lobe, a narrow neck
    bridge, and the head. The two lobes end at different coronal coordinates, so the dorsal frame
    is turned by a few degrees rather than being a no-op, and the gap either side of the bridge is
    8 voxels deep, which clears the notch search's ``thresh=5``.

    :return: The binary section as ``uint8``, indexed ``(sagittal, coronal)``.
    """
    mask = np.zeros((70, 46), dtype=np.uint8)
    mask[6:25, 6:31] = 1       # trochanteric lobe
    mask[26:32, 13:23] = 1     # the neck bridge
    mask[33:60, 9:34] = 1      # the head
    return mask


def notch_in_frame(mask: np.ndarray, centre: np.ndarray) -> tuple:
    """Run the whole notch read-out on a section, as ``contour_femoral_neck`` does.

    :param mask: A 2-D binary section.
    :param centre: The head centre as an integer ``(sagittal, coronal)`` index pair.
    :return: ``(angle, section_shape, notch_relative_to_the_head_cell, distance)``.
    """
    angle = _dorsal_frame_angle(np.argwhere(mask > 0), centre)
    sampled, anchor_cell = _sample_section_in_frame(mask, centre, angle)
    notch = find_notch(sampled, percentage=0.8, thresh=5, break_after_first=True)
    assert notch is not None, 'the phantom must resolve a notch'
    relative = np.asarray(notch, dtype=float) - anchor_cell
    return angle, sampled.shape, relative, float(np.linalg.norm(relative))


def shifted(mask: np.ndarray, pad) -> np.ndarray:
    """Pad a section at the low end of both axes, which shifts every index by ``pad``."""
    return np.pad(mask, [(pad[0], 0), (pad[1], 0)])


def test_the_phantom_is_not_vacuous(monkeypatch):
    """The phantom's own frame angle must be nonzero, or it proves nothing.

    The replaced implementation short-circuited both the resampling and the map-back when the
    accumulated angle came out exactly 0 (``if angle == 0: notch = notch_rot``), so a section whose
    two dorsal points are already level is trivially invariant under it and would pass every probe
    below against the defect. This phantom's first pass alone turns the frame by 5.5 degrees, and
    the pre-change code moves on all four pads of :data:`PADS` (``a`` by up to 0.17, the mapped-back
    notch by up to 6.4 voxels, the accumulated angle by 2.4 degrees).
    """
    mask = section()
    monkeypatch.setattr(F, '_NECK_FRAME_PASSES', 1)
    first_pass = _dorsal_frame_angle(np.argwhere(mask > 0), PHANTOM_CENTRE)
    assert abs(first_pass) > 1.0, f'the phantom is level: first pass {first_pass} degrees'
    monkeypatch.undo()
    assert abs(_dorsal_frame_angle(np.argwhere(mask > 0), PHANTOM_CENTRE)) > 1.0


def test_first_pass_reproduces_the_published_raster_rule(monkeypatch):
    """One pass of the coordinate frame is bitwise the raster rule it replaces.

    ``utils.dorsal_points_rotation_angle`` was already exactly invariant (100 of 100 reader-study
    halves under four pads), so the first pass is deliberately *not* redesigned: the coordinate
    version reproduces it, quirks included (see :func:`morphometry.femur._dorsal_frame_angle`), and
    the whole movement of the change belongs to the resampling the later passes used to run on.
    """
    mask = section()
    monkeypatch.setattr(F, '_NECK_FRAME_PASSES', 1)
    published = dorsal_points_rotation_angle(mask, np.array(center_of_mass(mask)).astype(np.int16))
    assert _dorsal_frame_angle(np.argwhere(mask > 0), PHANTOM_CENTRE) == published


def test_frame_angle_and_notch_are_invariant_under_a_pad():
    """A pad shifts every index and moves no anatomy, so nothing reported may change."""
    mask = section()
    base = notch_in_frame(mask, PHANTOM_CENTRE)
    for pad in PADS:
        got = notch_in_frame(shifted(mask, pad), PHANTOM_CENTRE + np.asarray(pad))
        assert got[0] == base[0], f'pad {pad} moved the frame angle'
        assert got[1] == base[1], f'pad {pad} moved the window shape'
        np.testing.assert_array_equal(got[2], base[2], err_msg=f'pad {pad} moved the notch')
        assert got[3] == base[3], f'pad {pad} moved the head-centre distance'


def test_frame_angle_and_notch_are_invariant_under_a_high_end_pad():
    """A pad at the *high* end changes no anatomy index at all, so it is the strictest probe."""
    mask = section()
    base = notch_in_frame(mask, PHANTOM_CENTRE)
    got = notch_in_frame(np.pad(mask, [(0, 6), (0, 9)]), PHANTOM_CENTRE)
    assert got[0] == base[0] and got[1] == base[1] and got[3] == base[3]
    np.testing.assert_array_equal(got[2], base[2])


def test_frame_angle_and_notch_are_invariant_under_a_bounding_box_crop():
    """Cropping to the mask's own bounding box removes no labelled voxel, so nothing may change."""
    mask = section()
    base = notch_in_frame(mask, PHANTOM_CENTRE)
    points = np.argwhere(mask > 0)
    low = points.min(axis=0)
    cropped = mask[low[0]:points[:, 0].max() + 1, low[1]:points[:, 1].max() + 1]
    got = notch_in_frame(cropped, PHANTOM_CENTRE - low)
    assert got[0] == base[0] and got[1] == base[1] and got[3] == base[3]
    np.testing.assert_array_equal(got[2], base[2])


def test_the_phantom_pins_the_frame_and_the_distance():
    """The construction itself, so a redefinition cannot pass as an invariance fix."""
    angle, shape, relative, distance = notch_in_frame(section(), PHANTOM_CENTRE)
    assert angle == pytest.approx(6.345109818094518, abs=1e-9)
    assert shape == (60, 31)
    np.testing.assert_allclose(relative, [-17.5, 4.0])
    assert distance / PHANTOM_RADIUS == pytest.approx(1.3808710054627877, abs=1e-9)


def test_sampled_section_keeps_a_zero_margin():
    """No mask voxel may reach the window border, or ``get_contour`` would invent a straight edge.

    ``utils.get_contour`` is a ``binary_erosion`` with the default ``border_value=0``, so a voxel on
    the border reads as contour. The margin is what the notch search's ``percentage`` and the dorsal
    point both rest on.
    """
    mask = section()
    points = np.argwhere(mask > 0)
    angle = _dorsal_frame_angle(points, PHANTOM_CENTRE)
    sampled, _ = _sample_section_in_frame(mask, PHANTOM_CENTRE, angle)
    assert not sampled[0].any() and not sampled[-1].any()
    assert not sampled[:, 0].any() and not sampled[:, -1].any()
    # and the window comes from the anatomy's extent rather than the array's: it is the rotated
    # extent plus the margin, up to the cell the extreme point rounds into on each side
    rotated = rotate_indices(points, PHANTOM_CENTRE, angle)
    extent = np.ceil(rotated.max(axis=0)) - np.floor(rotated.min(axis=0)) + 1
    np.testing.assert_array_less(np.asarray(sampled.shape), extent + 2 * _NECK_SECTION_MARGIN + 1)


def test_sampled_section_is_the_rotated_coordinates():
    """The window's basis, round-tripped: it is exactly the rotation ``rotate_indices`` applies.

    Forwards, every mask point's rotated cell is set in the window; backwards, every set cell maps
    onto a mask voxel. Together these pin the sense of the rotation, which the replaced
    ``skimage.transform.rotate`` call used to define (and which cannot be guessed: the opposite
    sense would also produce a plausible-looking section).
    """
    mask = section()
    points = np.argwhere(mask > 0)
    angle = _dorsal_frame_angle(points, PHANTOM_CENTRE)
    sampled, anchor_cell = _sample_section_in_frame(mask, PHANTOM_CENTRE, angle)

    cells = np.rint(anchor_cell + rotate_indices(points, PHANTOM_CENTRE, angle)).astype(int)
    assert sampled[cells[:, 0], cells[:, 1]].all(), 'a mask point did not land on a set cell'

    set_cells = np.argwhere(sampled > 0)
    back = np.rint(np.asarray(PHANTOM_CENTRE, dtype=float)
                   + rotate_indices(set_cells - anchor_cell, np.zeros(2), -angle)).astype(int)
    assert mask[back[:, 0], back[:, 1]].all(), 'a set cell did not come from a mask voxel'


def test_anchor_lands_on_an_integer_cell():
    """The head centre is the window's origin, which is what removes the inverse transform."""
    mask = section()
    angle = _dorsal_frame_angle(np.argwhere(mask > 0), PHANTOM_CENTRE)
    sampled, anchor_cell = _sample_section_in_frame(mask, PHANTOM_CENTRE, angle)
    assert anchor_cell.dtype.kind in 'iu'
    assert sampled[tuple(anchor_cell)] == mask[tuple(PHANTOM_CENTRE)]


def test_contour_femoral_neck_radii_are_invariant():
    """The two ring radii, the only thing the notch reaches, are unchanged by a pad."""
    mask = np.stack([section()] * 3, axis=-1)
    centre = np.array([PHANTOM_CENTRE[0], PHANTOM_CENTRE[1], 1])
    _, _, r_1, r_2 = contour_femoral_neck(mask, get_contour(mask), 1, centre, PHANTOM_RADIUS)
    for pad in PADS:
        padded = np.pad(mask, [(pad[0], 0), (pad[1], 0), (0, 0)])
        moved = np.array([centre[0] + pad[0], centre[1] + pad[1], 1])
        _, _, p_1, p_2 = contour_femoral_neck(padded, get_contour(padded), 1, moved, PHANTOM_RADIUS)
        assert (p_1, p_2) == (r_1, r_2), f'pad {pad} moved the neck ring'


def test_the_ring_crop_is_clamped_not_skipped():
    """The ring's nulling must reach the array edge, not be abandoned when the ring overruns it.

    The replaced code guarded each nulling with ``center[1] + c_a < mask.shape[1]`` /
    ``center[1] - c_a > 0`` and skipped it outright when the ring reached past the array, so on an
    array cropped close around the hip part of the contour the ring is meant to cut survived into
    the fit (and a negative anterior bound wrapped the slice round to null nearly the whole row).
    Here the anatomy sits hard against both coronal edges, so the ring must overrun them: what is
    kept has to be the ring's own band, with no contour left outside it.
    """
    mask = np.stack([section()[:, 4:36]] * 3, axis=-1)   # trimmed in coronal, ring overruns both ends
    centre = np.array([PHANTOM_CENTRE[0], PHANTOM_CENTRE[1] - 4, 1])
    kept, contour, r_1, r_2 = contour_femoral_neck(mask, get_contour(mask), 1, centre, PHANTOM_RADIUS)

    rows, cols = np.nonzero(contour[:, :, 1])
    distance = np.sqrt((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2)
    # one voxel of slack: the band's bounds are rounded to whole coronal indices per row
    assert distance.size and distance.min() >= r_1 - 1.5 and distance.max() <= r_2 + 1.5, (
        f'contour survives outside the ring [{r_1:.2f}, {r_2:.2f}]: '
        f'[{distance.min():.2f}, {distance.max():.2f}]')


def test_a_section_without_a_second_condyle_raises_runtime_error():
    """A section that does not reach past its own centre of mass has no second dorsal point.

    It must raise ``RuntimeError`` and nothing else: ``compute_torsion.py`` and the batch scripts
    catch ``(RuntimeError, AssertionError, ValueError)`` per measurement, so any other exception
    type aborts a cohort run instead of writing a NaN row. (The replaced implementation raised
    ``IndexError`` here, from ``argsort`` on an empty contour.)
    """
    mask = np.zeros((40, 30), dtype=np.uint8)
    mask[18:22, 10:20] = 1  # a sliver, narrower than twice the exclusion band
    with pytest.raises(RuntimeError):
        _dorsal_frame_angle(np.argwhere(mask > 0), np.array([20, 15]))


def test_an_empty_section_raises_runtime_error():
    """An empty layer has no extent to size a window from."""
    with pytest.raises(RuntimeError):
        _sample_section_in_frame(np.zeros((20, 20), dtype=np.uint8), np.array([10, 10]), 0.0)

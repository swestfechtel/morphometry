"""The Lee neck-layer selection: it must not depend on where the array ends.

``femur.get_femoral_neck_center_lee`` walks the axial layers from the top and takes the first on
which the femur is a *single outline* and a circle of radius ``2 * r`` about the head centre still
meets bone. The single-outline test was ``len(find_contours(layer, 0.8)) == 1`` on the raw layer,
and ``skimage.measure.find_contours`` does not close a region that reaches the array edge: it
returns one **open** polyline per disjoint interval of border contact. So a femur touching two sides
of its layer read as two contours, failed the test, and the walk ran on to a later layer -- a
property of the array bounds, not of the anatomy. On the 100 Augsburg reader-study halves, cropping
the femur to its own bounding box at margin 0 moved the selected layer on 12 of 100 while the
circle test flipped on 0 and the head centre and radius were bit-identical on all 100.

The fix evaluates ``find_contours`` on a :data:`morphometry.femur._NECK_LAYER_BORDER`-voxel zero
border, which puts the region strictly inside the evaluated array whatever the real array bounds
are. ``points_on_circle`` is left on the raw layer because it is unaffected by construction (it
keeps a sample only if that sample is in bounds *and* on bone, so an out-of-bounds sample is already
equivalent to a not-bone sample); these tests pin that too, so the asymmetry in the loop is not
taken on trust.

The phantom below breaks the pre-change code: it selects layer 3 there and layer 2 both with the
border and with a one-voxel margin round the anatomy.
"""
import numpy as np
import pytest

from morphometry import femur as F
from morphometry.femur import _NECK_LAYER_BORDER, get_femoral_neck_center_lee
from morphometry.utils import points_on_circle

from skimage.measure import find_contours

#: The phantom's head centre, as the integer voxel index the Lee path hands down.
PHANTOM_CENTRE = np.array([10, 10, 2], dtype=np.int16)
#: The phantom's head radius in voxels. The layer test's circle has radius ``2 * r``, so this has
#: to be small enough that a circle of 8 voxels about :data:`PHANTOM_CENTRE` meets bone on both of
#: the phantom's bone layers -- otherwise the circle test, not the contour test, would decide.
PHANTOM_RADIUS = 4.0
#: The layer whose bone touches two sides of the array, i.e. the one the pre-change code skips.
BORDERED_LAYER = 2
#: The layer the pre-change code falls through to.
FALLBACK_LAYER = 3


class _Reached(Exception):
    """Sentinel raised once the layer has been selected, so the phantom need go no further."""


def _disc(shape, centre, radius) -> np.ndarray:
    """A filled disc in a 2-D array, clipped by the array bounds.

    :param shape: The ``(sagittal, coronal)`` shape of the array.
    :param centre: The disc's ``(sagittal, coronal)`` centre, which may sit near an edge so that
        the disc is cut by it.
    :param radius: The disc's radius in voxels.
    :return: The binary disc as ``uint8``.
    """
    s, c = np.ogrid[:shape[0], :shape[1]]
    return ((s - centre[0]) ** 2 + (c - centre[1]) ** 2 <= radius ** 2).astype(np.uint8)


def phantom() -> np.ndarray:
    """A 3-D phantom whose proximal bone layer is cut by the array border.

    Two empty layers (which the walk skips, having no contour at all), then a head-like disc
    centred near the ``(0, 0)`` corner so that it is cut by **two** sides of the array -- the
    situation of a femoral head reaching the edge of the field of view, and the one a bounding-box
    crop at margin 0 creates on real data -- and then three layers holding a disc that sits strictly
    inside the array. A circle of radius ``2 * PHANTOM_RADIUS`` about :data:`PHANTOM_CENTRE` meets
    bone on every one of the four bone layers, so the single-outline test is the only thing that can
    separate them.

    :return: The binary phantom as ``uint8``, indexed ``(sagittal, coronal, axial)``.
    """
    mask = np.zeros((40, 40, 6), dtype=np.uint8)
    mask[:, :, BORDERED_LAYER] = _disc(mask.shape[:2], (10, 10), 12)
    for layer in range(FALLBACK_LAYER, mask.shape[2]):
        mask[:, :, layer] = _disc(mask.shape[:2], (20, 20), 12)
    return mask


def bordered(mask: np.ndarray, margin: int = 1) -> np.ndarray:
    """Pad a phantom in plane, which moves no anatomy but takes it off the array border.

    This is the reference the selected layer has to agree with: the same bone, in an array with
    room around it, where ``find_contours`` closes every region whatever the code does.

    :param mask: A 3-D phantom.
    :param margin: Voxels of zero margin to add at both ends of both in-plane axes.
    :return: The padded phantom. Axial indices are untouched, so layer numbers are comparable.
    """
    return np.pad(mask, [(margin, margin), (margin, margin), (0, 0)])


def selected_layer(mask: np.ndarray, centre: np.ndarray, monkeypatch) -> int:
    """The layer the real walk selects, without running the notch search that follows it.

    ``femur.contour_femoral_neck`` takes ``layer_selected`` as its third argument, so intercepting
    it reads the selection out of the shipped function itself rather than out of a replica of its
    loop -- and raising from the interceptor keeps the phantom from having to satisfy the notch
    search as well.

    :param mask: A 3-D binary femur mask.
    :param centre: The head centre as a voxel index; only its first two components are read.
    :param monkeypatch: pytest's ``monkeypatch`` fixture.
    :return: The selected axial layer index.
    :raises AssertionError: If the walk found no layer at all.
    """
    seen = {}

    def intercept(_mask, _contour, layer_selected, *args, **kwargs):
        seen['layer'] = int(layer_selected)
        raise _Reached

    monkeypatch.setattr(F, 'contour_femoral_neck', intercept)
    with pytest.raises(_Reached):
        get_femoral_neck_center_lee(mask, centre, PHANTOM_RADIUS)
    monkeypatch.undo()
    assert 'layer' in seen, 'the walk selected no layer'
    return seen['layer']


def test_the_phantom_is_not_vacuous():
    """The phantom must actually exercise the defect, or it proves nothing.

    Three things have to hold for this phantom to discriminate: the bordered layer's bone has to
    touch two sides of the array (one side still reads as a single open polyline and passes the
    test, so it would not fire), its raw contour count has to differ from its bordered one, and the
    circle test has to pass on it -- otherwise the pre-change code would be skipping that layer for
    a reason this change does not address.
    """
    mask = phantom()
    section = mask[:, :, BORDERED_LAYER]
    sides = [section[0].any(), section[-1].any(), section[:, 0].any(), section[:, -1].any()]
    assert sum(bool(s) for s in sides) >= 2, 'the phantom does not reach two array sides'
    assert len(find_contours(section, 0.8)) > 1, 'the raw layer is already a single outline'
    assert len(find_contours(np.pad(section, _NECK_LAYER_BORDER), 0.8)) == 1
    assert len(points_on_circle(section, PHANTOM_RADIUS * 2, PHANTOM_CENTRE[:2])) > 0
    # ... and the fallback layer passes both tests either way, so the walk has somewhere to go.
    fallback = mask[:, :, FALLBACK_LAYER]
    assert len(find_contours(fallback, 0.8)) == 1
    assert len(points_on_circle(fallback, PHANTOM_RADIUS * 2, PHANTOM_CENTRE[:2])) > 0


def test_selected_layer_matches_the_bordered_anatomy(monkeypatch):
    """The layer must be the one chosen for the same bone with room around it.

    This is the property the change buys, and it is what fails against the pre-change code: there
    the raw phantom selects :data:`FALLBACK_LAYER` while its padded twin -- identical anatomy, every
    index shifted -- selects :data:`BORDERED_LAYER`.
    """
    mask = phantom()
    reference = selected_layer(bordered(mask), PHANTOM_CENTRE[:2] + 1, monkeypatch)
    assert reference == BORDERED_LAYER
    assert selected_layer(mask, PHANTOM_CENTRE, monkeypatch) == reference


@pytest.mark.parametrize('margin', [1, 2, 5])
def test_selected_layer_is_invariant_under_an_in_plane_margin(monkeypatch, margin):
    """Adding zero margin moves no anatomy, so the selection may not move either.

    Several margins, because the border is zero everywhere: once the region is strictly inside the
    evaluated array no further margin can change a contour count, and the selection must therefore
    be flat in the margin rather than merely right at one value of it.
    """
    mask = phantom()
    base = selected_layer(mask, PHANTOM_CENTRE, monkeypatch)
    padded = selected_layer(bordered(mask, margin), PHANTOM_CENTRE[:2] + margin, monkeypatch)
    assert padded == base


def test_the_circle_test_needs_no_border():
    """``points_on_circle`` is unaffected by the array bounds, which is why it is left raw.

    It keeps a sample only if the sample is in bounds **and** on bone, so a sample outside the array
    is already treated exactly like a sample off the bone. Checked here on the bordered layer, whose
    ``2 * r`` circle is the one that could lose samples: its verdict is the same on the raw layer and
    on a padded one, so the asymmetry in the walk is a property of the two tests and not an
    oversight.
    """
    section = phantom()[:, :, BORDERED_LAYER]
    raw = points_on_circle(section, PHANTOM_RADIUS * 2, PHANTOM_CENTRE[:2])
    for margin in (1, 2, 5):
        padded = points_on_circle(np.pad(section, margin), PHANTOM_RADIUS * 2,
                                  PHANTOM_CENTRE[:2] + margin)
        assert len(padded) == len(raw)


def test_no_suitable_layer_still_raises_value_error(monkeypatch):
    """A mask with no single-outline layer must raise, and raise ``ValueError``.

    The batch scripts and ``docker/compute_torsion.py`` catch
    ``(RuntimeError, AssertionError, ValueError)`` per measurement and write a NaN row; anything
    else would abort a cohort run. The border must not turn that failure into something else -- nor
    into a silent success, which it would if a zero border could conjure an outline out of an empty
    layer.
    """
    mask = np.zeros((40, 40, 4), dtype=np.uint8)
    mask[5:12, 5:12, 1] = 1
    mask[20:27, 20:27, 1] = 1  # two disconnected pieces on every bone layer
    with pytest.raises(ValueError):
        get_femoral_neck_center_lee(mask, PHANTOM_CENTRE, PHANTOM_RADIUS)

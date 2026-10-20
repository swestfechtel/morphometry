"""Unit tests for the posterior-condylar reference line and its no-notch fallback.

``morphometry.knee.get_knee_reference_line`` uses ``find_notch`` for exactly one thing: the
sagittal coordinate that separates the two condyles, so that the posterior-most mask point of one
condyle can be told from the posterior-most point of the other. Sections that do not resolve the
intercondylar gap at all (a solid, roughly convex outline) have no notch to find, and the old
fallback rotated such a section so that its *longest contour chord* pointed along ``+x`` -- taking
that undirected chord in its ``-x`` representative, which turned the section by ~140-181 degrees
and therefore reported the ANTERIOR cortex as "dorsal". The fallback now uses the section centroid
as the divider and leaves the frame alone (the mask is in the LPI index frame, where the coronal
axis increases posteriorly, so "posterior" is given and no 180 degree ambiguity arises).

The phantoms here are deliberately asymmetric in the anterior-posterior direction, so that the
anterior tangent and the posterior tangent of the same section have *different* tilts: that is what
makes the tests able to tell the posterior cortex from the anterior one, which is the defect.
"""
import logging
import math

import numpy as np
import pytest
from scipy.ndimage import binary_erosion
from skimage.draw import disk, ellipse, polygon

from morphometry.knee import get_condylar_divider, get_knee_reference_line
from morphometry.utils import find_notch

SHAPE = (130, 130)
# A quadrilateral plateau: posterior edge (20, 60) -> (90, 70), anterior edge (20, 20) -> (90, 10).
# Its posterior tangent tilts by +8.13 degrees and its anterior tangent by -8.13, so the sign of the
# reported tilt alone says which cortex the reference line sits on.
QUAD = ((20, 20), (90, 10), (90, 70), (20, 60))
QUAD_POSTERIOR_TILT = math.degrees(math.atan2(70 - 60, 90 - 20))


def _fill(vertices, shape=SHAPE) -> np.ndarray:
    """Rasterise a polygon into a 2D binary section.

    :param vertices: Sequence of ``(sagittal, coronal)`` vertices.
    :param shape: Shape of the section.
    :return: The filled ``uint8`` mask.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    rr, cc = polygon(np.array([v[0] for v in vertices], dtype=float),
                     np.array([v[1] for v in vertices], dtype=float), shape)
    mask[rr, cc] = 1
    return mask


def _rotate(vertices, degrees: float, centre=(55, 40)):
    """Rotate polygon vertices about a centre, so a rotated section can be rasterised exactly.

    Rotating the vertices (rather than resampling the raster) keeps the phantom's edges straight
    and its notch status unchanged, which an interpolated rotation does not.
    :param vertices: Sequence of ``(sagittal, coronal)`` vertices.
    :param degrees: Rotation angle; positive turns ``+x`` towards ``+y``.
    :param centre: Centre of rotation.
    :return: The rotated vertices.
    """
    a = math.radians(degrees)
    out = []
    for x, y in vertices:
        dx, dy = x - centre[0], y - centre[1]
        out.append((centre[0] + dx * math.cos(a) - dy * math.sin(a),
                    centre[1] + dx * math.sin(a) + dy * math.cos(a)))
    return out


def _volume(section: np.ndarray, layers: int = 3) -> np.ndarray:
    """Embed a 2D section in a small 3D stack whose middle layer is the widest one.

    :param section: The 2D section.
    :param layers: Number of layers in the stack (the section goes into layer 1).
    :return: A 3D ``uint8`` mask; ``get_layer_with_biggest_convex_area`` selects layer 1.
    """
    volume = np.zeros(section.shape + (layers,), dtype=np.uint8)
    volume[:, :, 1] = section
    volume[:, :, 0] = binary_erosion(section, iterations=4)
    volume[:, :, 2] = binary_erosion(section, iterations=6)
    return volume


def _notched_plateau(shape=SHAPE) -> np.ndarray:
    """Two condyles joined by an anterior bridge: an intercondylar gap that closes anteriorly.

    :param shape: Shape of the section.
    :return: The ``uint8`` mask; ``find_notch`` resolves its notch.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for cx in (32, 78):
        rr, cc = disk((cx, 48), 17, shape=shape)
        mask[rr, cc] = 1
    rr, cc = polygon(np.array([18, 92, 92, 18]), np.array([18, 18, 44, 44]), shape)
    mask[rr, cc] = 1
    return mask


def _tilt(start, end) -> float:
    """In-plane tilt of the (undirected) line through two points, in degrees.

    :param start: First endpoint (index coordinates; only x and y are used).
    :param end: Second endpoint.
    :return: The tilt in (-90, 90], measured from the sagittal axis towards +y (posterior).
    """
    d = np.asarray(end, dtype=float)[:2] - np.asarray(start, dtype=float)[:2]
    if d[0] < 0:
        d = -d
    return math.degrees(math.atan2(d[1], d[0]))


def _cortex_distances(section: np.ndarray, *points):
    """Distance of each point to the posterior and to the anterior cortex of the section.

    This is the independent cue the reference line is scored against: per column of the section,
    the posterior cortex is the mask voxel with the largest coronal index and the anterior cortex
    the one with the smallest. A posterior condylar point must be on the posterior cortex (distance
    ~0 -- the back-rotation and the final ``uint16`` cast can put it a voxel outside the mask) and
    far from the anterior one, which is where the old chord-alignment fallback put it.
    :param section: The 2D section the points were found on.
    :param points: The points to score (index coordinates; only x and y are used).
    :return: List of ``(distance to posterior cortex, distance to anterior cortex)`` pairs in voxels.
    """
    columns = np.nonzero(section.any(axis=1))[0]
    posterior = np.array([[x, np.nonzero(section[x])[0].max()] for x in columns], dtype=float)
    anterior = np.array([[x, np.nonzero(section[x])[0].min()] for x in columns], dtype=float)
    out = []
    for point in points:
        p = np.asarray(point, dtype=float)[:2]
        out.append((float(np.linalg.norm(posterior - p, axis=1).min()),
                    float(np.linalg.norm(anterior - p, axis=1).min())))
    return out


def test_divider_lies_between_the_two_condyles():
    """The fallback divider separates the condyles of a section whose notch is resolvable."""
    section = _notched_plateau()
    divider = get_condylar_divider(section)
    assert divider.dtype == float  # a continuous index pair, not a voxel
    assert 32 < divider[0] < 78  # between the two condyle centres


def test_notched_section_uses_the_notch_and_logs_nothing(caplog):
    """A section with a resolvable notch takes the notch path and reports a posterior line."""
    section = _notched_plateau()
    assert find_notch(section, percentage=0.5, thresh=2) is not None
    with caplog.at_level(logging.WARNING, logger='morphometry'):
        layer, start, end = get_knee_reference_line(_volume(section), 'tibia')
    assert 'notch' not in caplog.text
    assert layer == 1
    # the two condyles sit at the same coronal level, so the condylar line is horizontal
    assert abs(_tilt(start, end)) < 2
    assert all(dp <= 1 < da for dp, da in _cortex_distances(section, start, end))


def test_section_without_a_notch_falls_back_to_the_centroid(caplog):
    """A section that does not resolve the intercondylar gap warns and still yields a line."""
    section = _fill(QUAD)
    assert find_notch(section, percentage=0.5, thresh=2) is None
    with caplog.at_level(logging.WARNING, logger='morphometry'):
        _, start, end = get_knee_reference_line(_volume(section), 'tibia')
    assert 'No intercondylar notch resolved' in caplog.text
    assert np.isfinite(np.asarray(start, dtype=float)).all()
    assert np.isfinite(np.asarray(end, dtype=float)).all()


@pytest.mark.parametrize('bone', ['tibia', 'femur'])
def test_fallback_line_lies_on_the_posterior_cortex(bone):
    """The fallback resolves the 180 degree ambiguity: its line is the POSTERIOR tangent.

    The phantom's anterior tangent tilts the other way (-8.13 degrees), which is what the old
    chord-alignment fallback reported. Both the tilt and the per-column offsets are checked, so
    this fails for an anterior line even if the two tangents happened to be parallel.
    """
    section = _fill(QUAD)
    percentage, thresh = (0.7, 1) if bone == 'femur' else (0.5, 2)
    assert find_notch(section, percentage=percentage, thresh=thresh) is None
    _, start, end = get_knee_reference_line(_volume(section), bone)
    assert _tilt(start, end) == pytest.approx(QUAD_POSTERIOR_TILT, abs=1)
    assert all(dp <= 2 < da for dp, da in _cortex_distances(section, start, end))


@pytest.mark.parametrize('degrees', [-20, -10, 0, 10, 20])
def test_fallback_is_covariant_with_an_in_plane_rotation(degrees):
    """Turning the section turns the fallback reference line by the same angle.

    The frame is taken from the image (posterior = +y), not from the section's own chords, so the
    reported line follows the anatomy instead of jumping by 180 degrees when a chord's orientation
    flips. Measured error over this range: 0.02-0.57 degrees, with the sweep running 0 or 1 steps,
    so the residual is the rasterisation of the +-8 degree edges, not the 1 degree sweep step; the
    1 degree tolerance is therefore tight enough to catch a small regression.

    The covariance holds only while the anatomy's anterior-posterior axis is the image's: rotated
    past about +45 degrees (or to +-90) this phantom's reported line is 80-82 degrees off, because
    "posterior = +y" is then simply false. That premise is shared with the notch path
    (``get_dorsal_mask_point`` takes the largest coronal coordinate) and is the reason masks must be
    reoriented to LPI before measurement.
    """
    section = _fill(_rotate(QUAD, degrees))
    assert find_notch(section, percentage=0.5, thresh=2) is None
    _, start, end = get_knee_reference_line(_volume(section), 'tibia')
    assert _tilt(start, end) == pytest.approx(QUAD_POSTERIOR_TILT + degrees, abs=1)


def test_featureless_section_resolves_instead_of_raising():
    """A plain ellipse has no notch on any frame; it yields its posterior tangent, not an error.

    The old fallback searched for a notch a second time in its rotated frame and raised
    ``RuntimeError('No notch found.')`` when that failed too, which aborted the measurement on 5 of
    the 100 Augsburg reader-study tibial halves.
    """
    section = np.zeros(SHAPE, dtype=np.uint8)
    rr, cc = ellipse(60, 50, 40, 22, shape=SHAPE)
    section[rr, cc] = 1
    assert find_notch(section, percentage=0.5, thresh=2) is None
    _, start, end = get_knee_reference_line(_volume(section), 'tibia')
    assert abs(_tilt(start, end)) < 3  # the posterior tangent of an axis-aligned ellipse
    assert all(dp <= 2 < da for dp, da in _cortex_distances(section, start, end))


def test_centrally_peaked_section_keeps_its_points_on_the_posterior_cortex():
    """Degenerate case: the posterior-most point sits at the divider itself.

    No knee section looks like this (a posterior cortex peaking in the intercondylar area), but the
    construction must not fall over or wander onto the anterior cortex when it happens: the line
    then runs from the posterior apex to where it grazes the far side of the divider, and both
    endpoints still lie on the posterior boundary.
    """
    section = _fill(((20, 30), (60, 75), (100, 30), (100, 10), (20, 10)))
    assert find_notch(section, percentage=0.5, thresh=2) is None
    _, start, end = get_knee_reference_line(_volume(section), 'tibia')
    # 3 voxels, not the 2 of the plateau phantoms: here the line ends on a sharp corner of the
    # outline, where the back-rotation of the swept endpoint lands up to two voxels outside it.
    assert all(dp <= 3 < da for dp, da in _cortex_distances(section, start, end))

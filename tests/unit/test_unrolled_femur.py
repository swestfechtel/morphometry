"""Unit tests for the unrolled femoral coordinate (``morphometry.cartilage.deformation``).

The test femur is the phantom of :mod:`tests.unit.test_femur_regions`: two condyle cylinders
of radius 22 mm about the medial-lateral axis, joined anteriorly by a grooved trochlear ridge,
separated posteriorly by an open intercondylar fossa, with a proximal shaft. Its sagittal bone
profile at a condylar medial-lateral level is a circle, so arc length along it is known in
closed form, and the phantom can be rotated into any pose to check that the coordinate belongs
to the bone rather than to the scanner.
"""
import dataclasses

import numpy as np
import pytest
import pyvista as pv
from scipy.spatial.transform import Rotation

from morphometry.cartilage import surface as S
from morphometry.cartilage.deformation import UNMAPPED_CAUSES, UnrolledFemur
from morphometry.cartilage.femur_regions import FEMUR_REGION_KEYS, FemoralFrame, subdivide_femur
from tests.unit.test_femur_regions import (
    CONDYLE_INNER, CONDYLE_OUTER, CONDYLE_RADIUS, phantom as femur_phantom,
)

#: Flexion-like pose of the intrinsic-coordinate test: rotation about world x (the phantom's
#: medial-lateral axis) through the origin, which is where :func:`femur_phantom` keeps the
#: anatomy, plus a translation applied to the subdivision afterwards.
FLEXION = Rotation.from_euler('x', 60, degrees=True).as_matrix()


@pytest.fixture(scope='module')
def base_sub():
    """The phantom in its base pose, subdivided once."""
    return subdivide_femur(femur_phantom())


@pytest.fixture(scope='module')
def base_unrolled(base_sub):
    """The unrolled coordinate of the base pose, built once."""
    return UnrolledFemur.from_reference(base_sub.bone_mesh, base_sub.frame)


@pytest.fixture(scope='module')
def flexed_sub():
    """The same phantom flexed by 60 degrees about its medial-lateral axis, subdivided."""
    return subdivide_femur(femur_phantom(rotation=FLEXION))


def test_arc_is_zero_at_notch_and_grows_posteriorly(base_sub, base_unrolled):
    """The trochlea reads <= 0, the posterior condyle > 0 and further along than the trochlea."""
    c = base_unrolled.coordinates(S.triangle_centroids(base_sub.mesh))
    codes = base_sub.mesh.cell_data['region']
    for side in ('l', 'r'):
        alf = c[codes == FEMUR_REGION_KEYS.index(f'a{side}f'), 1]
        plf = c[codes == FEMUR_REGION_KEYS.index(f'p{side}f'), 1]
        assert np.isfinite(alf).mean() > 0.95 and np.isfinite(plf).mean() > 0.95
        assert np.nanmax(alf) <= 1.0
        assert np.nanmin(plf) > 0
        assert np.nanmedian(plf) > np.nanmedian(alf)


def test_condylar_points_get_finite_coordinates(base_sub, base_unrolled):
    """Every tAB triangle of the weight-bearing condyles is parametrised."""
    c = base_unrolled.coordinates(S.triangle_centroids(base_sub.mesh))
    codes = base_sub.mesh.cell_data['region']
    condylar = np.isin(codes, [FEMUR_REGION_KEYS.index(k) for k in FEMUR_REGION_KEYS[:6]])
    assert np.isfinite(c[condylar]).all()


def test_arc_matches_the_circular_profile(base_sub, base_unrolled):
    """
    On a condylar level the profile is a circle of radius ``CONDYLE_RADIUS``, so the arc from
    the notch plane's distal crossing to a point at polar angle ``theta`` (0 = distal,
    + = anterior) is ``R * (theta_notch - theta)``.
    """
    frame = base_sub.frame
    theta_notch = np.arcsin(frame.notch_ap / CONDYLE_RADIUS)
    theta = np.radians(np.array([60.0, 30.0, 0.0, -45.0, -90.0, -110.0]))
    for u in (-20.0, 20.0):
        points = np.stack([np.full_like(theta, u), CONDYLE_RADIUS * np.sin(theta),
                           -CONDYLE_RADIUS * np.cos(theta)], axis=1)
        c = base_unrolled.coordinates(points)
        assert np.allclose(c[:, 0], (points - frame.notch_point) @ frame.ml_axis, atol=1e-9)
        # The smoothed surface sits a fraction of a voxel inside the nominal circle.
        assert np.allclose(c[:, 1], CONDYLE_RADIUS * (theta_notch - theta), atol=0.6)


def test_articular_surface_beyond_the_registration_crop_is_mapped(base_sub, base_unrolled):
    """
    Regression (final review, finding 1): the coordinate was built on the distal 40 mm of the
    bone only (``REGISTRATION_CROP_MM``), so articular surface further up -- the proximal
    trochlea, up to 50.9 mm above the distal end on the real reference femurs -- had no
    coordinate. The coordinate is now built on the whole bone: on the phantom (bone from
    ``w = -22`` to the shaft top) points of the condyle circle 36-43 mm above its distal end
    read the circular arc like every other point.
    """
    frame = base_sub.frame
    theta_notch = np.arcsin(frame.notch_ap / CONDYLE_RADIUS)
    theta = np.radians(np.array([130.0, 150.0, 165.0]))        # w = 14.1, 19.1, 21.3 mm
    distal = (np.asarray(base_sub.bone_mesh.points) @ frame.proximal_axis).min()
    for u in (-20.0, 20.0):
        points = np.stack([np.full_like(theta, u), CONDYLE_RADIUS * np.sin(theta),
                           -CONDYLE_RADIUS * np.cos(theta)], axis=1)
        assert (points @ frame.proximal_axis - distal).max() > 42.0
        c = base_unrolled.coordinates(points)
        assert np.isfinite(c).all()
        assert np.allclose(c[:, 1], CONDYLE_RADIUS * (theta_notch - theta), atol=0.6)
        assert not base_unrolled.unmapped_causes(points).any()


def test_unmapped_causes_agree_with_coordinates(base_sub, base_unrolled):
    """
    ``unmapped_causes`` is 0 exactly where ``coordinates`` maps a point; a point far from the
    bone reads ``'off_bone'``.
    """
    rng = np.random.default_rng(3)
    vertices = np.asarray(base_sub.bone_mesh.points)
    points = vertices[rng.choice(len(vertices), 4000, replace=False)]
    points = points + rng.normal(scale=1.5, size=points.shape)
    codes = base_unrolled.unmapped_causes(points)
    mapped = np.isfinite(base_unrolled.coordinates(points)).all(axis=1)
    assert np.array_equal(codes == 0, mapped)
    assert set(np.unique(codes)) <= set(range(len(UNMAPPED_CAUSES) + 1))
    far = base_sub.frame.notch_point + 20 * base_sub.frame.ap_axis
    assert base_unrolled.unmapped_causes(far[None])[0] == 1 + UNMAPPED_CAUSES.index('off_bone')


def test_ml_column_is_patient_right_from_the_notch(base_sub, base_unrolled):
    """Column 0 is the M-L offset from the notch point, positive to the patient's right."""
    frame = base_sub.frame
    # Distal-most points of the two condyles, 20 mm either side of the notch point.
    points = np.array([[frame.notch_point[0] + du, 0.0, -CONDYLE_RADIUS] for du in (-20, 20)])
    c = base_unrolled.coordinates(points)
    assert c[0, 0] == pytest.approx(-20.0) and c[1, 0] == pytest.approx(20.0)


def _flexed_and_shifted(flexed_sub, shift):
    """The flexed phantom's unrolled coordinate after translating its bone and frame."""
    bone = flexed_sub.bone_mesh.copy()
    bone.points = bone.points + shift
    frame = dataclasses.replace(
        flexed_sub.frame, notch_point=flexed_sub.frame.notch_point + shift,
        notch_ap=flexed_sub.frame.notch_ap + float(shift @ flexed_sub.frame.ap_axis))
    return UnrolledFemur.from_reference(bone, frame)


SHIFT = np.array([7.0, -12.0, 25.0])


@pytest.fixture(scope='module')
def flexed_pairs(base_sub, base_unrolled, flexed_sub):
    """
    Every base tAB triangle centroid with its coordinate in the base pose and the coordinate
    of the same anatomical point (rotated by ``FLEXION`` about the origin, then shifted by
    ``SHIFT``) in the flexed, shifted pose.
    """
    flexed = _flexed_and_shifted(flexed_sub, SHIFT)
    points = S.triangle_centroids(base_sub.mesh)
    return (points, base_unrolled.coordinates(points),
            flexed.coordinates(points @ FLEXION.T + SHIFT))


def test_coordinates_are_intrinsic_to_the_bone(flexed_pairs):
    """
    The same anatomical points get the same coordinates when the femur is flexed by 60 degrees
    and translated: the coordinate is built in the femur's own frame from its own bone.

    Checked on every tAB triangle whose sagittal level cuts the condyles (|u| >=
    ``CONDYLE_INNER``): the weight-bearing condyles, the posterior zones and the trochlea on
    the condylar part of the ridge. The residual is the two poses' voxelisations, smoothing and
    re-estimated frames (median 0.15 mm, p99 0.51 mm when written): max 0.59 mm on the
    condyles' interior, 0.72 mm within 1 mm of their medial-lateral edges, where the sagittal
    plane grazes the rounded edge of the cylinder and the profile changes fastest with ``u``.
    """
    points, c_base, c_rot = flexed_pairs
    u = np.abs(points[:, 0])
    condylar = u >= CONDYLE_INNER
    interior = (u >= CONDYLE_INNER + 1) & (u <= CONDYLE_OUTER - 1)
    assert np.isfinite(c_base[condylar]).all() and np.isfinite(c_rot[condylar]).all()
    error = np.abs(c_rot - c_base)
    assert np.max(error[interior]) < 0.6
    assert np.max(error[condylar]) < 0.8


def test_groove_levels_are_finite_and_close(flexed_pairs):
    """
    The intercondylar band (|u| < ``CONDYLE_INNER``): the trochlear groove and the corners of
    the fossa. The phantom's fossa wall is a plane 0.5 mm *in front of* and parallel to the
    notch plane, so whether its smoothed surface crosses the plane, and how far up the wall,
    differs between the two voxelisations: the origin is either such a crossing (kept while it
    lies within ``NOTCH_CORNER_DROP_MM`` = 2 mm above the wall's foot) or the foot itself (see
    :func:`morphometry.cartilage.deformation._origin_candidates`), so the arc of a groove level
    can move by up to about that much (max 2.05 mm when written). A real fossa roof is oblique
    to the notch plane. Pinned looser, but finite and without a jump to another part of the
    profile (which was ~20 mm).
    """
    points, c_base, c_rot = flexed_pairs
    groove = np.abs(points[:, 0]) < CONDYLE_INNER
    assert np.isfinite(c_base[groove]).all() and np.isfinite(c_rot[groove]).all()
    assert np.max(np.abs(c_rot - c_base)[groove]) < 2.5


def test_points_off_the_bone_are_nan(base_sub, base_unrolled):
    """A point 20 mm anterior of the notch is far from any bone surface."""
    far = base_sub.frame.notch_point + 20 * base_sub.frame.ap_axis
    assert np.isnan(base_unrolled.coordinates(far[None])).all()


def test_points_beyond_the_ml_extent_are_nan(base_sub, base_unrolled):
    """Beyond the outermost computed sagittal level there is no contour to read arc from."""
    point = np.array([CONDYLE_OUTER + 1.5, 0.0, -CONDYLE_RADIUS])
    assert np.isnan(base_unrolled.coordinates(point[None])).all()


def test_coordinates_accept_a_single_point_and_empty_input(base_unrolled):
    """Shape handling: ``(3,)`` gives ``(1, 2)``, ``(0, 3)`` gives ``(0, 2)``."""
    assert base_unrolled.coordinates(np.array([20.0, 0.0, -CONDYLE_RADIUS])).shape == (1, 2)
    assert base_unrolled.coordinates(np.zeros((0, 3))).shape == (0, 2)


def _cylinder_reference(radius=10.0, notch_ap=3.0):
    """
    A closed cylinder along world x (a condyle with no shaft) and the frame of an
    upright femur: ml = +x, ap = +y, proximal = +z, the notch plane at ``y = notch_ap``.
    Short enough (20 mm) that the 40 mm crop keeps all of it, so every sagittal profile is a
    *closed* circle.
    """
    # A tube sampled every 0.5 mm along x and every 0.5 degree round (no caps: a sagittal
    # cut through its interior is a closed circle either way, and the vertices must be dense
    # because coordinates() reads the nearest one).
    theta = np.radians(np.arange(0.0, 360.0, 0.5))
    x = np.arange(-10.0, 10.01, 0.5)
    xx, tt = np.meshgrid(x, np.append(theta, theta[0]), indexing='ij')
    grid = pv.StructuredGrid(xx[..., None], (radius * np.cos(tt))[..., None],
                             (radius * np.sin(tt))[..., None])
    mesh = grid.extract_surface().triangulate().clean()
    notch_point = np.array([0.0, notch_ap, -np.sqrt(radius ** 2 - notch_ap ** 2)])
    frame = FemoralFrame(proximal_axis=np.array([0.0, 0, 1]), ml_axis=np.array([1.0, 0, 0]),
                         ap_axis=np.array([0.0, 1, 0]), notch_point=notch_point,
                         notch_ap=notch_ap, divider=(0.0, 0.0))
    return mesh, frame


def test_closed_profile_is_opened_at_its_most_proximal_point():
    """
    A closed sagittal circle: arc 0 at the *distal* notch-plane crossing (not the proximal
    one), positive round the back up to the top of the circle, negative round the front, and
    the jump between the two at the top.
    """
    radius, notch_ap = 10.0, 3.0
    mesh, frame = _cylinder_reference(radius, notch_ap)
    unrolled = UnrolledFemur.from_reference(mesh, frame, max_distance_mm=0.5)
    theta_notch = np.arcsin(notch_ap / radius)
    theta = np.radians(np.array([170.0, 90.0, 0.0, -90.0, -170.0]))   # 0 = distal, + anterior
    points = np.stack([np.zeros_like(theta), radius * np.sin(theta),
                       -radius * np.cos(theta)], axis=1)
    arc = unrolled.coordinates(points)[:, 1]
    # 720 facets: the polygon's arc is within 0.01 mm of the circle's over these lengths.
    assert np.allclose(arc, radius * (theta_notch - theta), atol=0.05)
    assert arc[0] < 0 < arc[-1]


def test_invalid_slice_spacing_raises():
    """A non-positive level spacing is refused."""
    mesh, frame = _cylinder_reference()
    with pytest.raises(ValueError):
        UnrolledFemur.from_reference(mesh, frame, slice_mm=0.0)


@pytest.mark.parametrize('lowest_in_front_mm', [3.0, 0.5])
def test_condyle_lowest_anterior_of_notch_keeps_the_crossing(lowest_in_front_mm):
    """
    Regression: on a real femur the lowest point of a condyle can lie a few mm *anterior* of
    the notch plane. The origin must still be the notch-plane crossing on the distal side --
    not the lowest vertex (which, 3 mm in front, used to drop the level entirely, and 0.5 mm in
    front silently moved the origin by ~0.5 mm of arc).
    """
    radius = 10.0
    notch_ap = -lowest_in_front_mm          # the tube's lowest point is at ap = 0
    mesh, frame = _cylinder_reference(radius, notch_ap)
    unrolled = UnrolledFemur.from_reference(mesh, frame, max_distance_mm=0.5)
    theta_notch = np.arcsin(notch_ap / radius)
    theta = np.radians(np.array([60.0, 0.0, -60.0, -120.0]))
    points = np.stack([np.zeros_like(theta), radius * np.sin(theta),
                       -radius * np.cos(theta)], axis=1)
    arc = unrolled.coordinates(points)[:, 1]
    assert np.isfinite(arc).all()
    assert np.allclose(arc, radius * (theta_notch - theta), atol=0.05)


def _side_wall_reference(side_wall=lambda x: x >= 0, extent=10.0, wall_centre=-12.0,
                         block=True):
    """
    A two-part bone along world x (ml = +x, ap = +y, proximal = +z, notch plane ``y = 0``),
    reproducing the outer edge of a real condyle (K1_30_unbelastet, patient-right, 33.5-36 mm
    from the notch point):

    * ``x < 0`` -- a condyle: the disc ``(y + 5)^2 + z^2 <= 10^2``. Its profile crosses the
      notch plane on the distal arc, 1.3 mm above its lowest point.
    * ``x >= 0`` -- the condyle has moved back, ``(y + 12)^2 + z^2 <= 10^2``, so its distal arc
      no longer reaches the notch plane, and a block ``-12 <= y <= 8, 6 <= z <= 15`` in front
      of it (the side wall a sagittal plane grazes) does. The only notch-plane crossings are on
      the block, 16 mm above the condyle's lowest point, and the profile from there to the
      condyle runs along the block's underside and down the condyle's front: 26.1 mm of
      profile for a 20 mm chord.

    Voxelised at 0.5 mm, contoured and Taubin-smoothed like a real bone surface.

    :param side_wall: Function of the x coordinate selecting the side-wall part (default
        ``x >= 0``); e.g. ``|x| <= 3`` puts it in the middle, between two normal condyle parts.
    :param extent: Half-length of the bone along x (mm).
    :param wall_centre: ``y`` of the side-wall part's condyle centre (radius 10, so ``<= -10``
        keeps its distal arc behind the notch plane).
    :param block: Add the side-wall block; without it the side-wall part's profile does not
        reach the notch plane at all (no crossing, no corner).
    """
    step = 0.5
    x = np.arange(-extent - 2.0, extent + 2.01, step)
    y = np.arange(-26.0, 12.01, step)
    z = np.arange(-14.0, 19.01, step)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    wall = side_wall(xx) & (np.abs(xx) <= extent)
    inner = (np.abs(xx) <= extent) & ~wall & ((yy + 5) ** 2 + zz ** 2 <= 100)
    outer = wall & (((yy - wall_centre) ** 2 + zz ** 2 <= 100)
                    | (block & (yy >= wall_centre) & (yy <= 8) & (zz >= 6) & (zz <= 15)))
    image = pv.ImageData(dimensions=xx.shape, spacing=(step,) * 3,
                         origin=(x[0], y[0], z[0]))
    image.point_data['mask'] = (inner | outer).astype(float).ravel(order='F')
    mesh = image.contour([0.5], scalars='mask').triangulate().clean()
    mesh = mesh.smooth_taubin(n_iter=20, pass_band=0.1)
    notch_point = np.array([0.0, 0.0, -np.sqrt(100.0 - 25.0)])
    frame = FemoralFrame(proximal_axis=np.array([0.0, 0, 1]), ml_axis=np.array([1.0, 0, 0]),
                         ap_axis=np.array([0.0, 1, 0]), notch_point=notch_point,
                         notch_ap=0.0, divider=(0.0, 0.0))
    return mesh, frame


def test_level_anchored_on_a_side_wall_is_not_used():
    """
    Regression (the femoral strip on K1): a level whose only notch-plane crossings lie on a
    side wall, with a fold between them and the condyle's distal arc, used to anchor its arc
    there, so every point of its posterior condyle read ~10-30 mm too far along -- a strip of
    inflated arc beside the condyle on every map. Such a level is not used: the outer part's
    condyle is unmapped (NaN, counted as dropped by the delta maps), the inner part unchanged.
    """
    mesh, frame = _side_wall_reference()
    unrolled = UnrolledFemur.from_reference(mesh, frame)
    theta = np.radians(np.array([0.0, -45.0, -90.0]))          # distal, posterior-distal, posterior
    for u in (3.0, 6.0, 9.0):
        outer = np.stack([np.full_like(theta, u), -12 + 10 * np.sin(theta),
                          -10 * np.cos(theta)], axis=1)
        assert np.isnan(unrolled.coordinates(outer)).all()
    theta_notch = np.arcsin(5.0 / 10.0)
    inner = np.stack([np.full_like(theta, -5.0), -5 + 10 * np.sin(theta),
                      -10 * np.cos(theta)], axis=1)
    arc = unrolled.coordinates(inner)[:, 1]
    assert np.isfinite(arc).all()
    # voxelised and smoothed: the surface is within a fraction of a voxel of the circle
    assert np.allclose(arc, 10.0 * (theta_notch - theta), atol=0.8)


def test_a_gap_of_unused_levels_is_not_bridged():
    """
    Regression: a run of unused (side-wall) levels inside the M-L range used to be filled by a
    linear blend of the two kept levels either side, however far apart. Here the side wall
    spans |x| <= 4 (8 mm of unused levels) between two normal condyle parts; its condyle sits
    5.5 mm further back, so its lowest point is within ``max_distance_mm`` of the kept levels'
    profiles and the blend used to give it an arc. A vertex more than one slice into the gap now
    has no coordinate; one within a slice of a kept level, and the condyle either side, keep
    theirs.
    """
    mesh, frame = _side_wall_reference(side_wall=lambda x: np.abs(x) <= 4.0, extent=12.0,
                                       wall_centre=-10.5)
    unrolled = UnrolledFemur.from_reference(mesh, frame)
    lowest = np.array([[u, -10.5, -10.0] for u in (-2.0, 0.0, 2.5)])
    assert np.isnan(unrolled.coordinates(lowest)).all()
    theta = np.radians(np.array([0.0, -45.0, -90.0]))
    theta_notch = np.arcsin(5.0 / 10.0)
    for u in (-8.0, 8.0):
        normal = np.stack([np.full_like(theta, u), -5 + 10 * np.sin(theta),
                           -10 * np.cos(theta)], axis=1)
        arc = unrolled.coordinates(normal)[:, 1]
        assert np.allclose(arc, 10.0 * (theta_notch - theta), atol=0.8), u


def test_unmapped_causes_name_why_a_level_is_not_used():
    """
    A level dropped for its detour reads ``'detour'``; one whose profile never reaches the
    notch plane (the same condyle, moved back, without the side wall) reads ``'no_crossing'``
    -- the gap cap applies to both kinds of unused level alike. The normal part is mapped.
    """
    theta = np.radians(np.array([0.0, -45.0, -90.0]))
    detour = 1 + UNMAPPED_CAUSES.index('detour')
    no_crossing = 1 + UNMAPPED_CAUSES.index('no_crossing')
    for block, expected in ((True, detour), (False, no_crossing)):
        mesh, frame = _side_wall_reference(block=block)
        unrolled = UnrolledFemur.from_reference(mesh, frame)
        outer = np.stack([np.full_like(theta, 6.0), -12 + 10 * np.sin(theta),
                          -10 * np.cos(theta)], axis=1)
        assert np.isnan(unrolled.coordinates(outer)).all()
        assert (unrolled.unmapped_causes(outer) == expected).all(), block
        inner = np.stack([np.full_like(theta, -5.0), -5 + 10 * np.sin(theta),
                          -10 * np.cos(theta)], axis=1)
        assert not unrolled.unmapped_causes(inner).any()
    mesh, frame = _side_wall_reference(side_wall=lambda x: np.abs(x) <= 4.0, extent=12.0,
                                       wall_centre=-10.5, block=False)
    unrolled = UnrolledFemur.from_reference(mesh, frame)
    lowest = np.array([[u, -10.5, -10.0] for u in (-2.0, 0.0, 2.5)])
    assert np.isnan(unrolled.coordinates(lowest)).all()
    assert (unrolled.unmapped_causes(lowest) == no_crossing).all()

"""Unit tests for the Wirth & Eckstein femoral subregions (``morphometry.cartilage.femur_regions``).

The phantom is a simplified distal femur voxelised from a geometry defined in world mm: two
condyles as cylinders along the medial-lateral axis (label 1), joined anteriorly by a
trochlear ridge with a groove milled into it, separated posteriorly by an open intercondylar
fossa, carrying a cartilage shell (label 3) and continued proximally by a shaft cylinder.

Because the geometry lives in world space, the same femur can be sampled with any voxel size
and direction matrix and rotated into any pose -- in particular rotated about the
medial-lateral axis, which is what knee flexion does to it in the scanner -- and the region
shares must not change beyond discretisation error.

The condyles are *cylinders*, so a coronal cross-section of one condyle's subchondral bone is a
straight segment: dividing it at a third and two thirds of its arc length gives exactly a third
of the area each, which is what pins :func:`arc_length_coordinate`.
"""
import nibabel as nib
import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from morphometry.image_io import Segmentation
from morphometry.cartilage import surface as S
from morphometry.cartilage.femur_regions import (
    subdivide_femur, estimate_shaft_axis, posterior_condylar_axis, label_points_world,
    find_trochlear_notch, arc_length_coordinate, articular_arc_fraction,
    anatomical_femur_names, rename_femur_regions,
    CONDYLE_KEYS, FEMUR_REGION_KEYS,
)
from morphometry.cartilage.knee import Femur

BONE, CARTILAGE = 1, 3
LPI = np.diag([-1.0, -1.0, -1.0])  # direction of a knee after transform_coordinate_system()

# Phantom geometry in its local frame (u = patient right, v = anterior, w = proximal), all mm.
CONDYLE_RADIUS = 22.0        # radius of the two condyle cylinders about the u axis
CONDYLE_INNER = 8.0          # |u| at which a condyle starts (the fossa / groove is inside)
CONDYLE_OUTER = 32.0         # |u| at which it ends
NOTCH_V = 6.0                # anterior of this the fossa is roofed by the trochlear ridge
GROOVE_DEPTH = 5.0           # how deep the trochlear groove is milled into that ridge
CARTILAGE_MM = 2.0           # cartilage shell thickness
SHAFT_RADIUS = 14.0
SHAFT_TOP = 60.0
# Articular arc, measured from distal (0 deg) towards anterior (+) / posterior (-).
ARC = (-120.0, 100.0)
WEIGHTBEARING = 0.6
# What the geometry implies: the condyles' posterior end, and the 60% border behind the notch.
EXPECTED_POSTERIOR_V = -CONDYLE_RADIUS
EXPECTED_BOUNDARY_V = NOTCH_V - WEIGHTBEARING * (NOTCH_V - EXPECTED_POSTERIOR_V)


def _labels(local, arc=ARC, denuded=None):
    """
    Label the phantom at points given in its local frame (u = patient right, v = anterior,
    w = proximal; the condyle cylinders' axis is the u axis through the origin).

    :param local: ``(N, 3)`` local coordinates (mm).
    :param arc: ``(start, end)`` of the articular arc in degrees, 0 = distal, + = anterior.
    :param denuded: Optional ``(u, v, r)`` disc in which the cartilage is missing (denuded
        subchondral bone inside the weight-bearing condyle).
    :return: ``(N,)`` uint8 labels.
    """
    u, v, w = local.T
    rho = np.hypot(v, w)
    theta = np.degrees(np.arctan2(v, -w))

    # The trochlear ridge closes the fossa anterior of NOTCH_V, with a groove milled into it
    # that fades out at |u| = CONDYLE_INNER so the condyle surface itself stays cylindrical.
    groove = GROOVE_DEPTH * np.maximum(0.0, 1.0 - (u / CONDYLE_INNER) ** 2)
    surface = CONDYLE_RADIUS - np.where(v >= NOTCH_V, groove, 0.0)
    fossa = (np.abs(u) < CONDYLE_INNER) & (v < NOTCH_V)
    span = np.abs(u) <= CONDYLE_OUTER

    labels = np.zeros(len(local), dtype=np.uint8)
    labels[span & (rho <= surface) & ~fossa] = BONE
    labels[(u ** 2 + v ** 2 <= SHAFT_RADIUS ** 2) & (w >= 0) & (w <= SHAFT_TOP)] = BONE
    cartilage = (span & ~fossa & (rho > surface) & (rho <= surface + CARTILAGE_MM)
                 & (theta >= arc[0]) & (theta <= arc[1]))
    if denuded is not None:
        cartilage &= (u - denuded[0]) ** 2 + (v - denuded[1]) ** 2 > denuded[2] ** 2
    labels[cartilage] = CARTILAGE
    return labels


def phantom(spacing=(0.5, 0.5, 0.5), direction=LPI, rotation=None, reflect_array=False,
            crop_proximal=0, **kw):
    """
    Voxelise the phantom.

    :param spacing: Voxel size (mm).
    :param direction: 3x3 direction matrix of the image.
    :param rotation: Optional 3x3 rotation of the *anatomy* in world space (a rotation about
        world x is the femur flexing against the scanner).
    :param reflect_array: Reverse the array along index axis 0 afterwards (same affine),
        which mirrors the anatomy left-right.
    :param crop_proximal: Drop this many slices from the start of index axis 2, which in an
        LPI image cuts the proximal end of the shaft off at an array face -- obliquely, once
        the anatomy is rotated, as a real field of view does.
    :param kw: Passed to :func:`_labels`.
    :return: A nibabel-backed ``Segmentation``.
    """
    rot = np.eye(3) if rotation is None else np.asarray(rotation)
    m = np.asarray(direction, float) @ np.diag(spacing)
    # Index-space bounding box of the (rotated) anatomy, with a margin.
    corners = np.array([[u, v, w] for u in (-34, 34) for v in (-27, 27)
                        for w in (-26, SHAFT_TOP + 2)]) @ rot.T
    idx = corners @ np.linalg.inv(m).T
    lo = np.floor(idx.min(0)) - 3
    shape = (np.ceil(idx.max(0)) - lo + 4).astype(int)
    affine = np.eye(4)
    affine[:3, :3] = m
    affine[:3, 3] = m @ lo                 # array index 0 = index lo: the anatomy stays at world 0
    world = S.indices_to_world(affine, np.indices(shape).reshape(3, -1).T)
    labels = _labels(world @ rot, **kw).reshape(shape)   # world -> local: R^T p
    if reflect_array:
        labels = labels[::-1].copy()
    if crop_proximal:
        labels = labels[:, :, crop_proximal:].copy()
        affine = affine.copy()
        affine[:3, 3] = affine[:3, 3] + affine[:3, 2] * crop_proximal
    return Segmentation.from_nibabel(nib.Nifti1Image(labels, affine))


def _shares(sub):
    """Region shares keyed by (condyle side initial, letter)."""
    return {(key[2], key[0]): share for key, share in sub.region_shares.items()}


def _zone_shares(sub):
    """Extension-zone areas as fractions of the whole femoral tAB."""
    areas = sub.region_areas
    return {key: areas[key] / sub.tab_area for key in ('alf', 'plf', 'arf', 'prf')}


@pytest.fixture(scope='module')
def base():
    return subdivide_femur(phantom())


# ------------------------------------------------------------------------ the femoral frame

def test_frame_axes(base):
    """The frame is the phantom's own: shaft along +w, M-L along +u, A-P along +v."""
    frame = base.frame
    assert frame.proximal_axis @ [0, 0, 1] > 0.999
    assert frame.ml_axis @ [1, 0, 0] > 0.999
    assert frame.ap_axis @ [0, 1, 0] > 0.999
    assert np.linalg.det(np.stack([frame.ml_axis, frame.ap_axis, frame.proximal_axis])) \
        == pytest.approx(1.0)
    # The posterior condylar line joins the two condyles' most posterior points.
    for side, point in frame.posterior_points.items():
        assert point[1] == pytest.approx(EXPECTED_POSTERIOR_V, abs=0.6)
    assert frame.posterior_points['left'][0] < 0 < frame.posterior_points['right'][0]


def test_shaft_axis_is_recovered_in_any_pose():
    """The shaft axis is found to within a couple of degrees whatever the femur's pose."""
    for angle in (0, 30, 60, 90):
        rotation = Rotation.from_euler('x', angle, degrees=True).as_matrix()
        seg = phantom(rotation=rotation)
        axis, residual, bins = estimate_shaft_axis(label_points_world(seg, BONE),
                                                   label_points_world(seg, CARTILAGE))
        expected = rotation @ np.array([0.0, 0.0, 1.0])
        assert np.degrees(np.arccos(np.clip(axis @ expected, -1, 1))) < 2.5, angle
        assert residual < 1.0 and bins >= 5


def test_posterior_condylar_axis_tracks_a_rotated_femur():
    """Turning the femur in the scanner turns the posterior condylar line with it."""
    rotation = Rotation.from_euler('z', 20, degrees=True).as_matrix()
    seg = phantom(rotation=rotation)
    bone = S.bone_surface_mesh(seg, BONE)
    axis = rotation @ np.array([0.0, 0.0, 1.0])
    points = np.asarray(bone.points)
    distal = float((label_points_world(seg, CARTILAGE) @ axis).max())
    ml, _ = posterior_condylar_axis(points[points @ axis <= distal], axis)
    assert np.degrees(np.arccos(abs(ml @ (rotation @ np.array([1.0, 0.0, 0.0]))))) < 2.0


def test_notch_and_sixty_percent_border(base):
    """The anterior border sits at the trochlear notch, the posterior one 60% behind it."""
    assert base.frame.notch_ap == pytest.approx(NOTCH_V, abs=1.5)
    for condyle in base.condyles.values():
        assert condyle.posterior_ap == pytest.approx(EXPECTED_POSTERIOR_V, abs=0.6)
        assert condyle.boundary_ap == pytest.approx(EXPECTED_BOUNDARY_V, abs=1.2)
    # The notch landmark is the floor of the trochlear groove: on the midline, one groove
    # depth less distal than the condyles.
    notch = base.frame.notch_point
    assert abs(notch[0]) < 2.0
    assert notch[2] == pytest.approx(-(CONDYLE_RADIUS - GROOVE_DEPTH), abs=2.0)


def test_weightbearing_fraction_parameter():
    """A larger fraction reaches further back and takes more of the condyle."""
    seg = phantom()
    deeper = subdivide_femur(seg, weightbearing_fraction=0.75)
    expected = NOTCH_V - 0.75 * (NOTCH_V - EXPECTED_POSTERIOR_V)
    for side, condyle in deeper.condyles.items():
        assert condyle.boundary_ap == pytest.approx(expected, abs=1.2)
        assert condyle.tab_area > subdivide_femur(seg).condyles[side].tab_area
    with pytest.raises(ValueError, match='weightbearing_fraction'):
        subdivide_femur(seg, weightbearing_fraction=1.0)


# ------------------------------------------------------------------------ the thirds

def test_thirds_hold_a_third_of_the_condyle_each(base):
    """A cylindrical condyle gives exact arc-length thirds (the paper: 33.2-33.5%)."""
    for side, condyle in base.condyles.items():
        shares = condyle.region_shares
        for letter in ('i', 'c', 'e'):
            assert shares[f'{letter}c{side[0]}f'] == pytest.approx(1 / 3, abs=0.01), (side, letter)
        assert sum(condyle.region_areas.values()) == pytest.approx(condyle.tab_area, rel=1e-9)
    # The labelled mesh carries the same areas.
    areas = np.bincount(base.mesh.cell_data['region'], weights=S.triangle_areas(base.mesh),
                        minlength=len(FEMUR_REGION_KEYS))
    for code, key in enumerate(FEMUR_REGION_KEYS):
        assert areas[code] == pytest.approx(base.region_areas[key], rel=1e-9)
    assert areas.sum() == pytest.approx(base.tab_area, rel=1e-9)


def test_internal_third_faces_the_other_condyle(base):
    """Internal is the third towards the intercondylar notch, external the outer one."""
    regions = base.mesh.cell_data['region']
    centroids = S.triangle_centroids(base.mesh)
    for side in ('left', 'right'):
        internal = centroids[regions == FEMUR_REGION_KEYS.index(f'ic{side[0]}f')]
        external = centroids[regions == FEMUR_REGION_KEYS.index(f'ec{side[0]}f')]
        if side == 'left':                      # patient left is at negative world x
            assert internal[:, 0].min() > external[:, 0].max()
        else:
            assert internal[:, 0].max() < external[:, 0].min()


def test_proximal_wrap_is_not_part_of_the_weightbearing_condyle():
    """Cartilage wrapping up behind the condyle stays out of cF (and keeps the thirds exact)."""
    far = subdivide_femur(phantom(arc=(-165.0, 100.0)))
    for side, condyle in far.condyles.items():
        for letter in ('i', 'c', 'e'):
            assert condyle.region_shares[f'{letter}c{side[0]}f'] == pytest.approx(1 / 3, abs=0.015)
    # The extra wrap is counted with the posterior condyle zone, which grows accordingly.
    plain = subdivide_femur(phantom())
    assert far.region_areas['plf'] > plain.region_areas['plf'] + 100
    assert far.condyles['left'].tab_area == pytest.approx(plain.condyles['left'].tab_area, rel=0.1)


# ------------------------------------------------------------------------ invariance

def _assert_shares_close(a, b, atol, area_rtol=0.03):
    """
    Compare two subdivisions of the same femur: the thirds' shares, the extension zones'
    shares of the whole tAB, and each weight-bearing condyle's absolute area.

    ``atol`` has to allow for the phantom's sharp intercondylar wall: how much of that wall
    counts as cartilage-covered depends on how the voxel grid cuts it, and the wall is exactly
    where the internal third ends, so the thirds move by up to ~1 percentage point between
    poses that sample it differently. Rotations about the condyles' own axis (flexion) sample
    it identically and stay well inside half of that.
    """
    sa, sb = _shares(a), _shares(b)
    worst = max(abs(sa[k] - sb[k]) for k in sa)
    assert worst < atol, f'max share difference {worst:.4f}'
    za, zb = _zone_shares(a), _zone_shares(b)
    worst = max(abs(za[k] - zb[k]) for k in za)
    assert worst < 5 * atol, f'max extension-zone share difference {worst:.4f}'
    for side in ('left', 'right'):
        assert a.condyles[side].tab_area == pytest.approx(b.condyles[side].tab_area,
                                                          rel=area_rtol), side


@pytest.mark.parametrize('angle', [30, 60, 90])
def test_shares_invariant_to_flexion(base, angle):
    """Rotating the femur about the medial-lateral axis (flexion) changes nothing."""
    rotation = Rotation.from_euler('x', angle, degrees=True).as_matrix()
    _assert_shares_close(base, subdivide_femur(phantom(rotation=rotation)), atol=0.006,
                         area_rtol=0.01)


@pytest.mark.parametrize('rotation', [
    Rotation.from_euler('z', 20, degrees=True).as_matrix(),     # leg turned in the scanner
    Rotation.from_euler('y', 10, degrees=True).as_matrix(),     # varus / valgus tilt
    Rotation.from_euler('zx', [-15, 45], degrees=True).as_matrix(),
])
def test_shares_invariant_to_anatomy_pose(base, rotation):
    _assert_shares_close(base, subdivide_femur(phantom(rotation=rotation)), atol=0.012)


def test_shares_invariant_to_sampling(base):
    """Anisotropic voxels and an oblique, reflecting direction matrix keep the shares.

    The absolute areas are allowed a few percent here (see below); only the shares are held to
    the same tolerance as the other poses.
    """
    direction = Rotation.from_euler('xyz', [10, -15, 30], degrees=True).as_matrix() @ np.diag([1, -1, 1])
    # The absolute areas may differ by a few percent here: coarser, anisotropic voxels
    # discretise the condyle's sharp cartilage rim differently. The *shares* must not.
    _assert_shares_close(base, subdivide_femur(phantom(spacing=(0.75, 0.4, 0.4))), atol=0.012,
                         area_rtol=0.05)
    _assert_shares_close(base, subdivide_femur(phantom(spacing=(0.6, 0.45, 0.5),
                                                       direction=direction)), atol=0.012,
                         area_rtol=0.05)


def test_mirror_swaps_condyles(base):
    """A left-right mirrored femur swaps the condyles; internal stays internal."""
    mirrored = _shares(subdivide_femur(phantom(reflect_array=True)))
    original = _shares(base)
    for letter in ('i', 'c', 'e'):
        assert mirrored[('l', letter)] == pytest.approx(original[('r', letter)], abs=0.008)
        assert mirrored[('r', letter)] == pytest.approx(original[('l', letter)], abs=0.008)


def test_simpleitk_backend_matches_nibabel(base):
    """The SimpleITK-backed copy of the same volume gives the same subdivision."""
    seg = phantom()
    affine = seg.image.affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    itk = sitk.GetImageFromArray(np.asarray(seg.image.dataobj).transpose(2, 1, 0))
    itk.SetSpacing(spacing.tolist())
    itk.SetDirection((ras_to_lps @ affine[:3, :3] / spacing).ravel().tolist())
    itk.SetOrigin((ras_to_lps @ affine[:3, 3]).tolist())
    _assert_shares_close(base, subdivide_femur(Segmentation.from_sitk(itk)), atol=1e-6)


# ------------------------------------------------------------------------ tAB / failures

def test_denuded_bone_inside_the_condyle_is_part_of_tab():
    """A cartilage-free disc inside one condyle stays in that condyle's tAB."""
    hole = (20.0, -2.0, 4.0)
    sub = subdivide_femur(phantom(denuded=hole))
    plain = subdivide_femur(phantom())
    right = sub.condyles['right']
    assert right.tab_area - right.cab_area == pytest.approx(np.pi * hole[2] ** 2, rel=0.3)
    assert right.tab_area == pytest.approx(plain.condyles['right'].tab_area, rel=0.02)
    assert sub.condyles['left'].tab_area - sub.condyles['left'].cab_area < 10.0


def test_shaft_axis_survives_an_oblique_field_of_view_cut():
    """An obliquely cut proximal end: dropping the incomplete cross-sections is what saves it.

    The phantom is tilted and then cut at an array face, so the cut plane is oblique to the
    shaft and the proximal cross-sections are lenses whose centroids are pulled sideways --
    the situation ``min_bin_ratio`` exists for, and the only one that exercises it.
    """
    rotation = Rotation.from_euler('x', 25, degrees=True).as_matrix()
    seg = phantom(rotation=rotation, crop_proximal=90)
    bone, cartilage = label_points_world(seg, BONE), label_points_world(seg, CARTILAGE)
    expected = rotation @ np.array([0.0, 0.0, 1.0])
    angle = lambda v: np.degrees(np.arccos(np.clip(v @ expected, -1, 1)))

    kept, _, bins = estimate_shaft_axis(bone, cartilage)
    naive, _, naive_bins = estimate_shaft_axis(bone, cartilage, min_bin_ratio=0.0)
    assert bins < naive_bins, 'no incomplete cross-section was dropped'
    assert angle(kept) < 6.0
    assert angle(naive) > 15.0


def test_a_field_of_view_without_shaft_raises():
    """Cut away enough of the shaft and the axis is refused rather than guessed."""
    seg = phantom(rotation=Rotation.from_euler('x', 25, degrees=True).as_matrix(),
                  crop_proximal=125)
    with pytest.raises(ValueError, match='mm of femur above the cartilage'):
        subdivide_femur(seg)


def test_cartilage_on_one_condyle_only_raises(base):
    """A condyle with no covered bone is reported, not silently left out."""
    seg = phantom()
    array = np.asarray(seg.image.dataobj).copy()
    world = S.indices_to_world(seg.image.affine, np.indices(array.shape).reshape(3, -1).T)
    array[(world[:, 0] < 0).reshape(array.shape) & (array == CARTILAGE)] = 0   # patient left
    with pytest.raises(ValueError, match='patient-left femoral condyle has no subchondral'):
        subdivide_femur(Segmentation.from_nibabel(nib.Nifti1Image(array, seg.image.affine)))


def test_a_bone_without_a_notch_raises(base):
    """Fed a plain cylinder (the shaft), the notch search reports that there is no notch."""
    frame = base.frame
    shaft = label_points_world(phantom(), BONE)
    shaft = shaft[shaft[:, 2] > 25.0]            # proximal of the condyles: no fossa there
    with pytest.raises(ValueError, match='no intercondylar notch'):
        find_trochlear_notch(shaft, frame.ap_axis, frame.ml_axis, frame.proximal_axis,
                             anterior_limit=1e3)


def test_arc_length_needs_at_least_two_cross_sections(base):
    """A patch with no anterior-posterior extent cannot be parametrised across."""
    frame = base.frame
    flat = pv.PolyData(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
                       np.array([3, 0, 1, 2]))
    with pytest.raises(ValueError, match='too few usable cross-sections'):
        arc_length_coordinate(flat, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                              np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))


def test_missing_labels_raise():
    seg = phantom()
    array = np.asarray(seg.image.dataobj).copy()
    array[array == BONE] = 0
    with pytest.raises(ValueError, match='bone label'):
        subdivide_femur(Segmentation.from_nibabel(nib.Nifti1Image(array, seg.image.affine)))
    array = np.asarray(seg.image.dataobj).copy()
    array[array == CARTILAGE] = 0
    with pytest.raises(ValueError, match='covered by cartilage'):
        subdivide_femur(Segmentation.from_nibabel(nib.Nifti1Image(array, seg.image.affine)))


# ------------------------------------------------------------------------ naming / Femur

def test_anatomical_names():
    right = anatomical_femur_names('right')     # right knee: patient-left condyle is medial
    assert right['iclf'] == 'icMF' and right['cclf'] == 'ccMF' and right['ecrf'] == 'ecLF'
    assert right['alf'] == 'aMF' and right['left_posterior_zone'] == 'pMF'
    left = anatomical_femur_names('left')
    assert left['iclf'] == 'icLF' and left['icrf'] == 'icMF'
    assert anatomical_femur_names(None) is None
    with pytest.raises(ValueError):
        anatomical_femur_names('medial')
    assert rename_femur_regions({'iclf': 1, 'right_anterior_zone': 2}, 'right') == \
        {'icMF': 1, 'aLF': 2}
    with pytest.raises(ValueError, match='unknown femoral region key'):
        rename_femur_regions({'clt': 1}, 'right')


def test_femur_labels_voxels_by_nearest_region():
    """Femur transfers the regions to its cartilage voxels; condyles by patient side."""
    seg = phantom()
    femur = Femur(seg, CARTILAGE, laterality='right')
    femur.calculate_landmarks()
    femur.extract_subregions()
    assert femur.anatomical_names['iclf'] == 'icMF'
    world = S.array_indices_to_world(seg, femur.point_cloud)
    for key in CONDYLE_KEYS:
        points = getattr(femur, key)
        assert len(points) > 0, key
        side = np.sign(S.array_indices_to_world(seg, points)[:, 0])
        assert (side < 0).all() if key.endswith('lf') else (side > 0).all(), key
    for key in ('alf', 'plf', 'arf', 'prf'):
        assert len(getattr(femur, key)) > 0, key
    # classify_point agrees with the precomputed labels.
    for i in (0, len(world) // 2, len(world) - 1):
        assert femur.classify_point(femur.point_cloud[i]) == femur.point_labels[i]


def test_articular_arc_fraction_runs_anterior_to_posterior():
    """The A-P arc-length coordinate starts at 0 on the trochlea and grows posteriorly."""
    sub = subdivide_femur(phantom())
    frame = sub.frame
    points = np.asarray(sub.mesh.points, dtype=float)
    ap = points @ frame.ap_axis
    middle = np.abs(points @ frame.ml_axis - 20.0) < 1.0     # one sagittal band of a condyle
    anterior = points[middle][np.argmax(ap[middle])]
    posterior = points[middle][np.argmin(ap[middle])]
    fractions = articular_arc_fraction(sub, np.stack([anterior, posterior]))
    assert fractions[0] == pytest.approx(0.0, abs=0.06)
    # Not 1.0: the phantom's articular arc runs to 120 deg past distal, so the surface carries
    # on curving proximally behind its most posterior *point* -- which is the whole reason the
    # coordinate is an arc length and not an anterior-posterior coordinate.
    assert 0.8 < fractions[1] < 1.0
    # Monotone along the arc: sampling the band from anterior to posterior never goes back.
    band = points[middle][np.argsort(-ap[middle])][::40]
    values = articular_arc_fraction(sub, band)
    assert np.isfinite(values).all()
    assert np.corrcoef(np.arange(len(values)), values)[0, 1] > 0.9


def test_articular_arc_fraction_clamps_just_outside_and_drops_far_outside():
    """A point just past the medial-lateral edge is clamped; one well past it is NaN.

    An unclamped cut plane through the very edge of the mesh intersects nothing, which would
    make the coordinate NaN exactly where the tibiofemoral contact runs off the condyle. Far
    outside, though, the outermost cross-section is a near-degenerate sliver, and a fraction
    read off it is not comparable with one from a full contour.
    """
    sub = subdivide_femur(phantom())
    across = np.asarray(sub.mesh.points, dtype=float) @ sub.frame.ml_axis
    edge = np.asarray(sub.mesh.points, dtype=float)[int(np.argmax(across))]
    just_outside = edge + 0.5 * sub.frame.ml_axis
    inside = edge - 1.0 * sub.frame.ml_axis
    clamped = articular_arc_fraction(sub, just_outside.reshape(1, 3))[0]
    assert np.isfinite(clamped)
    assert clamped == pytest.approx(articular_arc_fraction(sub, inside.reshape(1, 3))[0],
                                    abs=0.1)
    far = edge + 30.0 * sub.frame.ml_axis
    assert np.isnan(articular_arc_fraction(sub, far.reshape(1, 3))[0])
    # The cut-off is a parameter, so a caller that wants the old clamp can have it.
    assert np.isfinite(articular_arc_fraction(sub, far.reshape(1, 3), max_outside_mm=100.0)[0])


def test_transfer_qc_flags_cartilage_far_from_the_bone(caplog):
    """Cartilage far from the tAB (a broken segmentation) is reported; a clean femur is not."""
    clean = Femur(phantom(), CARTILAGE)
    with caplog.at_level('WARNING', logger='morphometry'):
        clean.calculate_landmarks()
    assert not caplog.records
    assert clean.transfer_qc['max_mm'] < 4.0 and clean.transfer_qc['far_fraction'] == 0.0

    # A detached cartilage blob floating well proximal of the articular surface; outlier
    # removal is disabled so it stays in the point cloud.
    seg = phantom()
    array = np.asarray(seg.image.dataobj).copy()
    world = S.indices_to_world(seg.image.affine, np.indices(array.shape).reshape(3, -1).T)
    blob = ((np.abs(world[:, 0] - 20) <= 4) & (np.abs(world[:, 1]) <= 4)
            & (world[:, 2] > 36) & (world[:, 2] <= 44))
    array[blob.reshape(array.shape)] = CARTILAGE
    broken = Femur(Segmentation.from_nibabel(nib.Nifti1Image(array, seg.image.affine)),
                   CARTILAGE, outlier_ratio=0)
    caplog.clear()
    with caplog.at_level('WARNING', logger='morphometry'):
        broken.calculate_landmarks()
    assert broken.transfer_qc['far_fraction'] > 0.01
    assert broken.transfer_qc['max_mm'] > 8.0
    assert any('subchondral bone area' in r.getMessage() for r in caplog.records)

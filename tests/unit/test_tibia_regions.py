"""Unit tests for the Wirth & Eckstein tibial subregions (``morphometry.cartilage.tibia_regions``).

The phantoms are label volumes voxelised from a geometry defined in world mm: a tibial bone
block (label 2) with two cartilage plates (label 4) lying on its top face. Because the
geometry lives in world space, the same knee can be sampled with any voxel size and
direction matrix, rotated in the axial plane or tilted, and the region shares must not
change beyond discretisation error.
"""
import nibabel as nib
import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from morphometry.image_io import Segmentation
from morphometry.cartilage import surface as S
from morphometry.cartilage.tibia_regions import (
    subdivide_tibia, anatomical_tibia_names, rename_tibia_regions, area_moments,
    TIBIA_REGION_KEYS, REGION_LETTERS,
)
from morphometry.cartilage.knee import Tibia, split_tibia_thickness_by_compartment

BONE, CARTILAGE = 2, 4
LPI = np.diag([-1.0, -1.0, -1.0])  # direction of a knee after transform_coordinate_system()

# Two dissimilar plates, (centre u, centre v, semi-axis u, semi-axis v) in mm. The patient-left
# plate sits at -u (world x is RAS: +x = patient right). Each has a bulge towards the other
# plate, so internal and external wedges differ and a mirror that swapped them would show.
EGG_PLATES = ((-15.0, 1.0, 9.0, 13.0), (16.0, -1.0, 11.0, 14.0))
CIRCLE_PLATES = ((-15.0, 0.0, 10.0, 10.0), (15.0, 0.0, 10.0, 10.0))


def _labels(local, plates, bulge=True, bridges=False, hole=None):
    """
    Label the phantom at points given in its local frame (u = patient right, v = anterior,
    w = superior; bone top face at w = 0).

    :param local: ``(N, 3)`` local coordinates (mm).
    :param plates: ``((cu, cv, au, av), ...)`` elliptical plates.
    :param bulge: Add a disc on each plate's inner side (towards the other plate).
    :param bridges: Join the plates by cartilage strips anteriorly and posteriorly,
        enclosing the intercondylar area.
    :param hole: Optional ``(cu, cv, r)`` disc where the cartilage is missing (denuded bone).
    :return: ``(N,)`` uint8 labels.
    """
    u, v, w = local.T
    labels = np.zeros(len(local), dtype=np.uint8)
    labels[(np.abs(u) <= 32) & (np.abs(v) <= 22) & (w >= -12) & (w <= 0)] = BONE
    layer = (w > 0) & (w <= 2.0)
    cart = np.zeros(len(local), dtype=bool)
    for cu, cv, au, av in plates:
        cart |= ((u - cu) / au) ** 2 + ((v - cv) / av) ** 2 <= 1
        if bulge:
            inner = cu - np.sign(cu) * 0.9 * au
            cart |= (u - inner) ** 2 + (v - cv - 3.0) ** 2 <= 5.0 ** 2
    if bridges:
        cart |= (np.abs(u) <= 16) & (np.abs(np.abs(v) - 10.5) <= 1.5)
    if hole is not None:
        cart &= (u - hole[0]) ** 2 + (v - hole[1]) ** 2 > hole[2] ** 2
    labels[layer & cart] = CARTILAGE
    return labels


def phantom(spacing=(0.5, 0.5, 0.5), direction=LPI, rotation=None, reflect_array=False,
            plates=EGG_PLATES, **kw):
    """
    Voxelise the phantom.

    :param spacing: Voxel size (mm).
    :param direction: 3x3 direction matrix of the image.
    :param rotation: Optional 3x3 rotation of the *anatomy* in world space.
    :param reflect_array: Reverse the array along index axis 0 afterwards (same affine),
        which mirrors the anatomy left-right.
    :param plates: Plate geometry, see :func:`_labels`.
    :param kw: Passed to :func:`_labels`.
    :return: A nibabel-backed ``Segmentation``.
    """
    rot = np.eye(3) if rotation is None else np.asarray(rotation)
    m = np.asarray(direction, float) @ np.diag(spacing)
    # Index-space bounding box of the (rotated) anatomy, with a margin.
    corners = np.array([[u, v, w] for u in (-34, 34) for v in (-24, 24) for w in (-14, 4)]) @ rot.T
    idx = corners @ np.linalg.inv(m).T
    lo = np.floor(idx.min(0)) - 3
    shape = (np.ceil(idx.max(0)) - lo + 4).astype(int)
    affine = np.eye(4)
    affine[:3, :3] = m
    affine[:3, 3] = m @ lo                 # array index 0 = index lo: the anatomy stays at world 0
    world = S.indices_to_world(affine, np.indices(shape).reshape(3, -1).T)
    labels = _labels(world @ rot, plates, **kw).reshape(shape)   # world -> local: R^T p
    if reflect_array:
        labels = labels[::-1].copy()
    return Segmentation.from_nibabel(nib.Nifti1Image(labels, affine))


def _shares(sub):
    """Region shares keyed by (plate side, letter)."""
    return {(key[1], key[0]): share for key, share in sub.region_shares.items()}


@pytest.fixture(scope='module')
def base():
    return subdivide_tibia(phantom())


# ------------------------------------------------------------------------ central region

def test_central_region_holds_twenty_percent(base):
    """Central share is 20% of each plate's tAB area (to the bisection tolerance)."""
    for side, frame in base.plates.items():
        shares = frame.region_shares
        assert shares[f'c{side[0]}t'] == pytest.approx(0.2, abs=1e-4)
        assert sum(frame.region_areas.values()) == pytest.approx(frame.tab_area, rel=1e-9)
        assert all(a > 0 for a in frame.region_areas.values())
    # The labelled mesh carries the same areas.
    areas = np.bincount(base.mesh.cell_data['region'], weights=S.triangle_areas(base.mesh),
                        minlength=10)
    for code, key in enumerate(TIBIA_REGION_KEYS):
        assert areas[code] == pytest.approx(base.region_areas[key], rel=1e-9)


@pytest.mark.parametrize('fraction', [0.1, 0.5])
def test_central_fraction_parameter(fraction):
    sub = subdivide_tibia(phantom(), central_fraction=fraction)
    assert sub.region_shares['clt'] == pytest.approx(fraction, abs=1e-4)
    assert sub.region_shares['crt'] == pytest.approx(fraction, abs=1e-4)


def test_frame_and_plate_sides(base):
    """S on the plate, b3 = superior normal, right-handed frame, patient left at -x (RAS)."""
    left, right = base.plates['left'], base.plates['right']
    assert left.centroid[0] < 0 < right.centroid[0]
    for frame in (left, right):
        assert frame.axes[2] @ [0, 0, 1] > 0.999            # flat plates: normal = +z
        assert np.linalg.det(frame.axes) == pytest.approx(1.0)
        assert frame.anterior_direction @ [0, 1, 0] > 0.99
        assert frame.centroid[2] == pytest.approx(0.0, abs=0.3)   # on the bone's top face
    assert left.ml_direction @ [1, 0, 0] > 0.99 and right.ml_direction @ [-1, 0, 0] > 0.99
    # The central ellipse follows the plate's aspect: longer A-P than M-L for these plates.
    assert abs(left.axes[0] @ [0, 1, 0]) > 0.9
    assert left.semi_axes[0] > left.semi_axes[1]


def test_ellipse_follows_plate_aspect():
    """
    Pin the ellipse-aspect reading: semi-axes ~ sqrt(0.2) x the plate's own semi-axes.

    An elliptical plate's in-plane standard deviations are in the ratio of its semi-axes, so
    the 20% ellipse is a scaled copy of the plate (area ratio 0.2 -> length ratio sqrt(0.2)).
    Squaring the ratio (variances) or swapping it (moments of inertia) would fail this.
    """
    sub = subdivide_tibia(phantom(bulge=False))
    for (_, _, au, av), side in zip(EGG_PLATES, ('left', 'right')):
        expected = np.sqrt(0.2) * np.array([max(au, av), min(au, av)])
        assert np.allclose(sub.plates[side].semi_axes, expected, rtol=0.03), side


@pytest.mark.parametrize('angle,raises', [(50, False), (70, True)])
def test_anterior_guard(angle, raises):
    """Plates whose M-L line is > 60 deg off the world M-L axis cannot be told a/p apart."""
    seg = phantom(rotation=Rotation.from_euler('z', angle, degrees=True).as_matrix())
    if raises:
        with pytest.raises(ValueError, match='60 deg'):
            subdivide_tibia(seg)
    else:
        subdivide_tibia(seg)


def test_area_moments_exact_for_a_square():
    """Exact triangle moments: a unit square's variance is 1/12 per axis, however triangulated."""
    plane = pv.Plane(i_resolution=7, j_resolution=3).triangulate()
    centroid, cov, area = area_moments(plane)
    assert area == pytest.approx(1.0)
    assert np.allclose(centroid, 0, atol=1e-12)
    assert np.allclose(np.diag(cov)[:2], 1 / 12) and abs(cov[0, 1]) < 1e-12


# ------------------------------------------------------------------------ peripheral wedges

def test_circular_plate_gives_equal_wedges():
    """A circular plate centred on S: the four 90-degree wedges hold 20% each."""
    sub = subdivide_tibia(phantom(plates=CIRCLE_PLATES, bulge=False))
    for key, share in sub.region_shares.items():
        assert share == pytest.approx(0.2, abs=0.002), key


def test_internal_wedge_faces_other_plate(base):
    """The bulges point at the other plate, so internal > external on both plates."""
    shares = base.region_shares
    assert shares['ilt'] > shares['elt'] + 0.008
    assert shares['irt'] > shares['ert'] + 0.008
    # Every triangle of the internal wedge lies on the side of S facing the other plate.
    regions = base.mesh.cell_data['region']
    centroids = S.triangle_centroids(base.mesh)
    for side in ('left', 'right'):
        frame = base.plates[side]
        code = TIBIA_REGION_KEYS.index(f'i{side[0]}t')
        along = (centroids[regions == code] - frame.centroid) @ frame.ml_direction
        assert along.min() > -1e-6


# ------------------------------------------------------------------------ invariance

def _assert_shares_close(a, b, atol):
    sa, sb = _shares(a), _shares(b)
    worst = max(abs(sa[k] - sb[k]) for k in sa)
    assert worst < atol, f'max share difference {worst:.4f}'


@pytest.mark.parametrize('rotation', [
    Rotation.from_euler('z', 20, degrees=True).as_matrix(),     # knee turned in the scanner
    Rotation.from_euler('x', 7, degrees=True).as_matrix(),      # posterior slope
    Rotation.from_euler('zx', [-25, -5], degrees=True).as_matrix(),
    # Steep tilts: a cut in the world axial plane instead of the plate plane would show here.
    Rotation.from_euler('x', 45, degrees=True).as_matrix(),
    Rotation.from_euler('y', 30, degrees=True).as_matrix(),
])
def test_shares_invariant_to_anatomy_pose(base, rotation):
    """Rotating / tilting the anatomy in world space leaves the shares unchanged."""
    _assert_shares_close(base, subdivide_tibia(phantom(rotation=rotation)), atol=0.006)


def test_shares_invariant_to_sampling(base):
    """Anisotropic voxels and an oblique, reflecting direction matrix: same shares."""
    direction = Rotation.from_euler('xyz', [10, -15, 30], degrees=True).as_matrix() @ np.diag([1, -1, 1])
    other = subdivide_tibia(phantom(spacing=(0.75, 0.3125, 0.3125)))
    oblique = subdivide_tibia(phantom(spacing=(0.6, 0.4, 0.45), direction=direction))
    _assert_shares_close(base, other, atol=0.006)
    _assert_shares_close(base, oblique, atol=0.006)


def test_mirror_swaps_plates():
    """A left-right mirrored knee swaps the plates' shares; internal stays internal."""
    original = _shares(subdivide_tibia(phantom()))
    mirrored = _shares(subdivide_tibia(phantom(reflect_array=True)))
    for letter in REGION_LETTERS:
        assert mirrored[('l', letter)] == pytest.approx(original[('r', letter)], abs=0.002)
        assert mirrored[('r', letter)] == pytest.approx(original[('l', letter)], abs=0.002)


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
    other = subdivide_tibia(Segmentation.from_sitk(itk))
    _assert_shares_close(base, other, atol=1e-6)
    # Array indices map to the same world points on both backends.
    nib_world = S.array_indices_to_world(seg, np.argwhere(seg.array == CARTILAGE))
    itk_seg = Segmentation.from_sitk(itk)
    itk_world = S.array_indices_to_world(itk_seg, np.argwhere(itk_seg.array == CARTILAGE))
    assert np.allclose(np.sort(nib_world, axis=0), np.sort(itk_world, axis=0))


# ------------------------------------------------------------------------ plates / holes

def test_intercondylar_area_never_filled():
    """Cartilage bridges enclosing the intercondylar area: it stays out of tAB."""
    seg = phantom(bridges=True, bulge=False)
    sub = subdivide_tibia(seg)
    assert sub.plate_split == 'two_means'

    centroids = S.triangle_centroids(sub.mesh)
    gap = (np.abs(centroids[:, 0]) < 3) & (np.abs(centroids[:, 1]) < 7)
    assert not gap.any(), 'intercondylar area filled into tAB'
    for frame in sub.plates.values():
        assert frame.tab_area - frame.cab_area < 2.0      # nothing big was filled

    # Sanity: the plate-unaware step-1 rule *would* fill it.
    bone = sub.bone_mesh
    covered = S.cartilage_coverage(bone, seg, CARTILAGE)
    naive = S.total_subchondral_area(bone, covered)
    naive_centroids = S.triangle_centroids(bone)[naive]
    assert ((np.abs(naive_centroids[:, 0]) < 3) & (np.abs(naive_centroids[:, 1]) < 7)).any()


def test_denuded_hole_inside_a_plate_is_filled():
    """A cartilage-free disc inside one plate is part of that plate's tAB."""
    sub = subdivide_tibia(phantom(hole=(14.0, -4.0, 3.0)))
    right = sub.plates['right']
    assert right.tab_area - right.cab_area == pytest.approx(np.pi * 9, rel=0.25)
    assert sub.plates['left'].tab_area - sub.plates['left'].cab_area < 2.0
    assert sub.region_shares['crt'] == pytest.approx(0.2, abs=1e-4)


def test_missing_bone_label_raises():
    seg = phantom()
    array = np.asarray(seg.image.dataobj).copy()
    array[array == BONE] = 0
    with pytest.raises(ValueError, match='bone label'):
        subdivide_tibia(Segmentation.from_nibabel(nib.Nifti1Image(array, seg.image.affine)))


# ------------------------------------------------------------------------ naming / Tibia

def test_anatomical_names():
    right = anatomical_tibia_names('right')     # right knee: patient-left plate is medial
    assert right['clt'] == 'cMT' and right['plt'] == 'pMT' and right['crt'] == 'cLT'
    left = anatomical_tibia_names('left')
    assert left['clt'] == 'cLT' and left['irt'] == 'iMT'
    assert anatomical_tibia_names(None) is None
    with pytest.raises(ValueError):
        anatomical_tibia_names('medial')
    renamed = rename_tibia_regions({'cLT': 1, 'crt': 2}, 'right')
    assert renamed == {'cMT': 1, 'cLT': 2}
    # A re-keyed dict must not be mistaken for patient-side keys.
    with pytest.raises(ValueError, match='medial/lateral'):
        split_tibia_thickness_by_compartment({'cMT': {}, 'cLT': {}})


def test_tibia_labels_voxels_by_nearest_region():
    """Tibia transfers the regions to its cartilage voxels; plates by patient side."""
    seg = phantom()
    tibia = Tibia(seg, CARTILAGE, laterality='right')
    tibia.get_surface_points()
    tibia.calculate_landmarks()
    tibia.extract_subregions()
    assert tibia.anatomical_names['clt'] == 'cMT'
    world = S.array_indices_to_world(seg, tibia.point_cloud)
    is_left = np.char.endswith(tibia.point_labels.astype(str), 'LT')
    assert (world[is_left, 0] < 0).all() and (world[~is_left, 0] > 0).all()
    for key in TIBIA_REGION_KEYS:
        assert len(getattr(tibia, key)) > 0, key
    # classify_point agrees with the precomputed labels.
    for i in (0, len(world) // 2, len(world) - 1):
        assert tibia.classify_point(tibia.point_cloud[i]) == tibia.point_labels[i]
    # Central voxels sit round S, laterally inside the plate.
    centre = S.array_indices_to_world(seg, tibia.clt).mean(axis=0)
    assert np.linalg.norm(centre[:2] - tibia.left_landmarks.centroid[:2]) < 1.5


def test_transfer_qc_flags_cartilage_far_from_the_bone(caplog):
    """Cartilage far above the tAB (a broken segmentation) is reported; a clean knee is not."""
    clean = Tibia(phantom(), CARTILAGE)
    with caplog.at_level('WARNING', logger='morphometry'):
        clean.calculate_landmarks()
    assert not caplog.records
    assert clean.transfer_qc['max_mm'] < 3.0 and clean.transfer_qc['far_fraction'] == 0.0

    # A detached cartilage disc floating 2.5-4.5 mm above the bone's anterolateral corner,
    # ~10 mm from either plate; outlier removal is disabled so it stays in the point cloud.
    seg = phantom()
    array = np.asarray(seg.image.dataobj).copy()
    world = S.indices_to_world(seg.image.affine, np.indices(array.shape).reshape(3, -1).T)
    slab = (((world[:, 0] - 27) ** 2 + (world[:, 1] - 17) ** 2 <= 3 ** 2)
            & (world[:, 2] > 2.5) & (world[:, 2] <= 4.5))
    array[slab.reshape(array.shape)] = CARTILAGE
    broken = Tibia(Segmentation.from_nibabel(nib.Nifti1Image(array, seg.image.affine)),
                   CARTILAGE, outlier_ratio=0)
    caplog.clear()
    with caplog.at_level('WARNING', logger='morphometry'):
        broken.calculate_landmarks()
    assert broken.transfer_qc['far_fraction'] > 0.01
    assert broken.transfer_qc['max_mm'] > 8.0
    assert any('subchondral bone area' in r.getMessage() for r in caplog.records)

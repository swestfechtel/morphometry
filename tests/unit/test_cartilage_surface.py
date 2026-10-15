"""Unit tests for the bone-surface geometry behind the Wirth & Eckstein subregions.

``morphometry.cartilage.surface`` builds a bone mesh in world mm, flags the triangles covered
by cartilage (cAB), fills enclosed denuded holes (tAB) and splits meshes exactly. All phantoms
are synthetic: anisotropic voxels and an oblique, reflecting direction matrix, so any place
where index space leaks into the result shows up as a wrong position or area.
"""
import os
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pytest
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from morphometry.image_io import Segmentation
from morphometry.cartilage import surface as S

BONE, CARTILAGE = 1, 3
ANISO = (0.75, 0.3125, 0.3125)  # dfg_knees voxel size


def _affine(spacing, euler=None, reflect=False, origin=(12.0, -40.0, 7.5)):
    """Index-to-world matrix with the given spacing, rotation (deg) and optional x reflection."""
    direction = np.eye(3) if euler is None else Rotation.from_euler('xyz', euler, degrees=True).as_matrix()
    if reflect:
        direction = direction @ np.diag([-1.0, 1.0, 1.0])
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def _sphere(spacing, radius=10.0, **kw):
    """A voxelised sphere of ``radius`` mm centred in the array; returns (segmentation, centre)."""
    spacing = np.asarray(spacing, float)
    shape = np.ceil(2 * radius / spacing + 6).astype(int)
    affine = _affine(spacing, **kw)
    world = S.indices_to_world(affine, np.indices(shape).reshape(3, -1).T)
    centre = S.indices_to_world(affine, [(shape - 1) / 2])[0]
    mask = (np.linalg.norm(world - centre, axis=1) <= radius).reshape(shape).astype(np.uint8)
    return Segmentation.from_nibabel(nib.Nifti1Image(mask, affine)), centre


# ---------------------------------------------------------------------------- bone mesh

@pytest.mark.parametrize('euler,reflect', [(None, False), ((10, 25, -35), True)])
def test_sphere_mesh_in_world_mm(euler, reflect):
    """Mesh sits at the sphere's world position with its radius, analytic area, outward normals."""
    seg, centre = _sphere(ANISO, euler=euler, reflect=reflect)
    mesh = S.bone_surface_mesh(seg, BONE)
    areas = S.triangle_areas(mesh)
    centroids = S.triangle_centroids(mesh)

    area_centroid = (areas[:, None] * centroids).sum(axis=0) / areas.sum()
    assert np.linalg.norm(area_centroid - centre) < 0.05
    radii = np.linalg.norm(mesh.points - centre, axis=1)
    assert abs(radii.mean() - 10.0) < 0.1 and radii.max() < 10.5 and radii.min() > 9.5
    assert areas.sum() == pytest.approx(4 * np.pi * 100, rel=0.02)
    assert mesh.n_open_edges == 0  # nothing touches the array border: closed
    outward = np.einsum('ij,ij->i', mesh.cell_data['Normals'], centroids - centre)
    assert np.all(outward > 0)


def _as_sitk(seg):
    """
    Wrap a nibabel segmentation as the SimpleITK image SimpleITK would read from the same file.

    nibabel's affine maps to RAS; SimpleITK stores LPS geometry, so its direction and origin
    are the RAS ones with the first two world axes negated.
    """
    affine = seg.image.affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    itk = sitk.GetImageFromArray(np.asarray(seg.image.dataobj).transpose(2, 1, 0))
    itk.SetSpacing(spacing.tolist())
    itk.SetDirection((ras_to_lps @ affine[:3, :3] / spacing).ravel().tolist())
    itk.SetOrigin((ras_to_lps @ affine[:3, 3]).tolist())
    return Segmentation.from_sitk(itk)


def _assert_same_points(a, b, atol=1e-4):
    """Every point of ``a`` has a point of ``b`` within ``atol`` and vice versa."""
    assert np.max(cKDTree(b).query(a)[0]) < atol
    assert np.max(cKDTree(a).query(b)[0]) < atol


def test_simpleitk_backend_matches_nibabel():
    """SimpleITK's LPS geometry is converted: both backends give the same RAS world mesh."""
    seg, centre = _sphere(ANISO, euler=(10, 25, -35), reflect=True)
    itk = _as_sitk(seg)
    assert np.allclose(S.index_to_world_affine(itk), S.index_to_world_affine(seg))
    mesh_nib = S.bone_surface_mesh(seg, BONE)
    mesh_itk = S.bone_surface_mesh(itk, BONE)
    _assert_same_points(mesh_itk.points, mesh_nib.points)
    assert S.triangle_areas(mesh_itk).sum() == pytest.approx(S.triangle_areas(mesh_nib).sum(), rel=1e-6)


DFG_SAMPLE = Path(os.environ.get(
    'MORPH_DFG_KNEES_SAMPLE',
    '/home/simon/Data/dfg_knees/monai_segmentations/labels/original/K1_0_belastet.nii.gz'))


def test_backends_agree_on_a_real_file():
    """The same NIfTI read by nibabel and by SimpleITK gives the same RAS geometry and labels."""
    if not DFG_SAMPLE.exists():
        pytest.skip(f'dfg_knees sample not found at {DFG_SAMPLE} (set MORPH_DFG_KNEES_SAMPLE)')
    nib_seg = Segmentation.from_nibabel(nib.load(DFG_SAMPLE))
    itk_seg = Segmentation.from_sitk(sitk.ReadImage(str(DFG_SAMPLE)))
    assert np.allclose(S.index_to_world_affine(itk_seg), S.index_to_world_affine(nib_seg), atol=1e-4)
    assert np.array_equal(np.asarray(S.label_array_xyz(itk_seg)), np.asarray(S.label_array_xyz(nib_seg)))


def test_mesh_area_independent_of_spacing():
    """The same object sampled at very different voxel sizes has the same surface area."""
    a_iso = S.triangle_areas(S.bone_surface_mesh(_sphere((1.0, 1.0, 1.0))[0], BONE)).sum()
    a_aniso = S.triangle_areas(S.bone_surface_mesh(_sphere(ANISO, euler=(0, 0, 30))[0], BONE)).sum()
    assert a_aniso == pytest.approx(a_iso, rel=0.01)


def test_fov_cut_leaves_open_boundary_without_cap():
    """A bone running out of the array is cut open exactly at the array face, with no cap."""
    a = np.zeros((40, 40, 30), np.uint8)
    a[10:30, 10:30, 0:20] = BONE  # touches the z = 0 border: a 10 x 10 x 10 mm box at 0.5 mm
    affine = _affine((0.5, 0.5, 0.5), euler=(0, 30, 0))
    seg = Segmentation.from_nibabel(nib.Nifti1Image(a, affine))
    mesh = S.bone_surface_mesh(seg, BONE)
    assert mesh.n_open_edges > 0
    # The cut is at the outer voxel boundary (index z = -0.5), and nothing lies beyond it.
    z = S.indices_to_world(np.linalg.inv(affine), mesh.points)[:, 2]
    assert z.min() == pytest.approx(-0.5, abs=1e-9)
    # Four 10 x 10 mm sides + the 10 x 10 mm top; marching cubes and smoothing round the
    # 80 mm of box edges (about 4% of the area on so small a box).
    assert S.triangle_areas(mesh).sum() == pytest.approx(5 * 100, rel=0.05)
    # The shaft is not pulled in towards the cut: the open rim is the 4 x 10 mm cross-section.
    rim = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                     manifold_edges=False, non_manifold_edges=False)
    assert rim.compute_cell_sizes(length=True, area=False, volume=False)['Length'].sum() == \
        pytest.approx(40.0, rel=0.03)
    # Cutting keeps the outward winding.
    centre = S.indices_to_world(affine, [(19.5, 19.5, 9.5)])[0]
    cells = S.triangle_centroids(mesh)
    assert np.all(np.einsum('ij,ij->i', mesh.cell_data['Normals'], cells - centre) > 0)
    closed = S.bone_surface_mesh(seg, BONE, open_at_fov=False)
    assert closed.n_open_edges == 0
    assert S.triangle_areas(closed).sum() == pytest.approx(6 * 100, rel=0.05)


def test_largest_component_only():
    """A stray voxel blob with the bone label produces no surface of its own."""
    a = np.zeros((30, 30, 30), np.uint8)
    a[5:20, 5:20, 5:20] = BONE
    a[25:27, 25:27, 25:27] = BONE
    seg = Segmentation.from_nibabel(nib.Nifti1Image(a, np.eye(4)))
    mesh = S.bone_surface_mesh(seg, BONE)
    assert mesh.points.max() < 21
    assert len(S.bone_surface_mesh(seg, BONE, largest_component=False).points) > len(mesh.points)


# ---------------------------------------------------------------------------- cAB / tAB

SPACING = (0.5, 0.4, 0.3)
CAP = (slice(20, 60), slice(20, 60))     # 20 x 16 mm cartilage footprint
HOLE = (slice(35, 45), slice(35, 45))    # 5 x 4 mm denuded hole in it
CAP_AREA, HOLE_AREA = 20 * 16, 5 * 4


def _capped_bone(z0):
    """
    A box bone with a flat top at index z 30 carrying a cartilage cap with a hole, plus a
    cartilage blob floating 2 voxels (0.6 mm) above the bone elsewhere.

    :param z0: First z index of the bone (0 = the bone runs out of the array, i.e. a FOV cut).
    """
    a = np.zeros((80, 80, 45), np.uint8)
    a[10:70, 10:70, z0:31] = BONE
    a[CAP[0], CAP[1], 31:35] = CARTILAGE
    a[HOLE[0], HOLE[1], 31:35] = 0
    a[62:68, 62:68, 33:36] = CARTILAGE  # separated from the bone by z 31, 32
    affine = _affine(SPACING, euler=(5, -10, 20), reflect=True)
    return Segmentation.from_nibabel(nib.Nifti1Image(a, affine)), affine


@pytest.mark.parametrize('z0', [0, 5], ids=['fov_cut', 'closed'])
def test_coverage_and_hole_filling(z0):
    """cAB excludes the hole and the floating blob; tAB adds exactly the hole, not the bone."""
    seg, affine = _capped_bone(z0)
    mesh = S.bone_surface_mesh(seg, BONE)
    areas = S.triangle_areas(mesh)
    covered = S.cartilage_coverage(mesh, seg, CARTILAGE)
    total = S.total_subchondral_area(mesh, covered)

    # Every covered triangle lies on the top face, under the cap and outside the hole.
    idx = S.indices_to_world(np.linalg.inv(affine), S.triangle_centroids(mesh))
    cx, cy, cz = idx[covered].T
    assert np.all(np.abs(cz - 30.5) < 0.5)
    assert np.all((cx > 19) & (cx < 60) & (cy > 19) & (cy < 60))
    top = np.abs(idx[:, 2] - 30.5) < 0.5
    in_hole = top & (idx[:, 0] > 36) & (idx[:, 0] < 43) & (idx[:, 1] > 36) & (idx[:, 1] < 43)
    assert not covered[in_hole].any() and total[in_hole].all()
    under_blob = top & (idx[:, 0] > 62) & (idx[:, 0] < 67) & (idx[:, 1] > 62) & (idx[:, 1] < 67)
    assert under_blob.any() and not covered[under_blob].any() and not total[under_blob].any()

    # Areas: the covered patch may reach a sub-voxel band (< 0.3 mm) past the cartilage rim
    # (72 mm outside, 18 mm around the hole), never less than the cartilage footprint.
    rim_out, rim_hole, band = 72.0, 18.0, 0.3
    c_ab, t_ab = areas[covered].sum(), areas[total].sum()
    assert CAP_AREA - HOLE_AREA <= c_ab <= CAP_AREA - HOLE_AREA + band * (rim_out + rim_hole)
    assert CAP_AREA <= t_ab <= CAP_AREA + band * rim_out
    assert HOLE_AREA - band * rim_hole <= t_ab - c_ab <= HOLE_AREA
    assert t_ab < 0.2 * areas.sum()  # the rest of the bone is not tAB


def test_large_uncovered_region_is_not_a_hole():
    """An enclosed uncovered region larger than max_hole_fraction x cAB is not filled."""
    seg, _ = _capped_bone(0)
    mesh = S.bone_surface_mesh(seg, BONE)
    covered = S.cartilage_coverage(mesh, seg, CARTILAGE)
    total = S.total_subchondral_area(mesh, covered, max_hole_fraction=0.01)
    assert np.array_equal(total, covered)


def test_no_cartilage():
    """An absent cartilage label gives empty cAB and tAB."""
    seg, _ = _capped_bone(0)
    mesh = S.bone_surface_mesh(seg, BONE)
    covered = S.cartilage_coverage(mesh, seg, 99)
    assert not covered.any() and not S.total_subchondral_area(mesh, covered).any()


# ---------------------------------------------------------------------------- splitting

def _flat_rect():
    """A 10 x 6 mm rectangle in the z = 0 plane (corner at the origin), 17 x 11 quads split in two."""
    x, y = np.meshgrid(np.linspace(0, 10, 18), np.linspace(0, 6, 12), indexing='ij')
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    i, j = np.meshgrid(np.arange(17), np.arange(11), indexing='ij')
    p00 = (i * 12 + j).ravel()
    p10, p01, p11 = p00 + 12, p00 + 1, p00 + 13
    tris = np.concatenate([np.stack([p00, p10, p11], 1), np.stack([p00, p11, p01], 1)])
    rect = pv.PolyData(points, np.hstack([np.full((len(tris), 1), 3), tris]).ravel())
    rect.cell_data['flag'] = np.arange(rect.n_cells) % 3 == 0
    return rect


def test_plane_split_is_exact():
    """A plane cut conserves area and puts the analytic share on each side, keeping cell data."""
    rect = _flat_rect()
    total = S.triangle_areas(rect).sum()
    neg, pos = S.split_mesh_by_plane(rect, origin=(3.7, 0, 0), normal=(2, 0, 0))
    a_neg, a_pos = S.triangle_areas(neg).sum(), S.triangle_areas(pos).sum()
    assert neg.is_all_triangles and pos.is_all_triangles
    assert (a_neg + a_pos) / total == pytest.approx(1, abs=1e-9)
    assert a_neg / total == pytest.approx(0.37, abs=1e-9)
    for part in (neg, pos):  # pieces keep the parent's +z winding
        f = np.asarray(part.faces).reshape(-1, 4)[:, 1:]
        p = part.points[f]
        assert np.all(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] > 0)
    # The cut line's points are shared, so each part is one connected piece.
    assert neg.connectivity().point_data['RegionId'].max() == 0
    assert 'flag' in neg.cell_data and 'flag' in pos.cell_data
    # The flag's area is conserved through the cut as well.
    flag_area = S.triangle_areas(rect)[rect.cell_data['flag'].astype(bool)].sum()
    split_flag = sum(S.triangle_areas(p)[p.cell_data['flag'].astype(bool)].sum() for p in (neg, pos))
    assert split_flag == pytest.approx(flag_area, rel=1e-9)


def test_oblique_scalar_split_is_exact():
    """A linear point scalar (an oblique line) splits the rectangle into the analytic shares."""
    rect = _flat_rect()
    total = S.triangle_areas(rect).sum()
    s = rect.points[:, 0] + rect.points[:, 1]  # cut along x + y = 6: a triangle of area 18
    below, above = S.split_mesh_by_scalar(rect, s, 6.0)
    a_below, a_above = S.triangle_areas(below).sum(), S.triangle_areas(above).sum()
    assert (a_below + a_above) / total == pytest.approx(1, abs=1e-9)
    assert a_below == pytest.approx(18.0, rel=1e-9)


def test_radial_level_set_split():
    """A nonlinear (radial) level set conserves area and gives ~the analytic disc fraction."""
    disc = pv.PolyData(pv.Disc(inner=0, outer=10, r_res=40, c_res=180).triangulate())
    disc.points = disc.points.astype(float)
    total = S.triangle_areas(disc).sum()
    radius = np.linalg.norm(disc.points[:, :2], axis=1)
    inner, outer = S.split_mesh_by_scalar(disc, radius, 5.0)
    a_in, a_out = S.triangle_areas(inner).sum(), S.triangle_areas(outer).sum()
    assert (a_in + a_out) / total == pytest.approx(1, abs=1e-9)
    assert a_in / total == pytest.approx(0.25, rel=1e-3)
    assert np.linalg.norm(inner.points[:, :2], axis=1).max() <= 5.0 + 1e-6


def test_split_with_everything_on_one_side():
    """A level outside the scalar range returns the whole mesh on one side and nothing on the other."""
    rect = _flat_rect()
    below, above = S.split_mesh_by_scalar(rect, rect.points[:, 0], 100.0)
    assert above.n_cells == 0
    assert S.triangle_areas(below).sum() == pytest.approx(60.0, rel=1e-9)


def test_split_through_vertices_adds_no_seams():
    """A level through mesh vertices reuses them: no duplicate points, no extra open edges."""
    x, y = np.meshgrid(np.arange(11.0), np.arange(7.0), indexing='ij')  # 1 mm grid, 10 x 6 mm
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    i, j = np.meshgrid(np.arange(10), np.arange(6), indexing='ij')
    p00 = (i * 7 + j).ravel()
    tris = np.concatenate([np.stack([p00, p00 + 7, p00 + 8], 1), np.stack([p00, p00 + 8, p00 + 1], 1)])
    grid = pv.PolyData(points, np.hstack([np.full((len(tris), 1), 3), tris]).ravel())
    s = grid.points[:, 0] + grid.points[:, 1]  # x + y = 6 runs through grid vertices
    below, above = S.split_mesh_by_scalar(grid, s, 6.0)
    for part in (below, above):
        assert np.all(S.triangle_areas(part) > 0)
        assert part.n_open_edges == part.clean().n_open_edges  # nothing left to merge
    assert S.triangle_areas(below).sum() == pytest.approx(18.0, rel=1e-12)
    assert S.triangle_areas(below).sum() + S.triangle_areas(above).sum() == pytest.approx(60.0, rel=1e-12)


def _grid_with_coverage(covered_box, hole_box):
    """
    A 10 x 6 mm open 1 mm grid with the triangles whose centroid lies in ``covered_box`` but
    not in ``hole_box`` flagged covered. Boxes are ``(x0, x1, y0, y1)`` in mm.
    """
    x, y = np.meshgrid(np.arange(11.0), np.arange(7.0), indexing='ij')
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    i, j = np.meshgrid(np.arange(10), np.arange(6), indexing='ij')
    p00 = (i * 7 + j).ravel()
    tris = np.concatenate([np.stack([p00, p00 + 7, p00 + 8], 1), np.stack([p00, p00 + 8, p00 + 1], 1)])
    grid = pv.PolyData(points, np.hstack([np.full((len(tris), 1), 3), tris]).ravel())
    c = S.triangle_centroids(grid)

    def inside(box):
        return (c[:, 0] > box[0]) & (c[:, 0] < box[1]) & (c[:, 1] > box[2]) & (c[:, 1] < box[3])

    return grid, inside(covered_box) & ~inside(hole_box), inside(hole_box)


def test_uncovered_components():
    """An enclosed hole and the open-boundary remainder are two components with their areas."""
    grid, covered, in_hole = _grid_with_coverage((2, 8, 1, 5), (4, 6, 2, 4))
    labels, touches, areas = S.uncovered_components(grid, covered)
    assert np.all(labels[covered] == -1)
    assert len(areas) == 2
    hole = labels[in_hole][0]
    assert np.all(labels[in_hole] == hole)
    assert areas[hole] == pytest.approx(4.0) and not touches[hole]
    assert areas[1 - hole] == pytest.approx(60 - 24) and touches[1 - hole]
    total = S.total_subchondral_area(grid, covered)
    assert np.array_equal(total, covered | in_hole)


def test_hole_reaching_the_open_boundary_is_not_filled():
    """Denuded bone that runs out to the mesh's open (FOV) rim is not an enclosed hole."""
    # Covered strip from the bottom rim (y = 0) up; the 'hole' reaches that rim too.
    grid, covered, in_hole = _grid_with_coverage((2, 8, 0, 5), (4, 6, 0, 2))
    labels, touches, _ = S.uncovered_components(grid, covered)
    assert touches[labels[in_hole][0]]
    assert np.array_equal(S.total_subchondral_area(grid, covered), covered)

"""Unit tests for the area-weighted surface cartilage thickness (Wirth & Eckstein 2008, II-C).

The thickness is the area-weighted Euclidean distance between a bone's subchondral bone area
(cAB / tAB) and its articular cartilage surface (AC), in both directions
(:mod:`morphometry.cartilage.thickness`). These tests pin it on phantoms whose thickness is
known analytically:

* the **tibial** phantom of :mod:`tests.unit.test_tibia_regions` -- a flat bone block carrying
  two elliptical cartilage plates of a constant 2.0 mm;
* the **femoral** phantom of :mod:`tests.unit.test_femur_regions` -- two condyle cylinders
  carrying a 2.0 mm shell, with an optional denuded disc;

both voxelised from a geometry defined in world mm, so they can be sampled with any voxel size
and direction matrix and rotated into any pose. Note what the phantoms' *rasterisation* implies
for the truth: the bone surface sits half a voxel outside the outermost bone voxel centre and
the cartilage surface half a voxel outside the outermost cartilage voxel centre, so a 2.0 mm
layer sampled at 0.8 mm through-plane holds only two voxel layers and is 1.6 mm thick as
rasterised, not 2.0.
"""
import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from morphometry.image_io import Segmentation
from morphometry.cartilage import surface as S
from morphometry.cartilage.femur_regions import FEMUR_REGION_KEYS as FEMUR_REGION_KEYS_
from morphometry.cartilage.knee import Femur, Tibia, find_thinnest_area
from morphometry.cartilage.thickness import (
    DEFAULT_GRID_MM, KneeFrame, _extreme_mean, knee_frame_from_femur, knee_frame_from_tibia,
    measure_thickness, weighted_statistics,
)
from tests.unit.test_femur_regions import phantom as femur_phantom
from tests.unit.test_tibia_regions import phantom as tibia_phantom

TIBIA_BONE, TIBIA_CARTILAGE = 2, 4
FEMUR_CARTILAGE = 3
TRUE_MM = 2.0            # the constant cartilage thickness of both phantoms
EXTREME = 0.01           # the paper's top / bottom 1% for aMav / aMiv


# ----------------------------------------------------------------------- weighted statistics

def test_weighted_statistics_of_a_known_distribution():
    """aMe / aSD / aMax / aMin on a two-valued sample, and the 1% tails."""
    values = np.array([1.0, 3.0])
    stats = weighted_statistics(values, np.array([3.0, 1.0]))      # 3:1 area ratio
    assert stats['aMe'] == pytest.approx(1.5)
    assert stats['aSD'] == pytest.approx(np.sqrt(0.75 * 0.25 + 0.25 * 2.25))
    assert (stats['aMax'], stats['aMin']) == (3.0, 1.0)
    # The extreme 1% of the area lies wholly inside one of the two triangles.
    assert stats['aMav'] == pytest.approx(3.0)
    assert stats['aMiv'] == pytest.approx(1.0)


def test_weighted_statistics_ignores_zero_weights_and_nan():
    """A zero-area entry must not reach aMax / aMin (the denuded placeholder of a clean region)."""
    stats = weighted_statistics([1.0, 2.0, 0.0, np.nan], [1.0, 1.0, 0.0, 1.0])
    assert stats['aMe'] == pytest.approx(1.5)
    assert (stats['aMin'], stats['aMax']) == (1.0, 2.0)
    assert np.isnan(weighted_statistics([], [])['aMe'])
    assert np.isnan(weighted_statistics([1.0], [0.0])['aMe'])


def test_extreme_mean_splits_the_straddling_triangle_by_area():
    """The top fraction takes only the part of the boundary triangle's area it needs."""
    # Areas 1, 1, 8 with values 10, 8, 1: the top 10% of the area (1.0) is exactly the first.
    values, weights = np.array([10.0, 8.0, 1.0]), np.array([1.0, 1.0, 8.0])
    assert _extreme_mean(values, weights, 0.1, largest=True) == pytest.approx(10.0)
    # The top 15% is the first triangle plus half of the second: (10 * 1 + 8 * 0.5) / 1.5.
    assert _extreme_mean(values, weights, 0.15, largest=True) == pytest.approx(14.0 / 1.5)
    assert _extreme_mean(values, weights, 1.0, largest=True) == pytest.approx(
        np.average(values, weights=weights))


def test_region_statistics_pool_the_two_directions_by_area():
    """Hand-checked arithmetic of the pooling scheme, independent of any mesh.

    Four 1 mm^2 tAB triangles -- three covered, at 2, 2 and 3 mm, one denuded -- and two
    1.5 mm^2 AC triangles at 2 and 4 mm.
    """
    from morphometry.cartilage.thickness import _region_statistics

    stats = _region_statistics(np.array([2.0, 2.0, 3.0, 0.0]), np.ones(4),
                               np.array([1, 1, 1, 0], bool),
                               np.array([2.0, 4.0]), np.full(2, 1.5), EXTREME)
    assert stats['cMe'] == pytest.approx(7 / 3)          # area-weighted over cAB only
    assert stats['bMe'] == pytest.approx(3.0)            # area-weighted over AC
    assert stats['aMe'] == pytest.approx((7 / 3 + 3.0) / 2)
    assert stats['tab_aMe'] == pytest.approx(stats['aMe'] * 0.75)   # diluted by the denuded 1/4
    assert stats['denuded_fraction'] == pytest.approx(0.25)
    # The two families are kept apart: the bare names describe the cartilage that is there,
    # the tab_ ones the same with denuded bone counted as 0 mm.
    assert (stats['aMin'], stats['aMax']) == (2.0, 4.0)
    assert (stats['tab_aMin'], stats['tab_aMax']) == (0.0, 4.0)
    assert stats['tab_aSD'] > stats['aSD']          # the denuded zeros widen the distribution

    # Fully covered: the tAB statistics are the cAB ones, and the 0 mm denuded placeholder
    # (weight 0) must not be read as the minimum.
    clean = _region_statistics(np.array([2.0, 3.0]), np.ones(2), np.ones(2, bool),
                               np.array([2.0, 4.0]), np.full(2, 1.5), EXTREME)
    assert clean['tab_aMe'] == pytest.approx(clean['aMe'])
    assert clean['aMin'] == pytest.approx(2.0)
    assert clean['tab_aMin'] == pytest.approx(2.0)


# ------------------------------------------------------------------------------ the frame

def test_tibial_frame_is_the_plateau_plane():
    """The shared frame's axes are the phantom's own: +u patient right, +v anterior, +w up."""
    tibia = Tibia(tibia_phantom(), TIBIA_CARTILAGE)
    frame = tibia.knee_frame()
    assert frame.source == 'tibia' and frame.grid_mm == DEFAULT_GRID_MM
    # The phantom's two plate centres are 2 mm apart in v over 31 mm in u, so the frame's
    # right axis is tilted ~4 degrees from world +x -- as it should be, it joins the plates.
    assert frame.axes[0] @ [1, 0, 0] > 0.99       # patient right
    assert frame.axes[1] @ [0, 1, 0] > 0.99       # anterior
    assert frame.axes[2] @ [0, 0, 1] > 0.999      # the plate normal, superior
    assert np.linalg.det(frame.axes) == pytest.approx(1.0)
    # The origin is the midpoint of the two plates, i.e. near the phantom's own centre.
    assert np.linalg.norm(frame.origin[:2]) < 3.0


def test_frame_cells_round_half_up_and_invert():
    """A cell index is the rounded in-plane offset in grid units, and maps back to its centre."""
    frame = KneeFrame(origin=np.zeros(3), axes=np.eye(3), grid_mm=0.5)
    # The third coordinate (the direction the projection collapses) is ignored.
    cells = frame.cells(np.array([[0.0, 0.0, 0.0], [0.26, -0.26, 5.0], [1.0, 2.0, 0.0]]))
    assert cells.tolist() == [[0, 0], [1, -1], [2, 4]]
    # Exact half-grid ties go up on both sides, so the mapping is monotone (np.rint would
    # round -0.25 and +0.25 mm to the same cell 0).
    assert frame.cells(np.array([[0.25, -0.25, 0.0]])).tolist() == [[1, 0]]
    assert np.allclose(frame.cell_centre((2, 4)), [1.0, 2.0, 0.0])


def test_frames_of_different_source_do_not_match():
    """A femur-only frame is never mistaken for the shared tibial one, however alike it looks."""
    femur = Femur(femur_phantom(), FEMUR_CARTILAGE)
    femur.calculate_landmarks()
    femoral = knee_frame_from_femur(femur.subdivision)
    assert femoral.source == 'femur'
    assert femoral.matches(knee_frame_from_femur(femur.subdivision))
    assert not femoral.matches(KneeFrame(origin=femoral.origin, axes=femoral.axes,
                                         grid_mm=femoral.grid_mm, source='tibia'))
    tibia = Tibia(tibia_phantom(), TIBIA_CARTILAGE)
    tibia.calculate_landmarks()
    assert not femoral.matches(knee_frame_from_tibia(tibia.subdivision))


# -------------------------------------------------------------- a constant-thickness layer

def _tibial_thickness(**kw):
    """Measure the tibial phantom and return ``(ThicknessMaps, Tibia)``."""
    tibia = Tibia(tibia_phantom(**kw), TIBIA_CARTILAGE)
    return tibia.calculate_thickness(), tibia


def _bone_mean(statistics, keys) -> float:
    """tAB-area-weighted mean of ``aMe`` over some subregions (the whole-bone thickness)."""
    areas = np.array([statistics[k]['tab_area_mm2'] for k in keys])
    values = np.array([statistics[k]['aMe'] for k in keys])
    return float(np.average(values, weights=areas))


def test_flat_plate_reads_its_true_thickness():
    """A constant 2 mm layer on a flat plate reads 2 mm, in both directions and per region.

    The *interior* is where the method is exact, and the central subregions are the interior:
    they are pinned to 0.5%. The peripheral ones carry the rim deficit documented on
    :func:`~morphometry.cartilage.surface.articular_surface_mesh` -- a fixed-width band at the
    cartilage edge, so it costs a subregion in proportion to its perimeter-to-area ratio --
    and are pinned only loosely, to catch a gross regression rather than to certify accuracy.
    """
    maps, _ = _tibial_thickness()
    whole = _bone_mean(maps.statistics, list(maps))
    assert whole == pytest.approx(TRUE_MM, rel=0.05)
    for key in ('clt', 'crt'):
        assert maps.statistics[key]['aMe'] == pytest.approx(TRUE_MM, rel=0.005), key
    for key, stats in maps.statistics.items():
        assert stats['aMe'] == pytest.approx(TRUE_MM, rel=0.12), key
        if key not in ('clt', 'crt'):      # the rim deficit, never an excess
            assert stats['aMe'] < TRUE_MM, key
        # Both directions agree: the layer is the same thickness measured either way.
        assert stats['cMe'] == pytest.approx(stats['bMe'], abs=0.25), key
        # No denuded bone, so the tAB statistics equal the cAB ones.
        assert stats['denuded_fraction'] == pytest.approx(0.0, abs=0.01), key
        assert stats['tab_aMe'] == pytest.approx(stats['aMe'], rel=0.02), key
        assert stats['aMax'] <= TRUE_MM + 0.2, key


def test_central_region_is_uniform_and_the_extremes_are_ordered():
    """Inside the central region the 2 mm layer is uniform: aSD ~ 0 and aMin ~ aMax ~ 2 mm."""
    maps, _ = _tibial_thickness()
    for key in ('clt', 'crt'):
        stats = maps.statistics[key]
        assert stats['aSD'] < 0.1, key
        assert stats['aMin'] == pytest.approx(TRUE_MM, abs=0.3), key
        assert stats['aMiv'] <= stats['aMe'] <= stats['aMav'] <= stats['aMax'], key


def test_a_wedge_reads_its_mean_thickness(monkeypatch):
    """A linearly varying layer reads its mean thickness, not its extremes."""
    # The phantom's layer is w in (0, 2]; thinning it linearly from 2.0 mm anteriorly to
    # 1.0 mm posteriorly makes a wedge whose mean thickness the area-weighted aMe must find.
    from tests.unit import test_tibia_regions as tr

    flat = tr._labels

    def wedge(local, plates, **kw):
        labels = flat(local, plates, **kw)
        _u, v, w = local.T
        thickness = 1.5 + 0.5 * np.clip(v / 10.0, -1.0, 1.0)
        labels[(labels == TIBIA_CARTILAGE) & (w > thickness)] = 0
        return labels

    monkeypatch.setattr(tr, '_labels', wedge)
    seg = tibia_phantom(plates=tr.CIRCLE_PLATES)
    maps = Tibia(seg, TIBIA_CARTILAGE).calculate_thickness()

    # Against an independent quantity: for a layer whose two surfaces are parallel, the
    # area-weighted mean thickness *is* its volume divided by its base area. (Not against the
    # nominal 1.5 mm: a 0.5 mm voxel grid can only rasterise the wedge in 0.5 mm steps, which
    # floors each column and reads ~0.25 mm low.)
    voxels = int((np.asarray(seg.image.dataobj) == TIBIA_CARTILAGE).sum())
    volume = voxels * float(np.prod(S.voxel_spacing(seg)))
    area = sum(s['tab_area_mm2'] for s in maps.statistics.values())
    assert _bone_mean(maps.statistics, list(maps)) == pytest.approx(volume / area, rel=0.1)
    # Anterior (+v) is thick, posterior (-v) thin, which a mean over the whole plate would hide.
    assert maps.statistics['alt']['aMe'] > maps.statistics['plt']['aMe'] + 0.3
    assert maps.statistics['art']['aMe'] > maps.statistics['prt']['aMe'] + 0.3


def test_curved_shell_reads_its_true_thickness():
    """A 2 mm shell on the cylindrical condyles reads 2 mm in the central weight-bearing third.

    The phantom's sharp artificial features -- the flat faces where the condyle cylinders are
    cut off at |u| = 32 and the vertical walls of the intercondylar fossa -- carry bone surface
    that is *beside* rather than *under* cartilage, where a nearest-distance measurement
    legitimately reads less than the layer thickness. The central thirds are away from both.
    """
    femur = Femur(femur_phantom(), FEMUR_CARTILAGE)
    maps = femur.calculate_thickness()
    for key in ('cclf', 'ccrf'):
        assert maps.statistics[key]['aMe'] == pytest.approx(TRUE_MM, rel=0.02), key
        # The shell is uniform; what is left of aSD is the 0.5 mm voxel grid's staircase on a
        # curved surface.
        assert maps.statistics[key]['aSD'] < 0.35, key


def test_curved_shell_outer_thirds_lose_only_to_the_phantoms_own_sharp_edges():
    """The -10% on the condyles' outer thirds is phantom geometry, not an edge effect to pin.

    They read 1.80 against 2.00, but the deficit is **mirror-symmetric in the phantom's
    medial-lateral axis** rather than in internal/external: at 0.5 and 0.4 mm isotropic the
    patient-left condyle's *internal* third and the patient-right condyle's *external* third
    (both on +ml) agree to a thousandth of a millimetre, and the other two (both on -ml) to a
    hundredth, while internal and external differ by a tenth. That is the signature
    of the two artificial sharp edges the phantom has and a knee does not -- the flat face
    where the condyle cylinders are cut off and the vertical wall of the open intercondylar
    fossa -- being cut by the voxel grid at a different phase on each side. The thirds away
    from both (the central ones) converge on the truth as the grid refines.
    """
    coarse = Femur(femur_phantom(spacing=(0.5,) * 3), FEMUR_CARTILAGE).calculate_thickness()
    fine = Femur(femur_phantom(spacing=(0.4,) * 3), FEMUR_CARTILAGE).calculate_thickness()
    for maps in (coarse, fine):
        stats = maps.statistics
        assert stats['iclf']['aMe'] == pytest.approx(stats['ecrf']['aMe'], abs=0.002)
        assert stats['eclf']['aMe'] == pytest.approx(stats['icrf']['aMe'], abs=0.01)
        assert stats['iclf']['aMe'] < stats['eclf']['aMe'] - 0.05
    # Refining the grid moves the central thirds towards the truth, the outer ones hardly.
    assert fine.statistics['cclf']['aMe'] > coarse.statistics['cclf']['aMe']
    assert abs(fine.statistics['cclf']['aMe'] - TRUE_MM) < 0.02


# ----------------------------------------------------------------------------- denuded bone

def test_denuded_bone_dilutes_the_tab_mean_by_its_area():
    """tAB.aMe = cAB.aMe x (cAB / tAB): denuded triangles count as 0.0 mm."""
    # A 5 mm denuded disc inside the patient-right weight-bearing condyle (u = +20, v = -8).
    femur = Femur(femur_phantom(denuded=(20.0, -8.0, 5.0)), FEMUR_CARTILAGE)
    maps = femur.calculate_thickness()
    affected = [k for k, s in maps.statistics.items() if s['denuded_fraction'] > 0.02]
    assert affected, 'the denuded disc should leave a measurable denuded fraction somewhere'
    for key, stats in maps.statistics.items():
        covered = 1.0 - stats['denuded_fraction']
        assert stats['tab_aMe'] == pytest.approx(stats['aMe'] * covered, rel=1e-6), key
        assert stats['cab_area_mm2'] == pytest.approx(stats['tab_area_mm2'] * covered,
                                                      rel=1e-6), key
    # Where bone is denuded the *tAB* minimum is 0 mm by definition, while the cAB minimum
    # still describes the thinnest cartilage that is actually there.
    for key in affected:
        assert maps.statistics[key]['tab_aMin'] == pytest.approx(0.0)
        assert maps.statistics[key]['tab_aMiv'] == pytest.approx(0.0, abs=0.05)
        assert maps.statistics[key]['aMin'] > 0.0


# -------------------------------------------------------------------------- area weighting

def _plane_pair(refine: bool):
    """
    Two meshes of the same 2 x 1 plane, one uniformly triangulated and one refined on its
    left half, both carrying a thickness of 1 mm on the left and 3 mm on the right.

    :param refine: Subdivide the left half, so it holds many more (and much smaller) triangles.
    :return: ``(values, weights)`` ready for :func:`weighted_statistics`.
    """
    left = pv.Plane(center=(-0.5, 0, 0), direction=(0, 0, 1), i_size=1, j_size=1,
                    i_resolution=1, j_resolution=1).triangulate()
    if refine:
        left = left.subdivide(3)
    right = pv.Plane(center=(0.5, 0, 0), direction=(0, 0, 1), i_size=1, j_size=1,
                     i_resolution=1, j_resolution=1).triangulate()
    values = np.concatenate([np.full(left.n_cells, 1.0), np.full(right.n_cells, 3.0)])
    weights = np.concatenate([S.triangle_areas(left), S.triangle_areas(right)])
    return values, weights


def test_area_weighting_is_insensitive_to_the_triangulation():
    """Refining one half of a mesh must not move the mean; a per-triangle average would."""
    coarse = weighted_statistics(*_plane_pair(refine=False))
    fine = weighted_statistics(*_plane_pair(refine=True))
    assert coarse['aMe'] == pytest.approx(2.0)
    assert fine['aMe'] == pytest.approx(coarse['aMe'], abs=1e-9)
    assert fine['aSD'] == pytest.approx(coarse['aSD'], abs=1e-9)
    # The unweighted mean of the refined mesh is nowhere near 2 mm, which is what makes the
    # test meaningful.
    values, _ = _plane_pair(refine=True)
    assert abs(values.mean() - 2.0) > 0.5


# ------------------------------------------------------------- the joint-facing sheet

def test_joint_facing_sheet_keeps_one_side_of_a_fold():
    """Two sheets in one cell: only the one nearest the joint survives."""
    from morphometry.cartilage.thickness import _joint_facing_sheet

    cells = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0]])
    along = np.array([0.0, 0.4, 30.0, 10.0, 11.0])      # cell (0,0) folds, cell (1,0) does not
    low = _joint_facing_sheet(cells, along, facing_sign=-1, sheet_gap_mm=5.0)
    assert low.tolist() == [True, True, False, True, True]
    high = _joint_facing_sheet(cells, along, facing_sign=+1, sheet_gap_mm=5.0)
    assert high.tolist() == [False, False, True, True, True]
    # facing_sign 0 keeps everything, which is what a caller with no fold wants.
    assert _joint_facing_sheet(cells, along, facing_sign=0, sheet_gap_mm=5.0).all()


def test_femoral_maps_keep_only_the_joint_facing_sheet():
    """The femoral surface folds onto itself in the shared frame; the maps must not.

    Projected along the tibial plate normal, the condyle's articular surface wraps proximally
    past its own most distal point, so one in-plane cell collects triangles from both limbs of
    the wrap. Without the sheet selection the cell averages two surfaces far apart along the
    projection axis, and its recorded articular position lands between them -- inside the bone
    -- which is where the thinnest-point marker and the femoral arc coordinate read it from.
    """
    from scipy.spatial import cKDTree

    # The default phantom's articular arc barely wraps past vertical, so it folds by only a
    # couple of millimetres; widening the arc gives it a fold of the order a real femoral
    # horseshoe has (tens of millimetres).
    seg = femur_phantom(arc=(-150.0, 140.0))
    femur = Femur(seg, FEMUR_CARTILAGE)
    femur.calculate_landmarks()
    # A tibial-style frame: project along the phantom's proximal axis, which is what a real
    # tibial plateau normal does to a femur at full extension.
    frame = KneeFrame(origin=np.zeros(3), axes=np.eye(3), grid_mm=DEFAULT_GRID_MM,
                      source='tibia')
    kwargs = dict(bone_label=1, cartilage_label=FEMUR_CARTILAGE, frame=frame, bone='femur')
    folded = measure_thickness(seg, femur.subdivision, FEMUR_REGION_KEYS_,
                               facing_sign=0, **kwargs)
    sheeted = measure_thickness(seg, femur.subdivision, FEMUR_REGION_KEYS_,
                                facing_sign=-1, **kwargs)

    # Every recorded position should lie on the surface it claims to describe.
    surface = cKDTree(np.asarray(femur.subdivision.mesh.points, dtype=float))
    off_folded = surface.query(np.array(list(folded.positions.values())))[0]
    off_sheeted = surface.query(np.array(list(sheeted.positions.values())))[0]
    assert off_folded.max() > 8.0, 'the phantom should fold onto itself in this frame'
    # Not zero even without a fold: where the surface is near vertical it crosses several
    # millimetres inside one 0.5 mm cell, so the cell's area-weighted mean sits off it.
    assert off_sheeted.max() < 3.0
    assert off_sheeted.max() < 0.4 * off_folded.max()
    # The kept sheet is the one nearest the joint, i.e. the lowest along the projection axis.
    shared = set(folded.positions) & set(sheeted.positions)
    lower = [sheeted.positions[c][2] <= folded.positions[c][2] + 1e-9 for c in shared]
    assert all(lower)
    # The regional statistics are per triangle and must not move at all.
    for key in folded.statistics:
        assert sheeted.statistics[key]['aMe'] == pytest.approx(
            folded.statistics[key]['aMe'], abs=1e-12), key


# ------------------------------------------------------------------------------ invariance

@pytest.mark.parametrize('rotation', [
    Rotation.from_euler('z', 20, degrees=True).as_matrix(),      # leg turned in the scanner
    Rotation.from_euler('y', 10, degrees=True).as_matrix(),      # varus / valgus tilt
    Rotation.from_euler('zx', [-15, 12], degrees=True).as_matrix(),
])
def test_thickness_is_invariant_to_the_knees_pose(rotation):
    """Turning the whole knee in the scanner must not change the measured thickness."""
    base, _ = _tibial_thickness()
    other, _ = _tibial_thickness(rotation=rotation)
    # Per subregion, to 0.15 mm: the layer itself is reproduced exactly, but how the voxel grid
    # cuts the plate's rim -- where a nearest-distance thickness falls towards zero -- turns
    # with the anatomy, which moves the peripheral subregions by a tenth of a millimetre.
    for key in base.statistics:
        assert other.statistics[key]['aMe'] == pytest.approx(base.statistics[key]['aMe'],
                                                             abs=0.15), key
    assert _bone_mean(other.statistics, list(other)) == pytest.approx(
        _bone_mean(base.statistics, list(base)), abs=0.06)


def test_thickness_is_invariant_to_voxel_size_and_oblique_sampling():
    """Sampling the same knee differently gives the same thickness, to discretisation error."""
    base, _ = _tibial_thickness()
    # Isotropic but finer, and an oblique, reflecting direction matrix.
    finer, _ = _tibial_thickness(spacing=(0.4, 0.4, 0.4))
    oblique_direction = (Rotation.from_euler('xyz', [10, -15, 30], degrees=True).as_matrix()
                         @ np.diag([1.0, -1.0, 1.0]))
    oblique, _ = _tibial_thickness(spacing=(0.45, 0.5, 0.55), direction=oblique_direction)
    for maps in (finer, oblique):
        assert _bone_mean(maps.statistics, list(maps)) == pytest.approx(
            _bone_mean(base.statistics, list(base)), abs=0.1)
        for key in base.statistics:
            assert maps.statistics[key]['aMe'] == pytest.approx(
                base.statistics[key]['aMe'], abs=0.15), key


def test_anisotropic_sampling_reads_the_rasterised_layer():
    """At 0.8 mm through-plane a 2.0 mm layer rasterises to two voxels, i.e. 1.6 mm."""
    maps, _ = _tibial_thickness(spacing=(0.3, 0.3, 0.8))
    assert _bone_mean(maps.statistics, list(maps)) == pytest.approx(1.6, abs=0.1)


def test_simpleitk_backend_agrees_with_nibabel():
    """The SimpleITK backend measures the same knee identically (its LPS is converted to RAS)."""
    seg = tibia_phantom()
    base = Tibia(seg, TIBIA_CARTILAGE).calculate_thickness()
    itk = sitk.GetImageFromArray(np.asarray(seg.image.dataobj).transpose(2, 1, 0))
    affine = seg.image.affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    itk.SetSpacing([float(s) for s in spacing])
    itk.SetDirection([float(c) for c in
                      (np.diag([-1.0, -1.0, 1.0]) @ affine[:3, :3] / spacing).ravel()])
    itk.SetOrigin([float(c) for c in np.diag([-1.0, -1.0, 1.0]) @ affine[:3, 3]])
    other = Tibia(Segmentation.from_sitk(itk), TIBIA_CARTILAGE).calculate_thickness()
    for key in base.statistics:
        assert other.statistics[key]['aMe'] == pytest.approx(base.statistics[key]['aMe'],
                                                             abs=1e-6), key


def test_mirroring_swaps_the_two_plates_thicknesses():
    """A left-right mirrored knee returns the same numbers with the plates exchanged."""
    base, _ = _tibial_thickness()
    mirrored, _ = _tibial_thickness(reflect_array=True)
    for letter in ('c', 'i', 'e', 'a', 'p'):
        assert mirrored.statistics[f'{letter}rt']['aMe'] == pytest.approx(
            base.statistics[f'{letter}lt']['aMe'], abs=0.05), letter
        assert mirrored.statistics[f'{letter}lt']['aMe'] == pytest.approx(
            base.statistics[f'{letter}rt']['aMe'], abs=0.05), letter


# ------------------------------------------------------- the shared frame keeps the overlap

def _condyle_arc_points(flexion_deg: float, step_mm: float = 0.2) -> np.ndarray:
    """
    A cylindrical femoral condyle's articular surface, rotated by a flexion angle.

    The surface is an arc of a cylinder of radius 22 mm about the medial-lateral axis
    (world +x), spanning 100 degrees anterior to 120 degrees posterior of distal, as in the
    femoral phantom. Flexion rotates it about that same axis, which is what the knee does.

    :param flexion_deg: Rotation about the medial-lateral axis in degrees.
    :param step_mm: Sampling step along both the axis and the arc, fine enough that the
        sampling never limits which grid cells the surface covers.
    :return: ``(N, 3)`` world mm.
    """
    radius = 22.0
    theta = np.radians(np.arange(-120.0, 100.0, np.degrees(step_mm / radius)) + flexion_deg)
    u = np.arange(-30.0, 30.0, step_mm)
    uu, tt = np.meshgrid(u, theta, indexing='ij')
    return np.stack([uu.ravel(), radius * np.sin(tt).ravel(),
                     -radius * np.cos(tt).ravel()], axis=1)


def _plateau_points(step_mm: float = 0.2) -> np.ndarray:
    """A flat 60 x 44 mm tibial plateau in the world z = 0 plane, centred on the origin."""
    u, v = np.arange(-30.0, 30.0, step_mm), np.arange(-22.0, 22.0, step_mm)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    return np.stack([uu.ravel(), vv.ravel(), np.zeros(uu.size)], axis=1)


def _overlap_mm2(frame: KneeFrame, a: np.ndarray, b: np.ndarray) -> float:
    """Area (mm^2) of the frame cells that two point sets have in common."""
    cells_a = set(map(tuple, frame.cells(a).tolist()))
    cells_b = set(map(tuple, frame.cells(b).tolist()))
    return len(cells_a & cells_b) * frame.grid_mm ** 2


@pytest.mark.parametrize('flexion', [0.0, 30.0, 60.0, 90.0])
def test_the_tibial_frame_keeps_the_tibiofemoral_overlap_under_flexion(flexion):
    """
    The shared frame must be the tibial plateau plane, not a femoral one.

    The tibia stays put (its plateau is the world z = 0 plane) and the femur flexes: its
    articular surface is an arc of a cylinder about the medial-lateral axis, rotated by the
    flexion angle, so the part facing the tibia migrates from the distal condyle to the
    posterior one. Projected along the **tibial plateau normal** both bones keep a large common
    footprint at every angle, which is the overlap :func:`find_thinnest_area` searches.
    Projected along the **femoral shaft axis**, which turns with the femur, the tibial plateau
    is seen more and more edge-on and the common footprint collapses -- a femoral frame would
    lose the overlap at exactly the flexion angles the thinnest-point analysis is about.
    """
    rotation = Rotation.from_euler('x', flexion, degrees=True).as_matrix()
    tibial = KneeFrame(origin=np.zeros(3), axes=np.eye(3), grid_mm=DEFAULT_GRID_MM,
                       source='tibia')
    # The femoral frame's axes rotate with the femur: its rows are R @ e1, R @ e2, R @ e3.
    femoral = KneeFrame(origin=np.zeros(3), axes=rotation.T, grid_mm=DEFAULT_GRID_MM,
                        source='femur')
    condyle, plateau = _condyle_arc_points(flexion), _plateau_points()

    assert _overlap_mm2(tibial, condyle, plateau) > 1500.0, 'lost the overlap in the tibial frame'
    femoral_overlap = _overlap_mm2(femoral, condyle, plateau)
    if flexion == 0.0:
        assert femoral_overlap > 1500.0        # at full extension the two frames coincide
    if flexion == 90.0:
        assert femoral_overlap < 300.0         # the plateau is edge-on: the overlap is a sliver


def test_both_bones_overlap_in_the_shared_frame_on_the_phantoms():
    """A tibial and a femoral phantom measured in the tibial frame overlap and can be searched."""
    seg = tibia_phantom()
    tibia = Tibia(seg, TIBIA_CARTILAGE)
    tibial = tibia.calculate_thickness()
    # Reuse the tibial plates as a stand-in "femur" by measuring the same bone in the same
    # frame: the point is the frame plumbing, not the anatomy.
    other = measure_thickness(seg, tibia.subdivision, tuple(tibial),
                              bone_label=TIBIA_BONE, cartilage_label=TIBIA_CARTILAGE,
                              frame=tibia.knee_frame(), bone='femur')
    result = find_thinnest_area(tibial, other, seg)
    assert result['kernel_size'] % 2 == 1
    assert np.isfinite(result['combined_thickness'])


def test_maps_in_different_frames_are_refused():
    """Overlapping a tibial-frame map with a femoral-frame one raises instead of pairing up
    unrelated positions."""
    seg = femur_phantom()
    femur = Femur(seg, FEMUR_CARTILAGE)
    femoral = femur.calculate_thickness()            # no tibia -> a femur-only frame
    assert femoral.frame.source == 'femur'
    tibia = Tibia(tibia_phantom(), TIBIA_CARTILAGE)
    tibial = tibia.calculate_thickness()
    with pytest.raises(ValueError, match='different in-plane frames'):
        find_thinnest_area(tibial, femoral, seg)
    # A plain dict mixed with a framed one is refused too.
    with pytest.raises(ValueError, match='is not'):
        find_thinnest_area({'clt': {(0, 0): 1.0}}, femoral, seg)


# ------------------------------------------------------------------ the removed 'method' arg

def test_the_old_method_argument_raises():
    """``method='knn'`` / ``'mesh'`` must fail loudly, naming the replacement."""
    tibia = Tibia(tibia_phantom(), TIBIA_CARTILAGE)
    for value in ('knn', 'mesh'):
        with pytest.raises(ValueError, match="'method' argument is gone"):
            tibia.calculate_thickness(value)
    femur = Femur(femur_phantom(), FEMUR_CARTILAGE)
    with pytest.raises(ValueError, match="'method' argument is gone"):
        femur.calculate_thickness(method='knn')
    # The tibia is still the first positional parameter of the femoral call.
    with pytest.raises(TypeError, match='as the tibia'):
        femur.calculate_thickness('knn')


# --------------------------------------------------------------------------- map structure

def test_maps_cover_the_subchondral_area_and_carry_the_frame():
    """Every subregion's map holds about one cell per grid square of its tAB."""
    maps, _ = _tibial_thickness()
    assert maps.bone == 'tibia' and maps.frame.source == 'tibia'
    assert set(maps) == set(maps.statistics)
    for key, stats in maps.statistics.items():
        expected = stats['tab_area_mm2'] / maps.frame.grid_mm ** 2
        assert 0.5 * expected < len(maps[key]) < 1.5 * expected, key
        assert all(np.isfinite(v) for v in maps[key].values()), key
    # The per-triangle thicknesses are aligned with the subdivision mesh, for the heatmap, and
    # every map cell has an articular-surface position recorded for the 3D markers.
    tibia = Tibia(tibia_phantom(), TIBIA_CARTILAGE)
    tibia.calculate_landmarks()
    assert len(maps.triangle_thickness) == tibia.subdivision.mesh.n_cells
    assert set().union(*[set(m) for m in maps.values()]) <= set(maps.positions)

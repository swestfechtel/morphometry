"""Unit tests for the reference-anchored thickness maps and their per-knee deltas.

:mod:`morphometry.cartilage.deformation` carries each configuration's per-triangle cartilage
thickness into the reference scan's world frame (rigid bone registration) and bins it into
the reference's map coordinates; the delta map is configuration minus reference per cell.

Most tests build :class:`BoneMeasurement` objects directly from synthetic meshes -- a curved
plate standing in for the tibial tAB (with ``region`` / ``covered`` cell data and a smooth
thickness field) and a separate, asymmetric bone surface for the registration -- so they need
no image and run in well under a second each. One test measures the voxelised femoral
phantom of :mod:`tests.unit.test_femur_regions` and compares it with itself.
"""
import pickle

import numpy as np
import pytest
import pyvista as pv
from scipy.spatial.transform import Rotation

from morphometry.cartilage import surface as S
from morphometry.cartilage.deformation import (
    COMPARISONS, REFERENCE, BoneDeltaMaps, BoneMeasurement, MapGrid, bone_measurement,
    knee_delta_maps, rasterise, rasterise_labels,
)
from morphometry.cartilage.knee import Femur
from morphometry.cartilage.thickness import KneeFrame
from tests.unit.test_femur_regions import phantom as femur_phantom

KEY = (0, 'belastet')


# ----------------------------------------------------------------------------- synthetic data

def _rot(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """A rotation matrix from extrinsic x-y-z Euler angles in degrees."""
    return Rotation.from_euler('xyz', [x_deg, y_deg, z_deg], degrees=True).as_matrix()


def _bone_mesh() -> pv.PolyData:
    """An asymmetric closed bone surface below the plate: an ellipsoid with an off-centre bump."""
    mesh = pv.ParametricEllipsoid(25.0, 18.0, 12.0, u_res=90, v_res=60, w_res=10).triangulate()
    mesh = mesh.clean()
    p = np.asarray(mesh.points, dtype=float).copy()
    top = p[:, 2] > 0
    p[top, 2] += 5.0 * np.exp(-((p[top, 0] - 9.0) ** 2 + (p[top, 1] - 6.0) ** 2) / 40.0)
    p[:, 2] -= 14.0                          # the plate sits on top of the bone
    mesh.points = p
    return mesh


def _plate_thickness(xy: np.ndarray) -> np.ndarray:
    """The smooth synthetic thickness field (mm) at in-plane plate coordinates."""
    return 2.0 + 0.02 * xy[:, 0] + 0.01 * xy[:, 1]


def _synthetic_tibia_measurement(denuded=None, thinned=None) -> BoneMeasurement:
    """
    A synthetic tibial measurement in its own (identity) knee frame.

    :param denuded: Optional ``(cx, cy, r)``: triangles with centroid inside this disc are
        uncovered (denuded) and carry thickness 0, as ``ThicknessMaps.triangle_thickness`` does.
    :param thinned: Optional ``(cx, cy, r, by)``: thickness reduced by ``by`` mm inside the disc.
    :return: A :class:`BoneMeasurement`.
    """
    plate = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=40, j_size=30,
                     i_resolution=80, j_resolution=60).triangulate()
    p = np.asarray(plate.points, dtype=float).copy()
    p[:, 2] = 0.01 * p[:, 0] ** 2 - 0.005 * p[:, 1] ** 2          # gently curved
    plate.points = p
    centroids = S.triangle_centroids(plate)
    xy = centroids[:, :2]
    plate.cell_data['region'] = (xy[:, 0] > 0).astype(np.int64)
    covered = np.ones(plate.n_cells, dtype=bool)
    thickness = _plate_thickness(xy)
    if thinned is not None:
        cx, cy, r, by = thinned
        thickness[np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < r] -= by
    if denuded is not None:
        cx, cy, r = denuded
        covered[np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < r] = False
    thickness[~covered] = 0.0
    plate.cell_data['covered'] = covered
    frame = KneeFrame(origin=np.zeros(3), axes=np.eye(3), grid_mm=0.5, source='tibia')
    return BoneMeasurement(bone='tibia', tab_mesh=plate, bone_mesh=_bone_mesh(),
                           triangle_thickness=thickness, statistics={},
                           frame_origin=np.zeros(3), frame_axes=np.eye(3), knee_frame=frame)


def _moved(m: BoneMeasurement, rotation: np.ndarray, translation,
           frame_error: np.ndarray = None) -> BoneMeasurement:
    """
    The same measurement rigidly moved in world space (a repositioned knee).

    :param m: The measurement to move.
    :param rotation: ``(3, 3)`` rotation.
    :param translation: ``(3,)`` translation (mm).
    :param frame_error: Optional extra rotation applied to the *frame* only (an imperfect
        initial frame the ICP has to correct).
    :return: A new :class:`BoneMeasurement`.
    """
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)

    def move(mesh):
        out = mesh.copy()
        out.points = np.asarray(mesh.points, dtype=float) @ rotation.T + translation
        return out

    axes = m.frame_axes @ rotation.T
    if frame_error is not None:
        axes = axes @ np.asarray(frame_error).T
    origin = rotation @ m.frame_origin + translation
    knee_frame = KneeFrame(origin=origin, axes=axes, grid_mm=m.knee_frame.grid_mm,
                           source='tibia')
    return BoneMeasurement(bone=m.bone, tab_mesh=move(m.tab_mesh), bone_mesh=move(m.bone_mesh),
                           triangle_thickness=m.triangle_thickness.copy(),
                           statistics=dict(m.statistics), frame_origin=origin, frame_axes=axes,
                           knee_frame=knee_frame)


@pytest.fixture(scope='module')
def ref():
    """The synthetic reference tibia."""
    return _synthetic_tibia_measurement()


# ------------------------------------------------------------------------------------ constants

def test_reference_and_comparison_keys():
    """The reference and the seven comparisons, in the global order."""
    assert REFERENCE == (30, 'unbelastet')
    assert COMPARISONS == ((0, 'unbelastet'), (60, 'unbelastet'), (90, 'unbelastet'),
                           (0, 'belastet'), (30, 'belastet'), (60, 'belastet'),
                           (90, 'belastet'))


# ------------------------------------------------------------------------------------ map grid

def test_rasterise_area_weighted_mean():
    """Two points in one cell average by weight; other cells stay NaN."""
    g = MapGrid(np.zeros(2), 1.0, (3, 3))
    out = rasterise(g, np.array([[0.1, 0.0], [0.2, 0.1]]), np.array([1.0, 3.0]),
                    np.array([1.0, 3.0]))
    assert out.shape == (3, 3)
    assert out[0, 0] == pytest.approx(2.5) and np.isnan(out[1, 1])
    assert np.isfinite(out).sum() == 1


def test_rasterise_drops_nan_coordinates_and_values():
    """A NaN coordinate or value does not reach any cell."""
    g = MapGrid(np.zeros(2), 1.0, (3, 3))
    coords = np.array([[0.0, 0.0], [np.nan, 0.0], [1.0, 1.0], [2.0, 2.0]])
    out = rasterise(g, coords, np.array([1.0, 100.0, np.nan, 4.0]), np.ones(4))
    assert out[0, 0] == 1.0 and np.isnan(out[1, 1]) and out[2, 2] == 4.0


def test_map_grid_cells_round_half_up_and_index_columns():
    """Cell ``(i, j)``: ``i`` from coordinate column 0, ``j`` from column 1, half-up rounding."""
    g = MapGrid(np.array([-2.0, 10.0]), 0.5, (20, 20))
    cells = g.cells(np.array([[-2.0, 10.0], [-1.75, 10.0], [-2.0, 11.0], [-1.76, 10.24]]))
    assert cells.tolist() == [[0, 0], [1, 0], [0, 2], [0, 0]]
    assert g.centre_mm(1, 2) == pytest.approx([-1.5, 11.0])


def test_map_grid_covering_contains_every_point_with_padding():
    """``covering`` spans all finite points of all sets plus ``pad_cells`` on every side."""
    rng = np.random.default_rng(1)
    a = rng.uniform(-7.3, 4.1, size=(200, 2))
    b = np.vstack([rng.uniform(2.0, 9.9, size=(100, 2)), [[np.nan, np.nan]]])
    g = MapGrid.covering([a, b], grid_mm=1.0, pad_cells=2)
    cells = g.cells(np.vstack([a, b[:-1]]))
    assert cells.min() == 2
    assert np.all(cells.max(axis=0) == np.array(g.shape) - 3)
    centres = np.array([g.centre_mm(i, j) for i, j in cells])
    assert np.all(np.abs(centres - np.vstack([a, b[:-1]])) <= 0.5 + 1e-12)
    with pytest.raises(ValueError):
        MapGrid.covering([np.full((3, 2), np.nan)], grid_mm=1.0)


def test_rasterise_labels_majority_by_weight():
    """The label carrying the larger total weight wins its cell; empty cells are -1."""
    g = MapGrid(np.zeros(2), 1.0, (2, 2))
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [1.0, 1.0], [np.nan, 0.0]])
    labels = np.array([3, 5, 5, 7, 9])
    out = rasterise_labels(g, coords, labels, np.array([5.0, 2.0, 2.0, 1.0, 50.0]))
    assert out.dtype.kind == 'i'
    assert out.tolist() == [[3, -1], [-1, 7]]


# ------------------------------------------------------------------------------ knee delta maps

def test_rigidly_moved_identical_scan_gives_zero_delta(ref):
    """The same plate rigidly moved reads Δ ~ 0 everywhere, and ICP barely turns the start."""
    cfg = _moved(ref, _rot(4, -3, 7), [5.0, 2.0, -3.0])
    d = knee_delta_maps('tibia', ref, {KEY: cfg})
    assert isinstance(d, BoneDeltaMaps) and d.bone == 'tibia'
    delta = d.deltas[KEY]
    assert np.isfinite(delta).sum() > 500
    assert np.nanmax(np.abs(delta)) < 1e-6
    assert d.registrations[KEY].rms_mm < 0.05
    assert d.rotation_change_deg[KEY] < 0.1          # exact frames: ICP keeps its start
    assert d.dropped_fraction[KEY] == 0.0
    assert d.errors == {}
    # Footprints coincide: Δ defined exactly where both maps are.
    both = np.isfinite(d.reference) & np.isfinite(d.configurations[KEY])
    assert np.array_equal(np.isfinite(delta), both)


def test_icp_corrects_an_imperfect_initial_frame(ref):
    """A frame rotated 3° off the bone: ICP recovers the pose and reports a ~3° change."""
    cfg = _moved(ref, _rot(4, -3, 7), [5.0, 2.0, -3.0], frame_error=_rot(0, 0, 3))
    d = knee_delta_maps('tibia', ref, {KEY: cfg})
    transform = d.registrations[KEY].transform
    rotation, translation = _rot(4, -3, 7), np.array([5.0, 2.0, -3.0])
    # The fitted transform is the exact inverse of the motion, not the 3° off frame start.
    assert np.allclose(transform.rotation, rotation.T, atol=1e-6)
    assert np.allclose(transform.translation, -rotation.T @ translation, atol=1e-5)
    assert d.registrations[KEY].rms_mm < 0.05
    # The thickness field varies by 0.02 mm/mm, so mapping through the 3° off start would read
    # up to ~0.02 mm; the exact pose reads 0 to rounding.
    assert np.nanmax(np.abs(d.deltas[KEY])) < 1e-6
    assert d.rotation_change_deg[KEY] == pytest.approx(3.0, abs=0.2)


def test_thinned_patch_appears_where_it_was_made(ref):
    """A 0.5 mm thinning inside a 4 mm disc shows as -0.5 there and ~0 far from it."""
    cfg = _moved(_synthetic_tibia_measurement(thinned=(5.0, 3.0, 4.0, 0.5)),
                 _rot(-5, 2, 10), [-4.0, 6.0, 1.0])
    d = knee_delta_maps('tibia', ref, {KEY: cfg})
    delta = d.deltas[KEY]
    inside, outside = [], []
    for i, j in zip(*np.nonzero(np.isfinite(delta))):
        r = np.hypot(*(d.grid.centre_mm(i, j) - np.array([5.0, 3.0])))
        if r < 3.0:
            inside.append(delta[i, j])
        elif r > 5.5:
            outside.append(delta[i, j])
    assert len(inside) > 15 and len(outside) > 500
    assert np.allclose(inside, -0.5, atol=0.05)
    assert np.max(np.abs(outside)) < 0.05


def test_denuded_triangles_are_excluded_not_zero(ref):
    """Denuded triangles leave their cells empty in the configuration map, never 0."""
    cfg = _moved(_synthetic_tibia_measurement(denuded=(-6.0, -4.0, 4.0)),
                 _rot(2, 6, -4), [1.0, -2.0, 3.0])
    d = knee_delta_maps('tibia', ref, {KEY: cfg})
    config = d.configurations[KEY]
    centres = np.array([[d.grid.centre_mm(i, j) for j in range(d.grid.shape[1])]
                        for i in range(d.grid.shape[0])])
    mask_denuded = np.hypot(centres[..., 0] + 6.0, centres[..., 1] + 4.0) < 3.0
    assert mask_denuded.sum() > 15
    assert np.all(np.isnan(config[mask_denuded]))
    assert np.all(np.isnan(d.deltas[KEY][mask_denuded]))
    assert np.all(np.isfinite(d.reference[mask_denuded]))     # the reference is covered there
    assert not np.any(np.isclose(config[np.isfinite(config)], 0.0))


def test_reference_positions_and_region_codes(ref):
    """Per cell: the mean reference tAB position lies in that cell; region codes follow x."""
    cfg = _moved(ref, _rot(4, -3, 7), [5.0, 2.0, -3.0])
    d = knee_delta_maps('tibia', ref, {KEY: cfg})
    assert d.reference_positions.shape == d.grid.shape + (3,)
    finite = np.isfinite(d.reference)
    assert np.array_equal(np.all(np.isfinite(d.reference_positions), axis=2), finite)
    i, j = np.nonzero(finite)
    in_plane = ref.knee_frame.in_plane(d.reference_positions[i, j])
    centres = np.array([d.grid.centre_mm(a, b) for a, b in zip(i, j)])
    assert np.all(np.abs(in_plane - centres) <= 0.5 * d.grid.grid_mm + 1e-9)
    # The positions lie on the reference plate (z = 0.01 x^2 - 0.005 y^2), up to the bin's
    # curvature.
    p = d.reference_positions[i, j]
    assert np.all(np.abs(p[:, 2] - (0.01 * p[:, 0] ** 2 - 0.005 * p[:, 1] ** 2)) < 0.02)
    codes = d.region_codes[i, j]
    assert set(np.unique(codes)) == {0, 1}
    assert np.all(codes[centres[:, 0] < -0.6] == 0) and np.all(codes[centres[:, 0] > 0.6] == 1)
    assert np.all(d.region_codes[~finite] == -1)


def test_failed_registration_is_recorded_not_raised(ref):
    """A configuration whose registration raises is omitted and its error recorded."""
    good = _moved(ref, _rot(4, -3, 7), [5.0, 2.0, -3.0])
    bad = _moved(ref, np.eye(3), np.zeros(3))
    bad.bone_mesh = pv.PolyData()
    d = knee_delta_maps('tibia', ref, {KEY: good, (90, 'belastet'): bad})
    assert set(d.configurations) == set(d.deltas) == set(d.registrations) == {KEY}
    assert set(d.rotation_change_deg) == set(d.dropped_fraction) == {KEY}
    assert list(d.errors) == [(90, 'belastet')]
    assert 'ValueError' in d.errors[(90, 'belastet')]


def test_invalid_input_raises(ref):
    """Wrong bone names and missing frames are input errors, raised immediately."""
    with pytest.raises(ValueError):
        knee_delta_maps('patella', ref, {})
    with pytest.raises(ValueError):
        knee_delta_maps('femur', ref, {})                     # tibia passed as a femur
    no_frame = _moved(ref, np.eye(3), np.zeros(3))
    no_frame.knee_frame = None
    with pytest.raises(ValueError):
        knee_delta_maps('tibia', no_frame, {})
    other = _moved(ref, np.eye(3), np.zeros(3))
    other.bone = 'femur'
    with pytest.raises(ValueError):
        knee_delta_maps('tibia', ref, {KEY: other})


def test_results_pickle(ref):
    """Measurements and delta maps round-trip through pickle (results are cached to disk)."""
    cfg = _moved(ref, _rot(4, -3, 7), [5.0, 2.0, -3.0])
    d = knee_delta_maps('tibia', ref, {KEY: cfg})
    m2 = pickle.loads(pickle.dumps(cfg))
    assert m2.tab_mesh.n_cells == cfg.tab_mesh.n_cells
    assert np.array_equal(m2.tab_mesh.cell_data['covered'], cfg.tab_mesh.cell_data['covered'])
    d2 = pickle.loads(pickle.dumps(d))
    assert np.array_equal(d2.deltas[KEY], d.deltas[KEY], equal_nan=True)
    assert d2.grid == d.grid


# ------------------------------------------------------------------------ real subdivision

def test_femur_phantom_against_itself_is_exactly_zero():
    """A femur measured once and compared with itself: identity registration, Δ == 0."""
    femur = Femur(femur_phantom(), 3)
    maps = femur.calculate_thickness()
    m = bone_measurement('femur', femur.subdivision, maps)
    frame = femur.subdivision.frame
    assert np.allclose(m.frame_axes, np.stack([frame.ml_axis, frame.ap_axis,
                                               frame.proximal_axis]))
    assert np.allclose(m.frame_origin, frame.notch_point)
    assert m.femoral_frame is frame and m.tab_mesh is femur.subdivision.mesh
    assert np.array_equal(m.triangle_thickness, maps.triangle_thickness)

    d = knee_delta_maps('femur', m, {KEY: m})
    transform = d.registrations[KEY].transform
    assert np.allclose(transform.rotation, np.eye(3), atol=1e-9)
    assert np.allclose(transform.translation, 0.0, atol=1e-6)
    assert d.rotation_change_deg[KEY] < 1e-6
    delta = d.deltas[KEY]
    assert np.isfinite(delta).sum() > 1000
    assert np.nanmax(np.abs(delta)) < 1e-9
    assert np.array_equal(np.isfinite(delta), np.isfinite(d.reference))
    # Some regions of the subdivision show up in the map.
    assert len(set(np.unique(d.region_codes)) - {-1}) >= 6

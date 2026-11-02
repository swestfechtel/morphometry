# Knee Cartilage Delta Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per knee, compare the cartilage thickness of every configuration against unloaded 30° — per Wirth–Eckstein subregion and as bone-anchored per-position Δ maps with a largest-thinning hotspot — and summarise it over the cohort.

**Architecture:** Library code in `morphometry/cartilage/` (a shared robust-argmin helper extracted from `find_thinnest_area`; rigid bone registration; the unrolled femoral coordinate, Δ grids and hotspot search; matplotlib figures) and one batch script `scripts/analyse_cartilage_deformation.py` that measures every scan once (cached), registers each configuration's bones onto the reference's, builds the per-knee figures/CSVs, then the cohort tables and report.

**Tech Stack:** Python 3.10 venv (`numpy==1.26.4`, scipy, pyvista 0.44.1, matplotlib, pytest); no new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-02-knee-cartilage-delta-maps-design.md`

All paths below are relative to the worktree `/home/simon/Work/morphometry-delta-maps` (branch `knee-cartilage-delta-maps`). Run tests from there with the main checkout's venv: `/home/simon/Work/morphometry/.venv/bin/python -m pytest ...` (abbreviated `pytest` below).

## Global Constraints

- Input label set: `/home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations` (113 scans, 15 knees). Never mix in `/home/simon/Data/dfg_knees/monai_segmentations/labels/{original,final}`; all eight configurations of a knee come from one directory.
- Labels: femur bone 1, tibia bone 2, femoral cartilage 3, tibial cartilage 4. Load like the other knee scripts: `Segmentation('nibabel').read_image(p)` + `transform_coordinate_system()`, **no** `remove_outliers()`.
- Reference = `(30, 'unbelastet')`. Comparisons, in this order: `(0,'unbelastet'), (60,'unbelastet'), (90,'unbelastet'), (0,'belastet'), (30,'belastet'), (60,'belastet'), (90,'belastet')`.
- Δ = configuration − reference, mm; negative = thinner. "Largest deformation" = most negative smoothed Δ (largest thinning).
- Per-subregion Δ uses each scan's **own** subdivision: `aMe(config, key) − aMe(ref, key)`; % = `100 * Δ / aMe(ref, key)`.
- Δ maps: denuded triangles are **excluded**, not 0; a cell is defined only where both scans have a measured cell.
- Alignment comes from **bone** (rigid ICP on bone surfaces); initial frames may use cartilage, the refinement must not.
- Subregions/compartments stay patient-side (`clt`…`prt`, `iclf`…`prf`) unless `--laterality-table` is given.
- A missing configuration file (seven knees have no 90° loaded) is status `missing`, never NaN-as-failure; a failed one is status `failed`.
- Flagged scans/comparisons are written but excluded from cohort statistics unless `--include-flagged`; Wilcoxon p-values are reported **uncorrected** and labelled so.
- No change to any existing measurement value: `find_thinnest_area` must stay bit-identical (knee goldens unchanged).
- Docstrings verbose (`:param:` / `:return:` for every parameter); update `README.md` and `CLAUDE.md`.
- Per-measurement failures caught as `(RuntimeError, AssertionError, ValueError)` and written as NaN/`failed`, never aborting the run.

## Review Focus

1. **Reference scan fails or is QC-flagged** — every comparison of that knee becomes `failed` (resp. flagged) and the knee is listed; other knees unaffected. Pinned in Task 7.
2. **A bone fails in one scan but not the other** (e.g. tibia subdivision raises "plates overlap") — only that bone's comparison rows are `failed`; the other bone's rows and figure still appear. Pinned in Task 7.
3. **Registration that does not converge or lands far off** (FOV cut differs, poor initial frame) — residual is reported and the comparison flagged `registration_rms:<mm>` rather than silently producing a sheared Δ map. Pinned in Task 2 (bad-initialisation test) and Task 7.
4. **Δ map with an edge mismatch** — the hotspot must not sit on the rim where one scan's cartilage ends. Pinned in Task 5.
5. **Unrolled coordinate for points off the reference bone's articular region** (cartilage extending beyond the reference's) — returns NaN rather than a wrong coordinate, and such points are dropped from the map, counted in the output. Pinned in Task 3.

---

### Task 1: Extract the robust smoothed-argmin search from `find_thinnest_area`

**Files:**
- Create: `morphometry/cartilage/robust_search.py`
- Modify: `morphometry/cartilage/knee.py:877-1035` (`find_thinnest_area` body only)
- Test: `tests/unit/test_robust_search.py`

**Interfaces:**
- Produces: `smoothed_argmin(grid: np.ndarray, valid: np.ndarray, spacing: float, neighbourhood_mm: float, radius_mm: Optional[float] = None, restrict: Optional[np.ndarray] = None) -> dict` with keys `'center_index'` (`(i, j)` ints), `'point_index'` (`(i, j)`), `'smoothed'` (`(H, W)` float, NaN where undefined), `'searchable'` (`(H, W)` bool actually used), `'kernel_size'` (int), `'fallback'` (bool). Exactly the current algorithm: `k = round(neighbourhood_mm/spacing)`, forced odd, ≥ 3; NaN-aware `generic_filter(np.nanmedian, size=k, mode='reflect')`; `searchable = minimum_filter(valid, size=k, mode='constant', cval=False) & ~isnan(smoothed)` `& restrict` (if given); if empty → `fallback=True`, `searchable = valid & ~isnan(smoothed)`; argmin of smoothed over searchable → centre; refinement disc radius `max(1, round((neighbourhood_mm/2 if radius_mm is None else radius_mm)/spacing))` cells, argmin of raw `grid` over `disc & searchable`. No logging inside (callers log their own warning).

- [ ] **Step 1: Write the failing tests**

```python
def test_single_outlier_does_not_win():
    g = np.full((15, 15), 2.0); g[7, 7] = 0.1; g[3:6, 3:6] = 1.0
    r = smoothed_argmin(g, ~np.isnan(g), spacing=1.0, neighbourhood_mm=3.0)
    assert r['center_index'] == (4, 4) and r['kernel_size'] == 3 and not r['fallback']

def test_window_must_lie_inside_valid_footprint():
    g = np.full((9, 9), 2.0); g[0, :] = 0.0          # low rim row
    r = smoothed_argmin(g, np.ones_like(g, bool), 1.0, 3.0)
    assert r['center_index'][0] >= 1 and r['point_index'][0] >= 1

def test_restrict_and_fallback():
    g = np.full((9, 9), 2.0); g[4, 4] = 1.0
    none = np.zeros_like(g, bool)
    r = smoothed_argmin(g, np.ones_like(g, bool), 1.0, 3.0, restrict=none)
    assert r['fallback']
```

- [ ] **Step 2: Run** `pytest tests/unit/test_robust_search.py -v` — Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement `smoothed_argmin`** in `morphometry/cartilage/robust_search.py` (move the code, keep its comments).
- [ ] **Step 4: Route `find_thinnest_area` through it**: build `restrict` from the edge-margin and min-thickness masks exactly as now, keep its WARNING text when `fallback`, keep the returned dict identical.
- [ ] **Step 5: Run** `pytest tests/unit/test_robust_search.py tests/unit -k "thinnest or knee" -v` and `MORPH_DFG_KNEES_DIR=/home/simon/Data/dfg_knees/monai_segmentations/labels/original pytest tests/characterization/test_knee_cartilage.py -v` — Expected: all PASS, no golden moved (the goldens are pinned on the old label set; keep pointing them there).
- [ ] **Step 6: Commit** `refactor(knee): extract the robust smoothed-argmin search`.

### Task 2: Rigid bone registration

**Files:**
- Create: `morphometry/cartilage/registration.py`
- Test: `tests/unit/test_bone_registration.py`

**Interfaces:**
- Produces:
  - `@dataclass RigidTransform(rotation: np.ndarray (3,3), translation: np.ndarray (3,))` with `apply(points (N,3)) -> (N,3)`, `inverse() -> RigidTransform`, `@classmethod identity()`, `@classmethod from_frames(moving_origin, moving_axes, fixed_origin, fixed_axes) -> RigidTransform` (axes are `(3,3)` row-orthonormal frames; maps a point with local coordinates `c` in the moving frame to the point with the same `c` in the fixed frame: `p_f = fixed_origin + fixed_axes.T @ (moving_axes @ (p_m - moving_origin))`).
  - `@dataclass RegistrationResult(transform: RigidTransform, rms_mm: float, median_mm: float, p95_mm: float, n_points: int, iterations: int, converged: bool)`.
  - `crop_to_articular_end(mesh: pv.PolyData, axis: np.ndarray, length_mm: float, end: str) -> pv.PolyData` — keeps cells whose centroid lies within `length_mm` of the extreme along `axis` (`end='min'` for the femur with `axis=proximal_axis`, `end='max'` for the tibia with `axis=e3`).
  - `register_bone(moving: pv.PolyData, fixed: pv.PolyData, initial: Optional[RigidTransform] = None, max_iterations: int = 60, trim_fraction: float = 0.9, max_pair_mm: float = 5.0, tolerance_mm: float = 1e-4, max_points: int = 20000, seed: int = 0) -> RegistrationResult`.
- Default constant: `REGISTRATION_CROP_MM = 40.0`, `REGISTRATION_MAX_RMS_MM = 1.0` (provisional; Task 9 confirms it against the residual distribution, never against Δ).

Algorithm (not determined by the signature): trimmed point-to-plane ICP. Subsample `moving` vertices to `max_points` (deterministic RNG from `seed`); per iteration transform them, query nearest `fixed` vertex (`cKDTree`), drop pairs > `max_pair_mm`, keep the `trim_fraction` closest; solve the linearised point-to-plane least squares for `(ω, t)` using `fixed` point normals (`compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=False)`), compose the small rotation via Rodrigues and re-orthonormalise (SVD); stop when the update moves points < `tolerance_mm`. Residuals = `surface_distances` (`morphometry.cartilage.surface`) of the trimmed final moving points to `fixed`.

- [ ] **Step 1: Write the failing tests** — use a feature-rich synthetic surface (e.g. union of an ellipsoid and an offset smaller ellipsoid, triangulated via `pv.ParametricEllipsoid`/`merge`, or the femur phantom bone surface from `tests/unit/test_femur_regions.py::phantom` through `bone_surface_mesh`):

```python
def test_recovers_known_rigid_motion():
    fixed = _surface(); T = RigidTransform(_rot(8, 5, -6), np.array([3.0, -2.0, 4.0]))
    moving = fixed.copy(); moving.points = T.inverse().apply(fixed.points)
    r = register_bone(moving, fixed)
    assert r.converged and r.rms_mm < 0.05
    assert np.allclose(r.transform.apply(moving.points), fixed.points, atol=0.05)

def test_from_frames_maps_frame_to_frame():
    T = RigidTransform.from_frames(np.zeros(3), np.eye(3), np.array([1., 2, 3]), _rot(0, 0, 90))
    assert np.allclose(T.apply(np.array([[1., 0, 0]])), [1, 2, 3] + _rot(0, 0, 90)[0])

def test_partial_overlap_fov_cut_still_registers():
    # moving is cut 15 mm shorter along the long axis (different FOV) -> trimmed ICP copes
    ...; assert r.rms_mm < 0.1 and _angle(r.transform.rotation, true_R) < 0.5

def test_bad_initialisation_is_visible_in_the_residual():
    # 120 degree initial error: either converges back or reports rms_mm > REGISTRATION_MAX_RMS_MM
    ...; assert r.rms_mm < 0.1 or r.rms_mm > REGISTRATION_MAX_RMS_MM
```

- [ ] **Step 2: Run** `pytest tests/unit/test_bone_registration.py -v` — Expected: FAIL (import error).
- [ ] **Step 3: Implement** the dataclasses, `crop_to_articular_end` and `register_bone` in `morphometry/cartilage/registration.py`.
- [ ] **Step 4: Run** the tests — Expected: PASS (each < 10 s).
- [ ] **Step 5: Commit** `feat(cartilage): rigid ICP registration of bone surfaces`.

### Task 3: Unrolled femoral coordinate

**Files:**
- Create: `morphometry/cartilage/deformation.py`
- Test: `tests/unit/test_unrolled_femur.py`

**Interfaces:**
- Consumes: `FemoralFrame` (`femur_regions`), `FemurSubdivision.bone_mesh`, helpers `_slice_contour` / `_polyline_components` in `femur_regions.py` (import them; do not copy).
- Produces: `class UnrolledFemur` with `@classmethod from_reference(bone_mesh: pv.PolyData, frame: FemoralFrame, slice_mm: float = 0.5, max_distance_mm: float = 2.0) -> UnrolledFemur` and `coordinates(points_world: np.ndarray) -> np.ndarray` returning `(N, 2)` mm: column 0 = `(p − notch_point)·ml_axis` (patient right positive), column 1 = signed arc length along the sagittal bone contour at that M-L level, **0 where the contour crosses the notch plane `ap = notch_ap` on its distal side** (the crossing with the smallest proximal coordinate), **positive posteriorly** (towards decreasing `ap`, round the distal condyle to the posterior condyle). NaN for points farther than `max_distance_mm` from the reference bone surface or outside the M-L extent of computed slices.

Build once: for every M-L level `k*slice_mm` across the bone, take the sagittal contour (plane normal `ml_axis`), its connected component containing the distal notch-plane crossing, cumulative arc length signed from that crossing; then give every reference bone-mesh vertex `(ml, arc)` from its nearest contour point in the two nearest levels, linearly interpolated in ml. `coordinates()` = nearest-vertex lookup (`cKDTree`) with the distance check. Compute it on the bone surface cropped with `crop_to_articular_end(..., REGISTRATION_CROP_MM, 'min')` so the proximal shaft's open FOV edge never enters a contour.

- [ ] **Step 1: Write the failing tests** (femur phantom from `tests/unit/test_femur_regions.py::phantom`, subdivided with `subdivide_femur`):

```python
def test_arc_is_zero_at_notch_and_grows_posteriorly(base_sub):
    u = UnrolledFemur.from_reference(base_sub.bone_mesh, base_sub.frame)
    c = u.coordinates(triangle_centroids(base_sub.mesh))
    codes = base_sub.mesh.cell_data['region']
    alf = c[codes == FEMUR_REGION_KEYS.index('alf'), 1]
    plf = c[codes == FEMUR_REGION_KEYS.index('plf'), 1]
    assert np.nanmax(alf) <= 1.0 and np.nanmin(plf) > 0 and np.nanmedian(plf) > np.nanmedian(alf)

def test_coordinates_are_intrinsic_to_the_bone():
    # phantom rotated by 60 deg flexion-like + translated: same anatomical points -> same coords
    ...; assert np.nanmax(np.abs(c_rot - c_base)) < 0.6   # mm, one slice + mesh noise

def test_points_off_the_bone_are_nan(base_sub):
    far = base_sub.frame.notch_point + 20 * base_sub.frame.ap_axis
    assert np.isnan(UnrolledFemur.from_reference(base_sub.bone_mesh, base_sub.frame).coordinates(far[None])).all()
```

- [ ] **Step 2: Run** `pytest tests/unit/test_unrolled_femur.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `UnrolledFemur`** in `deformation.py`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(cartilage): unrolled femoral coordinate anchored at the trochlear notch`.

### Task 4: Bone measurements, map grids and per-knee Δ maps

**Files:**
- Modify: `morphometry/cartilage/deformation.py`
- Test: `tests/unit/test_delta_maps.py`

**Interfaces:**
- Consumes: Task 2 (`RigidTransform`, `register_bone`, `crop_to_articular_end`, `REGISTRATION_CROP_MM`), Task 3 (`UnrolledFemur`), `ThicknessMaps.triangle_thickness` / `.statistics`, `KneeFrame`, `triangle_centroids` / `triangle_areas` (`surface.py`).
- Produces:
  - Constants `REFERENCE = (30, 'unbelastet')` and `COMPARISONS` (the seven keys of Global Constraints, in that order).
  - `@dataclass BoneMeasurement(bone: str, tab_mesh: pv.PolyData, bone_mesh: pv.PolyData, triangle_thickness: np.ndarray, statistics: dict, frame_origin: np.ndarray, frame_axes: np.ndarray, femoral_frame: Optional[FemoralFrame] = None, knee_frame: Optional[KneeFrame] = None)` — picklable.
  - `bone_measurement(bone: str, subdivision, thickness: ThicknessMaps) -> BoneMeasurement` — femur: `frame_axes = rows(ml, ap, proximal)`, `frame_origin = notch_point`; tibia: `knee_frame_from_tibia(subdivision)` axes/origin.
  - `@dataclass MapGrid(origin_mm: np.ndarray (2,), grid_mm: float, shape: Tuple[int, int])` with `@classmethod covering(coordinate_sets: Sequence[np.ndarray], grid_mm: float, pad_cells: int = 2)`, `cells(coords (N,2)) -> (N,2) int` (half-up rounding as `KneeFrame.cells`), `centre_mm(i, j) -> (2,)`.
  - `rasterise(grid: MapGrid, coords: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray` — `(H,W)` area-weighted mean, NaN where empty; NaN coords dropped.
  - `rasterise_labels(grid, coords, labels: np.ndarray, weights) -> np.ndarray` — `(H,W)` int, majority by weight, `-1` where empty.
  - `@dataclass BoneDeltaMaps(bone: str, grid: MapGrid, reference: np.ndarray, region_codes: np.ndarray, configurations: Dict[tuple, np.ndarray], deltas: Dict[tuple, np.ndarray], registrations: Dict[tuple, RegistrationResult], dropped_fraction: Dict[tuple, float])` — keys are `(angle, load)`.
  - `knee_delta_maps(bone: str, reference: BoneMeasurement, configurations: Dict[tuple, BoneMeasurement], grid_mm: float = 1.0, crop_mm: float = REGISTRATION_CROP_MM) -> BoneDeltaMaps` — per configuration: initial `RigidTransform.from_frames(cfg.frame_origin, cfg.frame_axes, ref.frame_origin, ref.frame_axes)`, `register_bone(crop(cfg.bone_mesh), crop(ref.bone_mesh, crop_mm + 10), initial)`; map the configuration's **covered** tAB centroids into the reference; coordinates = `ref.knee_frame.in_plane(...)` (tibia) or `UnrolledFemur.from_reference(ref.bone_mesh, ref.femoral_frame).coordinates(...)` (femur, built once); one `MapGrid.covering` over reference + all configurations; `deltas[key] = configurations[key] − reference` (NaN where either is NaN); `dropped_fraction` = area share of covered triangles with NaN coordinates.

- [ ] **Step 1: Write the failing tests** (build `BoneMeasurement`s directly from synthetic meshes; no image needed):

```python
def test_rasterise_area_weighted_mean():
    g = MapGrid(np.zeros(2), 1.0, (3, 3))
    out = rasterise(g, np.array([[0.1, 0.0], [0.2, 0.1]]), np.array([1.0, 3.0]), np.array([1.0, 3.0]))
    assert out[0, 0] == pytest.approx(2.5) and np.isnan(out[1, 1])

def test_rigidly_moved_identical_scan_gives_zero_delta():
    ref = _synthetic_tibia_measurement()          # plate mesh + bone mesh + thickness field
    cfg = _moved(ref, _rot(4, -3, 7), [5., 2., -3.])
    d = knee_delta_maps('tibia', ref, {(0, 'belastet'): cfg})
    assert np.nanmax(np.abs(d.deltas[(0, 'belastet')])) < 0.05
    assert d.registrations[(0, 'belastet')].rms_mm < 0.05

def test_thinned_patch_appears_where_it_was_made():
    cfg = _moved(_thinned(ref, centre=[5., 3.], radius=4., by=0.5), ...)
    delta = knee_delta_maps('tibia', ref, {(0, 'belastet'): cfg}).deltas[(0, 'belastet')]
    # cells inside the patch read about -0.5, cells far away about 0
    ...

def test_denuded_triangles_are_excluded_not_zero():
    ...; assert not np.any(np.isclose(d.configurations[k][mask_denuded], 0.0))
```

- [ ] **Step 2: Run** `pytest tests/unit/test_delta_maps.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** the dataclasses and functions above in `deformation.py`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(cartilage): bone-anchored thickness delta maps against a reference scan`.

### Task 5: Largest-thinning hotspot per compartment

**Files:**
- Modify: `morphometry/cartilage/deformation.py`
- Test: `tests/unit/test_thinning_hotspot.py`

**Interfaces:**
- Consumes: Task 1 `smoothed_argmin`; Task 4 `BoneDeltaMaps`, `MapGrid`.
- Produces:
  - `COMPARTMENTS = ('both', 'left', 'right')`; `compartment_codes(bone: str, compartment: str) -> Tuple[int, ...]` — tibia `'left'` = codes of keys ending `'lt'`, `'right'` = `'rt'`; femur `'left'` = keys whose side letter is `l` (`iclf, cclf, eclf, alf, plf`), `'right'` likewise; `'both'` = all.
  - `find_largest_thinning(maps: BoneDeltaMaps, key: tuple, compartment: str = 'both', neighbourhood_mm: float = 3.0, radius_mm: Optional[float] = None) -> Optional[dict]` — runs `smoothed_argmin` on `maps.deltas[key]` with `valid = isfinite(delta)` and `restrict = isin(maps.region_codes, compartment_codes(...))`; returns `{'cell': (i, j), 'centre': (i, j), 'map_mm': (x, y), 'smoothed_delta_mm', 'delta_mm', 'subregion': key name of region_codes[cell] or '', 'kernel_size', 'fallback'}`, `None` if the compartment has no finite Δ cell. Logs a WARNING on `fallback`.

- [ ] **Step 1: Write the failing tests**

```python
def test_hotspot_finds_the_thinned_area_not_a_single_cell():
    # delta 0 everywhere, a 5x5 patch at -0.4, one isolated cell at -2.0
    h = find_largest_thinning(_maps(delta), key)
    assert _inside(h['centre'], patch) and h['smoothed_delta_mm'] == pytest.approx(-0.4)

def test_hotspot_stays_off_an_edge_mismatch():
    # configuration footprint 2 cells smaller on one side; its rim column reads -1.5
    h = find_largest_thinning(_maps(delta_with_rim), key)
    assert h['cell'][1] < rim_col - 1

def test_compartments_are_searched_separately():
    hl = find_largest_thinning(m, key, 'left'); hr = find_largest_thinning(m, key, 'right')
    assert hl['subregion'].endswith('lt') and hr['subregion'].endswith('rt')
```

- [ ] **Step 2: Run** `pytest tests/unit/test_thinning_hotspot.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `compartment_codes` and `find_largest_thinning`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(cartilage): robust largest-thinning hotspot per compartment`.

### Task 6: Figures

**Files:**
- Create: `morphometry/cartilage/deformation_plot.py`
- Test: `tests/unit/test_deformation_plot.py`

**Interfaces:**
- Consumes: Task 4 `BoneDeltaMaps`, Task 5 hotspot dicts.
- Produces:
  - `plot_knee_deltas(maps: BoneDeltaMaps, knee: str, hotspots: Optional[Dict[tuple, dict]] = None, comparisons: Sequence[tuple] = COMPARISONS, reference_key: tuple = REFERENCE, panel_titles: Optional[Dict[tuple, str]] = None, path: Optional[str] = None) -> matplotlib.figure.Figure` — 2 × 4 grid (rows `unbelastet`, `belastet`; columns 0/30/60/90). Reference cell: absolute thickness, `viridis`, own colourbar. Other seven: Δ, `RdBu_r` (thinning blue), one shared symmetric limit = 95th percentile of |Δ| over all seven panels (at least 0.1 mm), one shared colourbar. Reference subregion borders drawn on every panel (lines where `region_codes` changes between neighbouring cells). Hotspot marked with an `x`. A configuration absent from `maps.configurations` gets a blank panel titled "no scan"; a failed one is likewise absent and the caller passes `panel_titles={key: 'failed'}`. Tibia axes: x = e1 (patient right →), y = e2 (anterior ↑). Femur axes: x = M-L (patient right →), y = arc mm with the trochlea (negative) at the top. Axis labels in mm.
  - `plot_cohort_hotspots(rows: Sequence[dict], bone: str, path: Optional[str] = None) -> Figure` — one panel per comparison (7), scatter of `map_x_mm`/`map_y_mm` per compartment colour, same axis conventions; rows already mirrored by the caller when a laterality table is used.
  - `COMPARISONS` and `REFERENCE` are imported from `deformation.py` (Task 4).

- [ ] **Step 1: Write the failing tests** (Agg backend):

```python
def test_grid_layout_and_no_scan_panel(tmp_path):
    fig = plot_knee_deltas(_maps_without((90, 'belastet')), 'K7', path=tmp_path / 'f.png')
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert any('no scan' in t for t in titles) and (tmp_path / 'f.png').exists()

def test_delta_panels_share_a_symmetric_scale():
    lims = {im.get_clim() for im in _delta_images(plot_knee_deltas(_maps(), 'K1'))}
    assert len(lims) == 1 and next(iter(lims))[0] == -next(iter(lims))[1]
```

- [ ] **Step 2: Run** `pytest tests/unit/test_deformation_plot.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** both functions.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(cartilage): per-knee delta-map figures and cohort hotspot scatter`.

### Task 7: Script — measurement, per-knee processing, per-knee outputs

**Files:**
- Create: `scripts/analyse_cartilage_deformation.py`
- Test: `tests/unit/test_cartilage_deformation_script.py` (load the script with `importlib.util.spec_from_file_location`, as `test_thinnest_flexion_script.py` does)

**Interfaces:**
- Consumes: Tasks 2–6; `Tibia`, `Femur`, `scan_qc_flags` (`knee.py`).
- Produces (module-level, used by Task 8):
  - `parse_scan_name(name) -> Tuple[Optional[str], Optional[int], Optional[str]]` (regex `^(K\d+)_(\d+)_(belastet|unbelastet)$` on the suffix-stripped name); `discover_knees(paths) -> Dict[str, Dict[tuple, str]]` (`knee -> {(angle, load): path}`, natural knee order).
  - `measure_scan(path, femur_label=3, tibia_label=4, femur_bone_label=1, tibia_bone_label=2) -> dict` with `{'scan', 'tibia': BoneMeasurement|None, 'femur': BoneMeasurement|None, 'qc_flags': str, 'errors': {bone: str}}`; per-bone failures caught (`RuntimeError, AssertionError, ValueError`).
  - Cache: `<out>/cache/<scan>.pkl` holding that dict plus `{'commit', 'dirty'}`; reused only when the commit matches and the checkout was clean; `--no-cache` disables.
  - `process_knee(knee, measurements: Dict[tuple, dict], grid_mm, neighbourhood_mm, max_rms_mm) -> dict` → `{'subregion_rows', 'hotspot_rows', 'registration_rows', 'maps': {bone: BoneDeltaMaps}}`; row `status` ∈ `ok | missing | failed`; `qc_flags` = `;`-join of the reference's and configuration's scan flags plus `registration_rms:<mm>` when `rms_mm > max_rms_mm`.
  - Columns (exact):
    - `SUBREGION_COLUMNS = ['knee', 'angle', 'load', 'bone', 'subregion', 'anatomical_subregion', 'aMe_ref_mm', 'aMe_mm', 'delta_mm', 'delta_pct', 'status', 'qc_flags']`
    - `HOTSPOT_COLUMNS = ['knee', 'angle', 'load', 'bone', 'compartment', 'map_x_mm', 'map_y_mm', 'world_x', 'world_y', 'world_z', 'subregion', 'smoothed_delta_mm', 'delta_mm', 'kernel_size', 'fallback', 'status', 'qc_flags']` (world = reference scan's RAS mm: area-weighted mean of the reference's covered tAB centroids falling in the hotspot cell, NaN if none)
    - `REGISTRATION_COLUMNS = ['knee', 'angle', 'load', 'bone', 'rms_mm', 'median_mm', 'p95_mm', 'n_points', 'iterations', 'converged', 'dropped_fraction', 'flagged']`
  - CLI: `-i DIR|FILE ... -o DIR [--workers N --knees K1 K3 ... --include-flagged --exclude SCAN ... --laterality-table CSV --grid-mm 1.0 --neighbourhood-mm 3.0 --max-registration-rms 1.0 --no-cache]`. Writes `subregion_delta.csv`, `hotspots.csv`, `registration.csv`, `<knee>_tibia_delta.png`, `<knee>_femur_delta.png`. Scans measured in a `ProcessPoolExecutor(--workers)`; knees processed in a second pool.

- [ ] **Step 1: Write the failing tests** (pure halves, fake `BoneMeasurement`s / monkeypatched `knee_delta_maps`):

```python
def test_discover_knees_groups_by_knee_and_configuration(tmp_path): ...
def test_missing_configuration_is_status_missing_not_failed(): ...   # K7 without 90 belastet
def test_failed_reference_fails_every_comparison_of_that_knee(): ...
def test_one_bone_failing_keeps_the_other_bones_rows(): ...
def test_registration_residual_over_threshold_flags_the_comparison(): ...
def test_subregion_delta_and_percent(): ...   # aMe_ref 2.0, aMe 1.8 -> -0.2 mm, -10 %
```

- [ ] **Step 2: Run** `pytest tests/unit/test_cartilage_deformation_script.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** the functions, cache and CLI above.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(scripts): per-knee cartilage deformation against unloaded 30 degrees`.

### Task 8: Script — cohort statistics, laterality, report

**Files:**
- Modify: `scripts/analyse_cartilage_deformation.py`
- Test: `tests/unit/test_cartilage_deformation_script.py`

**Interfaces:**
- Consumes: Task 7 rows.
- Produces:
  - `load_laterality_table(path) -> Dict[str, str]` (same CSV format and errors as `scripts/compute_knee_cartilage_thickness.py::load_laterality_table`, keyed by knee id `K<n>`; duplicate the 20 lines, scripts do not import each other).
  - `select_rows(rows, include_flagged, exclude) -> list` — keeps `status == 'ok'`, drops rows with non-empty `qc_flags` unless `include_flagged`, drops rows whose configuration or reference scan is in `exclude`.
  - `to_anatomical(rows, laterality) -> list` — with a table: `subregion` renamed via `rename_tibia_regions` / `rename_femur_regions` maps (key-level), compartment `left/right` → `medial/lateral`, hotspot `map_x_mm` mirrored so medial is positive (`x_med = -x` for a right knee, `+x` for a left one); knees missing from the table dropped and listed. Without a table: unchanged.
  - `subregion_summary(rows) -> List[dict]` with `SUMMARY_COLUMNS = ['table', 'angle', 'load', 'bone', 'region', 'unit', 'n', 'mean', 'sd', 'median', 'min', 'max', 'wilcoxon_p_uncorrected', 'count']`: one row per comparison × bone × subregion × unit (`mm`, `%`); `wilcoxon_p_uncorrected` = `scipy.stats.wilcoxon(values).pvalue` (two-sided), NaN for n < 2 or all-zero.
  - `hotspot_frequency(rows) -> List[dict]` (`table='hotspot_frequency'`, `count` per comparison × bone × compartment × subregion).
  - Writes `cohort_summary.csv`, `<bone>_hotspots_cohort.png` (Task 6), `report.md` (per-comparison tables of mean Δ per subregion with n; hotspot frequency; a sentence that p-values are uncorrected; the same-side assumption when no laterality table; that Δ contains acquisition-to-acquisition spread; lists of missing, failed, flagged and excluded comparisons; registration residual summary), `report.meta.json` (commit, dirty flag, branch, host, date, args, inputs, counts — copy `_git`/`write_meta` pattern from `report_knee_cartilage_cohort.py`).

- [ ] **Step 1: Write the failing tests**

```python
def test_flagged_rows_are_excluded_by_default_and_included_on_request(): ...
def test_wilcoxon_is_reported_uncorrected_and_nan_for_tiny_n(): ...
def test_laterality_renames_and_mirrors(): ...  # right knee: clt -> medial name, map_x 3 -> -3
def test_knee_missing_from_laterality_table_is_dropped_and_listed(): ...
def test_hotspot_frequency_counts_per_subregion(): ...
def test_report_states_same_side_assumption_without_table(tmp_path): ...
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** the functions and the cohort stage of `main`.
- [ ] **Step 4: Run** `pytest tests/unit/test_cartilage_deformation_script.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(scripts): cohort statistics and report for cartilage deformation`.

### Task 9: Real-data check, QC threshold, documentation

**Files:**
- Modify: `tests/data_paths.py` (add `DFG_KNEES_CURRENT_DIR`, env `MORPH_DFG_KNEES_CURRENT_DIR`, default `/home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations`; do **not** change `DFG_KNEES_DIR`)
- Create: `tests/characterization/test_cartilage_deformation.py` (marker `needs_dfg_knees`, skip when the directory is absent)
- Modify: `morphometry/cartilage/registration.py` (final `REGISTRATION_MAX_RMS_MM`), `README.md`, `CLAUDE.md` (scripts section + a cartilage-deformation bullet under "Conventions worth preserving")

- [ ] **Step 1: Write the characterization tests** on `K1`:

```python
def test_reference_against_itself_is_exactly_zero(): ...   # identity transform, all deltas 0
def test_k1_loaded_30_registers_and_stays_plausible():
    # both bones: rms_mm < REGISTRATION_MAX_RMS_MM, converged; |median delta| < 0.3 mm
```

- [ ] **Step 2: Run** `pytest tests/characterization/test_cartilage_deformation.py -v` — Expected: PASS (fix code, not thresholds, if not).
- [ ] **Step 3: Run the cohort**: `python scripts/analyse_cartilage_deformation.py -i /home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations -o /home/simon/Data/dfg_knees/deformation_<short sha> --workers 16` — Expected: 15 knees, 105 comparisons per bone of which 7 `missing` (no 90° loaded scan).
- [ ] **Step 4: Set the residual threshold from `registration.csv` alone**: inspect the `rms_mm` distribution per bone; set `REGISTRATION_MAX_RMS_MM` at a visible gap above the bulk (or keep 1.0 mm if there is none), record the distribution (median / p90 / max per bone) and the rule in the constant's comment and `CLAUDE.md`. Do not look at Δ while choosing it. Re-run step 3 if it changed.
- [ ] **Step 5: Look at two knees' figures** (one with all eight scans, one without 90° loaded) and confirm the panels, the "no scan" panel, the borders and the hotspot markers render as specified.
- [ ] **Step 6: Document** in `README.md` and `CLAUDE.md`: the script's CLI and outputs, the data directory (and that the goldens/tests still use the outdated set — follow-up), the registration approach and its residual figures, the unrolled coordinate, the uncorrected p-values and the same-side assumption.
- [ ] **Step 7: Run the fast suite** `pytest -m "not needs_ct" -q` — Expected: all pass (same xfails as before).
- [ ] **Step 8: Commit** `docs(knee): cartilage deformation analysis, data set and registration QC`.

# NaKo Antetorsion Against the Inter-Head Line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a NaKo femoral-antetorsion variant measured against the line through both femoral head centres (the readers' reference), falling back to the image x-axis when that line cannot be established.

**Architecture:** A pure composition helper (four landmarks → two signed angles) plus a two-sided measurement `calculate_anteversion_head_line` in `morphometry/measurements/hip.py` that gets both Murphy lines via `get_proximal_reference_line`, maps the right half's landmarks into the parent frame (`+half` on x) and composes with `geometry.signed_axial_angle`. The NaKo scripts gain an `AT_murphy_headline` column and a fallback QC flag; `eval_nako.py` scores it against the readers.

**Tech Stack:** Python 3.10, numpy, pytest (golden/characterization harness in `tests/`).

**Spec:** `docs/superpowers/specs/2026-11-23-nako-head-line-antetorsion-design.md`

## Global Constraints

- `calculate_anteversion` and the `AT_murphy` column are not changed; `tests/golden/anteversion_left.json` must not move.
- Sign: positive = antetorsion, same as `calculate_anteversion` and femoral torsion.
- Axial plane only: z components of every landmark are ignored.
- Anisotropic in-plane spacing raises `ValueError` via `G.validate_in_plane_isotropy`.
- Returned values are for **image** halves (`left` = image-left = patient-right); the scripts map to patient side as they do for `AT_murphy`.
- New guards raise `ValueError`. No `side == 'right'` landmark branches (mirror-equivariance rule, CLAUDE.md).
- Verbose docstrings with `:param:` / `:return:` for every parameter (user CLAUDE.md).
- No parameter chosen against the 60 NaKo reader cases.

## Review Focus

1. One head centre fails (Murphy landmark error) → the other side still gets an x-axis value, the failed side raises/NaN; reference reported as `'x_axis'`. Pinned in Task 1 (`test_one_side_failing_falls_back_to_x_axis`).
2. Both sides fail → raises `ValueError` (scripts turn it into NaN for both). Pinned in Task 1.
3. Head centres coincide in the axial plane (degenerate line, axial length < 1e-6 voxel) → x-axis fallback, not a NaN/garbage angle. Pinned in Task 1 (`test_degenerate_head_line_falls_back`).
4. Pelvis rotated in the scanner → head-line value unchanged while `AT_murphy` changes. Pinned in Task 1 (`test_head_line_angle_is_rotation_invariant`).
5. Fallback rows reaching the eval silently mixed with head-line rows → scripts write `at_headline_fallback:x_axis` into `qc_flags`. Pinned in Task 2 (`test_table_writes_headline_column_and_fallback_flag`).

---

### Task 1: The measurement

**Files:**
- Modify: `morphometry/measurements/hip.py` (add after `calculate_anteversion`, ~line 347)
- Modify: `morphometry/measurements/__init__.py` (import + `__all__`)
- Test: `tests/unit/test_anteversion_head_line.py` (pure helper + monkeypatched fallback, no data)
- Test: `tests/characterization/test_anteversion_head_line.py` (`needs_nako`: golden, mirror equivariance, x-axis identity)
- Create: `tests/golden/anteversion_head_line.json` (via `MORPH_UPDATE_GOLDEN=1` on that node only)

**Interfaces:**
- Produces:
  - `_head_line_antetorsion(left_head: np.ndarray, left_base: np.ndarray, right_head: np.ndarray, right_base: np.ndarray, reference: np.ndarray | None = None) -> Tuple[float, float]` — all points in the **parent** index frame; `reference=None` means the inter-head line `left_head − right_head`; otherwise the given vector (the fallback passes `np.array([1., 0., 0.])`). Per side: `G.signed_axial_angle(reference, head − base, side)`.
  - `calculate_anteversion_head_line(segmentation_mask: Image, segmentation_label: int = 1, isotropic: bool = False, x_ratio: Optional[float] = None, plot: plt.Axes | bool = False) -> Tuple[float, float, str]` → `(left, right, reference)`, `reference ∈ {'head_line', 'x_axis'}`. A side whose own landmarks failed comes back as `np.nan` (only when the other side succeeded); both failing raises `ValueError`.

- [ ] **Step 1: Write the failing unit tests** in `tests/unit/test_anteversion_head_line.py`:

```python
def test_head_line_parallel_to_x_equals_x_axis_reference():
    lh, lb = np.array([60., 50, 30]), np.array([40., 44, 20])   # left: base anterior+lateral
    rh, rb = np.array([140., 50, 30]), np.array([160., 58, 20])
    assert _head_line_antetorsion(lh, lb, rh, rb) == pytest.approx(
        _head_line_antetorsion(lh, lb, rh, rb, reference=np.array([1., 0, 0])))

def test_sign_matches_calculate_anteversion_convention():
    # base anterior (smaller y) to head -> negative, as calculate_anteversion
    l, r = _head_line_antetorsion(np.array([60., 50, 0]), np.array([40., 44, 0]),
                                  np.array([140., 50, 0]), np.array([160., 44, 0]))
    assert l < 0 and r < 0 and l == pytest.approx(r)

def test_head_line_angle_is_rotation_invariant():
    # rotate all four points by 17 deg about z: values unchanged; x-axis reference values change
    ...  # assert_allclose(rotated, original, atol=1e-9); assert not allclose for reference=[1,0,0]

def test_mirror_swaps_sides():
    # x -> W-1-x applied to all points, left/right swapped: values swap exactly
    ...

def test_one_side_failing_falls_back_to_x_axis(monkeypatch):
    # monkeypatch hip.get_proximal_reference_line to raise ValueError for side='right'
    # on a tiny isotropic Segmentation; expect (finite, nan, 'x_axis')

def test_both_sides_failing_raises(monkeypatch): ...   # pytest.raises(ValueError)

def test_degenerate_head_line_falls_back(monkeypatch):
    # landmarks with identical axial (x, y) head centres in the parent frame -> reference 'x_axis'

def test_anisotropic_in_plane_spacing_is_refused(): ...  # ValueError
```

- [ ] **Step 2: Run** `pytest tests/unit/test_anteversion_head_line.py -v` — expected FAIL (ImportError).

- [ ] **Step 3: Implement** `_head_line_antetorsion` and `calculate_anteversion_head_line` in `morphometry/measurements/hip.py`. Split with `split_image_left_right`; call `get_proximal_reference_line(half, side=..., method='murphy', segmentation_label=..., isotropic=..., x_ratio=...)[:2]` per half, each in its own `try` catching `(ValueError, RuntimeError, AssertionError, IndexError)`; add `left_half.array.shape[0]` to the right landmarks' x. `x_ratio=None` → `spacing_z / spacing_x` as in `calculate_anteversion`. Log a WARNING when falling back, naming the reason. `plot` (an Axes) draws the axial slice at the mean head-centre z with both neck lines and the reference line. Export from `morphometry/measurements/__init__.py`.

- [ ] **Step 4: Run** `pytest tests/unit/test_anteversion_head_line.py tests/test_import_graph.py -v` — expected PASS.

- [ ] **Step 5: Write the NaKo characterization tests** in `tests/characterization/test_anteversion_head_line.py` (`pytestmark = pytest.mark.needs_nako`, fixture `nako_sample`):
  - `test_anteversion_head_line_golden`: `assert_golden("anteversion_head_line", list(values[:2]))` and `reference == 'head_line'`.
  - `test_anteversion_head_line_is_mirror_equivariant`: on `reflect_image_left_right(nako_sample)` values are swapped, `atol=1e-6`.
  - `test_x_axis_fallback_reproduces_calculate_anteversion`: monkeypatch the right side to fail; the left value equals `calculate_anteversion(left_half, 'left', 1, isotropic=True)` to `1e-9`.

- [ ] **Step 6: Capture** `MORPH_UPDATE_GOLDEN=1 pytest tests/characterization/test_anteversion_head_line.py::test_anteversion_head_line_golden`, then run the whole file plus `tests/characterization/test_hip.py` without the env var — expected PASS, `anteversion_left` unchanged.

- [ ] **Step 7: Commit** `feat(hip): antetorsion against the inter-head line with x-axis fallback`.

### Task 2: NaKo scripts and evaluation

**Files:**
- Modify: `scripts/process_nako_large.py` (`COLUMNS`, `_nan_result`, `f`, `build_table`)
- Modify: `scripts/process_nako.py` (`f`, table columns)
- Modify: `scripts/eval_nako.py` (Auto rows + `metrics` list)
- Test: `tests/unit/test_process_nako_large_script.py`

**Interfaces:**
- Consumes: `calculate_anteversion_head_line(mask, 1, isotropic=True)` → `(left, right, reference)` from Task 1.
- Produces: result-dict keys `fat_headline_left`, `fat_headline_right`, `fat_headline_reference` (`'head_line' | 'x_axis' | ''`, `''` when the call raised); table column `AT_murphy_headline`, appended **after** the historical columns and before `qc_flags`.

- [ ] **Step 1: Write the failing test** `test_table_writes_headline_column_and_fallback_flag` in `tests/unit/test_process_nako_large_script.py`: extend `_result` with the three keys; assert patient-right row's `AT_murphy_headline == round(fat_headline_left, 1)`, the historical column order is unchanged up to `Cartilage_thickness*`, and that with `fat_headline_reference='x_axis'` both rows' `qc_flags` contain `at_headline_fallback:x_axis` (joined with `;` after any Murphy flags), with `'head_line'` they do not. Also update `test_table_keeps_the_original_layout_and_side_mapping` for the new column position.

- [ ] **Step 2: Run** `pytest tests/unit/test_process_nako_large_script.py -v` — expected FAIL.

- [ ] **Step 3: Implement** in `process_nako_large.py`: one `try` around `calculate_anteversion_head_line` in `f` (bare `except Exception` like its neighbours → NaN, NaN, `''`); `_nan_result` gets the three keys; `build_table` writes the column and the flag. Mirror the column into `process_nako.py` (no QC column there; write the reference into an `AT_murphy_headline_reference` column instead). In `eval_nako.py` add Auto rows for metrics `AT_murphy_headline_right/left` scored against the same IRM `AT_murphy_*` reader means, and add both to `metrics`.

- [ ] **Step 4: Run** `pytest tests/unit -v` — expected PASS. Then smoke-run `f` from `process_nako_large.py` on one NaKo sample case from a Python shell and check `AT_murphy_headline` is finite and `fat_headline_reference == 'head_line'`.

- [ ] **Step 5: Commit** `feat(nako): AT_murphy_headline column, fallback flag, reader evaluation`.

### Task 3: Documentation

**Files:**
- Modify: `CLAUDE.md` (Hip section: one bullet on the head-line variant, its fallback and that `AT_murphy_headline` must not be pooled with `AT_murphy`; scripts bullet mentions the column/flag)
- Modify: `docs/scripts.md` (NaKo output columns and the `at_headline_fallback:x_axis` flag)
- Modify: `README.md` (measurement list)

- [ ] **Step 1:** Write the three edits; keep CLAUDE.md to ≤3 lines added.
- [ ] **Step 2: Run** `pytest -m "not needs_ct" -q` — expected all pass/skip as before.
- [ ] **Step 3: Commit** `docs: NaKo head-line antetorsion`.

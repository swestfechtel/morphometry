"""Tests for the largest-thinning hotspot search on thickness delta maps (synthetic grids)."""
import logging

import numpy as np
import pytest

from morphometry.cartilage.deformation import (
    BoneDeltaMaps, MapGrid, compartment_codes, find_largest_thinning,
)
from morphometry.cartilage.femur_regions import FEMUR_REGION_KEYS
from morphometry.cartilage.tibia_regions import TIBIA_REGION_KEYS

KEY = (60, 'unbelastet')
H = W = 40


def _maps(delta, bone='tibia', split=20, reference=None, configuration=None, grid_mm=0.5):
    """
    Minimal BoneDeltaMaps: patient-left region for i < split, patient-right otherwise.

    The reference and configuration maps default to the delta's own footprint (thickness
    1.0 mm there), as in a real ``knee_delta_maps`` result, where ``delta`` is finite exactly
    where both are.
    """
    keys = TIBIA_REGION_KEYS if bone == 'tibia' else FEMUR_REGION_KEYS
    left = keys.index('clt' if bone == 'tibia' else 'cclf')
    right = keys.index('crt' if bone == 'tibia' else 'ccrf')
    codes = np.full((H, W), -1, dtype=int)
    codes[:split] = left
    codes[split:] = right
    footprint = np.where(np.isfinite(delta), 1.0, np.nan)
    reference = footprint if reference is None else reference
    configuration = footprint + delta if configuration is None else configuration
    return BoneDeltaMaps(
        bone=bone, grid=MapGrid(origin_mm=(10.0, -5.0), grid_mm=grid_mm, shape=(H, W)),
        reference=reference, region_codes=codes, configurations={KEY: configuration},
        deltas={KEY: delta},
        registrations={}, dropped_fraction={}, reference_positions=np.full((H, W, 3), np.nan),
        rotation_change_deg={})


def test_hotspot_finds_the_thinned_area_not_a_single_cell():
    delta = np.zeros((H, W))
    delta[8:13, 8:13] = -0.4          # 5x5 patch
    delta[30, 30] = -2.0              # isolated outlier cell
    h = find_largest_thinning(_maps(delta), KEY)
    assert 8 <= h['centre'][0] < 13 and 8 <= h['centre'][1] < 13
    assert h['smoothed_delta_mm'] == pytest.approx(-0.4)
    assert 8 <= h['cell'][0] < 13 and 8 <= h['cell'][1] < 13
    assert h['delta_mm'] == pytest.approx(-0.4)
    assert h['map_mm'] == pytest.approx((10.0 + 0.5 * h['cell'][0], -5.0 + 0.5 * h['cell'][1]))
    assert h['subregion'] == 'clt' and h['fallback'] is False
    assert isinstance(h['cell'][0], int) and isinstance(h['delta_mm'], float)


def test_hotspot_stays_off_an_edge_mismatch():
    # the configuration footprint is 3 columns smaller on one side: columns 0-2 undefined and
    # the rim column 3 reads -1.5; a real (milder) thinning sits well inside
    delta = np.zeros((H, W))
    delta[:, :3] = np.nan
    rim_col = 3
    delta[:, rim_col] = -1.5
    delta[24:29, 22:27] = -0.3
    h = find_largest_thinning(_maps(delta), KEY)
    assert h['cell'][1] > rim_col + 1
    assert 24 <= h['cell'][0] < 29 and 22 <= h['cell'][1] < 27


def test_compartments_are_searched_separately():
    delta = np.zeros((H, W))
    delta[4:9, 10:15] = -0.5    # left
    delta[30:35, 10:15] = -0.2  # right
    m = _maps(delta)
    hl = find_largest_thinning(m, KEY, 'left')
    hr = find_largest_thinning(m, KEY, 'right')
    hb = find_largest_thinning(m, KEY, 'both')
    assert hl['subregion'].endswith('lt') and hr['subregion'].endswith('rt')
    assert hl['smoothed_delta_mm'] == pytest.approx(-0.5)
    assert hr['smoothed_delta_mm'] == pytest.approx(-0.2)
    assert hb['cell'][0] < 20


def test_missing_configuration_and_empty_compartment_give_none():
    delta = np.zeros((H, W))
    delta[20:] = np.nan  # right compartment has no finite cell
    m = _maps(delta)
    assert find_largest_thinning(m, (0, 'belastet')) is None
    assert find_largest_thinning(m, KEY, 'right') is None
    assert find_largest_thinning(m, KEY, 'left') is not None


def test_fallback_is_logged(caplog):
    delta = np.full((H, W), np.nan)
    delta[5:7, 5:7] = -0.1   # too small for any full smoothing window
    with caplog.at_level(logging.WARNING, logger='morphometry'):
        h = find_largest_thinning(_maps(delta), KEY)
    assert h['fallback'] is True
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_compartment_codes():
    assert [TIBIA_REGION_KEYS[c] for c in compartment_codes('tibia', 'left')] == \
        [k for k in TIBIA_REGION_KEYS if k.endswith('lt')]
    assert [TIBIA_REGION_KEYS[c] for c in compartment_codes('tibia', 'right')] == \
        [k for k in TIBIA_REGION_KEYS if k.endswith('rt')]
    assert {FEMUR_REGION_KEYS[c] for c in compartment_codes('femur', 'left')} == \
        {'iclf', 'cclf', 'eclf', 'alf', 'plf'}
    assert {FEMUR_REGION_KEYS[c] for c in compartment_codes('femur', 'right')} == \
        {'icrf', 'ccrf', 'ecrf', 'arf', 'prf'}
    assert len(compartment_codes('femur', 'both')) == len(FEMUR_REGION_KEYS)
    with pytest.raises(ValueError):
        compartment_codes('femur', 'middle')


def test_fallback_stays_inside_the_requested_compartment():
    delta = np.full((H, W), np.nan)
    delta[5:7, 5:7] = -0.5      # left: thinner, but only a sliver
    delta[30:32, 30:32] = -0.1  # right: no full smoothing window either
    m = _maps(delta)
    hr = find_largest_thinning(m, KEY, 'right')
    hl = find_largest_thinning(m, KEY, 'left')
    assert hr['fallback'] is True
    assert hr['cell'][0] >= 20 and hr['subregion'].endswith('rt')
    assert hr['delta_mm'] == pytest.approx(-0.1)
    assert hl['cell'][0] < 20 and hl['subregion'].endswith('lt')


def _rim_case():
    """1 mm grid, k = 3: a strong thinning in the second column from an edge, a milder one inside."""
    delta = np.zeros((H, W))
    delta[10:15, 0:2] = -1.0     # cells at 1 and 2 mm from the edge: clear the k=3 window rule
    delta[24:29, 22:27] = -0.3
    return delta


def test_edge_margin_keeps_the_hotspot_off_the_footprint_rim():
    delta = _rim_case()
    m = _maps(delta, grid_mm=1.0)
    off = find_largest_thinning(m, KEY, edge_margin_mm=0.0)
    assert off['cell'][1] == 1 and off['edge_distance_mm'] == pytest.approx(2.0)
    h = find_largest_thinning(m, KEY)      # default edge margin, 2 mm
    assert 24 <= h['cell'][0] < 29 and 22 <= h['cell'][1] < 27
    assert h['edge_distance_mm'] > 2.0 and h['fallback'] is False


def test_edge_margin_applies_to_each_scan_footprint():
    # the delta is defined everywhere the configuration is; the reference ends at column 5
    # (its rim lies inside the configuration's footprint), or the configuration does
    base = np.zeros((H, W))
    base[10:15, 5:7] = -1.0
    base[24:29, 22:27] = -0.3
    for which in ('reference', 'configuration'):
        delta = base.copy()
        delta[:, :5] = np.nan
        short = np.where(np.isfinite(delta), 1.0, np.nan)
        full = np.ones((H, W))
        if which == 'reference':
            m = _maps(delta, grid_mm=1.0, reference=short, configuration=full + np.nan_to_num(delta))
        else:
            m = _maps(delta, grid_mm=1.0, reference=full, configuration=short + delta)
        assert find_largest_thinning(m, KEY, edge_margin_mm=0.0)['cell'][1] in (5, 6)
        h = find_largest_thinning(m, KEY)
        assert 22 <= h['cell'][1] < 27, which


def test_edge_margin_fallback_stays_in_the_compartment():
    # a right compartment only 3 cells wide: nothing is 2 mm from its edge; the fallback must
    # stay right even though the left compartment holds a deeper, fully interior thinning
    delta = np.full((H, W), np.nan)
    delta[:20, :] = 0.0
    delta[5:10, 10:15] = -0.8
    delta[30:33, 10:20] = -0.1
    m = _maps(delta, grid_mm=1.0)
    hr = find_largest_thinning(m, KEY, 'right')
    assert hr['fallback'] is True and hr['cell'][0] >= 20 and hr['subregion'].endswith('rt')

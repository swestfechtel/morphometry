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


def _maps(delta, bone='tibia', split=20):
    """Minimal BoneDeltaMaps: patient-left region for i < split, patient-right otherwise."""
    keys = TIBIA_REGION_KEYS if bone == 'tibia' else FEMUR_REGION_KEYS
    left = keys.index('clt' if bone == 'tibia' else 'cclf')
    right = keys.index('crt' if bone == 'tibia' else 'ccrf')
    codes = np.full((H, W), -1, dtype=int)
    codes[:split] = left
    codes[split:] = right
    nan = np.full((H, W), np.nan)
    return BoneDeltaMaps(
        bone=bone, grid=MapGrid(origin_mm=(10.0, -5.0), grid_mm=0.5, shape=(H, W)),
        reference=nan, region_codes=codes, configurations={}, deltas={KEY: delta},
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

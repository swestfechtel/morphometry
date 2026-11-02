"""Unit tests for ``morphometry.cartilage.robust_search.smoothed_argmin``."""
import numpy as np

from morphometry.cartilage.robust_search import smoothed_argmin


def test_single_outlier_does_not_win():
    g = np.full((15, 15), 2.0); g[7, 7] = 0.1; g[3:6, 3:6] = 1.0
    r = smoothed_argmin(g, ~np.isnan(g), spacing=1.0, neighbourhood_mm=3.0)
    # The smoothed minimum (1.0) is attained on a plus-shaped set of cells around (4, 4)
    # (a window with >= 5 of 9 patch cells has median 1.0), so the first in C order, (3, 4),
    # wins the tie; what matters is that it is in the patch and not the isolated outlier.
    assert r['center_index'] == (3, 4) and r['kernel_size'] == 3 and not r['fallback']
    assert r['smoothed'][r['center_index']] == 1.0
    assert 3 <= r['point_index'][0] < 6 and 3 <= r['point_index'][1] < 6


def test_window_must_lie_inside_valid_footprint():
    g = np.full((9, 9), 2.0); g[0, :] = 0.0          # low rim row
    r = smoothed_argmin(g, np.ones_like(g, bool), 1.0, 3.0)
    assert r['center_index'][0] >= 1 and r['point_index'][0] >= 1


def test_restrict_and_fallback():
    g = np.full((9, 9), 2.0); g[4, 4] = 1.0
    none = np.zeros_like(g, bool)
    r = smoothed_argmin(g, np.ones_like(g, bool), 1.0, 3.0, restrict=none)
    assert r['fallback']

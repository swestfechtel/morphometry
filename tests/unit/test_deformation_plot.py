"""Tests for the delta-map figures (Agg backend, synthetic maps)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from morphometry.cartilage.deformation import BoneDeltaMaps, COMPARISONS, MapGrid, REFERENCE
from morphometry.cartilage.deformation_plot import plot_cohort_hotspots, plot_knee_deltas


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close('all')


def _maps(bone='tibia', drop=()):
    grid = MapGrid((-5.0, -4.0), 1.0, (10, 8))
    rng = np.random.default_rng(0)
    ref = rng.uniform(1.5, 3.0, grid.shape)
    codes = np.zeros(grid.shape, int)
    codes[5:, :] = 1
    codes[:, 6:] = -1
    deltas = {k: rng.normal(0, 0.3, grid.shape) for k in COMPARISONS if k not in drop}
    deltas[COMPARISONS[0]][0, 0] = np.nan
    return BoneDeltaMaps(bone, grid, ref, codes, {k: ref + d for k, d in deltas.items()},
                         deltas, {}, {}, np.zeros(grid.shape + (3,)), {})


def _maps_without(key, bone='tibia'):
    return _maps(bone, drop=(key,))


def _delta_images(fig):
    return [im for ax in fig.axes for im in ax.get_images() if im.get_cmap().name == 'RdBu_r']


def test_grid_layout_and_no_scan_panel(tmp_path):
    fig = plot_knee_deltas(_maps_without((90, 'belastet')), 'K7', path=tmp_path / 'f.png')
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert any('no scan' in t for t in titles) and (tmp_path / 'f.png').exists()
    assert len(_delta_images(fig)) == 6


def test_delta_panels_share_a_symmetric_scale():
    lims = {im.get_clim() for im in _delta_images(plot_knee_deltas(_maps(), 'K1'))}
    assert len(lims) == 1 and next(iter(lims))[0] == -next(iter(lims))[1]


def test_limit_has_floor_and_failed_title():
    m = _maps(drop=((0, 'belastet'),))
    for k in m.deltas:
        m.deltas[k] = np.full_like(m.deltas[k], np.nan)
    fig = plot_knee_deltas(m, 'K', panel_titles={(0, 'belastet'): 'failed'})
    assert next(iter({im.get_clim() for im in _delta_images(fig)})) == (-0.1, 0.1)
    assert any('failed' in a.get_title() for a in fig.axes)


def test_femur_axis_inverted_and_hotspot_marked():
    fig = plot_knee_deltas(_maps('femur'), 'K', hotspots={(60, 'unbelastet'): {'map_mm': (0.0, 1.0)}})
    ax = [a for a in fig.axes if a.get_title().startswith('60° unloaded')][0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1] and len(ax.lines) == 1


def test_cohort_scatter(tmp_path):
    rows = [dict(angle=60, load='belastet', bone='tibia', compartment='left', map_x_mm=1.0, map_y_mm=2.0),
            dict(angle=60, load='belastet', bone='tibia', compartment='right', map_x_mm=np.nan, map_y_mm=2.0),
            dict(angle=0, load='belastet', bone='femur', compartment='both', map_x_mm=1.0, map_y_mm=2.0)]
    fig = plot_cohort_hotspots(rows, 'tibia', path=tmp_path / 'c.png')
    n = sum(len(c.get_offsets()) for a in fig.axes for c in a.collections)
    assert n == 1 and (tmp_path / 'c.png').exists()

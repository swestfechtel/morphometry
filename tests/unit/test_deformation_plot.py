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
    grid = MapGrid((-5.0, -4.0), 0.5, (10, 8))
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


def _panel(fig, title):
    return [a for a in fig.axes if a.get_title().startswith(title)][0]


def test_extent_and_orientation_in_mm():
    # origin (-5, -4), 0.5 mm cells, (10, 8): x edges -5.25..-0.25, y edges -4.25..-0.25.
    ax = _panel(plot_knee_deltas(_maps(), 'K'), '60° unloaded')
    im = ax.get_images()[0]
    assert im.get_array().shape == (8, 10)  # transposed: rows = column 1
    np.testing.assert_allclose(im.get_extent(), [-5.25, -0.25, -4.25, -0.25])
    assert im.origin == 'lower'


def test_region_border_segments():
    ax = _panel(plot_knee_deltas(_maps(), 'K'), '60° unloaded')
    segs = np.array(ax.collections[0].get_segments())  # (n, 2 points, 2)
    vertical = segs[np.isclose(segs[:, 0, 0], segs[:, 1, 0])]
    horizontal = segs[np.isclose(segs[:, 0, 1], segs[:, 1, 1])]
    assert len(vertical) == 6 and len(horizontal) == 10 and len(segs) == 16
    # the 4|5 split of column 0 is at x = -2.75, over cells j = 0..5 (y -4.25..-1.25)
    np.testing.assert_allclose(vertical[:, :, 0], -2.75)
    np.testing.assert_allclose(np.sort(vertical[:, 0, 1]), -4.25 + 0.5 * np.arange(6))
    # the labelled / unlabelled edge at y = -1.25 spans every column
    np.testing.assert_allclose(horizontal[:, :, 1], -1.25)
    np.testing.assert_allclose(np.sort(horizontal[:, 0, 0]), -5.25 + 0.5 * np.arange(10))


def test_hotspot_marker_position():
    fig = plot_knee_deltas(_maps(), 'K', hotspots={(60, 'unbelastet'): {'map_mm': (-3.0, -2.0)}})
    x, y = _panel(fig, '60° unloaded').lines[0].get_data()
    assert (list(x), list(y)) == ([-3.0], [-2.0])


def test_cohort_scatter_anatomical(tmp_path):
    rows = [dict(angle=60, load='belastet', bone='tibia', compartment='medial', map_x_mm=1.0, map_y_mm=2.0),
            dict(angle=60, load='belastet', bone='tibia', compartment='lateral', map_x_mm=-1.0, map_y_mm=2.0)]
    fig = plot_cohort_hotspots(rows, 'tibia', anatomical=True)
    n = sum(len(c.get_offsets()) for a in fig.axes for c in a.collections)
    assert n == 2 and 'medial positive' in fig._suptitle.get_text()


def test_axis_and_colour_bar_labels_name_patient_right_and_the_one_sided_thickness():
    """
    Both bones' x axis is the patient-right coordinate (the femur's too, not "medial-lateral",
    which would suggest an anatomical side); the colour bars say the maps are the one-sided
    cAB -> AC distance, not the aMe of the tables.
    """
    for bone in ('tibia', 'femur'):
        fig = plot_knee_deltas(_maps(bone), 'K')
        panels = [a for a in fig.axes if a.get_title()]
        assert {a.get_xlabel() for a in panels} == {'patient right (mm)'}
        bars = [a.get_ylabel() for a in fig.axes if not a.get_title() and a.get_ylabel()]
        assert bars and all('cAB→AC' in label for label in bars)
        plt.close(fig)


def _figure_text(fig):
    return '\n'.join(t.get_text() for t in fig.texts)


def test_knee_caption_explains_sign_scale_and_search_settings():
    """The caption under the panels states the sign of delta, the scale and the hotspot search."""
    fig = plot_knee_deltas(_maps(), 'K', neighbourhood_mm=4.0, edge_margin_mm=1.5)
    text = _figure_text(fig).replace('\n', ' ')
    assert 'Δ < 0 (BLUE) = cartilage THINNER' in text
    assert 'Δ = thickness(this scan) − thickness(reference)' in text
    assert 'over 4 mm' in text and 'at least 1.5 mm' in text
    assert 'LT / RT' in text  # tibial abbreviations
    text = _figure_text(plot_knee_deltas(_maps('femur'), 'K')).replace('\n', ' ')
    assert 'trochlear notch' in text and 'LF / RF' in text


def test_reference_panel_names_subregions_and_no_data_is_grey():
    from morphometry.cartilage.deformation_plot import NO_DATA_GREY, region_display_name
    from matplotlib.colors import to_rgba
    fig = plot_knee_deltas(_maps(), 'K')
    ref = _panel(fig, '30° unloaded')
    # codes 0 and 1 of the synthetic map are 'clt' and 'ilt'
    assert sorted(t.get_text() for t in ref.texts) == ['cLT', 'iLT']
    assert not _panel(fig, '60° unloaded').texts
    assert _panel(fig, '60° unloaded').get_facecolor() == to_rgba(NO_DATA_GREY)
    assert region_display_name('iclf') == 'icLF' and region_display_name('alf') == 'aLF'
    femur_ref = _panel(plot_knee_deltas(_maps('femur'), 'K'), '30° unloaded')
    assert sorted(t.get_text() for t in femur_ref.texts) == ['cc', 'ic']  # thirds shortened


def test_cohort_whole_bone_hotspot_is_a_ring_drawn_last_and_caption_present():
    rows = [dict(angle=60, load='belastet', bone='tibia', compartment=c, map_x_mm=1.0, map_y_mm=2.0)
            for c in ('both', 'left')]
    fig = plot_cohort_hotspots(rows, 'tibia', neighbourhood_mm=3.0, edge_margin_mm=2.0)
    ax = _panel(fig, '60° loaded')
    last = ax.collections[-1]
    assert len(last.get_facecolors()) == 0 or last.get_facecolors()[0][3] == 0  # hollow
    assert 'ONE KNEE' in _figure_text(fig)

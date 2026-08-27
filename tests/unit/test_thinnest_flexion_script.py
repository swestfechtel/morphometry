"""Unit tests for the pure helpers in scripts/test_thinnest_point_flexion.py.

Only the filename parsing and the articular-surface arc-length fraction are unit-tested
here (they are pure and data-independent); the end-to-end cohort analysis is exercised
against real segmentations when the script is run.
"""
import importlib.util
import pathlib

import numpy as np

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'analyse_thinnest_point_flexion.py'
_spec = importlib.util.spec_from_file_location('thinnest_flexion', _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_parse_filename():
    assert mod.parse_filename('K1_0_belastet.nii.gz') == ('K1', 0, 'belastet')
    assert mod.parse_filename('K12_90_unbelastet.nii.gz') == ('K12', 90, 'unbelastet')
    assert mod.parse_filename('K3_30_belastet.nii') == ('K3', 30, 'belastet')
    assert mod.parse_filename('not_a_segmentation.txt') is None
    assert mod.parse_filename('K1_0.nii.gz') is None  # missing load


class _Femur:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud


def test_articular_ap_fraction_anterior_and_posterior():
    """A straight anterior->posterior articular surface gives fraction = y-position along it."""
    # One sagittal column x=5; the tibia-facing surface runs y=0..10 at constant z (flat),
    # so arc length is proportional to y and the fraction equals y / y_max.
    pts = np.array([[5, y, 20] for y in range(11)], float)
    femur = _Femur(pts)
    spacing = (1.0, 1.0, 1.0)
    assert mod.articular_ap_fraction(femur, (5.0, 0.0), spacing) == 0.0    # anterior end
    assert mod.articular_ap_fraction(femur, (5.0, 10.0), spacing) == 1.0   # posterior end
    assert mod.articular_ap_fraction(femur, (5.0, 5.0), spacing) == 0.5    # middle


def test_articular_ap_fraction_degenerate_returns_nan():
    """Too few points in the column -> NaN (skipped downstream)."""
    femur = _Femur(np.array([[5, 3, 20]], float))
    assert np.isnan(mod.articular_ap_fraction(femur, (5.0, 3.0), (1.0, 1.0, 1.0)))

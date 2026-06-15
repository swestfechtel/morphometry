"""Characterization of hip measurements on a NaKo sample (isotropic; femur=1,
cartilage=2, acetabulum=3).

Locks current numeric behaviour ahead of the measurements refactor. ``center
edge angle`` is currently invoked via a matplotlib Axes because the default
``plot=False`` path has a pre-existing bug (``if plot is not None`` instead of
``is not False``); the rewrite fixes that, but calling with an Axes exercises the
same calculation and keeps this golden valid across the change.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import pytest

from morphometry.image_io import Segmentation
from morphometry.measurements import hip as H
from tests.conftest import assert_golden, split_left_right

pytestmark = pytest.mark.needs_nako


def test_acetabular_anteversion(nako_sample):
    assert_golden("acetabular_anteversion",
                  H.calculate_acetabular_anteversion(nako_sample, 1, 3, isotropic=True, ct=False))


def test_center_edge_angle(nako_sample):
    fig, ax = plt.subplots()
    try:
        cea = H.calculate_center_edge_angle(nako_sample, 1, 3, isotropic=True, project=True, plot=ax)
    finally:
        plt.close(fig)
    assert_golden("center_edge_angle", cea)


def test_ccd_left(nako_sample):
    left, _ = split_left_right(nako_sample)
    assert_golden("ccd_left", H.calculate_ccd(left, side="left", isotropic=True))


def test_anteversion_left(nako_sample):
    left, _ = split_left_right(nako_sample)
    assert_golden("anteversion_left", H.calculate_anteversion(left, side="left", isotropic=True))


def test_alpha_angle_left(nako_sample):
    left, _ = split_left_right(nako_sample)
    assert_golden("alpha_angle_left", H.calculate_alpha_angle(left.array, side="left", isotropic=True))


def test_acetabular_depth_left(nako_sample):
    left, _ = split_left_right(nako_sample)
    assert_golden("acetabular_depth_left",
                  H.calculate_acetabular_depth(left.array, side="left", isotropic=True))


def test_min_distance_left(nako_sample):
    left, _ = split_left_right(nako_sample)
    assert_golden("min_distance_left",
                  H.calculate_min_distance_between_femoral_head_and_acetabulum(
                      left.array, side="left", isotropic=True))


def test_subchondral_distance_left(nako_sample):
    left, _ = split_left_right(nako_sample)
    mean, std, mn, mx, *_ = H.calculate_subchondral_distance_ray_tracing(
        left, side="left", isotropic=True)
    assert_golden("subchondral_distance_left", (mean, std, mn, mx))


@pytest.mark.stochastic
def test_cartilage_thickness_knn(nako_sample):
    assert_golden("cartilage_thickness_knn",
                  H.calculate_cartilage_thickness_knn(nako_sample.array, cartilage_label=2))

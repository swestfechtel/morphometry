"""Characterization of the whole-leg CT (`*_ct`) measurements.

Uses a single NMDID whole-leg CT segmentation (both legs, labels femur=1,
tibia=2, fibula=3, patella=5, hip=7) split into image halves. These goldens
guard the CT code paths through the measurements refactor -- in particular
``calculate_ccd_ct`` exercises ``get_femoral_neck_center_ct``, so it protects the
planned MRI/CT neck-finding deduplication.

Skipped unless the CT sample is available (set ``MORPH_CT_SAMPLE`` to override).
"""
import pytest

from morphometry import hip as Hg
from morphometry.measurements import hip as MH
from morphometry import whole_leg as W
from morphometry.measurements.femur import calculate_femoral_torsion_ct
from tests.conftest import assert_golden, split_left_right

pytestmark = pytest.mark.needs_ct


@pytest.fixture
def ct_left(ct_whole_leg):
    """The left image half of the whole-leg CT segmentation."""
    left, _ = split_left_right(ct_whole_leg)
    return left


def test_femoral_head_center_ct_radius(ct_left):
    radius, _ = Hg.get_femoral_head_center_ct(ct_left, 1, "left")
    assert_golden("fhc_ct_radius_left", radius)


def test_ccd_ct(ct_left):
    assert_golden("ccd_ct_left", MH.calculate_ccd_ct(ct_left, side="left"))


def test_femoral_torsion_ct(ct_left):
    assert_golden("femoral_torsion_ct_left",
                  calculate_femoral_torsion_ct(ct_left, ct_left, side="left"))


def test_hip_knee_ankle_angle_ct(ct_left):
    assert_golden("hka_ct_left", W.calculate_hip_knee_ankle_angle(ct_left, side="left"))


def test_mechanical_axis_deviation_ct(ct_left):
    assert_golden("mad_ct_left", W.calculate_mechanical_axis_deviation(ct_left, side="left"))


def test_bone_length_ct(ct_left):
    assert_golden("femur_length_ct_left", W.calculate_bone_length_ct(ct_left, 1))
    assert_golden("tibia_length_ct_left", W.calculate_bone_length_ct(ct_left, 2))


def test_subchondral_distance_ct(ct_left):
    mean, std, mn, mx, *_ = MH.calculate_subchondral_distance_ray_tracing_ct(ct_left, side="left")
    assert_golden("subchondral_distance_ct_left", (mean, std, mn, mx))

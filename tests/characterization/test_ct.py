"""Characterization of the whole-leg CT (`*_ct`) measurements.

Uses a single NMDID whole-leg CT segmentation (both legs, labels femur=1,
tibia=2, fibula=3, patella=5, hip=7) split into image halves. These goldens
guard the CT code paths through the measurements refactor -- in particular
``calculate_ccd_ct`` exercises ``get_femoral_neck_center_ct``, so it protects the
planned MRI/CT neck-finding deduplication.

The ``hka_ct_left`` / ``mad_ct_left`` goldens were re-captured when ``split_ct_image``
started giving its crops the correct affine: the knee and ankle landmarks had been
converted to physical points through the whole-leg affine, which placed them hundreds
of millimetres off (above the femoral head), so the old values were not measuring a
leg axis at all. Note that ``calculate_hip_knee_ankle_angle`` returns the *deviation*
from a straight axis; the clinical hip-knee-ankle angle is 180 minus this value.

Skipped unless the CT sample is available (set ``MORPH_CT_SAMPLE`` to override).
"""
import pytest

from morphometry import hip as Hg
from morphometry.image_io import split_ct_image
from morphometry.measurements import hip as MH
from morphometry.measurements import whole_leg as W
from morphometry.measurements.femur import calculate_femoral_torsion_ct
from morphometry.measurements.tibia import calculate_tibial_torsion
from tests.conftest import assert_golden, split_left_right

pytestmark = pytest.mark.needs_ct


@pytest.fixture
def ct_left(ct_whole_leg):
    """The left image half of the whole-leg CT segmentation."""
    left, _ = split_left_right(ct_whole_leg)
    return left


@pytest.fixture
def ct_right(ct_whole_leg):
    """The right image half of the whole-leg CT segmentation."""
    _, right = split_left_right(ct_whole_leg)
    return right


def test_femoral_head_center_ct_radius(ct_left):
    radius, _ = Hg.get_femoral_head_center_ct(ct_left, 1, "left")
    assert_golden("fhc_ct_radius_left", radius)


def test_ccd_ct(ct_left):
    assert_golden("ccd_ct_left", MH.calculate_ccd_ct(ct_left, side="left"))


def test_ccd_ct_right(ct_right):
    """Pin the CT CCD of the image-right half, measured as the reflection of a left one.

    The right half runs the left-side landmark code on the reflected leg, so this
    records the side that the equivariance probes only prove self-consistent.
    """
    assert_golden("ccd_ct_right", MH.calculate_ccd_ct(ct_right, side="right"))


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


def _ct_tibial_torsion(leg, side: str) -> float:
    """Tibial torsion from a whole-leg CT half, via the knee and ankle sub-volumes.

    :param leg: One image half of a whole-leg CT segmentation.
    :param side: The image side of ``leg``, 'left' or 'right'.
    :return: The tibial torsion angle in degrees.
    """
    _, masks = split_ct_image(leg, None, femur_label=1, hip_label=7, patella_label=5, fibula_label=3)
    return calculate_tibial_torsion(masks[1].array, masks[2].array, tibia_label_knee=2,
                                    tibia_label_ankle=2, fibula_label=3, side=side)


def test_tibial_torsion_ct_left(ct_left):
    """Pin the CT tibial torsion of the image-left half (no golden covered CT tibial torsion)."""
    assert_golden("tibial_torsion_ct_left", _ct_tibial_torsion(ct_left, "left"))


def test_tibial_torsion_ct_right(ct_right):
    """Pin the CT tibial torsion of the image-right half.

    Unlike ``tibial_torsion_right`` (Augsburg PA000001), this fixture's value moved
    when the right side became mirror-equivariant, so it is the first golden that
    actually records the right-side tibial torsion rather than a leg whose condylar
    vector happens to survive the mirror unchanged.
    """
    assert_golden("tibial_torsion_ct_right", _ct_tibial_torsion(ct_right, "right"))

"""Unit tests for scripts/subchondral_nako.py.

Two things in that script are easy to get wrong and expensive to notice later:

* it splits a both-hips mask and reports coordinates in the **whole**-image frame, so the
  right half's affine has to be shifted or every right-hip coordinate is off by half the
  sagittal extent -- 192 mm on the NaKo cohort, silently, with distances still correct;
* it calls the module-private `_subchondral_distance_ray_tracing` directly rather than the
  public wrapper, to avoid resolving the femoral head centre twice (which it must report,
  and which the wrapper does not return). That shortcut is only legitimate while it
  reproduces the wrapper exactly.
"""
import importlib.util
import pathlib

import nibabel as nib
import numpy as np
import pytest

from morphometry.image_io import Segmentation
from morphometry.measurements.hip import calculate_subchondral_distance_ray_tracing

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'subchondral_nako.py'
_spec = importlib.util.spec_from_file_location('subchondral_nako', _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _both_hips(shape=(96, 64, 64), head_radius=10, gap=3):
    """Build a both-hips mask: a femoral head per side under a concentric acetabular cup.

    :param shape: The array shape; axis 0 is split in half by the script.
    :param head_radius: The femoral head radius in voxels.
    :param gap: The joint space in voxels between head surface and cup.
    :return: A Segmentation in an LPI-like frame (femur=1, acetabulum=3).
    """
    array = np.zeros(shape, dtype=np.float64)
    half = shape[0] // 2
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1)

    for centre in ((half // 2, shape[1] // 2, shape[2] // 2),
                   (half + half // 2, shape[1] // 2, shape[2] // 2)):
        offset = grid - np.asarray(centre)
        distance = np.linalg.norm(offset, axis=-1)
        array[distance <= head_radius] = 1
        # the cup sits above the head (LPI: smaller axis-2 index is more superior)
        cup = (distance >= head_radius + gap) & (distance <= head_radius + gap + 4)
        array[cup & (offset[..., 2] <= 0)] = 3

    affine = np.diag([-1.0, -1.0, -1.0, 1.0])
    affine[:3, 3] = [40.0, 30.0, -20.0]
    return Segmentation.from_nibabel(nib.Nifti1Image(array, affine))


@pytest.mark.parametrize('side', ['left', 'right'])
def test_image_half_keeps_the_whole_image_physical_frame(side):
    """A voxel has the same physical location addressed through the half or the whole."""
    seg = _both_hips()
    half = seg.array.shape[0] // 2
    hip = mod.image_half(seg, side)

    offset = half if side == 'right' else 0
    for index in ([0, 0, 0], [5, 7, 11], [half - 1, 20, 30]):
        index = np.asarray(index, dtype=float)
        assert hip.transform_index_to_physical_point(index) == pytest.approx(
            seg.transform_index_to_physical_point(index + np.array([offset, 0, 0])))


def test_right_half_would_be_offset_without_the_shift():
    """Guard the bug being prevented: the naive half is wrong by exactly half the extent."""
    seg = _both_hips()
    half = seg.array.shape[0] // 2
    naive = Segmentation.from_nibabel(nib.Nifti1Image(seg.array[half:], seg.affine, seg.header))
    shifted = mod.image_half(seg, 'right')

    delta = (np.asarray(shifted.transform_index_to_physical_point([0., 0., 0.]))
             - np.asarray(naive.transform_index_to_physical_point([0., 0., 0.])))
    # axis 0 has a -1 mm spacing in this fixture, so the shift is -half millimetres
    assert delta == pytest.approx([-half, 0, 0])


@pytest.mark.parametrize('side', ['left', 'right'])
def test_direct_core_call_matches_the_public_wrapper(side):
    """The script's shortcut past the public wrapper must not change a single ray."""
    seg = _both_hips()
    hip = mod.image_half(seg, side)

    expected = calculate_subchondral_distance_ray_tracing(
        hip, side=side, femur_label=mod.FEMUR_LABEL, acetabulum_label=mod.ACETABULUM_LABEL,
        isotropic=True, n_rays=mod.N_RAYS, cone_angle=mod.CONE_ANGLE, min_ray_fraction=0.)

    from morphometry.hip import get_femoral_head_center
    radius, fhc = get_femoral_head_center(hip.array, side=side,
                                          segmentation_label=mod.FEMUR_LABEL, isotropic=True)
    actual = mod._subchondral_distance_ray_tracing(
        hip, fhc, radius,
        np.where(hip.array == mod.FEMUR_LABEL, 1, 0),
        np.where(hip.array == mod.ACETABULUM_LABEL, 1, 0),
        n_rays=mod.N_RAYS, cone_angle=mod.CONE_ANGLE, min_ray_fraction=0.)

    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want)


def _measure_synthetic(gap=3, head_radius=10, side='left'):
    """Run the script's measurement over a synthetic hip with a known joint space.

    :param gap: The joint space in voxels between head surface and cup.
    :param head_radius: The femoral head radius in voxels.
    :param side: The image side to measure.
    :return: ``(radius, fhc, hip, distances, exits)``.
    """
    from morphometry.hip import get_femoral_head_center
    seg = _both_hips(head_radius=head_radius, gap=gap)
    hip = mod.image_half(seg, side)
    radius, fhc = get_femoral_head_center(hip.array, side=side,
                                          segmentation_label=mod.FEMUR_LABEL, isotropic=True)
    _, _, _, _, distances, exits, _ = mod._subchondral_distance_ray_tracing(
        hip, fhc, radius,
        np.where(hip.array == mod.FEMUR_LABEL, 1, 0),
        np.where(hip.array == mod.ACETABULUM_LABEL, 1, 0),
        n_rays=mod.N_RAYS, cone_angle=mod.CONE_ANGLE, min_ray_fraction=0.)
    return radius, np.asarray(fhc), hip, distances, exits


@pytest.mark.parametrize('gap', [2, 3, 5])
def test_shortest_ray_recovers_the_synthetic_joint_space(gap):
    """End to end on a known geometry, via the invariant the method actually guarantees.

    Not the mean: rays are cast over a 45-degree half-angle cone, and an off-axis ray
    crosses the two concentric shells obliquely, so its chord exceeds the radial gap (3.85
    mm near the axis against 4.44 at 35-50 degrees on this fixture). The *shortest* ray is
    the well-aligned one, and it recovers the gap we built.
    """
    _, _, _, distances, _ = _measure_synthetic(gap=gap)
    assert len(distances) > 0
    assert distances.min() == pytest.approx(gap, abs=0.5)


def test_off_axis_rays_are_longer_than_near_axis_rays():
    """Pin the obliquity above, so a regression cannot quietly reintroduce it as bias."""
    radius, fhc, hip, distances, exits = _measure_synthetic()
    exit_index = np.array([hip.transform_physical_point_to_index(e) for e in exits])
    offset = exit_index - fhc
    # the cone is aimed at world +z, which is -k in this LPI-like fixture
    angle = np.degrees(np.arccos(np.clip(-offset[:, 2] / np.linalg.norm(offset, axis=1), -1, 1)))

    near, far = distances[angle < 15], distances[angle >= 35]
    assert len(near) and len(far)
    assert near.mean() < far.mean()


def test_exit_points_lie_on_the_femoral_head_surface():
    """Every ray leaves the head one radius out from the centre it was cast from."""
    radius, fhc, hip, _, exits = _measure_synthetic()
    radii = np.linalg.norm(np.array([hip.transform_physical_point_to_index(e) for e in exits])
                           - fhc, axis=1)
    assert radii.mean() == pytest.approx(radius, abs=1.5)

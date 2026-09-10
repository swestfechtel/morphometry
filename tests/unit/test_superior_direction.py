"""Unit tests for the subchondral ray-tracing cone axis and its ray-attrition warning.

``_superior_direction`` is a world-space constant rather than an affine-derived
vector, so the point of the first half of these tests is the *invariance*: the same
superior direction must come back no matter how the volume happens to be oriented on
disk. An affine-derived implementation would return an anterior vector for the
as-stored (L, I, P) case and an inverted one for LPS/RAS, so those cases pin the
constant against a regression back to index-axis stepping.

The second half covers ``min_ray_fraction``: with a fixed superior axis the cone is
no longer aimed at the label, so rays can clear the lateral acetabular rim and be
discarded. That is logged, never raised, and only in ``'superior'`` mode. All of it
runs on synthetic masks so it stays in the fast (no sample data) loop.
"""
import logging

import nibabel as nib
import numpy as np
import pytest

from morphometry.image_io import Segmentation
from morphometry.measurements import hip as hip_measurements
from morphometry.measurements.hip import _subchondral_distance_ray_tracing, _superior_direction

# axcodes -> the affine that maps index space to nibabel's RAS+ world space. Each
# column is the world direction of one index axis, e.g. 'L' means index axis 0 walks
# toward Left, which is -x in RAS+.
AXIS_TO_RAS = {"R": (0, 1.0), "L": (0, -1.0), "A": (1, 1.0), "P": (1, -1.0),
               "S": (2, 1.0), "I": (2, -1.0)}

# radius of the synthetic femoral head, in voxels and (on the 1 mm grids used here) mm
FEMUR_RADIUS = 10.0


def _affine(axcodes: str, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Build a RAS+ affine whose index axes run along the given anatomical directions.

    :param axcodes: Three axis codes (e.g. ``'LPI'``) giving the direction each index
        axis increases toward.
    :param spacing: Voxel spacing along each index axis, in mm.
    :return: A (4, 4) index->world affine.
    """
    affine = np.zeros((4, 4))
    affine[3, 3] = 1.0
    for index_axis, code in enumerate(axcodes):
        world_axis, sign = AXIS_TO_RAS[code]
        affine[world_axis, index_axis] = sign * spacing[index_axis]
    return affine


def _segmentation(axcodes: str, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Segmentation:
    """A tiny Segmentation carrying the affine for ``axcodes`` (content is irrelevant)."""
    array = np.zeros((4, 5, 6), dtype=np.uint8)
    return Segmentation.from_nibabel(nib.Nifti1Image(array, _affine(axcodes, spacing)))


def test_superior_direction_is_world_z_unit_vector():
    direction = _superior_direction()
    np.testing.assert_allclose(direction, [0.0, 0.0, 1.0])
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def _index_axis_2_superior(image: Segmentation) -> np.ndarray:
    """The superseded, affine-derived superior direction: negated index-axis-2 step.

    Kept in the tests only as the counter-example the constant replaced, so the
    reason for the constant is executable rather than a docstring claim.
    :param image: A Segmentation providing the index<->physical transform.
    :return: A (3,) unit vector: physical direction of decreasing index axis 2.
    """
    p0 = np.asarray(image.transform_index_to_physical_point([0, 0, 0]), dtype=float)
    p1 = np.asarray(image.transform_index_to_physical_point([0, 0, 1]), dtype=float)
    direction = -(p1 - p0)
    return direction / np.linalg.norm(direction)


def test_measurement_is_orientation_invariant():
    """End to end: the same physical hip must measure the same in any orientation.

    The synthetic hip is defined in *world* coordinates and then sampled onto four
    differently-oriented grids, so all four volumes hold the identical anatomy and only
    the index order and flips differ. A correct superior axis therefore has to yield the
    same rays and the same distances. This is the property the constant exists to
    guarantee; an affine-derived axis would aim inferiorly on LPS/RAS and anteriorly on
    the as-stored LIP volume, finding no acetabulum at all.
    """
    results = {codes: _oriented_hip_trace(codes) for codes in ("LPI", "LPS", "RAS", "LIP")}

    reference = results["LPI"]
    assert len(reference) > 0
    for codes, distances in results.items():
        assert len(distances) == len(reference), f"{codes} lost rays the LPI grid kept"
        np.testing.assert_allclose(np.sort(distances), np.sort(reference), atol=0,
                                   err_msg=f"{codes} measured a different hip than LPI")


def _oriented_hip_trace(axcodes: str, cap_angle: float = 40.0,
                        shape: tuple[int, int, int] = (60, 60, 60)) -> np.ndarray:
    """Measure one synthetic hip sampled onto a grid of the given orientation.

    The head and the acetabular cap are defined by world-space geometry (the cap is
    centred on world +z, i.e. Superior) and then rasterised through each orientation's
    affine, so every grid holds the same anatomy at a different index order. The volume
    is centred on the world origin so it stays in bounds whichever way the axes point.
    :param axcodes: Three axis codes giving the direction each index axis increases toward.
    :param cap_angle: Angular half-width of the acetabular cap, in degrees.
    :param shape: Volume shape in voxels.
    :return: The per-ray subchondral distances, in mm.
    """
    affine = _affine(axcodes)
    rotation = affine[:3, :3]
    affine[:3, 3] = -rotation @ (np.array(shape, dtype=float) / 2.0)  # volume centre at origin

    indices = np.indices(shape).reshape(3, -1).T.astype(float)
    world = indices @ rotation.T + affine[:3, 3]
    fhc_world = np.array([0.0, 0.0, -10.0])  # a little inferior of centre, cap above it
    offsets = world - fhc_world
    radii = np.linalg.norm(offsets, axis=1)
    cosine = np.where(radii > 0, offsets[:, 2] / np.maximum(radii, 1e-9), 0.0)  # about world +z
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    femur = (radii <= FEMUR_RADIUS).reshape(shape).astype(np.uint8)
    acetabulum = ((radii >= 12.0) & (radii <= 15.0)
                  & (angles <= cap_angle)).reshape(shape).astype(np.uint8)
    image = Segmentation.from_nibabel(
        nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine))
    fhc_index = (np.linalg.inv(affine) @ np.append(fhc_world, 1.0))[:3]

    return _subchondral_distance_ray_tracing(image, fhc_index, FEMUR_RADIUS, femur,
                                             acetabulum, n_rays=200,
                                             acetabulum_radius_factor=2.0,
                                             cone_axis="superior")[4]


@pytest.mark.parametrize("axcodes,agrees", [
    ("LPI", True),    # index axis 2 runs inferiorly, so the two formulations coincide
    ("LPS", False),   # index axis 2 runs superiorly -> inverted
    ("RAS", False),   # likewise inverted
    ("LIP", False),   # index axis 2 runs posteriorly -> an anterior vector
])
def test_constant_beats_index_axis_derivation_off_lpi(axcodes, agrees):
    """Pin *why* the constant is used: index-axis stepping is right only on LPI.

    This is the test with teeth. It fails if anyone reintroduces the affine-derived
    formulation, which returns a plausible but wrong direction — anterior on
    as-stored data, exactly inverted on an LPS or RAS image — with no error raised.
    """
    derived = _index_axis_2_superior(_segmentation(axcodes))
    assert bool(np.allclose(derived, _superior_direction())) is agrees


def test_superior_direction_ignores_anisotropic_spacing():
    """Spacing cancels in a normalised direction, so a thick-slice grid is unaffected."""
    _segmentation("LPI", spacing=(0.5, 0.5, 3.0))
    np.testing.assert_allclose(_superior_direction(), [0.0, 0.0, 1.0])


def test_unknown_cone_axis_raises():
    """A typo'd mode must fail loudly rather than silently falling back to a default."""
    image = _segmentation("LPI")
    empty = np.zeros(image.array.shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="cone_axis"):
        _subchondral_distance_ray_tracing(image, np.zeros(3), 1.0, empty, empty,
                                          cone_axis="sideways")


# --- ray attrition warning ------------------------------------------------------------
#
# Synthetic geometry, so these run in the fast loop without the NaKo sample: a spherical
# "femoral head" with a spherical-shell "acetabulum" cap above it, whose angular half-width
# is the knob. Rays outside the cap clear its rim and are dropped exactly as they do at a
# real lateral acetabular rim, so the surviving fraction is (1-cos cap)/(1-cos cone_angle).

FHC_INDEX = np.array([30.0, 30.0, 40.0])


def _synthetic_hip(cap_angle: float, shape: tuple[int, int, int] = (60, 60, 60)):
    """A spherical femoral head under an acetabular shell cap of the given half-angle.

    :param cap_angle: Angular half-width of the acetabular cap about the superior
        direction, in degrees. Below the measurement's cone half-angle, rays near the
        cone's edge miss the cap and are discarded.
    :param shape: Volume shape in voxels.
    :return: A tuple ``(image, femur_mask, acetabulum_mask)`` on an LPI 1 mm grid.
    """
    offsets = np.indices(shape).reshape(3, -1).T.astype(float) - FHC_INDEX
    radii = np.linalg.norm(offsets, axis=1)
    # superior is world +z; on an LPI grid that is the direction of decreasing index axis 2
    cosine = np.where(radii > 0, -offsets[:, 2] / np.maximum(radii, 1e-9), 0.0)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    femur = (radii <= FEMUR_RADIUS).reshape(shape).astype(np.uint8)
    acetabulum = ((radii >= 12.0) & (radii <= 15.0) & (angles <= cap_angle)).reshape(shape)
    image = Segmentation.from_nibabel(
        nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), _affine("LPI")))
    return image, femur, acetabulum.astype(np.uint8)


def _trace(cap_angle: float, **kwargs):
    """Run the ray tracer over the synthetic hip, with the peri-acetabular crop widened."""
    image, femur, acetabulum = _synthetic_hip(cap_angle)
    kwargs.setdefault("cone_axis", "superior")
    return _subchondral_distance_ray_tracing(image, FHC_INDEX, FEMUR_RADIUS, femur,
                                             acetabulum, n_rays=200,
                                             acetabulum_radius_factor=2.0, **kwargs)


def _warnings(caplog) -> list[str]:
    """The WARNING-level messages captured from the hip measurement logger."""
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "morphometry.measurements.hip"]


def test_heavy_ray_loss_warns_once_and_still_returns(caplog):
    """A narrow cap loses half the rays: exactly one WARNING, and a normal result."""
    with caplog.at_level(logging.WARNING, logger="morphometry.measurements.hip"):
        result = _trace(30.0)

    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "99 of 200 rays" in messages[0]        # (1-cos30)/(1-cos45) ~ 46%
    assert "70%" in messages[0]                    # the configured threshold
    assert "lateral acetabular rim" in messages[0]  # the actionable cause

    mean, std, mn, mx, distances, exits, hits = result  # never raises, arity unchanged
    assert len(distances) == 99
    assert mean == pytest.approx(np.mean(distances))
    assert exits.shape == hits.shape == (99, 3)


def test_moderate_ray_loss_does_not_warn(caplog):
    """Rays are lost but stay above the threshold, so nothing is logged."""
    with caplog.at_level(logging.WARNING, logger="morphometry.measurements.hip"):
        _, _, _, _, distances, _, _ = _trace(40.0)

    assert 0.7 * 200 < len(distances) < 200  # genuinely lossy, just not enough to warn
    assert _warnings(caplog) == []


def test_min_ray_fraction_zero_disables_the_warning(caplog):
    """The same heavy loss is silent once the threshold is switched off."""
    with caplog.at_level(logging.WARNING, logger="morphometry.measurements.hip"):
        _trace(30.0, min_ray_fraction=0.0)

    assert _warnings(caplog) == []


def test_acetabulum_axis_never_warns(caplog):
    """The legacy axis stays silent: attrition is only meaningful for a fixed axis.

    The 'acetabulum' axis is built by aiming at the label's own centroid, so a low
    surviving-ray count there says nothing about acetabular coverage and the 0.7
    threshold is not calibrated for it. The guard must stay scoped to 'superior'.
    """
    with caplog.at_level(logging.WARNING, logger="morphometry.measurements.hip"):
        _, _, _, _, distances, _, _ = _trace(30.0, cone_axis="acetabulum")

    assert len(distances) < 0.7 * 200  # the loss that would have warned in superior mode
    assert _warnings(caplog) == []


def test_public_wrapper_forwards_min_ray_fraction(monkeypatch):
    """The threshold reaches the tracer from the public API, not just the private one."""
    captured = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return (0.0, 0.0, 0.0, 0.0, np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3)))

    monkeypatch.setattr(hip_measurements, "_subchondral_distance_ray_tracing", _spy)
    monkeypatch.setattr(hip_measurements, "get_femoral_head_center",
                        lambda *a, **k: (FEMUR_RADIUS, FHC_INDEX))
    image, femur, _ = _synthetic_hip(30.0)
    image = Segmentation.from_nibabel(nib.Nifti1Image(femur, _affine("LPI")))

    hip_measurements.calculate_subchondral_distance_ray_tracing(
        image, cone_axis="superior", min_ray_fraction=0.42)

    assert captured["cone_axis"] == "superior"
    assert captured["min_ray_fraction"] == 0.42

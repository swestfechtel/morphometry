"""Unit tests for extracting the acetabular articular surface from a whole hip bone.

Whole-leg CT label maps carry one label for the entire hip bone. The rim searches were
written for masks that segment the acetabulum on its own, so on a hip bone they find the
iliac wing instead. ``get_acetabular_surface`` narrows the bone to the part that faces the
femoral head by keeping, per ray out of the head centre, only the first bone voxel.
"""
import numpy as np
import pytest

from morphometry.hip import get_acetabular_surface


def _hip_bone(shape=(80, 80, 80), center=(40, 40, 40), cup_radius=12, wing_offset=22):
    """Build a hip bone: a cup hugging the femoral head plus an iliac wing behind it.

    The wing is placed further out along the same superolateral directions as the cup, so
    a search that ignores which surface faces the head will prefer the wing (it is more
    lateral) and the cup is the only correct answer.
    :param shape: The array shape.
    :param center: The femoral head centre in index coordinates.
    :param cup_radius: The cup's inner radius in voxels.
    :param wing_offset: How far beyond the cup the wing sits, in voxels.
    :return: ``(mask, centre)`` -- the binary hip bone mask and the head centre.
    """
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1)
    offset = grid - np.asarray(center)
    distance = np.linalg.norm(offset, axis=-1)

    # the cup covers everything except the inferolateral opening (-x, +z)
    opening = (offset[..., 0] < 0) & (offset[..., 2] > 0)
    cup = (distance >= cup_radius) & (distance <= cup_radius + 3) & ~opening
    wing = (distance >= cup_radius + wing_offset) & (distance <= cup_radius + wing_offset + 3)
    return (cup | wing).astype(np.uint8), np.asarray(center)


def test_surface_keeps_the_cup_and_drops_the_wing():
    """Only the bone facing the head survives; the iliac wing does not."""
    mask, center = _hip_bone()
    surface = get_acetabular_surface(mask, center, radius=12, spacing=np.ones(3), radius_factor=1.4)

    kept = np.argwhere(surface)
    assert len(kept) > 0
    distances = np.linalg.norm(kept - center, axis=1)
    # every kept voxel is on the cup (radius 12-15), none on the wing (radius 34-37)
    assert distances.max() < 20


def test_the_wing_is_shadowed_where_the_cup_covers_it():
    """Widen the band past the wing: it still only appears through the cup's opening.

    A ray that the cup intercepts stops there, so behind the cup the wing is invisible no
    matter how far the search reaches. Only the rays that leave through the acetabular
    opening travel far enough to find it -- which is what the band exists to exclude.
    """
    mask, center = _hip_bone()
    surface = get_acetabular_surface(mask, center, radius=12, spacing=np.ones(3), radius_factor=4.)

    kept = np.argwhere(surface) - center
    far = np.linalg.norm(kept, axis=1) > 20  # only the wing is this far out
    assert far.any(), 'the wide band should reach the wing through the opening'

    # Deep inside the quadrant the cup covers, the wing is essentially never the first
    # hit. Not *exactly* never: a discrete ray grid over a voxelised shell always leaves a
    # few rays grazing the quadrant boundary, so the property is a small fraction rather
    # than zero. It is the magnitude that matters -- binning finer than the default pushes
    # this above 15%, which is the failure the default guards against.
    covered = (kept[:, 0] > 3) & (kept[:, 2] < -3)
    leaked = (far & covered).sum() / max(covered.sum(), 1)
    assert leaked < 0.02, f'the wing showed through the cup for {leaked:.1%} of it'


def test_surface_stops_at_the_acetabular_opening():
    """Rays leaving through the cup's opening find no cup, so the patch ends at the rim."""
    mask, center = _hip_bone()
    surface = get_acetabular_surface(mask, center, radius=12, spacing=np.ones(3), radius_factor=1.4)

    kept = np.argwhere(surface) - center
    through_opening = (kept[:, 0] < -3) & (kept[:, 2] > 3)
    assert not through_opening.any()


def test_search_band_bounds_the_surface():
    """Nothing beyond ``radius_factor`` head radii is returned."""
    mask, center = _hip_bone()
    surface = get_acetabular_surface(mask, center, radius=12, spacing=np.ones(3), radius_factor=1.1)

    kept = np.argwhere(surface)
    assert np.linalg.norm(kept - center, axis=1).max() <= 1.1 * 12


def test_anisotropic_spacing_is_measured_in_millimetres():
    """The search band is a physical distance, so a coarse axis reaches fewer voxels."""
    mask, center = _hip_bone()
    isotropic = get_acetabular_surface(mask, center, radius=12, spacing=np.ones(3), radius_factor=1.4)
    anisotropic = get_acetabular_surface(mask, center, radius=12, spacing=np.array([1., 1., 3.]),
                                         radius_factor=1.4)

    # with 3 mm slices the same millimetre band spans a third of the voxels along z
    assert np.ptp(np.argwhere(anisotropic)[:, 2]) < np.ptp(np.argwhere(isotropic)[:, 2])


def test_empty_mask_returns_empty_surface():
    """A missing hip bone yields an empty surface rather than raising."""
    surface = get_acetabular_surface(np.zeros((10, 10, 10), dtype=np.uint8), np.array([5, 5, 5]),
                                     radius=3, spacing=np.ones(3))
    assert not surface.any()


@pytest.mark.parametrize('resolution', [2., 5., 10.])
def test_one_voxel_per_ray(resolution):
    """Each angular bin contributes at most one voxel, so a coarser grid keeps fewer."""
    mask, center = _hip_bone()
    counts = [np.count_nonzero(get_acetabular_surface(mask, center, radius=12, spacing=np.ones(3),
                                                      radius_factor=4., angular_resolution=r))
              for r in (resolution, resolution * 2)]
    assert counts[0] > counts[1]

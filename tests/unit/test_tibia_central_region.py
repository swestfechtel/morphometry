"""Unit tests for the central tibial subregion radius (``calculate_ellipse``).

The central subregion (cLT / cRT) of each tibial plate is a sphere around the plate
centre that should hold ~20% of the plate's cartilage voxels. The former iterative
search halved its radius on the step that first reached 20% and so returned a constant
10 voxels on real knees (6-9% of the plate). These tests pin the quantile radius on
synthetic point clouds, where the expected share is known exactly.
"""
import numpy as np
import pytest

from morphometry.cartilage.knee import calculate_ellipse


def _plate(n_x: int, n_y: int, n_z: int = 4) -> np.ndarray:
    """
    Build a box-shaped synthetic plate point cloud on the integer voxel grid.

    Integer grids produce many tied distances, which is the realistic case for voxel
    point clouds and exercises the tie handling.

    :param n_x: Extent along axis 0 (voxels).
    :param n_y: Extent along axis 1 (voxels).
    :param n_z: Extent along axis 2 (voxels).
    :return: An ``(n_x * n_y * n_z, 3)`` float array of voxel indices.
    """
    grid = np.mgrid[0:n_x, 0:n_y, 0:n_z].reshape(3, -1).T
    return grid.astype(float)


def _share_inside(points: np.ndarray, center: np.ndarray, radius: float) -> int:
    """
    Count points strictly inside ``radius``, exactly as ``Tibia.classify_point`` tests.

    :param points: The point cloud.
    :param center: The sphere centre.
    :param radius: The sphere radius.
    :return: The number of points with ``norm(point - center) < radius``.
    """
    return int(np.sum(np.linalg.norm(points - center, axis=1) < radius))


def test_continuous_cloud_puts_exactly_fraction_inside():
    """Without ties, exactly ``int(N * fraction)`` points lie strictly inside."""
    rng = np.random.default_rng(0)
    points = rng.normal(size=(1001, 3)) * [10, 6, 2]
    center = points.mean(axis=0)
    radius = calculate_ellipse(points, center)
    assert _share_inside(points, center, radius) == int(len(points) * 0.2)


@pytest.mark.parametrize('fraction', [0.1, 0.2, 0.35, 0.5])
def test_voxel_grid_share_close_to_fraction(fraction):
    """On a voxel grid (many ties) the share is at most ``fraction`` and close to it."""
    points = _plate(40, 30)
    center = points.mean(axis=0)
    radius = calculate_ellipse(points, center, fraction=fraction)
    inside = _share_inside(points, center, radius)
    target = int(len(points) * fraction)
    assert inside <= target
    # The only shortfall is the tie group at the returned distance; on this grid a tie
    # group is small relative to the plate.
    assert inside / len(points) == pytest.approx(fraction, abs=0.02)


def test_radius_is_not_constant():
    """Plates of different size get different radii (the old bug returned 10.0 for both)."""
    small, large = _plate(30, 20), _plate(80, 60)
    r_small = calculate_ellipse(small, small.mean(axis=0))
    r_large = calculate_ellipse(large, large.mean(axis=0))
    assert r_large > r_small


def test_fraction_is_monotonic():
    """A larger fraction gives a larger radius."""
    points = _plate(40, 30)
    center = points.mean(axis=0)
    radii = [calculate_ellipse(points, center, fraction=f) for f in (0.1, 0.2, 0.4, 0.8)]
    assert radii == sorted(radii) and len(set(radii)) == len(radii)


def test_edge_cases():
    """Empty clouds and the 0 / 1 fractions are handled without errors."""
    points = _plate(10, 10)
    center = points.mean(axis=0)
    assert calculate_ellipse(np.empty((0, 3)), np.zeros(3)) == 0.0
    assert _share_inside(points, center, calculate_ellipse(points, center, fraction=0.0)) == 0
    assert _share_inside(points, center, calculate_ellipse(points, center, fraction=1.0)) == len(points)

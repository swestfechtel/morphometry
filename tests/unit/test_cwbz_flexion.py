"""Unit tests for the femoral CWBZ superior-inferior contact-cluster restriction.

At high knee flexion the femoral cartilage folds back on itself, so a single
anterior-posterior window (used to select the central weight-bearing zone) intersects
the cartilage at two superior-inferior levels: the true weight-bearing surface (near the
tibia) and the anterior trochlea folded into the same A-P range. ``_restrict_to_contact_cluster``
keeps only the S-I cluster nearest the tibial contact level, dropping the fold. These
tests exercise that helper directly with synthetic point sets (the correct answer is
known), without needing private flexion data.
"""
import numpy as np

from morphometry.cartilage.knee import _restrict_to_contact_cluster


def _blob(x0, y0, z0, n=200, spread=3.0, seed=0):
    """Build a small Nx3 cluster of integer voxel coordinates around ``(x0, y0, z0)``."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.normal(x0, spread, n),
        rng.normal(y0, spread, n),
        rng.normal(z0, spread, n),
    ])
    return np.round(pts)


def test_drops_far_superior_fold():
    """Two S-I clusters: the far-superior fold is dropped, the near-tibia one kept."""
    # High axis2 = inferior (near the tibia); low axis2 = superior (the folded trochlea).
    weight_bearing = _blob(80, 280, 250, n=400, seed=1)   # near the tibia contact
    anterior_fold = _blob(80, 280, 50, n=150, seed=2)     # far superior fold
    points = np.vstack([weight_bearing, anterior_fold])
    ref_si = 270.0  # tibial contact level, just inferior to the weight-bearing cluster

    kept = _restrict_to_contact_cluster(points, ref_si)

    assert len(kept) == len(weight_bearing)
    assert kept[:, 2].min() > 150, 'the far-superior fold should have been dropped'
    assert kept[:, 2].max() <= points[:, 2].max()


def test_keeps_single_cluster_unchanged():
    """A single S-I cluster (low flexion) is returned unchanged."""
    points = _blob(80, 300, 240, n=500, seed=3)
    ref_si = 250.0
    kept = _restrict_to_contact_cluster(points, ref_si)
    assert len(kept) == len(points)


def test_handles_degenerate_inputs():
    """Empty, single-point and all-same-level inputs are handled without error."""
    assert len(_restrict_to_contact_cluster(np.empty((0, 3)), 100.0)) == 0
    single = np.array([[1.0, 2.0, 3.0]])
    assert len(_restrict_to_contact_cluster(single, 3.0)) == 1
    flat = np.column_stack([np.arange(10), np.arange(10), np.full(10, 5.0)])
    assert len(_restrict_to_contact_cluster(flat, 5.0)) == 10

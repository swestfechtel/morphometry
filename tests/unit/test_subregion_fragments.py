"""Unit tests for subregion contiguity cleanup (``_remove_subregion_fragments``).

Subregion assignment (``Tibia.classify_point`` / the femoral zone splits) can bucket a
few edge voxels into a subregion where they sit spatially disconnected from its main
patch. A subregion is one anatomical patch, so small disconnected fragments are dropped.
"""
import numpy as np

from morphometry.cartilage.knee import _remove_subregion_fragments


def test_drops_small_disconnected_fragment():
    """A small detached fragment is removed; the main patch is kept."""
    main = np.argwhere(np.ones((20, 20, 6))).astype(float)          # 2400-voxel patch
    fragment = np.array([[40, 40, 0], [40, 41, 0], [41, 40, 0]], float)  # tiny, detached
    points = np.vstack([main, fragment])
    cleaned = _remove_subregion_fragments(points, ratio=0.2)
    assert len(cleaned) == len(main)
    assert not np.any((cleaned[:, 0] >= 40))                        # fragment gone


def test_keeps_genuine_two_lobed_subregion():
    """Two comparably-sized disconnected lobes are both retained (ratio not exceeded)."""
    lobe_a = np.argwhere(np.ones((10, 10, 6))).astype(float)
    lobe_b = np.argwhere(np.ones((10, 10, 6))).astype(float) + [40, 0, 0]
    points = np.vstack([lobe_a, lobe_b])
    cleaned = _remove_subregion_fragments(points, ratio=0.2)
    assert len(cleaned) == len(points)                             # both kept


def test_handles_empty_and_single_component():
    """Empty input and a single connected component pass through unchanged."""
    assert len(_remove_subregion_fragments(np.empty((0, 3)))) == 0
    single = np.argwhere(np.ones((8, 8, 4))).astype(float)
    assert len(_remove_subregion_fragments(single)) == len(single)

"""Unit tests for the geometry of the image left-right mirror.

``mirror_image_left_right`` reverses the voxels along index axis 0 and mirrors the geometry
with them, so every voxel keeps its physical location (mirrored index ``i`` is the parent's
``n - 1 - i``). These tests pin that for both image backends, including a SimpleITK direction
matrix that permutes the axes and oblique (rotated, non-axis-aligned) geometries with
anisotropic spacing for both backends.

Where it is used: no production code path uses it at present. It was introduced for the
right-side proximal-femur landmark search, which has since switched to
``reflect_image_left_right`` (reverse the voxels, keep the geometry: the physical mirror twin,
which the physical-space CT head-centre search needs). The helper is kept as the
index-relabelling counterpart and is used by the MRI probe of
``tests/characterization/test_femoral_torsion_equivariance.py``.
"""
import numpy as np
import pytest

from morphometry.image_io import mirror_image_left_right
from tests.unit.test_split_left_right_geometry import BUILDERS, _index_array, _indices


@pytest.fixture(params=[(backend, width) for backend in BUILDERS for width in (10, 11)],
                ids=lambda p: f'{p[0]}-width{p[1]}')
def parent(request):
    """A parent volume for each backend (incl. a permuted direction), even and odd widths."""
    backend, width = request.param
    builder, x_step = BUILDERS[backend]
    image = builder(width)
    image.x_step = np.asarray(x_step)
    image.builder = backend
    return image


def test_mirror_reverses_the_data_and_keeps_the_class(parent):
    """The voxel data is the parent reversed along index axis 0; class, backend and shape are kept."""
    mirrored = mirror_image_left_right(parent)
    assert type(mirrored) is type(parent) and mirrored.type == parent.type
    assert tuple(mirrored.shape) == tuple(parent.shape)
    assert np.array_equal(_index_array(mirrored), _index_array(parent)[::-1])


def test_mirrored_index_maps_to_the_partner_voxel_of_the_parent(parent):
    """Index ``i`` of the mirror is the parent's index ``n - 1 - i``, so every voxel keeps its physical location."""
    mirrored = mirror_image_left_right(parent)
    n = parent.shape[0]
    full, flipped = _index_array(parent), _index_array(mirrored)
    for index in _indices(parent.shape):
        partner = (n - 1 - index[0], index[1], index[2])
        assert np.asarray(mirrored.transform_index_to_physical_point(index)) == pytest.approx(
            np.asarray(parent.transform_index_to_physical_point(partner)))
        # the voxel value travels with its physical location
        assert flipped[index] == full[partner]


def test_mirror_keeps_spacing_and_flips_only_the_first_axis_direction(parent):
    """Spacing is unchanged; one index step now moves the opposite way along the same physical axis."""
    mirrored = mirror_image_left_right(parent)
    assert np.asarray(mirrored.spacing)[:3] == pytest.approx(np.asarray(parent.spacing)[:3])

    step = (np.asarray(mirrored.transform_index_to_physical_point((1, 0, 0)))
            - np.asarray(mirrored.transform_index_to_physical_point((0, 0, 0))))
    assert step == pytest.approx(-parent.x_step)
    for axis in (1, 2):
        unit = [0, 0, 0]
        unit[axis] = 1
        delta_m = (np.asarray(mirrored.transform_index_to_physical_point(unit))
                   - np.asarray(mirrored.transform_index_to_physical_point((0, 0, 0))))
        delta_p = (np.asarray(parent.transform_index_to_physical_point(unit))
                   - np.asarray(parent.transform_index_to_physical_point((0, 0, 0))))
        assert delta_m == pytest.approx(delta_p)


def test_physical_to_index_round_trips_through_the_mirror(parent):
    """A physical point resolves to mirrored index ``n - 1 - i`` where the parent resolves it to ``i``."""
    mirrored = mirror_image_left_right(parent)
    n = parent.shape[0]
    point = parent.transform_index_to_physical_point((2, 3, 4))
    assert np.asarray(mirrored.transform_physical_point_to_index(point)) == pytest.approx([n - 1 - 2, 3, 4])


def test_mirroring_twice_restores_the_parent(parent):
    """Mirror of the mirror has the parent's data and geometry; the parent itself is untouched."""
    before = _index_array(parent).copy()
    twice = mirror_image_left_right(mirror_image_left_right(parent))
    assert np.array_equal(_index_array(twice), before)
    assert np.array_equal(_index_array(parent), before)
    for index in _indices(parent.shape):
        assert np.asarray(twice.transform_index_to_physical_point(index)) == pytest.approx(
            np.asarray(parent.transform_index_to_physical_point(index)))


def _unit_direction(image) -> np.ndarray:
    """The 3x3 direction cosine matrix of an image, with the spacing divided out.

    :param image: The Image / Segmentation (nibabel or SimpleITK).
    :return: The direction matrix whose column ``j`` is the unit physical direction of index axis ``j``.
    """
    if image.type == 'nibabel':
        linear = np.asarray(image.affine)[:3, :3]
        return linear / np.linalg.norm(linear, axis=0)
    return np.asarray(image.image.GetDirection(), dtype=float).reshape(3, 3)


def test_mirror_every_voxel_maps_to_its_partner_and_flips_handedness(parent):
    """Every voxel of the mirror sits at its parent partner ``n - 1 - i``; spacing kept; direction det negated.

    The oblique builders are proper rotations (det +1) with anisotropic spacing, so their
    mirror's direction determinant must be exactly -1; for the axis-aligned builders the
    determinant must flip sign.
    """
    mirrored = mirror_image_left_right(parent)
    n = parent.shape[0]
    index = np.argwhere(np.ones(tuple(int(v) for v in parent.shape), dtype=bool))
    for i, j, k in index:
        assert np.asarray(mirrored.transform_index_to_physical_point((int(i), int(j), int(k)))) == pytest.approx(
            np.asarray(parent.transform_index_to_physical_point((n - 1 - int(i), int(j), int(k)))), abs=1e-9)

    assert np.asarray(mirrored.spacing)[:3] == pytest.approx(np.asarray(parent.spacing)[:3])
    parent_det = np.linalg.det(_unit_direction(parent))
    assert abs(parent_det) == pytest.approx(1.0)
    assert np.linalg.det(_unit_direction(mirrored)) == pytest.approx(-parent_det)
    if 'oblique' in parent.builder:
        # the oblique parents are proper rotations, so the mirror is exactly an improper one
        assert parent_det == pytest.approx(1.0)
        assert np.linalg.det(_unit_direction(mirrored)) == pytest.approx(-1.0)

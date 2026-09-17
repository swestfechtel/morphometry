"""Unit tests for the geometry of the image-side left / right split.

Every left/right split used to build the right half as ``array[half:]`` but wrap it
with the *parent's* affine, so index 0 of the right half mapped to the parent's index
0 and every right-side physical coordinate was off by ``half * spacing_x`` (195 mm on
the NMDID whole-leg CT fixture). ``split_image_left_right`` gives each half its own,
correct geometry; these tests pin that for both image backends.
"""
import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from morphometry.image_io import Image, Segmentation, split_image_left_right


def _array(width: int) -> np.ndarray:
    """A volume of distinct values, so a half that was cut in the wrong place cannot pass.

    :param width: Size of the first (image left-right) axis.
    :return: A ``(width, 6, 5)`` float array holding ``arange`` values.
    """
    return np.arange(width * 6 * 5, dtype=np.float64).reshape(width, 6, 5)


def _nibabel(width: int) -> Segmentation:
    """A nibabel Segmentation with an anisotropic, flipped, translated affine.

    :param width: Size of the first axis.
    :return: The Segmentation.
    """
    affine = np.diag([-0.8, -0.9, -3.0, 1.0])
    affine[:3, 3] = [150.0, 45.0, -560.0]
    return Segmentation.from_nibabel(nib.Nifti1Image(_array(width), affine))


def _sitk(width: int) -> Segmentation:
    """A SimpleITK Segmentation with anisotropic spacing, a non-trivial direction and origin.

    SimpleITK indexes ``(x, y, z)`` while ``GetArrayFromImage`` returns ``(z, y, x)``, so
    the image is built from the transposed array to keep index ``(i, j, k)`` holding the
    same value as in the nibabel variant.
    :param width: Size of the first (x) index axis.
    :return: The Segmentation.
    """
    image = sitk.GetImageFromArray(_array(width).transpose(2, 1, 0))
    image.SetSpacing((0.8, 0.9, 3.0))
    image.SetOrigin((150.0, 45.0, -560.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0))
    return Segmentation.from_sitk(image)


def _sitk_permuted(width: int) -> Segmentation:
    """A SimpleITK Segmentation whose direction matrix permutes (and flips) the axes.

    Index x runs along physical +y, index y along physical -z and index z along physical
    +x, so stepping along the split axis moves the physical point in y, not x. This catches
    an implementation that shifts ``origin[0]`` by ``half * spacing[0]`` instead of
    advancing the origin along the split axis' actual physical direction.
    :param width: Size of the first (x) index axis.
    :return: The Segmentation.
    """
    image = _sitk(width).image
    image.SetDirection((0.0, 0.0, 1.0,
                        1.0, 0.0, 0.0,
                        0.0, -1.0, 0.0))
    return Segmentation.from_sitk(image)


def _oblique_rotation() -> np.ndarray:
    """A proper rotation (det +1) about a non-axis-aligned axis, so no column is axis-aligned.

    :return: The 3x3 rotation matrix (35 degrees about the normalised axis (1, 2, 3)).
    """
    axis = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    angle = np.deg2rad(35.0)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k  # Rodrigues' formula


OBLIQUE_SPACING = np.array([0.8, 0.9, 3.0])


def _nibabel_oblique(width: int) -> Segmentation:
    """A nibabel Segmentation with a rotated (oblique), anisotropically scaled affine.

    :param width: Size of the first axis.
    :return: The Segmentation.
    """
    affine = np.eye(4)
    affine[:3, :3] = _oblique_rotation() * OBLIQUE_SPACING  # column j = direction j scaled by spacing j
    affine[:3, 3] = [150.0, 45.0, -560.0]
    return Segmentation.from_nibabel(nib.Nifti1Image(_array(width), affine))


def _sitk_oblique(width: int) -> Segmentation:
    """A SimpleITK Segmentation with a rotated (oblique) direction matrix and anisotropic spacing.

    :param width: Size of the first (x) index axis.
    :return: The Segmentation.
    """
    image = _sitk(width).image
    image.SetDirection(tuple(_oblique_rotation().ravel()))
    return Segmentation.from_sitk(image)


def _index_array(image: Image) -> np.ndarray:
    """The voxel array in ``(x, y, z)`` index order, whichever backend holds it.

    :param image: The Image / Segmentation.
    :return: The array such that ``array[i, j, k]`` is the voxel at index ``(i, j, k)``.
    """
    return image.array if image.type == 'nibabel' else image.array.transpose(2, 1, 0)


# builder, and the physical displacement of one index step along the split (first) axis
BUILDERS = {'nibabel': (_nibabel, [-0.8, 0.0, 0.0]),
            'sitk': (_sitk, [-0.8, 0.0, 0.0]),
            'sitk-permuted': (_sitk_permuted, [0.0, 0.8, 0.0]),
            'nibabel-oblique': (_nibabel_oblique, _oblique_rotation()[:, 0] * OBLIQUE_SPACING[0]),
            'sitk-oblique': (_sitk_oblique, _oblique_rotation()[:, 0] * OBLIQUE_SPACING[0])}


@pytest.fixture(params=[(backend, width) for backend in BUILDERS for width in (10, 11)],
                ids=lambda p: f'{p[0]}-width{p[1]}')
def parent(request):
    """A parent volume for each backend, with an even and an odd left-right width.

    The expected per-voxel step along the split axis is attached as ``x_step``.
    """
    backend, width = request.param
    builder, x_step = BUILDERS[backend]
    image = builder(width)
    image.x_step = np.asarray(x_step)
    return image


def _indices(shape) -> list:
    """Corner, centre and edge voxels of a volume.

    :param shape: The volume's index-space shape.
    :return: A list of integer index triples.
    """
    x, y, z = shape
    return [(0, 0, 0), (x - 1, y - 1, z - 1), (x // 2, y // 2, z // 2), (x - 1, 0, z - 1), (0, y - 1, 0)]


def test_halves_hold_the_plain_split(parent):
    """The voxel data is exactly the ``shape[0] // 2`` split, and the halves keep the parent's class."""
    left, right = split_image_left_right(parent)
    half = parent.shape[0] // 2
    full = _index_array(parent)

    assert type(left) is type(parent) and type(right) is type(parent)
    assert left.type == parent.type and right.type == parent.type
    assert np.array_equal(_index_array(left), full[:half])
    assert np.array_equal(_index_array(right), full[half:])
    assert tuple(left.shape) == (half, *parent.shape[1:])
    assert tuple(right.shape) == (parent.shape[0] - half, *parent.shape[1:])


def test_left_half_keeps_the_parent_geometry(parent):
    """Index ``i`` of the left half is the parent's index ``i``."""
    left, _ = split_image_left_right(parent)
    for index in _indices(left.shape):
        assert np.asarray(left.transform_index_to_physical_point(index)) == pytest.approx(
            np.asarray(parent.transform_index_to_physical_point(index)))


def test_right_half_is_translated_by_the_split_point(parent):
    """Index ``i`` of the right half is the parent's index ``i + [half, 0, 0]``."""
    _, right = split_image_left_right(parent)
    half = parent.shape[0] // 2
    for index in _indices(right.shape):
        shifted = (index[0] + half, index[1], index[2])
        assert np.asarray(right.transform_index_to_physical_point(index)) == pytest.approx(
            np.asarray(parent.transform_index_to_physical_point(shifted)))


def test_halves_keep_spacing_and_direction(parent):
    """Only the origin of the right half moves, and only along the split axis' direction."""
    left, right = split_image_left_right(parent)
    for part in (left, right):
        assert np.asarray(part.spacing)[:3] == pytest.approx(np.asarray(parent.spacing)[:3])
        assert np.asarray(part.direction).ravel() == pytest.approx(np.asarray(parent.direction).ravel())

    half = parent.shape[0] // 2
    shift = np.asarray(right.origin) - np.asarray(parent.origin)
    # the origin advances by ``half`` steps along the split axis' physical direction
    assert shift == pytest.approx(parent.x_step * half)
    assert np.asarray(left.origin) == pytest.approx(np.asarray(parent.origin))


def test_split_does_not_modify_the_parent(parent):
    """The parent volume keeps its data and geometry."""
    before_array = _index_array(parent).copy()
    before_origin = np.asarray(parent.origin).copy()
    split_image_left_right(parent)
    assert np.array_equal(_index_array(parent), before_array)
    assert np.asarray(parent.origin) == pytest.approx(before_origin)

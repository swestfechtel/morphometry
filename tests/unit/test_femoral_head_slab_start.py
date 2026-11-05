"""Where the femoral-head slab STARTS must not depend on how many EMPTY slices the array carries.

``tests/unit/test_femoral_head_slab.py`` pins the other end of the same slab, the bound that stops
the inferior column walk from running head into neck into shaft. This module pins its start.

``get_femoral_head_center`` starts its slab at the topmost slice holding a femur voxel. Until
2026-11-05 it then tried to skip a top-of-head slice with "too few mask points" by z-scoring the
per-slice voxel counts and advancing while the first slice's ``|z| > 2``::

    layer_sizes = np.zeros(segmentation_mask.shape[2])   # one zero per EMPTY slice
    for i in ...: layer_sizes[i] = np.sum(layer)
    layer_zscores = np.abs(zscore(layer_sizes))
    while layer_zscores[layer_high] > 2:
        layer_high += 1

``layer_sizes`` spanned the array's **full** z extent, so the mean and the standard deviation --
and hence every z-score -- were functions of the number of empty slices around the anatomy. Padding
an image with empty slices, or cropping them away, moved the head centre although not one labelled
voxel changed. That is what this module pins: the phantom below is translated in z *and* given a
varying number of empty slices on both ends, and the centre and radius must agree up to the
translation. Not *bitwise* -- see the tolerance note on
:func:`test_empty_z_slices_do_not_move_the_centre`; that word is reserved for the pre-change
versus post-change comparison on a fixed array, which is exact.

**These tests fail against the pre-change code, and this phantom is the regime in which the guard
actually fires** -- which is why it is the test that would have caught the defect. The guard fires
iff ``f > 0.8 * c^2``, where ``f`` is the fraction of the array's z extent holding femur and
``c^2 = 1 + CV^2`` the normalised second moment of the non-empty slice areas (see
``get_femoral_head_center``). The reference run is z-trimmed, so ``f = 1`` (102 of 102 slices
occupied) at an area CV of 0.329, i.e. ``c^2 = 1.108`` -- inside the firing region -- and the
measured first-slice ``|z|`` is 2.811 against the threshold of 2. Padding then walks it out: at
0 / 1 / 2 / 3 empty slices per end the shipped guard advanced ``layer_high`` by 3 / 2 / 2 / 1
slices, and from 5 on by none. (The closed form is approximate here, since this phantom's first
slice holds 95 voxels rather than being near-empty: it predicts 3.042 against the measured 2.811
and would call ``pad = 5`` a firing case, where the real ``|z|`` is 1.969 and it does not fire.)
Measured
against the pad-0 reference this test uses, that moved the fitted centre by up to **0.1012** voxel
on the isotropic path (radius 0.0340) and **0.0042** on the non-isotropic one (radius 0.0015) --
the same "one mechanism, two amplitudes" split the real cohorts show, since on the consensus path
the slab start only seeds a fit that then re-converges. The reference matters on the non-isotropic
path and is worth stating: against the guard-inert pad-40 run instead, the consensus figure is
0.0067 rather than 0.0042, a difference larger than the figure itself. The isotropic 0.1012 is the
same under either reference.

So the phantom demonstrates both halves of the defect: that the guard fires at all, and that
whether it fires is decided by the empty-slice count rather than by the anatomy. The real cohorts in
``tests/characterization/test_head_centre_empty_slice_trim.py`` show the same thing where their
slices are thin enough -- cropped, NaKo's first-slice ``|z|`` reaches a median 2.466 and the guard
fires on 108 of 120 halves -- while at Augsburg's 10 mm the apex slice is already half the mean
slice area and it fires on only 2 of 100. The phantom is 1 mm for exactly that reason.

The phantom is 1 mm isotropic on purpose. On that grid the topmost head slice is a small spherical
cap, which is the configuration the guard was meant for; on the 10 mm slices of the Augsburg hip
MRI the top slice is a large cross-section. Both code paths are exercised, since the guard sat
above the ``if isotropic:`` branch and reached both.
"""
import numpy as np
import pytest

from morphometry.hip import get_femoral_head_center

SPACING = (1.0, 1.0, 1.0)  #: 1 mm isotropic, like the NaKo hip segmentations
HEAD_RADIUS_MM = 22.0
NECK_RADIUS_MM = 14.0
NECK_ANGLE_DEG = 35.0  #: inclination of the neck axis below the horizontal, running laterally
SHAPE = (160, 90, 130)
HEAD_CENTRE = np.array([100.3, 45.4, 30.3])  #: off-grid head centre (voxel index) in the full volume

#: Empty slices added at BOTH ends of the z-trimmed phantom. 0-3 switch the pre-change guard on
#: (it advanced the slab start by 3 / 2 / 2 / 1 slices), 5 and up leave it inert.
PADDINGS = [0, 1, 2, 3, 5, 10, 20, 40]


def head_and_neck(head: np.ndarray, shape=SHAPE, spacing=SPACING) -> np.ndarray:
    """A femoral head with a neck and a trochanter block, as a left image half (medial = +x).

    The same construction as ``tests/unit/test_femoral_head_consensus.py``, on an isotropic grid.

    :param head: The head centre as a (possibly fractional) voxel index.
    :param shape: The volume shape in voxels.
    :param spacing: Voxel spacing in mm.
    :return: A uint8 mask with the femur labelled 1.
    """
    xx, yy, zz = np.indices(shape).astype(float)
    offset = np.stack([(xx - head[0]) * spacing[0], (yy - head[1]) * spacing[1],
                       (zz - head[2]) * spacing[2]], -1)
    sphere = (offset ** 2).sum(-1) <= HEAD_RADIUS_MM ** 2
    # the neck axis runs laterally (-x) and inferiorly (+z) out of the head centre
    axis = np.array([-np.cos(np.radians(NECK_ANGLE_DEG)), 0.0, np.sin(np.radians(NECK_ANGLE_DEG))])
    along = offset @ axis
    radial = np.linalg.norm(offset - along[..., None] * axis, axis=-1)
    neck = (radial <= NECK_RADIUS_MM) & (along >= 0) & (along <= 60)
    dx, dy, dz = offset[..., 0], offset[..., 1], offset[..., 2]
    trochanter = (dx <= -40) & (dx >= -65) & (np.abs(dy) <= 20) & (dz >= 15) & (dz <= 80)
    return (sphere | neck | trochanter).astype(np.uint8)


def _z_trimmed(mask: np.ndarray) -> np.ndarray:
    """Drop every leading and trailing all-zero z slice, keeping all labelled voxels.

    :param mask: A 3D mask.
    :return: The mask cropped along z to the slices that hold a labelled voxel.
    """
    occupied = np.argwhere(mask.sum(axis=(0, 1)) > 0)[:, 0]
    return mask[:, :, occupied.min():occupied.max() + 1]


@pytest.fixture(scope='module')
def trimmed_phantom() -> np.ndarray:
    """The phantom cropped to its occupied z slices, so a padding is the only empty-slice source."""
    return _z_trimmed(head_and_neck(HEAD_CENTRE))


@pytest.mark.parametrize('isotropic', [True, False], ids=['isotropic', 'consensus_fit'])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_empty_z_slices_do_not_move_the_centre(trimmed_phantom, side, isotropic):
    """Padding z with empty slices moves the centre by exactly the padding and keeps the radius.

    An empty slice holds no labelled voxel, so it cannot carry anatomical information: the centre
    of the reference run and of every padded run must agree once the padding is subtracted. The
    tolerance is 1e-9 rather than 0 because a padding shifts every z coordinate in the point cloud
    by a constant, so ``sphere_fit``'s ``np.linalg.lstsq`` sees a different design matrix and its
    SVD is not bit-exact under the shift; the residual measured here is 1.2e-11 voxel, against the
    0.1012 the removed guard produced.
    """
    mask = trimmed_phantom if side == 'left' else trimmed_phantom[::-1]
    kwargs = dict(side=side, segmentation_label=1, x_ratio=SPACING[2] / SPACING[0],
                  isotropic=isotropic, spacing=None if isotropic else SPACING)
    radius, centre = get_femoral_head_center(mask, **kwargs)
    centre = np.asarray(centre, dtype=float)
    for pad in PADDINGS:
        padded = np.pad(mask, [(0, 0), (0, 0), (pad, pad)])
        padded_radius, padded_centre = get_femoral_head_center(padded, **kwargs)
        shift = np.asarray(padded_centre, dtype=float) - centre
        np.testing.assert_allclose(shift, [0.0, 0.0, pad], atol=1e-9,
                                   err_msg=f'{pad} empty slices per end moved the centre by {shift}')
        assert padded_radius == pytest.approx(radius, abs=1e-9), \
            f'{pad} empty slices per end moved the radius by {padded_radius - radius}'


@pytest.mark.parametrize('isotropic', [True, False], ids=['isotropic', 'consensus_fit'])
def test_the_slab_starts_at_the_first_labelled_slice(trimmed_phantom, isotropic):
    """The slab's upper bound is the topmost slice that holds a femur voxel, with nothing skipped.

    The removed guard's whole effect was to advance that bound, so this is the direct statement of
    what changed: on this phantom it used to start 3 slices below the top of the head.
    """
    *_, layer_high, layer_low = get_femoral_head_center(
        trimmed_phantom, side='left', segmentation_label=1, return_layers=True,
        x_ratio=SPACING[2] / SPACING[0], isotropic=isotropic,
        spacing=None if isotropic else SPACING)
    assert layer_high == 0, f'the slab starts {layer_high} slices below the first labelled slice'
    assert layer_low > layer_high

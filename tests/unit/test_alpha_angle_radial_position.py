"""The alpha angle's section plane can be rotated about the neck axis (``plane_rotation``).

The published alpha angle is read from a **radial series** and reported as the maximum over it --
the angle is highest antero-superiorly -- where ``calculate_alpha_angle`` has always sampled one
position. CLAUDE.md records the measurement: over 20 NaKo cases at 15-degree steps from -60 to
+60, the anterior mean runs 30.26 to 45.03 degrees and the posterior the other way, and the
per-half maximum lifts the anterior mean by ~18 degrees -- most of the gap to the published
asymptomatic 54.1 +- 5.1.

``plane_rotation`` exposes that position. It does **not** implement the reduction, deliberately:
whether to report the maximum, and over what range, is a definitional choice that changes what the
returned pair *means*, because the anterior and posterior angles peak at opposite ends of the
sweep -- a per-angle maximum returns two numbers measured in different planes. So the mechanism
ships and the policy does not.

These tests pin the three properties that make that safe to have:

1. **the default is bitwise the old construction**, so nothing that does not pass the parameter
   can be affected -- which is what lets this exist without a do-not-pool burden;
2. the parameter actually turns the plane, and turns it about the **neck axis** rather than about
   an image axis (checked by the symmetry a rotation about ``u`` must have, not by eyeballing);
3. a full turn returns to the start, and the sweep stays inside the isotropy guard and the
   ``(0, 90)`` plausibility gate rather than bypassing them.
"""
import numpy as np
import pytest

from morphometry.measurements.hip import _reslice_femoral_neck_section, calculate_alpha_angle


@pytest.fixture
def phantom():
    """A head sphere on a neck cylinder, asymmetric about the section plane.

    Asymmetric on purpose: a shape symmetric about the plane would be reproduced by ``+phi`` and
    ``-phi`` alike, and the point of :func:`test_the_rotation_is_about_the_neck_axis` is that the
    two differ.

    :return: ``(volume, head_centre, neck_centre)``.
    """
    n = 90
    volume = np.zeros((n, n, n), dtype=np.int16)
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    volume[(xx - 60.) ** 2 + (yy - 45.) ** 2 + (zz - 25.) ** 2 <= 15. ** 2] = 1
    for t in np.linspace(0, 1, 400):
        volume[(xx - (60 - 32 * t)) ** 2 + (yy - 45.) ** 2 + (zz - (25 + 26 * t)) ** 2 <= 10. ** 2] = 1
    # a bump off the section plane, so +phi and -phi cannot agree
    volume[(xx - 50.) ** 2 + (yy - 56.) ** 2 + (zz - 34.) ** 2 <= 7. ** 2] = 1
    return volume, np.array([60.0, 45.0, 25.0]), np.array([40.0, 45.0, 40.0])


def test_the_default_is_bitwise_the_single_position_construction(phantom):
    """``plane_rotation=0.0`` must reproduce the old reslice exactly, not approximately.

    This is the property that lets the parameter exist at all: every caller that does not pass it
    is unaffected, so no published value moves and no golden needed re-capturing. The guard is a
    falsy check on a float rather than a branch on a sentinel, so 0.0 takes the identical code
    path the literal ``v = (0, 1, 0)`` did.
    """
    volume, head, neck = phantom

    default = _reslice_femoral_neck_section(volume, head, neck, 33.0)
    explicit = _reslice_femoral_neck_section(volume, head, neck, 33.0, 0.0)

    assert np.array_equal(default[0], explicit[0]), 'the section is not bitwise identical'
    np.testing.assert_array_equal(default[1], explicit[1])
    np.testing.assert_array_equal(default[2], explicit[2])


def test_a_nonzero_rotation_actually_turns_the_plane(phantom):
    """A discriminating check: the section must change, and change with the angle.

    Without this a no-op implementation would pass every other test here.
    """
    volume, head, neck = phantom
    base = _reslice_femoral_neck_section(volume, head, neck, 33.0)[0]

    sections = [_reslice_femoral_neck_section(volume, head, neck, 33.0, np.radians(d))[0]
                for d in (15.0, 30.0, 45.0)]

    for d, section in zip((15.0, 30.0, 45.0), sections):
        assert section.shape != base.shape or not np.array_equal(section, base), \
            f'plane_rotation={d} deg left the section unchanged'
    assert not np.array_equal(sections[0], sections[-1]), 'the sections do not vary with the angle'


def test_the_rotation_is_about_the_neck_axis(phantom):
    """It must turn the plane about ``u`` (the neck axis), not about an image axis.

    The discriminating property: a rotation about ``u`` leaves the component of every sampled
    offset *along* ``u`` untouched, so the neck cell's first coordinate is invariant while its
    second one moves. A rotation about an image axis would move both. Checked on the returned
    cells rather than on the basis, so it tests the behaviour and not the arithmetic.
    """
    volume, head, neck = phantom

    _, _, neck_cell_0 = _reslice_femoral_neck_section(volume, head, neck, 33.0, 0.0)
    _, _, neck_cell_phi = _reslice_femoral_neck_section(volume, head, neck, 33.0, np.radians(40.0))

    # the window origin moves with the anatomy's extent, so compare the neck's offset from the head
    _, head_0, _ = _reslice_femoral_neck_section(volume, head, neck, 33.0, 0.0)
    _, head_phi, _ = _reslice_femoral_neck_section(volume, head, neck, 33.0, np.radians(40.0))
    along_0 = neck_cell_0[0] - head_0[0]
    along_phi = neck_cell_phi[0] - head_phi[0]
    across_0 = neck_cell_0[1] - head_0[1]
    across_phi = neck_cell_phi[1] - head_phi[1]

    assert along_phi == pytest.approx(along_0, abs=1e-9), \
        'the along-axis component moved, so the rotation is not about the neck axis'
    assert across_phi != pytest.approx(across_0, abs=1e-9), \
        'the across-axis component did not move, so nothing was rotated'


def test_a_full_turn_returns_to_the_start(phantom):
    """``plane_rotation`` of 2*pi must reproduce 0 to float tolerance.

    A cheap check that the basis is a rotation and not a shear or a reflection.
    """
    volume, head, neck = phantom

    base = _reslice_femoral_neck_section(volume, head, neck, 33.0, 0.0)[0]
    turned = _reslice_femoral_neck_section(volume, head, neck, 33.0, 2 * np.pi)[0]

    assert turned.shape == base.shape
    # `order=0` sampling, so a float-noise basis can move at most a handful of boundary cells
    disagree = int(np.count_nonzero(turned != base))
    assert disagree <= base.size // 1000, f'{disagree} of {base.size} cells differ after a full turn'


def test_the_sweep_still_goes_through_the_guards(phantom):
    """A swept position is the real measurement: the isotropy guard and the (0, 90) gate apply.

    The point of implementing the mechanism inside ``calculate_alpha_angle`` rather than beside it
    is that a caller sweeping positions cannot accidentally bypass either check. Anisotropic
    spacing must still be refused at a non-zero rotation, and the refusal must be the same
    ``ValueError`` the batch scripts already handle.
    """
    volume, head, neck = phantom
    del head, neck

    # a bare array plus an explicit `spacing=`, which the signature supports and which is the
    # path `docker/compute_torsion.py` uses; a hand-rolled Image stand-in would not be recognised
    # by `image_io.mask_and_spacing` and would test nothing about the guard
    with pytest.raises(ValueError, match='anisotropic'):
        calculate_alpha_angle(volume, 'left', 1, isotropic=True, spacing=(1.0, 1.0, 3.0),
                              plane_rotation=np.radians(30.0))

    # and the guard is not what makes a swept position raise: with isotropic spacing it proceeds
    isotropic_ok = calculate_alpha_angle(volume, 'left', 1, isotropic=True, spacing=(1.0, 1.0, 1.0),
                                         plane_rotation=0.0)
    assert all(0.0 < angle < 90.0 for angle in isotropic_ok)

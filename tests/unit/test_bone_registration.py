"""Unit tests for the rigid bone-surface registration (``morphometry.cartilage.registration``).

The test surface is the bone of the femoral phantom of :mod:`tests.unit.test_femur_regions`
(two condyle cylinders, an open intercondylar fossa, a trochlear ridge with a groove and a
proximal shaft) meshed through :func:`morphometry.cartilage.surface.bone_surface_mesh`, i.e.
the same smoothed, FOV-open marching-cubes surface the real scans produce. It is asymmetric
under every rotation (the fossa is posterior, the shaft proximal), so the ICP has a unique
answer; a sphere-like surface would not pin the rotation at all.
"""
import pickle

import numpy as np
import pytest
import pyvista as pv
from scipy.spatial.transform import Rotation

from morphometry.cartilage import surface as S
from morphometry.cartilage.registration import (
    REGISTRATION_CROP_MM, REGISTRATION_MAX_RMS_MM, RegistrationResult, RigidTransform,
    crop_to_articular_end, register_bone,
)
from tests.unit.test_femur_regions import SHAFT_TOP, phantom as femur_phantom

BONE = 1


def _rot(x_deg, y_deg, z_deg):
    """Rotation matrix of extrinsic x, y, z Euler angles (degrees)."""
    return Rotation.from_euler('xyz', [x_deg, y_deg, z_deg], degrees=True).as_matrix()


def _angle(r_a, r_b):
    """Angle (degrees) of the relative rotation ``r_a @ r_b.T``."""
    return np.degrees(Rotation.from_matrix(r_a @ r_b.T).magnitude())


@pytest.fixture(scope='module')
def fixed_surface():
    """The femoral phantom's bone surface at 0.5 mm (~80k vertices)."""
    return S.bone_surface_mesh(femur_phantom(), BONE)


def _surface(fixed_surface):
    """A private copy of the module-scoped surface, so a test cannot modify the shared one."""
    return fixed_surface.copy()


# --- RigidTransform -----------------------------------------------------------------------

def test_from_frames_maps_frame_to_frame():
    T = RigidTransform.from_frames(np.zeros(3), np.eye(3), np.array([1., 2, 3]), _rot(0, 0, 90))
    assert np.allclose(T.apply(np.array([[1., 0, 0]])), np.array([1, 2, 3]) + _rot(0, 0, 90)[0])


def test_from_frames_general_round_trip():
    """A point's local coordinates in the moving frame equal its image's in the fixed frame."""
    rng = np.random.default_rng(1)
    mo, fo = rng.normal(size=3), rng.normal(size=3)
    ma, fa = _rot(10, -20, 33), _rot(-70, 5, 12)
    T = RigidTransform.from_frames(mo, ma, fo, fa)
    p = rng.normal(size=(5, 3)) * 10
    assert np.allclose((T.apply(p) - fo) @ fa.T, (p - mo) @ ma.T)


def test_inverse_identity_and_pickle():
    T = RigidTransform(_rot(30, -10, 5), np.array([1.0, -2.0, 3.0]))
    p = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(T.inverse().apply(T.apply(p)), p)
    assert np.allclose(RigidTransform.identity().apply(p), p)
    T2 = pickle.loads(pickle.dumps(T))
    assert np.array_equal(T2.rotation, T.rotation) and np.array_equal(T2.translation, T.translation)
    # A single point (3,) is accepted as well as (N, 3).
    assert T.apply(p[0]).shape == (3,)


# --- crop_to_articular_end ----------------------------------------------------------------

def test_crop_keeps_cells_near_the_chosen_end(fixed_surface):
    mesh = fixed_surface
    axis = np.array([0.0, 0.0, 2.0])               # not unit on purpose: it is normalised
    lo = crop_to_articular_end(mesh, axis, 20.0, 'min')
    hi = crop_to_articular_end(mesh, axis, 20.0, 'max')
    zc = mesh.cell_centers().points[:, 2]
    assert lo.n_cells == np.count_nonzero(zc <= zc.min() + 20.0)
    assert hi.n_cells == np.count_nonzero(zc >= zc.max() - 20.0)
    assert lo.cell_centers().points[:, 2].max() <= zc.min() + 20.0 + 1e-9
    assert hi.cell_centers().points[:, 2].min() >= zc.max() - 20.0 - 1e-9
    assert isinstance(lo, pv.PolyData) and lo.is_all_triangles
    with pytest.raises(ValueError):
        crop_to_articular_end(mesh, axis, 20.0, 'middle')


# --- register_bone ------------------------------------------------------------------------

def test_recovers_known_rigid_motion(fixed_surface):
    fixed = _surface(fixed_surface)
    T = RigidTransform(_rot(8, 5, -6), np.array([3.0, -2.0, 4.0]))
    moving = fixed.copy()
    moving.points = T.inverse().apply(fixed.points)
    r = register_bone(moving, fixed)
    assert isinstance(r, RegistrationResult)
    assert r.converged and r.rms_mm < 0.05
    assert np.allclose(r.transform.apply(moving.points), fixed.points, atol=0.05)
    assert r.n_points > 0 and r.iterations >= 1
    assert r.median_mm <= r.p95_mm


def test_registration_is_deterministic(fixed_surface):
    fixed = _surface(fixed_surface)
    moving = fixed.copy()
    moving.points = RigidTransform(_rot(3, -4, 2), np.array([1.0, 1.0, -1.0])).apply(fixed.points)
    a = register_bone(moving, fixed, max_points=5000, seed=3)
    b = register_bone(moving, fixed, max_points=5000, seed=3)
    assert np.array_equal(a.transform.rotation, b.transform.rotation)
    assert np.array_equal(a.transform.translation, b.transform.translation)
    assert a.rms_mm == b.rms_mm


def test_partial_overlap_fov_cut_still_registers(fixed_surface):
    """
    The moving scan's FOV cuts the shaft 15 mm shorter than the fixed scan's: the part of the
    fixed bone above the cut has no partner, and trimmed ICP must still recover the pose.
    """
    fixed = fixed_surface
    true = RigidTransform(_rot(6, -4, 7), np.array([2.0, -3.0, 1.5]))
    top = fixed.cell_centers().points[:, 2].max()
    short = fixed.extract_cells(
        np.flatnonzero(fixed.cell_centers().points[:, 2] <= top - 15.0)).extract_surface()
    moving = short.copy()
    moving.points = true.inverse().apply(short.points)
    r = register_bone(moving, fixed, initial=RigidTransform(_rot(2, 2, -2), np.zeros(3)))
    assert r.converged
    assert r.rms_mm < 0.1 and _angle(r.transform.rotation, true.rotation) < 0.5
    assert np.allclose(r.transform.apply(moving.points), short.points, atol=0.1)


def test_resampled_partial_overlap_reaches_the_discretisation_floor(fixed_surface):
    """
    A *different acquisition*: the anatomy sits in another pose in the scanner, so it is
    voxelised and meshed differently, and the FOV cuts the shaft at an array face (oblique
    once the anatomy is rotated) at least 14 mm shorter. Two independent 0.5 mm
    discretisations of the phantom's sharp edges do not coincide, so even the *true* pose
    leaves a trimmed rms of ~0.13 mm here: the bound is that floor, and the pose must be
    recovered to a fraction of a degree.
    """
    fixed = fixed_surface
    pose = _rot(6, -4, 7)
    moving = S.bone_surface_mesh(femur_phantom(rotation=pose, crop_proximal=50), BONE)
    assert (moving.points @ pose)[:, 2].max() < SHAFT_TOP - 14.0   # shorter everywhere
    shift = np.array([2.0, -3.0, 1.5])
    moving.points = moving.points + shift
    # World = pose @ local + shift on the moving side, world = local on the fixed side.
    true = RigidTransform(pose.T, -pose.T @ shift)
    d = np.sort(S.surface_distances(true.apply(moving.points), fixed))
    floor = np.sqrt(np.mean(d[:int(np.ceil(0.9 * len(d)))] ** 2))
    r = register_bone(moving, fixed, initial=RigidTransform(_rot(2, 2, -2), np.zeros(3)))
    assert r.converged
    assert r.rms_mm < floor + 0.02
    assert _angle(r.transform.rotation, true.rotation) < 0.5
    assert np.linalg.norm(r.transform.translation - true.translation) < 0.5


def test_bad_initialisation_is_visible_in_the_residual(fixed_surface):
    """
    A 120 degree initial error: either ICP converges back to the true pose, or the failure
    is visible in the residual (rms above the QC threshold). What must never happen is a
    wrong transform reported with a small residual.
    """
    fixed = _surface(fixed_surface)
    T = RigidTransform(_rot(5, -3, 4), np.array([1.0, 2.0, -1.0]))
    moving = fixed.copy()
    moving.points = T.inverse().apply(fixed.points)
    initial = RigidTransform(_rot(120, 0, 0), np.zeros(3))
    r = register_bone(moving, fixed, initial=initial)
    recovered = (_angle(r.transform.rotation, T.rotation) < 0.5
                 and np.allclose(r.transform.apply(moving.points), fixed.points, atol=0.1))
    assert r.rms_mm < 0.1 or r.rms_mm > REGISTRATION_MAX_RMS_MM
    if r.rms_mm < 0.1:
        assert recovered, 'small residual but wrong transform'
    else:
        assert not recovered


@pytest.mark.parametrize('axis_deg', [
    (120, 0, 0), (0, 120, 0),
    pytest.param((0, 0, 120), marks=pytest.mark.xfail(strict=True, reason=(
        'known limitation, pinned: the phantom is nearly symmetric under a 180 degree turn '
        'about its shaft axis (full condyle cylinders; only the small trochlear ridge differs), '
        'ICP lands in that mirror pose, and the ridge is less than the 10% of points the '
        'trimmed residual drops, so the wrong pose reads rms < 0.1 mm. A trimmed one-sided '
        'residual cannot see a near-symmetry covering more than trim_fraction of the surface; '
        'the initial frame must resolve it.'))),
])
def test_no_silently_wrong_transform_from_any_bad_start(fixed_surface, axis_deg):
    """Same requirement as above, about each axis: a small residual implies the true pose."""
    fixed = fixed_surface
    moving = fixed.copy()
    T = RigidTransform(_rot(4, 2, -3), np.array([-1.0, 1.0, 2.0]))
    moving.points = T.inverse().apply(fixed.points)
    r = register_bone(moving, fixed, initial=RigidTransform(_rot(*axis_deg), np.zeros(3)),
                      max_points=5000)
    if r.rms_mm <= REGISTRATION_MAX_RMS_MM:
        assert _angle(r.transform.rotation, T.rotation) < 0.5


def test_no_overlap_reports_failure_not_nan_crash(fixed_surface):
    """Surfaces 100 mm apart: no pair within ``max_pair_mm``; not converged, large residual."""
    fixed = fixed_surface
    moving = fixed.copy()
    moving.points = fixed.points + np.array([100.0, 0.0, 0.0])
    r = register_bone(moving, fixed, max_points=2000)
    assert not r.converged
    assert r.rms_mm > REGISTRATION_MAX_RMS_MM


def test_constants():
    assert REGISTRATION_CROP_MM == 40.0
    assert REGISTRATION_MAX_RMS_MM == 1.0

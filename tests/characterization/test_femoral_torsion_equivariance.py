"""Mirror equivariance of femoral torsion (MRI Lee / Murphy and whole-leg CT), stage by stage.

The two image halves hold mirror-image anatomy, so the right-side code must behave like the
left-side code on the mirror image. Two probes are checked for every stage:

* ``left_via_right``: the LEFT half mirrored and measured with ``side='right'`` must match the
  left half measured with ``side='left'``;
* ``right_via_left``: the RIGHT half mirrored and measured with ``side='left'`` must match the
  right half measured with ``side='right'``.

Landmarks must correspond by ``x -> n - 1 - x`` (``n`` the width of the image the landmark
indexes: the hip half for ``hip_*``, the knee half for ``knee_*``) and the torsion angles must
agree. Stages are tested separately so a failure points at one: head centre and neck point
(the proximal reference line), the knee condylar line, and the final angle. (The pure angle
composition is covered by ``tests/unit/test_torsion_angle_equivariance.py``.)

MRI inputs mirror the ``compute_torsion.py`` inputs (hip half as an Image, knee half as a plain
array, ``x_ratio``); the hip half is mirrored with ``mirror_image_left_right``, the knee array
with ``[::-1]``. Production reflects instead (``reflect_image_left_right``). For the MRI path the
two are equivalent: both produce the identical reversed voxel array, and the MRI landmark search
works in index space; the only geometry it touches is the trochanter-minor plausibility check,
which uses a distance between two points of the same image, and the two affines differ only in
the sign of their first column, which leaves such distances unchanged. The whole-leg CT
landmarks are computed in physical space, where lateral is a physical direction, so the CT probe
uses ``reflect_image_left_right``: a mirror only relabels indices (a left leg stays a left leg
physically), a reflection produces the mirror-twin leg.

What these tests can and cannot catch: right image halves are measured by reflecting them and
running the left-side code, so the tests guard against reintroducing a separate right-side
branch and verify the landmark mapping (``x -> n - 1 - x``) and the angle composition in the
reflected frame. They cannot catch defects in the shared left-side code itself, which a mirrored
probe reproduces identically on both sides -- e.g. its dependence on where the hip sits along x
(the head trim scales an absolute index). (The angle composition's former dependence on the
in-plane rotation of the image was such a defect; its rotation invariance is now tested directly in
``tests/unit/test_femoral_torsion_composition.py``.)
"""
import warnings
from functools import lru_cache

import numpy as np
import pytest

from morphometry.femur import get_proximal_reference_line, get_proximal_reference_line_ct
from morphometry.image_io import (mirror_image_left_right, reflect_image_left_right, split_ct_image,
                                  split_image_left_right)
from morphometry.measurements.femur import (calculate_femoral_torsion, calculate_femoral_torsion_ct,
                                            get_femoral_torsion_landmarks, get_femoral_torsion_landmarks_ct)
from tests import data_paths
from tests.conftest import _load_seg

OTHER = {'left': 'right', 'right': 'left'}
PROBES = {'left_via_right': 'left', 'right_via_left': 'right'}  # probe name -> side of the reference half


def _mirror_x(point, n: int) -> np.ndarray:
    """Map a voxel index of an ``n``-wide image to the mirrored image (``x -> n - 1 - x``).

    :param point: A length-3 index.
    :param n: Width of the image along x.
    :return: The mirrored index as a float array.
    """
    point = np.asarray(point, dtype=float).copy()
    point[0] = n - 1 - point[0]
    return point


def _same_pair(a, b, c, d, abs_tol: float) -> bool:
    """Whether the unordered point pairs ``{a, b}`` and ``{c, d}`` coincide within ``abs_tol``.

    :return: True when they match in either order.
    """
    close = lambda p, q: np.allclose(np.asarray(p, float), np.asarray(q, float), rtol=0, atol=abs_tol)
    return (close(a, c) and close(b, d)) or (close(a, d) and close(b, c))


# ----------------------------------------------------------------------------------------- MRI
@lru_cache(maxsize=None)
def _mri_inputs():
    """Augsburg PA000001 halves as ``compute_torsion.py`` feeds them, plus their mirror images."""
    base = data_paths.AUGSBURG_PA000001
    files = {n: base / f'{n}_seg.nii.gz' for n in ('hip', 'knee')}
    if not all(p.exists() for p in files.values()):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hip, knee = (_load_seg(files[n]) for n in ('hip', 'knee'))
    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
    halves = {}
    for side, h, k in zip(('left', 'right'), split_image_left_right(hip), split_image_left_right(knee)):
        halves[side] = {'hip': h, 'knee': k.array,
                        'mirrored_hip': mirror_image_left_right(h), 'mirrored_knee': k.array[::-1]}
    return x_ratio, halves


@pytest.fixture
def mri():
    inputs = _mri_inputs()
    if inputs is None:
        pytest.skip('Augsburg PA000001 data not found')
    return inputs


@lru_cache(maxsize=None)
def _mri_results(ref_side: str, method: str):
    """Reference (``ref_side`` half, ``ref_side`` code) and probe (its mirror, the other side's code)."""
    x_ratio, halves = _mri_inputs()
    half, probe_side = halves[ref_side], OTHER[ref_side]
    out = {}
    for role, h, k, side in (('ref', half['hip'], half['knee'], ref_side),
                             ('probe', half['mirrored_hip'], half['mirrored_knee'], probe_side)):
        np.random.seed(0)
        line = get_proximal_reference_line(h, side=side, method=method, x_ratio=x_ratio)
        lm = get_femoral_torsion_landmarks(h, k, side=side, method=method, x_ratio=x_ratio)
        angle = calculate_femoral_torsion(h, k, side=side, method=method, x_ratio=x_ratio)
        out[role] = {'line': line, 'landmarks': lm, 'angle': angle}
    return out, half['hip'].array.shape[0], half['knee'].shape[0]


MRI_CASES = [(probe, method) for probe in PROBES for method in ('lee', 'murphy')]


@pytest.mark.needs_augsburg
@pytest.mark.parametrize('probe,method', MRI_CASES)
def test_mri_head_centre_is_mirror_equivariant(mri, probe, method):
    """Proximal reference line start (femoral head centre) and head radius."""
    res, n_hip, _ = _mri_results(PROBES[probe], method)
    ref, prb = res['ref']['line'], res['probe']['line']
    assert np.array_equal(_mirror_x(prb[0], n_hip), np.asarray(ref[0], float)), (ref[0], prb[0])
    assert prb[2] == pytest.approx(ref[2], abs=1e-9)


@pytest.mark.needs_augsburg
@pytest.mark.parametrize('probe,method', MRI_CASES)
def test_mri_neck_point_is_mirror_equivariant(mri, probe, method):
    """Proximal reference line end (Lee neck centre / Murphy neck base)."""
    res, n_hip, _ = _mri_results(PROBES[probe], method)
    ref, prb = res['ref']['line'], res['probe']['line']
    assert np.array_equal(_mirror_x(prb[1], n_hip), np.asarray(ref[1], float)), (ref[1], prb[1])


@pytest.mark.needs_augsburg
@pytest.mark.parametrize('probe,method', MRI_CASES)
def test_mri_knee_condylar_line_is_mirror_equivariant(mri, probe, method):
    """Posterior condylar line endpoints (as an unordered pair; ordering is side-dependent)."""
    res, _, n_knee = _mri_results(PROBES[probe], method)
    ref, prb = res['ref']['landmarks'], res['probe']['landmarks']
    assert _same_pair(_mirror_x(prb['knee_start'], n_knee), _mirror_x(prb['knee_end'], n_knee),
                      ref['knee_start'], ref['knee_end'], abs_tol=0), (ref, prb)


@pytest.mark.needs_augsburg
@pytest.mark.parametrize('probe,method', MRI_CASES)
def test_mri_torsion_angle_is_mirror_equivariant(mri, probe, method):
    """The femoral torsion itself."""
    res, _, _ = _mri_results(PROBES[probe], method)
    assert res['probe']['angle'] == pytest.approx(res['ref']['angle'], abs=1e-9)


# ------------------------------------------------------------------------------------------ CT
@lru_cache(maxsize=None)
def _ct_inputs():
    """The CT fixture's halves with their knee crops, plus their reflections."""
    path = data_paths.CT_WHOLE_LEG
    if path is None or not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seg = _load_seg(path)
    halves = {}
    for side, leg in zip(('left', 'right'), split_image_left_right(seg)):
        reflected = reflect_image_left_right(leg)
        halves[side] = {'leg': leg, 'knee': split_ct_image(leg, None)[1][1],
                        'reflected_leg': reflected, 'reflected_knee': split_ct_image(reflected, None)[1][1]}
    return halves


@pytest.fixture
def ct():
    inputs = _ct_inputs()
    if inputs is None:
        pytest.skip('No whole-leg CT sample')
    return inputs


@lru_cache(maxsize=None)
def _ct_results(ref_side: str):
    half, probe_side = _ct_inputs()[ref_side], OTHER[ref_side]
    out = {}
    for role, leg, knee, side in (('ref', half['leg'], half['knee'], ref_side),
                                  ('probe', half['reflected_leg'], half['reflected_knee'], probe_side)):
        out[role] = {'line': get_proximal_reference_line_ct(leg, side=side),
                     'landmarks': get_femoral_torsion_landmarks_ct(leg, knee, side=side),
                     'angle': calculate_femoral_torsion_ct(leg, knee, side=side)}
    return out, half['leg'].array.shape[0], half['knee'].array.shape[0]


@pytest.mark.needs_ct
@pytest.mark.parametrize('probe', PROBES)
def test_ct_head_centre_is_mirror_equivariant(ct, probe):
    """CT head centre (a continuous index from the physical sphere fit)."""
    res, n_leg, _ = _ct_results(PROBES[probe])
    ref, prb = res['ref']['line'], res['probe']['line']
    assert _mirror_x(prb[0], n_leg) == pytest.approx(np.asarray(ref[0], float), abs=1e-6), (ref[0], prb[0])


@pytest.mark.needs_ct
@pytest.mark.parametrize('probe', PROBES)
def test_ct_neck_point_is_mirror_equivariant(ct, probe):
    """CT neck base (Murphy)."""
    res, n_leg, _ = _ct_results(PROBES[probe])
    ref, prb = res['ref']['line'], res['probe']['line']
    assert _mirror_x(prb[1], n_leg) == pytest.approx(np.asarray(ref[1], float), abs=1e-6), (ref[1], prb[1])


@pytest.mark.needs_ct
@pytest.mark.parametrize('probe', PROBES)
def test_ct_knee_condylar_line_is_mirror_equivariant(ct, probe):
    res, _, n_knee = _ct_results(PROBES[probe])
    ref, prb = res['ref']['landmarks'], res['probe']['landmarks']
    assert _same_pair(_mirror_x(prb['knee_start'], n_knee), _mirror_x(prb['knee_end'], n_knee),
                      ref['knee_start'], ref['knee_end'], abs_tol=0), (ref, prb)


@pytest.mark.needs_ct
@pytest.mark.parametrize('probe', PROBES)
def test_ct_torsion_angle_is_mirror_equivariant(ct, probe):
    res, _, _ = _ct_results(PROBES[probe])
    assert res['probe']['angle'] == pytest.approx(res['ref']['angle'], abs=1e-6)

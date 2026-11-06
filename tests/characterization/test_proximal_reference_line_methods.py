"""All three advertised proximal-reference-line methods resolve, and tomczak's values are pinned.

``get_proximal_reference_line`` accepts ``method`` in ``{'lee', 'murphy', 'tomczak'}`` -- named in
``calculate_femoral_torsion``'s docstring and admitted by its assertion -- but ``'tomczak'`` raised
``AttributeError: 'Segmentation' object has no attribute 'get_shape'`` for its whole advertised
life. ``get_trochanter_major`` read ``hip_image.get_shape()``, which no ``Image`` has ever defined;
the accessor is the ``shape`` property. No caller in the repo exercises tomczak, so nothing caught
it, and ``AttributeError`` is outside the ``(RuntimeError, AssertionError, ValueError)`` tuple the
batch scripts and ``compute_torsion.py`` catch -- a cohort run that asked for tomczak would have
aborted rather than written a NaN row.

**The tomczak goldens below are NOT validated against anything.** They are the first values the
method has ever produced, pinned only so that a future change to it is detected rather than silently
absorbed. Nothing in the repo compares them with a reader, a publication or another method, and the
trochanter-major landmark they rest on has no characterization test of its own. Do not quote them as
a measurement; treat them as a regression anchor.

Lee and Murphy are asserted here too, against the same fixture, because fixing the accessor touches
``get_trochanter_major`` and the shared head-centre resolve above the method branch -- so the point
of the pair below is that neither moved.
"""
import numpy as np
import pytest

from morphometry.femur import get_proximal_reference_line
from morphometry.image_io import split_image_left_right
from tests.conftest import assert_golden


@pytest.mark.needs_augsburg
@pytest.mark.parametrize('method', ['lee', 'murphy', 'tomczak'])
def test_every_advertised_method_resolves(augsburg_pa000001, method):
    """Each advertised method returns a usable reference line rather than raising.

    :param augsburg_pa000001: The PA000001 fixture loader.
    :param method: The proximal reference-line method under test.
    """
    hip = augsburg_pa000001['hip']
    left, _ = split_image_left_right(hip)
    out = get_proximal_reference_line(left, method=method, side='left')

    # tomczak additionally returns the trochanter-major radius, hence four values rather than three
    assert len(out) == (4 if method == 'tomczak' else 3)
    start, end, r_fh = out[0], out[1], out[2]
    assert np.all(np.isfinite(np.asarray(start, dtype=float)))
    assert np.all(np.isfinite(np.asarray(end, dtype=float)))
    assert r_fh > 0, 'the femoral head radius must be positive'
    assert not np.allclose(np.asarray(start, dtype=float), np.asarray(end, dtype=float)), \
        'start and end coincide, so the reference line has no direction'
    if method == 'tomczak':
        assert out[3] > 0, 'the trochanter-major radius must be positive'


@pytest.mark.needs_augsburg
def test_tomczak_reference_line_is_pinned(augsburg_pa000001):
    """Pin tomczak's newly-reachable values so a change to them is detected.

    Unvalidated: see the module docstring. This is a regression anchor, not a measurement.

    :param augsburg_pa000001: The PA000001 fixture loader.
    """
    hip = augsburg_pa000001['hip']
    left, _ = split_image_left_right(hip)
    start, end, r_fh, r_tm = get_proximal_reference_line(left, method='tomczak', side='left')
    # flat: assert_golden's _flatten cannot take an inhomogeneous (list, float, float) tuple
    assert_golden('proximal_reference_line_tomczak_left',
                  [*np.asarray(end, dtype=float).tolist(), float(r_fh), float(r_tm)])

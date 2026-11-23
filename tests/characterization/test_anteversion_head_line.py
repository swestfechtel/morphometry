"""Characterization of the NaKo antetorsion measured against the inter-head line.

``calculate_anteversion_head_line`` measures both image halves' Murphy neck axes against the line
through the two femoral head centres (the NaKo readers' reference). Pinned here on the first NaKo
sample hip: the golden values, mirror equivariance (a reflected hip gives the two values swapped),
and that the x-axis fallback reproduces ``calculate_anteversion`` exactly.
"""
import numpy as np
import pytest

from tests.conftest import assert_golden, split_left_right
from morphometry.image_io import reflect_image_left_right
from morphometry.measurements import hip as H

pytestmark = pytest.mark.needs_nako


def test_anteversion_head_line_golden(nako_sample):
    """Both image halves' head-line antetorsion on the NaKo sample, measured against the head line."""
    values = H.calculate_anteversion_head_line(nako_sample, 1, isotropic=True)
    assert values[2] == 'head_line'
    assert_golden("anteversion_head_line", list(values[:2]))


def test_anteversion_head_line_is_mirror_equivariant(nako_sample):
    """The reflected hip reads the original values with the two image sides swapped."""
    left, right, reference = H.calculate_anteversion_head_line(nako_sample, 1, isotropic=True)
    m_left, m_right, m_reference = H.calculate_anteversion_head_line(
        reflect_image_left_right(nako_sample), 1, isotropic=True)
    assert reference == m_reference == 'head_line'
    np.testing.assert_allclose((m_left, m_right), (right, left), atol=1e-6)


def test_x_axis_fallback_reproduces_calculate_anteversion(nako_sample, monkeypatch):
    """With the right side failing, the left value is exactly calculate_anteversion's."""
    left_half, _ = split_left_right(nako_sample)
    expected = H.calculate_anteversion(left_half, 'left', 1, isotropic=True)

    real = H.get_proximal_reference_line

    def right_fails(hip_image, side='left', **kwargs):
        if side == 'right':
            raise ValueError('simulated right-side landmark failure')
        return real(hip_image, side=side, **kwargs)

    monkeypatch.setattr(H, 'get_proximal_reference_line', right_fails)
    left, right, reference = H.calculate_anteversion_head_line(nako_sample, 1, isotropic=True)
    assert reference == 'x_axis'
    assert np.isnan(right)
    assert left == pytest.approx(expected, abs=1e-9)

"""The post-reorientation orientation check (``Image.orientation_ok``) and ``scan_qc_flags``.

``Image.transform_coordinate_system`` only permutes / flips voxel axes, so it cannot make an
oblique acquisition axis-aligned. Three dfg_knees scans (K7/K8/K11_90_belastet) have two
voxel axes ~45 deg between the anatomical A-P and S-I axes; nibabel labels axes greedily in
voxel-axis order (``io_orientation``), the near tie resolves differently once the axes are
permuted, and the "LPI" result is labelled e.g. LSP. The reorientation is deliberately left
as is; these tests pin that the mismatch is *recorded* and turned into a QC flag.
"""
import logging
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from morphometry.cartilage.knee import scan_qc_flags
from morphometry.cartilage.tibia_regions import TRANSFER_MAX_FAR_FRACTION
from morphometry.image_io import Segmentation

# Direction cosines (unit columns, RAS world) of K7_90_belastet's affine, rounded to three
# decimals. Column 0 is (0.086, -0.703, -0.706): its S-I and A-P components differ by 0.003.
_K7_DIRECTIONS = np.array([[0.086, -0.474, 0.877],
                           [-0.703, 0.595, 0.390],
                           [-0.706, -0.650, -0.282]])


def _segmentation(directions: np.ndarray, shape=(6, 7, 8), spacing=(0.3125, 0.3125, 0.75)):
    """
    A small labelled ``Segmentation`` with the given voxel-axis directions.

    :param directions: 3x3 matrix whose columns are the voxel axes in RAS world; it is
        replaced by its nearest orthonormal matrix (polar decomposition).
    :param shape: Array shape.
    :param spacing: Voxel spacing per axis (mm).
    :return: The ``Segmentation`` (not yet reoriented).
    """
    u, _, vt = np.linalg.svd(directions)
    affine = np.eye(4)
    affine[:3, :3] = (u @ vt) * np.asarray(spacing)
    affine[:3, 3] = (10.0, -20.0, 30.0)
    data = np.zeros(shape, dtype=np.float32)
    data[1:3, 2:5, 3:6] = 4
    return Segmentation.from_nibabel(nib.Nifti1Image(data, affine))


def test_axis_aligned_scan_lands_in_lpi():
    """An RAS-aligned image reoriented to LPI is LPI: ``orientation_ok`` is True, no flag."""
    seg = _segmentation(np.eye(3))
    assert seg.orientation_ok is None          # nothing requested yet
    seg.transform_coordinate_system()
    assert seg.achieved_axcodes == ('L', 'P', 'I')
    assert seg.orientation_ok is True
    assert scan_qc_flags(seg) == ''


def test_moderately_oblique_scan_lands_in_lpi():
    """A clear (non-tied) obliquity, like the 93 normal dfg_knees scans, still lands in LPI."""
    angle = np.deg2rad(30)                      # rotation about the L-R axis
    rotation = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    seg = _segmentation(rotation)
    seg.transform_coordinate_system()
    assert seg.orientation_ok is True


def test_near_tie_oblique_scan_is_flagged(caplog):
    """K7's geometry: the raw affine reads IAR, the reoriented one LSP instead of LPI."""
    seg = _segmentation(_K7_DIRECTIONS)
    assert nib.aff2axcodes(seg.image.affine) == ('I', 'A', 'R')
    with caplog.at_level(logging.WARNING, logger='morphometry'):
        seg.transform_coordinate_system()
    assert seg.axcodes == ('L', 'P', 'I')       # requested
    assert seg.achieved_axcodes == ('L', 'S', 'P')
    assert seg.orientation_ok is False
    assert any('LSP' in r.getMessage() for r in caplog.records)
    assert scan_qc_flags(seg) == 'non_lpi_orientation:LSP'


def test_reorientation_keeps_world_geometry_of_flagged_scan():
    """The flag is only a label problem: every voxel keeps its world position."""
    seg = _segmentation(_K7_DIRECTIONS)
    before = seg.image.affine @ np.r_[np.argwhere(seg.array == 4).mean(0), 1]
    seg.transform_coordinate_system()
    after = seg.image.affine @ np.r_[np.argwhere(seg.array == 4).mean(0), 1]
    np.testing.assert_allclose(after, before, atol=1e-9)


def test_orientation_ok_none_without_requested_axcodes():
    """A crop-like copy with no requested orientation does not claim to be (un)flagged."""
    seg = _segmentation(_K7_DIRECTIONS)
    assert seg.orientation_ok is None
    assert scan_qc_flags(seg) == ''


@pytest.mark.parametrize('far_fraction, expected', [
    (0.0, ''),
    (TRANSFER_MAX_FAR_FRACTION, ''),             # the threshold itself is not flagged
    (0.0342, 'tibia_transfer_far:0.034'),
])
def test_tibia_transfer_flag(far_fraction, expected):
    """The tibial transfer QC is flagged above TRANSFER_MAX_FAR_FRACTION only."""
    seg = _segmentation(np.eye(3))
    seg.transform_coordinate_system()
    tibia = SimpleNamespace(transfer_qc={'far_fraction': far_fraction})
    assert scan_qc_flags(seg, tibia) == expected


def test_flags_combine_in_order():
    """Both flags: orientation first, ';'-joined (the process_nmdid.py qc_flags format)."""
    seg = _segmentation(_K7_DIRECTIONS)
    seg.transform_coordinate_system()
    tibia = SimpleNamespace(transfer_qc={'far_fraction': 0.5})
    assert scan_qc_flags(seg, tibia) == 'non_lpi_orientation:LSP;tibia_transfer_far:0.500'
    # a tibia that failed (None) or never ran its landmarks (transfer_qc None) adds nothing
    assert scan_qc_flags(seg, None) == 'non_lpi_orientation:LSP'
    assert scan_qc_flags(seg, SimpleNamespace(transfer_qc=None)) == 'non_lpi_orientation:LSP'

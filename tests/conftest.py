"""Shared pytest fixtures and the golden-value comparison helper.

The characterization tests lock in the *current* numeric behaviour of the
measurement functions so that the measurements refactor (moving ``calculate_*``
into ``morphometry.measurements``) can be proven not to change results.

Golden values live in ``tests/golden/*.json`` and are compared with a tolerance.
Run with ``MORPH_UPDATE_GOLDEN=1`` to (re)capture them; without it the tests load
the committed goldens and assert against them (xfail if a golden is missing).
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from morphometry.image_io import Segmentation
from tests import data_paths

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE_GOLDEN = os.environ.get("MORPH_UPDATE_GOLDEN") == "1"


@pytest.fixture(autouse=True)
def _seed():
    """Seed numpy before every test so stochastic methods (KMeans, etc.) are stable."""
    np.random.seed(0)
    yield


def _load_seg(path: Path) -> Segmentation:
    """Load a NIfTI segmentation in the standard LPI orientation with outliers removed."""
    seg = Segmentation("nibabel")
    seg.read_image(str(path))
    seg.transform_coordinate_system()
    seg.remove_outliers()
    return seg


@pytest.fixture
def augsburg_pa000001():
    """The three Augsburg PA000001 segmentations (hip, knee, ankle), LPI + de-outliered."""
    base = data_paths.AUGSBURG_PA000001
    files = {n: base / f"{n}_seg.nii.gz" for n in ("hip", "knee", "ankle")}
    if not all(p.exists() for p in files.values()):
        pytest.skip(f"Augsburg PA000001 data not found under {base}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {n: _load_seg(p) for n, p in files.items()}


@pytest.fixture
def nako_sample():
    """The first NaKo hip segmentation (isotropic; femur=1, cartilage=2, acetabulum=3)."""
    d = data_paths.NAKO_SAMPLE_DIR
    if not d.exists():
        pytest.skip(f"NaKo sample dir not found at {d}")
    files = sorted(d.glob("*.nii.gz"))
    if not files:
        pytest.skip(f"No NaKo segmentations in {d}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _load_seg(files[0])


@pytest.fixture
def ct_whole_leg():
    """A whole-leg CT segmentation, or skip if MORPH_CT_SAMPLE is not set."""
    p = data_paths.CT_WHOLE_LEG
    if p is None or not p.exists():
        pytest.skip("No whole-leg CT sample (set MORPH_CT_SAMPLE to enable CT tests)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _load_seg(p)


def split_left_right(seg: Segmentation):
    """Split a both-sides mask into (left_image, right_image) Segmentations at shape[0]//2.

    Mirrors the image-side split used throughout the pipelines ("left"/"right" is
    the image side, not the patient side).
    """
    half = seg.array.shape[0] // 2
    left = Segmentation.from_nibabel(nib.Nifti1Image(seg.array[:half], seg.affine, seg.header))
    right = Segmentation.from_nibabel(nib.Nifti1Image(seg.array[half:], seg.affine, seg.header))
    return left, right


def _flatten(value):
    """Flatten a scalar / tuple / array measurement result into a list of floats."""
    arr = np.asarray(value, dtype=float).ravel()
    return [float(x) for x in arr]


def assert_golden(name: str, value, *, rtol: float = 1e-4, atol: float = 1e-3):
    """Compare a measurement result against the committed golden, or capture it.

    With ``MORPH_UPDATE_GOLDEN=1`` the value is written to ``tests/golden/<name>.json``.
    Otherwise the golden is loaded and compared element-wise within tolerance; a
    missing golden xfails so the harness can exist before goldens are captured.

    :param name: Golden file stem (one file per logical measurement group).
    :param value: Scalar, tuple, or array result to record/compare.
    :param rtol: Relative tolerance for the comparison.
    :param atol: Absolute tolerance for the comparison.
    """
    flat = _flatten(value)
    path = GOLDEN_DIR / f"{name}.json"
    if UPDATE_GOLDEN:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(flat, indent=2))
        return
    if not path.exists():
        pytest.xfail(f"golden {name} not captured (run with MORPH_UPDATE_GOLDEN=1)")
    expected = json.loads(path.read_text())
    np.testing.assert_allclose(flat, expected, rtol=rtol, atol=atol,
                               err_msg=f"measurement '{name}' drifted from golden")

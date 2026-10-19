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
import pytest

from morphometry.image_io import Segmentation, split_image_left_right
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
def augsburg_reader_study():
    """A loader for one Augsburg reader-study patient's hip and knee segmentations.

    :return: A function ``load(patient) -> {'hip': Segmentation, 'knee': Segmentation}`` (LPI,
        de-outliered) that skips the test when that patient's files are absent.
    """
    base = data_paths.AUGSBURG_READER_STUDY

    def load(patient: str) -> dict:
        files = {n: base / patient / f"{n}_seg_test.nii.gz" for n in ("hip", "knee")}
        if not all(p.exists() for p in files.values()):
            pytest.skip(f"Augsburg reader-study patient {patient} not found under {base}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {n: _load_seg(p) for n, p in files.items()}

    return load


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


def _load_knee_seg(path: Path) -> Segmentation:
    """Load a ``dfg_knees`` knee segmentation exactly as the knee scripts do.

    Unlike :func:`_load_seg` this does **not** call ``remove_outliers()``: the knee cartilage
    scripts load and reorient only by default (``compute_knee_cartilage_thickness.py`` has an
    opt-in ``--remove-outliers``; ``analyse_thinnest_point_flexion.py`` and
    ``report_knee_cartilage_cohort.py`` never call it), and ``Tibia`` / ``Femur`` clean their
    own cartilage masks anyway (``outlier_ratio``). Loading differently here would pin numbers
    no default run produces.

    :param path: Path to the segmentation NIfTI.
    :return: The LPI-oriented ``Segmentation``.
    """
    seg = Segmentation("nibabel")
    seg.read_image(str(path))
    seg.transform_coordinate_system()
    return seg


@pytest.fixture
def dfg_knees_dir():
    """The ``dfg_knees`` segmentation directory, or skip if it is absent.

    :return: ``Path`` of the directory holding ``K<subject>_<angle>_<load>.nii.gz``
        (override with ``MORPH_DFG_KNEES_DIR``).
    """
    d = data_paths.DFG_KNEES_DIR
    if not d.exists():
        pytest.skip(f"dfg_knees segmentation dir not found at {d} (set MORPH_DFG_KNEES_DIR)")
    return d


@pytest.fixture
def dfg_knee_file(dfg_knees_dir):
    """A locator for one ``dfg_knees`` scan file, skipping when that scan is absent.

    :return: A function ``path(name) -> Path`` for the *raw* file (not loaded), e.g.
        ``path('K1_0_belastet')``.
    """

    def path(name: str) -> Path:
        p = data_paths.dfg_knee_path(name)
        if not p.exists():
            pytest.skip(f"dfg_knees scan {name} not found at {p}")
        return p

    return path


@pytest.fixture
def dfg_knee(dfg_knee_file):
    """A loader for one ``dfg_knees`` scan, LPI-oriented (no outlier removal).

    :return: A function ``load(name) -> Segmentation`` that skips the test when the scan is
        absent (see :func:`_load_knee_seg` for why outliers are kept).
    """

    def load(name: str) -> Segmentation:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _load_knee_seg(dfg_knee_file(name))

    return load


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
    the image side, not the patient side) by delegating to
    ``morphometry.image_io.split_image_left_right``, so the right half carries its own,
    correctly translated geometry.
    :param seg: The both-sides segmentation.
    :return: ``(left, right)`` Segmentations.
    """
    return split_image_left_right(seg)


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

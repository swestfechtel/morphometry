"""The cartilage re-exports of ``morphometry.measurements`` are loaded lazily (PEP 562), so
importing any measurement (e.g. the torsion docker image's entry point) never loads cartilage code."""
import subprocess
import sys

import pytest

import morphometry.measurements as M

_CARTILAGE_NAMES = [
    "calculate_tibial_cartilage_thickness",
    "calculate_femoral_cartilage_thickness",
    "calculate_knee_cartilage_thickness",
    "calculate_thinnest_cartilage_area",
    "plot_knee_segments",
    "plot_knee_thickness",
    "find_thinnest_area",
    "find_thinnest_area_by_compartment",
    "split_tibia_thickness_by_compartment",
    "plot_thinnest_point",
]

_PROBE = """
import sys
sys.path.insert(0, {root!r})
exec(compile(__import__('ast').Module(body=[n for n in __import__('ast').parse(open({path!r}).read()).body
     if isinstance(n, (__import__('ast').Import, __import__('ast').ImportFrom))], type_ignores=[]), 'imports', 'exec'))
print([m for m in sys.modules if m.startswith('morphometry.cartilage') or m == 'morphometry.measurements.cartilage'])
"""


def test_torsion_imports_do_not_load_cartilage():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    path = root / "morphometry" / "docker" / "compute_torsion.py"
    out = subprocess.run([sys.executable, "-c", _PROBE.format(root=str(root), path=str(path))],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    assert out == "[]", out


@pytest.mark.parametrize("name", _CARTILAGE_NAMES)
def test_cartilage_names_still_importable(name):
    import morphometry.measurements.cartilage as C
    assert name in M.__all__
    assert callable(getattr(M, name))
    assert getattr(M, name) is getattr(C, name)
    assert name in dir(M)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        M.nope

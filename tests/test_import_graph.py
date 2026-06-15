"""Enforce the package dependency direction after the measurements refactor.

The intended acyclic layering is:

    measurements/*  ->  region get_* modules  ->  geometry / utils / constants / image_io / bresenham

So: region modules must NOT import from ``morphometry.measurements``; and the leaf
infra modules (``geometry``, ``utils``, ``constants``) must NOT import region or
measurement modules. Modules that do not exist yet (e.g. before a phase lands) are
simply skipped, so this test is valid throughout the migration.
"""
import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "morphometry"
REGION_MODULES = {"hip", "femur", "knee", "tibia", "ankle", "whole_leg"}
LEAF_MODULES = {"geometry", "utils", "constants"}


def _imported_morphometry_modules(py_file: Path) -> set[str]:
    """Return the set of `morphometry.*` dotted module paths imported by a file."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("morphometry"):
            found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("morphometry"):
                    found.add(alias.name)
    return found


@pytest.mark.parametrize("region", sorted(REGION_MODULES))
def test_region_modules_do_not_import_measurements(region):
    f = PKG / f"{region}.py"
    if not f.exists():
        pytest.skip(f"{region}.py absent")
    offenders = [m for m in _imported_morphometry_modules(f) if "measurements" in m]
    assert not offenders, f"{region}.py must not import measurements: {offenders}"


@pytest.mark.parametrize("leaf", sorted(LEAF_MODULES))
def test_leaf_modules_do_not_import_regions_or_measurements(leaf):
    f = PKG / f"{leaf}.py"
    if not f.exists():
        pytest.skip(f"{leaf}.py absent")
    imported = _imported_morphometry_modules(f)
    bad = [m for m in imported
           if "measurements" in m or any(m.endswith(f".{r}") for r in REGION_MODULES)]
    assert not bad, f"{leaf}.py must not import region/measurement modules: {bad}"


def test_measurements_package_layout():
    """Once measurements/ exists, it must be a per-region subpackage."""
    meas = PKG / "measurements"
    if not meas.exists():
        pytest.skip("measurements/ not created yet")
    assert (meas / "__init__.py").exists(), "measurements/ must be a package"

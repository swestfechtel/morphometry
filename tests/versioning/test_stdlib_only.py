"""``versioning/`` runs in CI with a bare interpreter, so it may import only the standard library."""
import ast
import sys
from pathlib import Path

import pytest

VERSIONING = Path(__file__).resolve().parents[2] / "versioning"


@pytest.mark.parametrize("path", sorted(VERSIONING.glob("*.py")), ids=lambda p: p.name)
def test_only_stdlib_imports(path: Path) -> None:
    """Fail on any top-level import in ``path`` that is not stdlib or intra-package.

    :param path: a module of the ``versioning`` package.
    """
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or node.module == "__future__":
                continue
            names = [node.module]
        else:
            continue
        bad += [n for n in names
                if n.split(".")[0] not in sys.stdlib_module_names and n.split(".")[0] != "versioning"]
    assert not bad, f"{path.name} imports non-stdlib modules: {bad}"

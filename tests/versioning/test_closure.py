"""Tests for the ``ast``-based import-closure walker (``versioning.closure``)."""
from pathlib import Path

import pytest

from versioning.closure import UnresolvedImport, import_closure

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(root: Path, rel: str, text: str) -> Path:
    """Write ``text`` to ``root/rel``, creating parent directories.

    :param root: directory the relative path is resolved against.
    :param rel: POSIX-style relative path of the file.
    :param text: file contents.
    :return: the absolute path written.
    """
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_closure_follows_imports(tmp_path):
    """``from pkg import mod``, function-local relative imports and package ``__init__`` all count."""
    _write(tmp_path, "morphometry/__init__.py", "")
    _write(tmp_path, "morphometry/a.py", "from morphometry import b\n")
    _write(tmp_path, "morphometry/b.py", "def g():\n    from .c import f\n    return f()\n")
    _write(tmp_path, "morphometry/c.py", "def f():\n    return 1\n")
    _write(tmp_path, "morphometry/unused.py", "import os\n")
    entry = _write(tmp_path, "entry.py", "import os\nimport numpy\nimport morphometry.a\n")

    closure = import_closure([entry], tmp_path)

    assert closure == {
        Path("morphometry/__init__.py"),
        Path("morphometry/a.py"),
        Path("morphometry/b.py"),
        Path("morphometry/c.py"),
    }


def test_closure_from_import_of_name_resolves_to_module(tmp_path):
    """``from pkg.mod import name`` (a name, not a submodule) adds ``pkg/mod.py`` and ``pkg/__init__``."""
    _write(tmp_path, "morphometry/__init__.py", "")
    _write(tmp_path, "morphometry/sub/__init__.py", "from .inner import thing\n")
    _write(tmp_path, "morphometry/sub/inner.py", "thing = 1\n")
    entry = _write(tmp_path, "entry.py", "from morphometry.sub import thing\n")

    assert import_closure([entry], tmp_path) == {
        Path("morphometry/__init__.py"),
        Path("morphometry/sub/__init__.py"),
        Path("morphometry/sub/inner.py"),
    }


def test_closure_unresolved_raises(tmp_path):
    """A ``from pkg import x`` where ``x`` is neither a module nor a name in ``__init__`` fails loudly."""
    _write(tmp_path, "morphometry/__init__.py", "defined = 1\n")
    entry = _write(tmp_path, "entry.py", "\nfrom morphometry import missing_mod\n")

    with pytest.raises(UnresolvedImport) as info:
        import_closure([entry], tmp_path)
    assert info.value.path == entry
    assert info.value.lineno == 2
    assert info.value.name == "morphometry.missing_mod"


def test_closure_unresolved_module_raises(tmp_path):
    """``import pkg.nope`` with no such file fails loudly."""
    _write(tmp_path, "morphometry/__init__.py", "")
    entry = _write(tmp_path, "entry.py", "import morphometry.nope\n")

    with pytest.raises(UnresolvedImport):
        import_closure([entry], tmp_path)


def test_closure_dynamic_import_raises(tmp_path):
    """``importlib.import_module("pkg.x")`` cannot be followed statically, so it fails loudly."""
    _write(tmp_path, "morphometry/__init__.py", "")
    _write(tmp_path, "morphometry/x.py", "")
    entry = _write(
        tmp_path, "entry.py", "import importlib\nm = importlib.import_module('morphometry.x')\n"
    )

    with pytest.raises(UnresolvedImport) as info:
        import_closure([entry], tmp_path)
    assert info.value.lineno == 2


def test_closure_dunder_import_raises(tmp_path):
    """``__import__("pkg...")`` is rejected the same way."""
    _write(tmp_path, "morphometry/__init__.py", "")
    entry = _write(tmp_path, "entry.py", "m = __import__('morphometry')\n")

    with pytest.raises(UnresolvedImport):
        import_closure([entry], tmp_path)


def test_closure_relative_dynamic_import_raises(tmp_path):
    """``import_module(".femur", "morphometry")`` names the package through its anchor argument."""
    entry = _write(
        tmp_path, "entry.py", "import importlib\nimportlib.import_module('.femur', 'morphometry')\n"
    )

    with pytest.raises(UnresolvedImport):
        import_closure([entry], tmp_path)


def test_closure_package_kwarg_dynamic_import_raises(tmp_path):
    """The ``package=`` keyword is inspected too."""
    entry = _write(
        tmp_path,
        "entry.py",
        "from importlib import import_module\nimport_module('.femur', package='morphometry')\n",
    )

    with pytest.raises(UnresolvedImport):
        import_closure([entry], tmp_path)


def test_closure_star_import_raises(tmp_path):
    """``from morphometry.x import *`` is refused rather than guessed at."""
    _write(tmp_path, "morphometry/__init__.py", "")
    _write(tmp_path, "morphometry/x.py", "A = 1\n")
    entry = _write(tmp_path, "entry.py", "from morphometry.x import *\n")

    with pytest.raises(UnresolvedImport) as info:
        import_closure([entry], tmp_path)
    assert info.value.name == "morphometry.x.*"


def test_closure_ignores_dynamic_import_of_other_packages(tmp_path):
    """A dynamic import of another package (or of a non-literal) is not this walker's concern."""
    entry = _write(
        tmp_path,
        "entry.py",
        "import importlib\nimportlib.import_module('api.tasks')\nimportlib.import_module(name)\n",
    )

    assert import_closure([entry], tmp_path) == set()


def test_real_torsion_closure():
    """The real torsion entry point reaches the measurement code but not the deformation research code.

    ``morphometry/cartilage/thickness.py`` *is* in the closure, and correctly so: importing
    ``morphometry.measurements.femur`` runs ``morphometry/measurements/__init__.py``, which
    imports ``measurements.cartilage`` → ``cartilage.thickness``. (Cross-checked against
    ``sys.modules`` after running compute_torsion.py's imports: identical module sets.)
    The cartilage deformation-map modules are imported by nothing on that path.
    """
    closure = import_closure([REPO_ROOT / "morphometry/docker/compute_torsion.py"], REPO_ROOT)

    assert Path("morphometry/measurements/femur.py") in closure
    assert Path("morphometry/image_io.py") in closure
    assert Path("morphometry/__init__.py") in closure
    assert Path("morphometry/cartilage/thickness.py") in closure
    assert Path("morphometry/cartilage/deformation.py") not in closure
    assert Path("morphometry/cartilage/registration.py") not in closure
    assert Path("morphometry/docker/test.py") not in closure

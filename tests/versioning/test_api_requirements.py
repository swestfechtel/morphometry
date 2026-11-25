"""The API image's pinned requirement file must be a closed, root-consistent set."""
import re
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[2]


def _norm(name: str) -> str:
    """Canonicalise a distribution name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(path: Path) -> dict[str, str]:
    """Parse ``name==version`` lines of a requirements file.

    :param path: requirements file.
    :return: mapping of normalised name to version (comments/blank lines ignored).
    """
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==")
            pins[_norm(name.strip())] = version.strip()
    return pins


def test_api_requirements_closed():
    """Every dependency of every listed package that applies on Linux/py3.10 is listed."""
    pins = _pins(REPO / "api" / "requirements.txt")
    missing = []
    for name in pins:
        for req in metadata.requires(name) or []:
            r = Requirement(req)
            if r.marker and not r.marker.evaluate({"extra": ""}):
                continue
            if _norm(r.name) not in pins:
                missing.append(f"{name} -> {r.name}")
    assert not missing, missing


def test_api_requirements_equal_root_pins():
    """Each pin equals the root requirements.txt version (where the root lists it)."""
    api = _pins(REPO / "api" / "requirements.txt")
    root = _pins(REPO / "requirements.txt")
    wrong = {k: (v, root[k]) for k, v in api.items() if k in root and root[k] != v}
    assert not wrong, wrong


def test_api_requirements_superset_of_morphometry():
    """Everything the library image pins is pinned identically here."""
    api = _pins(REPO / "api" / "requirements.txt")
    lib = _pins(REPO / "morphometry" / "requirements.txt")
    assert {k: v for k, v in lib.items() if api.get(k) != v} == {}


def test_api_requirements_exclude_inference_stack():
    """The inference stack runs in the nnunet image and must not be installed here."""
    api = _pins(REPO / "api" / "requirements.txt")
    assert not {"torch", "nnunetv2", "totalsegmentator"} & set(api)

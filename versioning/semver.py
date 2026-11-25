"""Plain ``MAJOR.MINOR.PATCH`` versions and release-tag lookup (standard library only)."""
import re
import subprocess
from pathlib import Path
from typing import NamedTuple

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class Version(NamedTuple):
    """An ordered ``MAJOR.MINOR.PATCH`` version."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def parse_version(text: str) -> Version:
    """Parse a plain ``MAJOR.MINOR.PATCH`` string.

    :param text: version text; no ``v`` prefix and no pre-release suffix allowed.
    :return: the parsed, orderable :class:`Version`.
    :raises ValueError: if ``text`` is not of the form ``<int>.<int>.<int>``.
    """
    match = _VERSION_RE.match(text)
    if match is None:
        raise ValueError(f"invalid version {text!r} (expected MAJOR.MINOR.PATCH)")
    return Version(*(int(g) for g in match.groups()))


def last_release_tag(repo: Path) -> str | None:
    """Return the newest ``v*`` release tag reachable from HEAD.

    :param repo: path of the git repository.
    :return: the tag name (e.g. ``"v1.0.0"``), or ``None`` if no ``v*`` tag exists at all.
    :raises RuntimeError: if ``v*`` tags exist but none is reachable from HEAD.
    """
    described = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
        cwd=repo, capture_output=True, text=True,
    )
    if described.returncode == 0:
        return described.stdout.strip()
    tags = subprocess.run(
        ["git", "tag", "-l", "v[0-9]*"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    if tags:
        raise RuntimeError("v* tags exist but none is reachable from HEAD (shallow clone?)")
    return None

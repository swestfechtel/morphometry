"""Plain ``MAJOR.MINOR.PATCH`` versions and release-tag lookup (standard library only)."""
import re
from pathlib import Path
from typing import NamedTuple

from versioning._git import git

_VERSION_RE = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")


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
    match = _VERSION_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid version {text!r} (expected MAJOR.MINOR.PATCH)")
    return Version(*(int(g) for g in match.groups()))


def last_release_tag(repo: Path, ref: str = "HEAD") -> str | None:
    """Return the newest ``v*`` release tag reachable from ``ref``.

    :param repo: path of the git repository.
    :param ref: commit-ish to search back from, default ``HEAD``.
    :return: the tag name (e.g. ``"v1.0.0"``), or ``None`` if no ``v*`` tag exists at all.
    :raises RuntimeError: if ``v*`` tags exist but none is reachable from ``ref``.
    """
    described = git(repo, "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", ref, check=False)
    if described.returncode == 0:
        return described.stdout.strip()
    tags = git(repo, "tag", "-l", "v[0-9]*").stdout.strip()
    if tags:
        raise RuntimeError(f"v* tags exist but none is reachable from {ref} (shallow clone?)")
    return None

"""Release manifest: which archived docker images belong to which release version.

``release/images.json`` maps ``"MAJOR.MINOR.PATCH"`` to ``{"nnunet": ref, "torsion": ref, ...}``
where each ref is a ``repo@sha256:...`` digest reference. Versions that did not rebuild an
image (e.g. the gated nnUNet image) are resolved by falling back to the newest lower entry.
"""
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "release" / "images.json"


def _parse(version: str) -> tuple[int, int, int]:
    """Parse ``MAJOR.MINOR.PATCH`` into an int tuple (local, so the api image needs no ``versioning/``)."""
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def load_manifest(path: Path | None = None) -> dict[str, dict]:
    """Read the release manifest.

    :param path: manifest file; defaults to the module-level ``MANIFEST_PATH`` (looked up at call time).
    :return: version -> image-entry mapping; ``{}`` if the file does not exist.
    """
    path = MANIFEST_PATH if path is None else path
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def images_for(version: str, manifest: dict) -> tuple[str | None, dict]:
    """Find the image entry that applies to ``version``.

    :param version: running release version, e.g. ``"1.4.0"``.
    :param manifest: version -> entry mapping as returned by :func:`load_manifest`.
    :return: ``(images_from, entry)``: the entry for ``version`` if present, else the entry of
        the newest version lower than it, else ``(None, {})``.
    """
    if version in manifest:
        return version, manifest[version]
    target = _parse(version)
    lower = [v for v in manifest if _parse(v) < target]
    if not lower:
        return None, {}
    best = max(lower, key=_parse)
    return best, manifest[best]


def released_digests(manifest: dict | None = None) -> set[str]:
    """Collect every image reference named in any manifest entry.

    :param manifest: version -> entry mapping; loaded from ``MANIFEST_PATH`` if ``None``.
    :return: the set of all image refs (string values) across all entries.
    """
    if manifest is None:
        manifest = load_manifest()
    return {ref for entry in manifest.values() for ref in entry.values() if isinstance(ref, str)}

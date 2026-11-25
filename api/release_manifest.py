"""Release manifest: which archived docker images belong to which release version.

``release/images.json`` maps ``"MAJOR.MINOR.PATCH"`` to ``{"nnunet": ref, "torsion": ref, ...}``
where each ref is a ``repo@sha256:...`` digest reference. Versions that did not rebuild an
image (e.g. the gated nnUNet image) are resolved by falling back to the newest lower entry.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "release" / "images.json"

# provenance stage key -> manifest image key
STAGE_IMAGE = {"segmentation": "nnunet", "torsion": "torsion"}


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


def sorted_versions(manifest: dict) -> list[str]:
    """Manifest versions in ascending order, skipping (with a warning) keys that are not ``MAJOR.MINOR.PATCH``.

    :param manifest: version -> entry mapping.
    :return: the parseable versions, oldest first.
    """
    good = []
    for v in manifest:
        try:
            good.append((_parse(v), v))
        except (ValueError, AttributeError):
            logger.warning("skipping unparseable release manifest version %r", v)
    return [v for _, v in sorted(good)]


def stage_digests(version: str, manifest: dict) -> dict[str, str]:
    """Image refs a release used per provenance stage.

    :param version: release version; must be a key of ``manifest`` (exact, no fallback).
    :param manifest: version -> entry mapping.
    :return: stage (``segmentation``/``torsion``) -> image ref; stages the entry lacks are omitted.
    :raises KeyError: if ``version`` is not in the manifest.
    """
    entry = manifest[version]
    return {stage: entry[key] for stage, key in STAGE_IMAGE.items() if isinstance(entry.get(key), str)}


def versions_for_digest(stage: str, digest: str | None, manifest: dict) -> list[str]:
    """Release versions whose image for ``stage`` equals ``digest``.

    :param stage: provenance stage, ``segmentation`` or ``torsion``.
    :param digest: recorded image digest; ``None`` matches nothing.
    :param manifest: version -> entry mapping.
    :return: matching versions, ascending; unparseable manifest keys are skipped with a warning.
    """
    if digest is None:
        return []
    key = STAGE_IMAGE[stage]
    return [v for v in sorted_versions(manifest) if manifest[v].get(key) == digest]

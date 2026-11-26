"""Release manifest: which archived docker images belong to which release version.

``release/images.json`` maps ``"MAJOR.MINOR.PATCH"`` to ``{"nnunet": ref, "torsion": ref, ...}``
where each ref is a ``repo@sha256:...`` digest reference. Versions that did not rebuild an
image (e.g. the gated nnUNet image) are resolved by falling back to the newest lower entry.

Where the file is read from: ``MORPH_API_RELEASE_MANIFEST`` (setting ``release_manifest``) when
set, else :data:`MANIFEST_PATH` in the source tree. The api image ships no ``release/`` directory
(an image can never contain its own manifest entry: the entry is written after the image is
pushed), so production compose bind-mounts the deploy checkout's ``release/images.json`` and points
the setting at it. A missing file is an empty manifest; an unreadable or corrupt one is logged as
an error and also treated as empty, so ``/version`` and the examination list never fail on it.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "release" / "images.json"

# provenance stage key -> manifest image key
STAGE_IMAGE = {"segmentation": "nnunet", "torsion": "torsion"}


def _parse(version: str) -> tuple[int, int, int]:
    """Parse ``MAJOR.MINOR.PATCH`` into an int tuple (local, so the api image needs no ``versioning/``).

    :param version: candidate version string.
    :return: ``(major, minor, patch)``.
    :raises ValueError: if ``version`` is not three dot-separated integers.
    :raises AttributeError: if ``version`` is not a string.
    """
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def manifest_path() -> Path:
    """Path the manifest is read from when no explicit path is given.

    :return: the ``release_manifest`` setting (``MORPH_API_RELEASE_MANIFEST``) if set, else the
        module-level :data:`MANIFEST_PATH` (looked up at call time, so tests can patch it).
    """
    from api.settings import get_settings  # local: api.settings imports this module

    configured = get_settings().release_manifest
    return MANIFEST_PATH if configured is None else configured


def load_manifest(path: Path | None = None) -> dict[str, dict]:
    """Read the release manifest.

    :param path: manifest file; defaults to :func:`manifest_path`.
    :return: version -> image-entry mapping; ``{}`` if the file does not exist, and ``{}`` (with an
        error logged) if it cannot be read or is not a JSON object.
    """
    path = manifest_path() if path is None else Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("cannot read release manifest %s (%s); treating it as empty", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.error("release manifest %s is not a JSON object; treating it as empty", path)
        return {}
    return data


def images_for(version: str, manifest: dict) -> tuple[str | None, dict]:
    """Find the image entry that applies to ``version``.

    A pre-release / build suffix (``1.4.0-dev``, the api Dockerfile's default ``0.0.0-dev``) is
    stripped before comparing; a version that is still not ``MAJOR.MINOR.PATCH`` matches nothing
    (logged as a warning) instead of raising.

    :param version: running release version, e.g. ``"1.4.0"``.
    :param manifest: version -> entry mapping as returned by :func:`load_manifest`.
    :return: ``(images_from, entry)``: the entry for ``version`` if present, else the entry of
        the newest version lower than it (unparseable manifest keys are skipped), else
        ``(None, {})``.
    """
    if version in manifest:
        return version, manifest[version]
    base = version.split("-", 1)[0].split("+", 1)[0] if isinstance(version, str) else version
    try:
        target = _parse(base)
    except (ValueError, AttributeError):
        logger.warning("running version %r is not MAJOR.MINOR.PATCH; no release manifest entry applies", version)
        return None, {}
    if base in manifest:
        return base, manifest[base]
    lower = [v for v in sorted_versions(manifest) if _parse(v) < target]
    if not lower:
        return None, {}
    best = lower[-1]
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

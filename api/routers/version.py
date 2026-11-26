"""Version / provenance endpoint (no auth)."""
import re

from fastapi import APIRouter

from api import release_manifest
from api.settings import get_settings
from api.version import get_git_commit, get_version, is_dirty

router = APIRouter()

_PLAIN_SEMVER = re.compile(r"\d+\.\d+\.\d+")


@router.get("/version")
def version() -> dict:
    """Report the running version, git provenance and the resolved docker images.

    :return: ``version`` (suffixed ``-dev`` when no manifest entry exists for exactly this
        version and it is plain ``MAJOR.MINOR.PATCH``; any other ``MORPH_VERSION``, such as the
        Dockerfile default ``0.0.0-dev``, is reported as is and never raises), ``images_from``
        (the manifest version the default images come from; ``None`` if none applies),
        ``git_commit``, ``dirty`` and the ``images`` the worker will run.
    """
    running = get_version()
    images_from, _ = release_manifest.images_for(running, release_manifest.load_manifest())
    settings = get_settings()
    # Only a plain MAJOR.MINOR.PATCH is suffixed; anything else (the Dockerfile default
    # "0.0.0-dev", a garbage value) is reported as is.
    exact = images_from == running or not _PLAIN_SEMVER.fullmatch(running)
    return {
        "version": running if exact else f"{running}-dev",
        "images_from": images_from,
        "git_commit": get_git_commit(),
        "dirty": is_dirty(),
        "images": {"nnunet": settings.resolved_nnunet_image, "torsion": settings.resolved_torsion_image},
    }


@router.get("/releases")
def releases() -> dict:
    """List released versions for the UI's version filter.

    :return: ``{"releases": [{"version", "date", "images": {nnunet, torsion, api, frontend}}]}``, newest
        first; manifest keys that are not ``MAJOR.MINOR.PATCH`` are skipped with a warning.
    """
    manifest = release_manifest.load_manifest()
    return {"releases": [
        {
            "version": v,
            "date": manifest[v].get("date"),
            "images": {k: manifest[v].get(k) for k in ("nnunet", "torsion", "api", "frontend")},
        }
        for v in reversed(release_manifest.sorted_versions(manifest))
    ]}

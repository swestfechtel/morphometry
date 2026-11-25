"""Version / provenance endpoint (no auth)."""
from fastapi import APIRouter

from api import release_manifest
from api.settings import get_settings
from api.version import get_git_commit, get_version, is_dirty

router = APIRouter()


@router.get("/version")
def version() -> dict:
    """Report the running version, git provenance and the resolved docker images.

    :return: ``version`` (suffixed ``-dev`` when no manifest entry exists for exactly this
        version), ``images_from`` (the manifest version the default images come from),
        ``git_commit``, ``dirty`` and the ``images`` the worker will run.
    """
    running = get_version()
    images_from, _ = release_manifest.images_for(running, release_manifest.load_manifest())
    settings = get_settings()
    return {
        "version": running if images_from == running else f"{running}-dev",
        "images_from": images_from,
        "git_commit": get_git_commit(),
        "dirty": is_dirty(),
        "images": {"nnunet": settings.resolved_nnunet_image, "torsion": settings.resolved_torsion_image},
    }

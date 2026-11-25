"""Per-stage provenance records: which image (by digest) produced a stage's output.

Provenance must never fail a job: every lookup here degrades to ``None`` fields and
a logged warning rather than raising.
"""
import json
import logging
import os
import subprocess
from datetime import datetime, timezone

from api.release_manifest import released_digests
from api.version import get_git_commit, get_version, is_dirty

logger = logging.getLogger("api")


def _repository(ref: str) -> str:
    """Return the repository part of an image ref (drop ``@digest`` and ``:tag``).

    A ``:`` before the last ``/`` belongs to a registry host port, not a tag.

    :param ref: image reference, e.g. ``host:5000/a/b:1.0`` or ``a/b@sha256:...``.
    :return: the repository, e.g. ``host:5000/a/b``.
    """
    ref = ref.split("@", 1)[0]
    head, _, last = ref.rpartition("/")
    last = last.split(":", 1)[0]
    return f"{head}/{last}" if head else last


def inspect_image(ref: str) -> tuple[str | None, str | None]:
    """Look up a local image's repo digest and image id via ``docker image inspect``.

    :param ref: image reference to inspect.
    :return: ``(repo_digest, image_id)``; the digest is the ``RepoDigests`` entry whose
        repository equals the ref's (``None`` if absent). ``(None, None)`` on any
        failure (a warning is logged; never raises).
    """
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}} {{.Id}}", ref],
            capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"docker image inspect exited {proc.returncode}")
        out = proc.stdout.strip()
        digests_json, _, image_id = out.rpartition(" ")
        digests = json.loads(digests_json)
        repo = _repository(ref)
        digest = next((d for d in digests if _repository(d) == repo), None)
        return digest, image_id or None
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a job
        logger.warning("Could not inspect image %s: %s", ref, exc)
        return None, None


def stage_record(image_ref: str) -> dict:
    """Build the provenance record for one pipeline stage.

    :param image_ref: the configured docker image ref the stage ran.
    :return: dict with ``api_version, image, digest, released, api_image, git_commit,
        dirty, ran_at``. ``digest`` is the repo digest, else the local image id, else
        ``None``; ``released`` says whether it is a digest in the release manifest.
        Never raises: on error the fields gathered so far are kept.
    """
    record = {"api_version": None, "image": image_ref, "digest": None, "released": False,
              "api_image": os.environ.get("MORPH_API_SELF_IMAGE") or None,
              "git_commit": None, "dirty": None,
              "ran_at": datetime.now(timezone.utc).isoformat()}
    try:
        record["api_version"] = get_version()
        commit = get_git_commit()
        # guard against non-JSON values (e.g. a mocked subprocess returning bytes)
        record["git_commit"] = commit if isinstance(commit, str) else None
        record["dirty"] = is_dirty()
        repo_digest, image_id = inspect_image(image_ref)
        record["digest"] = repo_digest or image_id
        record["released"] = bool(repo_digest) and repo_digest in released_digests()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Provenance incomplete for %s: %s", image_ref, exc)
    return record

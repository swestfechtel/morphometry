"""Examination read/update/delete endpoints."""
from typing import Literal

from fastapi import APIRouter, Depends, Response, status
from sqlmodel import Session

from api import release_manifest, serializers
from api.db import repository
from api.deps import get_session, get_store
from api.errors import NotFoundError
from api.schemas.examination import ExaminationList, ExaminationUpdate
from api.storage.store import Store

router = APIRouter(prefix="/examinations", tags=["examinations"])


def _build_version_filter(version: str, stage: str, manifest: dict) -> repository.VersionFilter:
    """Translate the ``version``/``stage`` query into a repository filter.

    :param version: a release version, or the special values ``unreleased`` / ``none``.
    :param stage: ``any``, ``torsion`` or ``segmentation``.
    :param manifest: version -> entry mapping.
    :return: the filter.
    :raises NotFoundError: for a version that is neither special nor a parseable released version in
        the manifest (lists the known ones).
    """
    stages = tuple(release_manifest.STAGE_IMAGE) if stage == "any" else (stage,)
    if version in ("unreleased", "none"):
        return repository.VersionFilter(stages=stages, mode=version)
    released = release_manifest.sorted_versions(manifest)  # only parseable MAJOR.MINOR.PATCH keys
    if version not in released:
        known = ", ".join(released) or "none"
        raise NotFoundError(f"Unknown version {version}; known versions: {known}")
    return repository.VersionFilter(stages=stages, mode="digest",
                                    digests=release_manifest.stage_digests(version, manifest))


@router.get("/", response_model=ExaminationList)
def list_examinations(version: str | None = None,
                      stage: Literal["any", "torsion", "segmentation"] = "any",
                      session: Session = Depends(get_session)):
    """List examinations, optionally only those produced by a release's images.

    :param version: release version (matched by recorded image digest), ``unreleased`` or ``none``;
        omitted means no filtering.
    :param stage: which stage's digest must match (``any`` = either).
    """
    # envelope shape {"examinations": [...]} preserved for the UI (reads result.examinations)
    manifest = release_manifest.load_manifest()
    vf = _build_version_filter(version, stage, manifest) if version else None
    active = repository.active_jobs_by_examination(session)
    return ExaminationList(examinations=[
        serializers.to_summary(row, active.get(row.id), manifest)
        for row in repository.list_examinations(session, version_filter=vf)
    ])


@router.get("/{examination_id}")
def get_examination(examination_id: str, session: Session = Depends(get_session),
                    store: Store = Depends(get_store)):
    row = repository.get_examination(session, examination_id)
    if row is None:
        raise NotFoundError(f"Examination {examination_id} not found")
    return serializers.to_detail(row, store)


@router.patch("/{examination_id}")
def update_examination(examination_id: str, update: ExaminationUpdate,
                       session: Session = Depends(get_session)):
    row = repository.get_examination(session, examination_id)
    if row is None:
        raise NotFoundError(f"Examination {examination_id} not found")
    if update.status is not None:
        row.status = update.status.value
    if update.landmarks is not None:
        row.landmarks = update.landmarks
    repository.upsert_examination(session, row)
    return {"status": "updated"}


@router.delete("/{examination_id}", status_code=status.HTTP_205_RESET_CONTENT)
def delete_examination(examination_id: str, session: Session = Depends(get_session),
                       store: Store = Depends(get_store)):
    if not repository.delete_examination(session, examination_id):
        raise NotFoundError(f"Examination {examination_id} not found")
    store.delete_examination(examination_id)
    # explicit empty body: a 205 must not carry content (avoids a Content-Length mismatch)
    return Response(status_code=status.HTTP_205_RESET_CONTENT)

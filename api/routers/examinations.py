"""Examination read/update/delete endpoints."""
from fastapi import APIRouter, Depends, Response, status
from sqlmodel import Session

from api import serializers
from api.db import repository
from api.deps import get_session, get_store
from api.errors import NotFoundError
from api.schemas.examination import ExaminationList, ExaminationUpdate
from api.storage.store import Store

router = APIRouter(prefix="/examinations", tags=["examinations"])


@router.get("/", response_model=ExaminationList)
def list_examinations(session: Session = Depends(get_session)):
    # envelope shape {"examinations": [...]} preserved for the UI (reads result.examinations)
    return ExaminationList(examinations=[serializers.to_summary(row) for row in repository.list_examinations(session)])


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

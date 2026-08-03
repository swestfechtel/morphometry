"""Upload endpoints (UI single/multi series, series-selection, Orthanc streaming)."""
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from api import runtime, serializers
from api.db import repository
from api.db.engine import session_scope
from api.deps import get_queue, get_settings, get_store
from api.errors import IngestError, NotFoundError
from api.ingest import dicom
from api.ingest.orthanc import load_rules, match_rule
from api.schemas.enums import ExaminationStatus, JobKind
from api.schemas.jobs import JobCreated
from api.schemas.uploads import ExaminationCreated, OrthancInstanceMeta, SeriesSelection
from api.settings import Settings
from api.storage.store import Store
from api.tasks.dispatch import dispatch_job
from api.tasks.queue import TaskQueue

router = APIRouter(tags=["uploads"])

_RULES_PATH = Path(__file__).resolve().parent.parent / "filter_rules.json"


async def _save_uploads(files: list[UploadFile], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for upload in files:
        name = Path(upload.filename).name
        if name == "VERSION":
            continue
        (dest / name).write_bytes(await upload.read())


async def _save_uploads_indexed(files: list[UploadFile], dest: Path) -> None:
    """Save every uploaded file under a unique index name.

    Series grouping is content-based (by SeriesInstanceUID), so original names are
    irrelevant — and a whole-examination export can nest files in sub-folders with
    colliding basenames. Indexed names sidestep the collisions; junk files are
    filtered out later when they fail to parse as DICOM.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for index, upload in enumerate(files):
        (dest / f"{index:06d}").write_bytes(await upload.read())


@router.post("/upload/", status_code=status.HTTP_201_CREATED, response_model=ExaminationCreated)
async def upload(examination_type: str = Form(...), files: list[UploadFile] = File(...)):
    if examination_type.lower() != "torsion":
        raise IngestError(f"Unknown examination_type: {examination_type}")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        await _save_uploads(files, tmpdir)
        examination_id = await run_in_threadpool(dicom.ingest_torsion_from_dir, tmpdir)
    return ExaminationCreated(examination_id=examination_id)


@router.post("/upload/torsion/series", status_code=status.HTTP_201_CREATED)
async def upload_torsion_series(files: list[UploadFile] = File(...),
                               store: Store = Depends(get_store)):
    """Phase 1: ingest a whole examination directory and return its candidate series.

    All uploaded files are scanned; DICOM is grouped by series and staged (with
    preview slices), non-DICOM is discarded, and a ``pending_selection`` examination
    is recorded. The response is the pending detail (series list + preview counts);
    the user then calls ``/upload/torsion/select`` to pick one and start processing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        await _save_uploads_indexed(files, tmpdir)
        examination_id = await run_in_threadpool(dicom.enumerate_torsion_series, tmpdir)
    with session_scope(runtime.get_engine()) as session:
        row = repository.get_examination(session, examination_id)
        return serializers.to_detail(row, store)


@router.post("/upload/torsion/select", status_code=status.HTTP_202_ACCEPTED, response_model=JobCreated)
async def select_torsion_series(selection: SeriesSelection, queue: TaskQueue = Depends(get_queue)):
    """Phase 2: materialize the chosen series and auto-start the full pipeline."""
    with session_scope(runtime.get_engine()) as session:
        row = repository.get_examination(session, selection.examination_id)
        if row is None:
            raise NotFoundError(f"Examination {selection.examination_id} not found")
        if row.status != ExaminationStatus.PENDING_SELECTION.value:
            raise IngestError(f"Examination {selection.examination_id} is not awaiting series selection")

    await run_in_threadpool(
        dicom.materialize_torsion_selection,
        selection.examination_id, selection.mode, selection.to_selection())
    # Auto-start segmentation + torsion (the exam is now 'unprocessed').
    return dispatch_job(selection.examination_id, JobKind.FULL, "full", queue)


@router.post("/upload/torsion/multi", status_code=status.HTTP_201_CREATED, response_model=ExaminationCreated)
async def upload_torsion_multi(
    hip_files: list[UploadFile] = File(...),
    knee_files: list[UploadFile] = File(...),
    ankle_files: list[UploadFile] = File(...),
):
    if not (hip_files and knee_files and ankle_files):
        raise IngestError("All three series (hip, knee, ankle) are required")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for region, fs in (("hip", hip_files), ("knee", knee_files), ("ankle", ankle_files)):
            await _save_uploads(fs, base / region)
        examination_id = await run_in_threadpool(
            dicom.ingest_torsion_multi_from_dirs, base / "hip", base / "knee", base / "ankle")
    return ExaminationCreated(examination_id=examination_id)


@router.post("/upload/orthanc", status_code=status.HTTP_202_ACCEPTED)
async def upload_orthanc(
    file: Annotated[bytes, File()],
    metadata: str = Body(...),
    settings: Settings = Depends(get_settings),
    store: Store = Depends(get_store),
    queue: TaskQueue = Depends(get_queue),
):
    """Stage one received DICOM instance and (re)schedule its debounced finalize."""
    tags = json.loads(metadata)
    meta = OrthancInstanceMeta.model_validate(tags)

    rule = match_rule(tags, load_rules(_RULES_PATH))
    if rule is None:
        raise IngestError(f"No routing rule matched instance for accession {meta.accession_number}")

    instance_uid = tags.get("0008,0018") or hashlib.sha1(file).hexdigest()
    store.stage_incoming(meta.accession_number, instance_uid, file)
    queue.enqueue_in(settings.orthanc_debounce_seconds, "api.tasks.orthanc.finalize_orthanc", meta.accession_number)
    return {"status": "accepted", "examination_id": meta.accession_number}

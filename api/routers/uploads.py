"""Upload endpoints (UI single/multi series, Orthanc instance streaming)."""
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from api.deps import get_queue, get_settings, get_store
from api.errors import IngestError
from api.ingest import dicom
from api.ingest.orthanc import load_rules, match_rule
from api.schemas.uploads import ExaminationCreated, OrthancInstanceMeta
from api.settings import Settings
from api.storage.store import Store
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


@router.post("/upload/", status_code=status.HTTP_201_CREATED, response_model=ExaminationCreated)
async def upload(examination_type: str = Form(...), files: list[UploadFile] = File(...)):
    etype = examination_type.lower()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        await _save_uploads(files, tmpdir)
        if etype == "torsion":
            examination_id = await run_in_threadpool(dicom.ingest_torsion_from_dir, tmpdir)
        elif etype == "x_ray_foot_ap":
            first = next(p for p in tmpdir.iterdir() if p.is_file())
            examination_id = await run_in_threadpool(dicom.ingest_xray, first)
        else:
            raise IngestError(f"Unknown examination_type: {examination_type}")
    return ExaminationCreated(examination_id=examination_id)


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

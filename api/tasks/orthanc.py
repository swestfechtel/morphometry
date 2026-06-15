"""Orthanc series finalization task.

After the debounce window, the staged DICOM instances for an accession are turned
into an examination and the torsion pipeline is run. Runs in the worker, so it is
durable and never blocks the API event loop. Idempotent: a no-op if nothing is
staged (e.g. a duplicate finalize fired).
"""
import logging
import time
import uuid

from api.db import repository
from api.db.engine import session_scope
from api.db.models import Job
from api.ingest.dicom import ingest_torsion_from_dir
from api.runtime import get_engine, get_settings, get_store
from api.schemas.enums import JobKind
from api.tasks.torsion import run_torsion

logger = logging.getLogger("api")


def finalize_orthanc(accession: str, debounce_seconds: int | None = None) -> None:
    """Build an examination from staged instances and run the torsion pipeline.

    Debounce without cancellation: if a new instance arrived within the debounce
    window, return early — a later scheduled finalize (fired after the dir goes
    quiet) does the work. Idempotent: a no-op if nothing is staged.
    """
    store, engine = get_store(), get_engine()
    files = store.incoming_files(accession)
    if not files:
        logger.info("No staged instances for accession %s; nothing to finalize.", accession)
        return

    window = get_settings().orthanc_debounce_seconds if debounce_seconds is None else debounce_seconds
    if time.time() - max(f.stat().st_mtime for f in files) < window:
        logger.debug("Accession %s still receiving instances; deferring finalize.", accession)
        return

    try:
        examination_id = ingest_torsion_from_dir(store.incoming_dir(accession))
    finally:
        store.clear_incoming(accession)

    job_id = str(uuid.uuid4())
    with session_scope(engine) as session:
        repository.create_job(session, Job(id=job_id, examination_id=examination_id, kind=JobKind.FULL.value))

    run_torsion(examination_id, job_id, "full")

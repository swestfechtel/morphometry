"""The torsion pipeline job functions executed by the RQ worker.

A job takes only JSON-serializable ids; it rebuilds engine/store/settings from
:mod:`api.runtime`, lazy-loads volumes from the Store, runs the two containers,
validates their JSON output, and persists results + encoded slices. Job and
examination status transitions are written to the DB so ``/jobs/{id}`` is durable.
"""
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib

from morphometry.image_io import Segmentation

from api.db import repository
from api.db.engine import session_scope
from api.domain.encode import encode_torsion_images
from api.runtime import get_engine, get_settings, get_store
from api.schemas.docker_io import ErrorsModel, LandmarksModel, ResultsModel
from api.schemas.enums import ExaminationStatus, JobState
from api.tasks.docker_run import run_segmentation_container, run_torsion_container

logger = logging.getLogger("api")


def _set_job(engine, job_id: str, status: JobState, *, started: bool = False,
             finished: bool = False, error: str | None = None) -> None:
    with session_scope(engine) as session:
        job = repository.get_job(session, job_id)
        if job is None:
            return
        job.status = status.value
        if started:
            job.started_at = datetime.now(timezone.utc)
        if finished:
            job.finished_at = datetime.now(timezone.utc)
        if error is not None:
            job.error = error
        repository.update_job(session, job)


def _set_examination_status(engine, examination_id: str, status: ExaminationStatus) -> None:
    with session_scope(engine) as session:
        ex = repository.get_examination(session, examination_id)
        if ex is not None:
            ex.status = status.value
            repository.upsert_examination(session, ex)


def run_torsion(examination_id: str, job_id: str, mode: str = "full") -> None:
    """Run the segmentation and/or torsion stages for an examination (RQ entrypoint)."""
    engine = get_engine()
    _set_job(engine, job_id, JobState.RUNNING, started=True)
    _set_examination_status(engine, examination_id, ExaminationStatus.RUNNING)
    try:
        if mode in ("full", "segmentation"):
            _segment(examination_id)
        if mode in ("full", "torsion"):
            _align(examination_id)
        _set_job(engine, job_id, JobState.FINISHED, finished=True)
    except Exception as exc:  # noqa: BLE001 - record and re-raise for RQ
        logger.exception("Job %s failed for examination %s", job_id, examination_id)
        _set_job(engine, job_id, JobState.FAILED, finished=True, error=str(exc))
        _set_examination_status(engine, examination_id, ExaminationStatus.FAILED)
        raise


def _segment(examination_id: str) -> None:
    """Run nnUNet segmentation, persist masks + encoded slices, mark 'segmented'."""
    engine, store, settings = get_engine(), get_store(), get_settings()
    with session_scope(engine) as session:
        ex = repository.get_examination(session, examination_id)
        source_paths = dict(ex.source_paths)

    masks: dict[str, Segmentation] = {}
    with tempfile.TemporaryDirectory() as tempdir:
        for region in ("hip", "knee", "ankle"):
            (Path(tempdir) / region / "input").mkdir(parents=True)
            (Path(tempdir) / region / "output").mkdir(parents=True)
            store.load_image(source_paths[region]).save_image(
                f"{tempdir}/{region}/input/{region}_0000.nii.gz")

        run_segmentation_container(tempdir, settings)

        mask_paths = {}
        for region in ("hip", "knee", "ankle"):
            seg = Segmentation.from_nibabel(nib.load(f"{tempdir}/{region}/output/{region}.nii.gz"))
            seg.transform_coordinate_system()
            masks[region] = seg
            mask_paths[region] = store.save_mask(examination_id, region, seg)

    transformed = store.load_image(source_paths["transformed"])
    image_b64, seg_b64 = encode_torsion_images(
        transformed, masks["hip"], masks["knee"], masks["ankle"], settings.encode_pool_size)
    encoded_paths = store.save_encoded(examination_id, image_b64, seg_b64)

    with session_scope(engine) as session:
        ex = repository.get_examination(session, examination_id)
        ex.mask_paths = mask_paths
        ex.encoded_paths = encoded_paths
        ex.status = ExaminationStatus.SEGMENTED.value
        repository.upsert_examination(session, ex)


def _align(examination_id: str) -> None:
    """Run the torsion container, validate its output, persist results, mark 'processed'."""
    engine, store, settings = get_engine(), get_store(), get_settings()
    with session_scope(engine) as session:
        ex = repository.get_examination(session, examination_id)
        mask_paths = dict(ex.mask_paths or {})

    with tempfile.TemporaryDirectory() as tempdir:
        for region in ("hip", "knee", "ankle"):
            store.load_segmentation(mask_paths[region]).save_image(f"{tempdir}/{region}_segmentation.nii.gz")

        run_torsion_container(tempdir, settings)

        results = ResultsModel.model_validate_json(Path(f"{tempdir}/results.json").read_text())
        landmarks = LandmarksModel.model_validate_json(Path(f"{tempdir}/landmarks.json").read_text())
        errors = ErrorsModel.model_validate_json(Path(f"{tempdir}/errors.json").read_text())

    if errors.errors:
        logger.error("Torsion computation reported errors for %s: %s", examination_id, errors.errors)

    with session_scope(engine) as session:
        ex = repository.get_examination(session, examination_id)
        ex.torsion_values = results.model_dump()
        ex.landmarks = landmarks.root
        ex.status = ExaminationStatus.PROCESSED.value
        repository.upsert_examination(session, ex)

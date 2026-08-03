"""Shared job-dispatch helper.

Creates a committed Job row, enqueues the worker task, and records the returned RQ
id. Extracted from the jobs router so both the model-dispatch endpoints and the
series-selection upload endpoint can start processing through the same path.
"""
import uuid

from api import runtime
from api.db import repository
from api.db.engine import session_scope
from api.db.models import Job
from api.errors import NotFoundError
from api.schemas.enums import JobKind
from api.schemas.jobs import JobCreated
from api.tasks.queue import TaskQueue

_TORSION_TASK = "api.tasks.torsion.run_torsion"


def dispatch_job(examination_id: str, kind: JobKind, mode: str, queue: TaskQueue) -> JobCreated:
    """Create a (committed) job row, enqueue it, then record the RQ id.

    The job row is committed before enqueueing so the worker (a separate process —
    or the eager in-process queue) reliably sees it.

    :param examination_id: The examination to process (must exist).
    :param kind: The job kind recorded on the row.
    :param mode: The ``run_torsion`` mode (``"full"``/``"segmentation"``/``"torsion"``).
    :param queue: The task queue to enqueue on.
    :return: The created-job envelope for the HTTP response.
    """
    engine = runtime.get_engine()
    job_id = str(uuid.uuid4())
    with session_scope(engine) as session:
        if repository.get_examination(session, examination_id) is None:
            raise NotFoundError(f"Examination {examination_id} not found")
        repository.create_job(session, Job(id=job_id, examination_id=examination_id, kind=kind.value))

    rq_id = queue.enqueue(_TORSION_TASK, examination_id, job_id, mode)

    with session_scope(engine) as session:
        job = repository.get_job(session, job_id)
        job.rq_job_id = rq_id
        repository.update_job(session, job)
    return JobCreated(job_id=job_id, examination_id=examination_id)

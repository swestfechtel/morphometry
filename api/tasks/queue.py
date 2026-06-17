"""Task-queue abstraction.

A tiny protocol over "enqueue a dotted-path job function with id args" lets the
API depend on an interface rather than RQ directly: the real implementation
(``RQQueue``) pushes to Redis; the test/dev implementation (``EagerQueue``) runs
the job inline. Swapping to Celery later means writing one more implementation.
"""
import importlib
import logging
import uuid
from typing import Protocol

from api.settings import Settings


def _resolve(dotted_path: str):
    module_name, attr = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


class TaskQueue(Protocol):
    def enqueue(self, dotted_path: str, *args) -> str:
        """Enqueue a job; return an RQ job id (for reconciliation)."""
        ...


class RQQueue:
    """Real queue backed by Redis + RQ (single GPU queue ⇒ one worker serializes jobs)."""

    def __init__(self, settings: Settings, connection=None):
        from redis import Redis
        from rq import Queue
        self._settings = settings
        self._connection = connection or Redis.from_url(settings.redis_url)
        self._queue = Queue(settings.gpu_queue_name, connection=self._connection,
                            default_timeout=settings.job_timeout)

    def enqueue(self, dotted_path: str, *args) -> str:
        job = self._queue.enqueue(dotted_path, *args, job_timeout=self._settings.job_timeout)
        return job.id

    def enqueue_in(self, delay_seconds: int, dotted_path: str, *args) -> str:
        from datetime import timedelta
        job = self._queue.enqueue_in(timedelta(seconds=delay_seconds), dotted_path, *args)
        return job.id


class EagerQueue:
    """Runs jobs synchronously in-process. For tests and single-user dev without Redis.

    Mirrors a real queue's contract: enqueue returns immediately with a job id and
    never propagates the *job's* exception to the caller — a failing job records its
    own failure (DB status), exactly as the RQ worker would.
    """

    def __init__(self):
        self._logger = logging.getLogger("api")

    def enqueue(self, dotted_path: str, *args) -> str:
        try:
            _resolve(dotted_path)(*args)
        except Exception:  # noqa: BLE001 - the job records its own failure; don't fail enqueue
            self._logger.exception("Eager job %s failed", dotted_path)
        return str(uuid.uuid4())

    def enqueue_in(self, delay_seconds: int, dotted_path: str, *args) -> str:  # ignore delay
        return self.enqueue(dotted_path, *args)


def make_queue(settings: Settings) -> TaskQueue:
    """Construct the real RQ-backed queue from settings."""
    return RQQueue(settings)

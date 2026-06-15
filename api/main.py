"""FastAPI application factory.

Replaces ``rest_api.py``. State lives in injected dependencies (DB, store, queue)
rather than module globals; long work runs on the queue, not the event loop.
Run with: ``uvicorn api.main:app`` (plus ``redis-server`` and an ``rq`` worker).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import runtime
from api.db import repository
from api.db.engine import session_scope
from api.deps import require_api_key
from api.errors import DomainError
from api.logging_config import configure_logging
from api.routers import examinations, health, jobs, uploads
from api.schemas.enums import JobState
from api.schemas.errors import ErrorResponse

logger = logging.getLogger("api")


def _reconcile_orphaned_jobs() -> None:
    """Mark jobs left 'running' by a previous process as failed (their worker is gone)."""
    with session_scope(runtime.get_engine()) as session:
        for job in repository.list_jobs_by_status(session, JobState.RUNNING.value):
            job.status = JobState.FAILED.value
            job.error = "Interrupted by restart"
            repository.update_job(session, job)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = runtime.get_settings()
    configure_logging(settings.log_dir, settings.log_level)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    runtime.get_engine()  # create tables
    _reconcile_orphaned_jobs()
    logger.info("API started (storage=%s, auth=%s)", settings.storage_dir, settings.auth_enabled)
    yield


def create_app() -> FastAPI:
    settings = runtime.get_settings()
    app = FastAPI(title="Morphometry API", lifespan=lifespan)

    allow_all = settings.cors_allow_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=not allow_all,  # '*' + credentials is invalid
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(DomainError)
    async def _domain_error(_request: Request, exc: DomainError):
        return JSONResponse(status_code=exc.status_code,
                            content=ErrorResponse(detail=str(exc), code=exc.code).model_dump())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422,
                            content=ErrorResponse(detail=str(exc), code="validation_error").model_dump())

    app.include_router(health.router)
    for router in (examinations.router, uploads.router, jobs.router):
        app.include_router(router, dependencies=[Depends(require_api_key)])

    return app


app = create_app()

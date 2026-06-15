"""FastAPI dependency providers.

Replaces the old module-global singletons (file_controller, executor, the job
dicts) with ``Depends``-injected dependencies, so they can be overridden in tests
(temp DB/store, eager queue) without monkeypatching globals.
"""
from collections.abc import Iterator

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlmodel import Session

from api import runtime
from api.settings import Settings
from api.storage.store import Store
from api.tasks.queue import TaskQueue, make_queue

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_settings() -> Settings:
    return runtime.get_settings()


def get_store() -> Store:
    return runtime.get_store()


def get_session() -> Iterator[Session]:
    """Yield a transactional DB session (commit on success, rollback on error)."""
    from api.db.engine import session_scope
    with session_scope(runtime.get_engine()) as session:
        yield session


def get_queue() -> TaskQueue:
    """The task queue (RQ-backed). Overridden with EagerQueue in tests."""
    return make_queue(runtime.get_settings())


def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Enforce the X-API-Key header when auth is enabled (non-empty api_keys)."""
    settings = runtime.get_settings()
    if not settings.auth_enabled:
        return
    if api_key is None or api_key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

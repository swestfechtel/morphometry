"""FastAPI dependency providers.

Replaces the old module-global singletons (file_controller, executor, the job
dicts) with ``Depends``-injected dependencies, so they can be overridden in tests
(temp DB/store, eager queue) without monkeypatching globals.
"""
from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, Query, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from sqlmodel import Session

from api import runtime
from api.auth import service as auth_service
from api.auth.service import UserView
from api.settings import Settings
from api.storage.store import Store
from api.tasks.queue import TaskQueue, make_queue

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def _bearer_token(authorization: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if authorization and authorization[:7].lower() == "bearer ":
        return authorization[7:].strip() or None
    return None


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


def current_user(authorization: str | None = Header(None)) -> UserView | None:
    """The user authenticated by a Bearer token, or None (no/invalid token)."""
    token = _bearer_token(authorization)
    return auth_service.verify_token(token) if token else None


def require_user(user: UserView | None = Depends(current_user)) -> UserView:
    """Require a logged-in user (a valid Bearer token). Used by /auth/me."""
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def require_auth(authorization: str | None = Header(None),
                 api_key: str | None = Security(_api_key_header)) -> None:
    """Enforce auth on protected endpoints when enabled: accept EITHER a valid user
    Bearer token OR a valid X-API-Key (machine clients). No-op when auth is disabled."""
    settings = runtime.get_settings()
    if not settings.auth_enabled:
        return
    token = _bearer_token(authorization)
    if token and auth_service.verify_token(token) is not None:
        return
    if api_key is not None and api_key in settings.api_keys:
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing credentials")


def require_volume_access(authorization: str | None = Header(None),
                          header_key: str | None = Security(_api_key_header),
                          query_key: str | None = Security(_api_key_query),
                          token_query: str | None = Query(None, alias="token")) -> None:
    """Auth for the media endpoints (volumes, series previews): accept a user token
    OR an API key, from a header OR a query param.

    ``<img>`` tags and Cornerstone's NIfTI loader fetch these URLs themselves and
    can't always set a header, so a query-param fallback is offered for both auth
    kinds: ``?token=`` (user) and ``?api_key=`` (machine). No-op when auth is disabled.
    """
    settings = runtime.get_settings()
    if not settings.auth_enabled:
        return
    token = _bearer_token(authorization) or token_query
    if token and auth_service.verify_token(token) is not None:
        return
    key = header_key or query_key
    if key is not None and key in settings.api_keys:
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing credentials")

"""User authentication + management service.

Shared by the HTTP auth endpoints and the ``python -m api.users`` CLI, so both go
through the same hashing/validation logic. Each call opens its own session and
returns a detached :class:`UserView` snapshot (the ORM object would be unusable
once the session closes).
"""
from dataclasses import dataclass

from api import runtime
from api.auth import passwords, tokens
from api.db import repository
from api.db.engine import session_scope
from api.db.models import User


# A real argon2 hash used only to equalize the verify cost on the unknown-user
# path, so response timing doesn't leak whether a username exists.
_DUMMY_HASH = passwords.hash_password("timing-equalization")


class AuthError(Exception):
    """A user-management operation failed (e.g. duplicate / missing user)."""


@dataclass(frozen=True)
class UserView:
    """A read-only snapshot of a user, safe to use after the session closes."""
    username: str
    is_active: bool
    token_version: int


def _view(user: User) -> UserView:
    return UserView(username=user.username, is_active=user.is_active, token_version=user.token_version)


# --- management (used by the CLI) -------------------------------------------
def create_user(username: str, password: str, *, is_active: bool = True) -> UserView:
    with session_scope(runtime.get_engine()) as session:
        if repository.get_user(session, username) is not None:
            raise AuthError(f"user {username!r} already exists")
        user = User(username=username, password_hash=passwords.hash_password(password), is_active=is_active)
        repository.upsert_user(session, user)
        return _view(user)


def set_password(username: str, password: str) -> UserView:
    """Set a new password and bump ``token_version`` (invalidates existing tokens)."""
    with session_scope(runtime.get_engine()) as session:
        user = repository.get_user(session, username)
        if user is None:
            raise AuthError(f"user {username!r} does not exist")
        user.password_hash = passwords.hash_password(password)
        user.token_version += 1
        repository.upsert_user(session, user)
        return _view(user)


def set_active(username: str, is_active: bool) -> UserView:
    with session_scope(runtime.get_engine()) as session:
        user = repository.get_user(session, username)
        if user is None:
            raise AuthError(f"user {username!r} does not exist")
        user.is_active = is_active
        repository.upsert_user(session, user)
        return _view(user)


def delete_user(username: str) -> None:
    with session_scope(runtime.get_engine()) as session:
        if not repository.delete_user(session, username):
            raise AuthError(f"user {username!r} does not exist")


def list_users() -> list[UserView]:
    with session_scope(runtime.get_engine()) as session:
        return [_view(u) for u in repository.list_users(session)]


# --- authentication (used by the API) ---------------------------------------
def authenticate(username: str, password: str) -> UserView | None:
    """Return the user if the credentials are valid and the account is active."""
    with session_scope(runtime.get_engine()) as session:
        user = repository.get_user(session, username)
        if user is None or not user.is_active:
            if user is None:
                passwords.verify_password(_DUMMY_HASH, password)  # equalize timing
            return None
        if not passwords.verify_password(user.password_hash, password):
            return None
        return _view(user)


def issue_token(user: UserView) -> str:
    return tokens.create_token(user.username, user.token_version)


def verify_token(token: str) -> UserView | None:
    """Return the user a valid token authenticates, else None.

    Checks the signature/expiry, then that the user still exists, is active, and the
    token's ``token_version`` still matches (so a password change invalidates it).
    """
    payload = tokens.read_token(token, runtime.get_settings().session_ttl_seconds)
    if not payload:
        return None
    with session_scope(runtime.get_engine()) as session:
        user = repository.get_user(session, payload.get("u", ""))
        if user is None or not user.is_active or user.token_version != payload.get("v"):
            return None
        return _view(user)

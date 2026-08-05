"""Authentication endpoints: login (issue a token) and whoami.

Login is intentionally left OUT of the auth-protected router group (you can't be
authenticated before you log in). User management (create/edit) is done via the
``python -m api.users`` CLI, not the API.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from api import runtime
from api.auth import service as auth_service
from api.auth.service import UserView
from api.deps import require_user
from api.schemas.auth import LoginRequest, TokenResponse, UserInfo

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    """Verify credentials and return a signed session token."""
    user = auth_service.authenticate(body.username, body.password)
    if user is None:
        # 401 with a generic message; do not reveal whether the username exists.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    return TokenResponse(
        access_token=auth_service.issue_token(user),
        username=user.username,
        expires_in=runtime.get_settings().session_ttl_seconds,
    )


@router.get("/me", response_model=UserInfo)
def me(user: UserView = Depends(require_user)):
    """Return the currently authenticated user (requires a valid Bearer token)."""
    return UserInfo(username=user.username, is_active=user.is_active)

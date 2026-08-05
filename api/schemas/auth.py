"""Auth request/response schemas."""
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    """Returned on successful login; the client sends the token back as
    ``Authorization: Bearer <token>`` (or ``?token=`` for media URLs)."""
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_in: int  # seconds until the token expires


class UserInfo(BaseModel):
    username: str
    is_active: bool

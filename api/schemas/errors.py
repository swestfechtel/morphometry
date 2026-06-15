"""Consistent error response schema."""
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Uniform error body returned by all exception handlers."""
    detail: str
    code: str = "error"

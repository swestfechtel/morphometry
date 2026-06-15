"""Schemas for validating the JSON the torsion container writes.

The worker must not blindly trust ``results.json`` / ``landmarks.json`` /
``errors.json``. These models validate structure before the values are persisted;
malformed output fails the job cleanly instead of writing garbage.
"""
import math
from typing import Any

from pydantic import BaseModel, RootModel


class ResultsModel(BaseModel):
    """The six torsion angles produced by the torsion container."""
    model_config = {"extra": "ignore"}

    femoral_torsion_left: float
    femoral_torsion_right: float
    femoral_torsion_left_murphy: float | None = None
    femoral_torsion_right_murphy: float | None = None
    tibial_torsion_left: float
    tibial_torsion_right: float


class ErrorsModel(BaseModel):
    """The per-measurement error list emitted by the torsion container."""
    errors: list[str] = []


def _sanitize(value: Any) -> Any:
    """Recursively replace NaN floats with 0 (mirrors the legacy UI sanitization)."""
    if isinstance(value, float):
        return 0 if math.isnan(value) else value
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    return value


class LandmarksModel(RootModel[dict]):
    """The (nested) landmark dict. Validated as a dict; values sanitized on demand."""

    def sanitized(self) -> dict:
        """Return the landmark dict with NaN coordinates replaced by 0."""
        return _sanitize(self.root)

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

    # The QC flag strings the container emits beside those angles. Declared so `extra="ignore"`
    # stops silently dropping them: the container has written them since `822181b` (Murphy) and
    # gained the Lee and condylar families since, and until now every one was computed and thrown
    # away on the API path. They need NO database change -- `Examination.torsion_values` is a JSON
    # column (`api/db/models.py:37`) and `api/tasks/torsion.py` persists `results.model_dump()`
    # wholesale -- so declaring them here is sufficient to store them. `''` means the landmark
    # looked sound; a non-empty string is `;`-joined `name:value` entries (see
    # `morphometry.femur.murphy_neck_base_qc`, `lee_neck_axis_qc` and
    # `morphometry.knee.knee_condylar_line_qc`). They never change a measurement: a flagged leg
    # keeps its angle, and the flag only says the row needs a look at the mask.
    #
    # NOTE the key names are PATIENT side, as the container writes them, where the goldens and the
    # library are IMAGE side -- so left and right cross over relative to `tests/golden/`.
    #
    # Default `""` and not `None`, for two reasons. `api/schemas/examination.py::TorsionValues`
    # declares these as `str`, and the serializer splats this model's dump straight into it
    # (`api/serializers.py:42`), so a `None` here would fail validation on every examination
    # produced by a container old enough not to emit the key. And `""` is the same value the
    # container writes for a sound landmark, which is the honest default: an empty flag string has
    # always meant "nothing to report" and never "the QC did not run", and nothing in the
    # container records provenance that would let the two be told apart.
    femoral_torsion_left_murphy_qc_flags: str = ""
    femoral_torsion_right_murphy_qc_flags: str = ""
    femoral_torsion_left_lee_qc_flags: str = ""
    femoral_torsion_right_lee_qc_flags: str = ""
    femur_condylar_line_left_qc_flags: str = ""
    femur_condylar_line_right_qc_flags: str = ""
    tibia_condylar_line_left_qc_flags: str = ""
    tibia_condylar_line_right_qc_flags: str = ""


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

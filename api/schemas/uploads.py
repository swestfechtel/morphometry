"""Upload request/response schemas."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from api.schemas.enums import ExaminationType


class ExaminationCreated(BaseModel):
    """Returned by the upload endpoints (201 Created)."""
    examination_id: str


class SeriesSelection(BaseModel):
    """The user's series pick for a pending torsion examination (phase 2 of upload).

    ``whole_leg`` selects one series (auto-split into hip/knee/ankle); ``regions``
    assigns three already-split series. The relevant UID fields are required for the
    chosen mode.
    """
    examination_id: str
    mode: Literal["whole_leg", "regions"]
    series_uid: str | None = None                 # whole_leg
    hip: str | None = None                         # regions
    knee: str | None = None                        # regions
    ankle: str | None = None                       # regions

    @model_validator(mode="after")
    def _check_required(self) -> "SeriesSelection":
        if self.mode == "whole_leg":
            if not self.series_uid:
                raise ValueError("series_uid is required for mode 'whole_leg'")
        elif not (self.hip and self.knee and self.ankle):
            raise ValueError("hip, knee and ankle series UIDs are required for mode 'regions'")
        return self

    def to_selection(self) -> dict:
        """The plain dict :func:`api.ingest.dicom.materialize_torsion_selection` expects."""
        if self.mode == "whole_leg":
            return {"series_uid": self.series_uid}
        return {"hip": self.hip, "knee": self.knee, "ankle": self.ankle}


class OrthancInstanceMeta(BaseModel):
    """Typed view of the simplified DICOM metadata forwarded by the Orthanc plugin.

    Only the fields the routing/ingest logic needs are declared; everything else is
    ignored. Tag keys in the simplified JSON are hex strings like '0008,0050'.
    """
    model_config = {"extra": "allow", "populate_by_name": True}

    accession_number: str = Field(alias="AccessionNumber")

    def tag(self, key: str):
        """Look up a raw simplified-JSON tag value (e.g. '0008,1030')."""
        return getattr(self, "__pydantic_extra__", {}).get(key)

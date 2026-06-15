"""Upload request/response schemas."""
from pydantic import BaseModel, Field

from api.schemas.enums import ExaminationType


class ExaminationCreated(BaseModel):
    """Returned by the upload endpoints (201 Created)."""
    examination_id: str


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

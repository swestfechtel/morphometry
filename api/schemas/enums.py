"""Shared string enums for examinations and jobs.

String values are kept identical to the pre-refactor codebase so the frontend and
any persisted/forwarded data keep working unchanged.
"""
from enum import Enum


class ExaminationType(str, Enum):
    TORSION = "torsion"


class ExaminationStatus(str, Enum):
    PENDING_SELECTION = "pending_selection"  # series enumerated, awaiting the user's pick
    UNPROCESSED = "unprocessed"
    RUNNING = "running"
    SEGMENTED = "segmented"
    PROCESSED = "processed"
    FAILED = "failed"


class JobKind(str, Enum):
    FULL = "full"          # segmentation + torsion
    SEGMENTATION = "segmentation"
    TORSION = "torsion"
    ORTHANC_FINALIZE = "orthanc_finalize"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

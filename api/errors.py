"""Domain exceptions mapped to HTTP responses by the app's exception handlers."""


class DomainError(Exception):
    """Base class for expected, client-facing errors."""
    status_code = 400
    code = "error"


class NotFoundError(DomainError):
    status_code = 404
    code = "not_found"


class DuplicateError(DomainError):
    status_code = 409
    code = "duplicate"


class IngestError(DomainError):
    status_code = 400
    code = "ingest_error"

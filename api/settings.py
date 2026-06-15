"""Application configuration via environment variables / .env.

All previously-hardcoded values (storage paths, database/redis URLs, docker image
tags, shared-memory size, API keys, CORS origins) live here so the service can be
deployed without code changes. Override any field with an ``MORPH_API_``-prefixed
environment variable or an entry in a local ``.env`` file.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_API_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Typed application settings, populated from the environment / .env."""

    model_config = SettingsConfigDict(env_prefix="MORPH_API_", env_file=".env", extra="ignore")

    # --- storage / persistence -------------------------------------------------
    #: Root directory for the SQLite DB and per-examination image files.
    storage_dir: Path = _API_DIR / "data"
    #: SQLAlchemy URL; defaults to a SQLite file under ``storage_dir`` if unset.
    database_url: str | None = None
    #: Directory for application logs.
    log_dir: Path = _API_DIR / "logs"
    log_level: str = "INFO"
    #: Duplicate-upload policy: 'reject' or 'replace'.
    on_duplicate: str = "reject"

    # --- task queue ------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    #: Name of the (single-worker) queue that serializes GPU jobs.
    gpu_queue_name: str = "gpu"
    #: RQ job timeout in seconds (long enough for nnUNet + torsion).
    job_timeout: int = 3600
    #: Debounce window for finalizing an Orthanc series, in seconds.
    orthanc_debounce_seconds: int = 10

    # --- docker model images ---------------------------------------------------
    nnunet_image: str = "swestfechtel/nnunet_torsion:latest"
    torsion_image: str = "swestfechtel/torsion:latest"
    docker_shm_size: str = "32G"

    # --- image encoding --------------------------------------------------------
    #: Worker-side matplotlib pool size for slice PNG rendering (None = os default).
    encode_pool_size: int | None = None

    # --- auth / CORS -----------------------------------------------------------
    #: Accepted ``X-API-Key`` values. Empty list disables auth (dev only).
    api_keys: list[str] = []
    #: Allowed CORS origins. Use explicit origins in production, not ['*'].
    cors_allow_origins: list[str] = ["*"]

    @field_validator("api_keys", "cors_allow_origins", mode="before")
    @classmethod
    def _split_csv(cls, value):
        """Allow comma-separated strings in env vars in addition to JSON lists."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return value  # let pydantic parse JSON
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def resolved_database_url(self) -> str:
        """The DB URL, defaulting to a SQLite file inside ``storage_dir``."""
        return self.database_url or f"sqlite:///{self.storage_dir / 'api.db'}"

    @property
    def incoming_dir(self) -> Path:
        """Staging directory for in-flight Orthanc DICOM instances."""
        return self.storage_dir / "_incoming"

    @property
    def auth_enabled(self) -> bool:
        return len(self.api_keys) > 0

    def examination_dir(self, examination_id: str) -> Path:
        """Per-examination storage directory (``storage_dir/{id}``)."""
        return self.storage_dir / examination_id


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()

"""Application configuration via environment variables / .env.

All previously-hardcoded values (storage paths, database/redis URLs, docker image
tags, shared-memory size, API keys, CORS origins) live here so the service can be
deployed without code changes. Override any field with an ``MORPH_API_``-prefixed
environment variable or an entry in a local ``.env`` file.
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from api import release_manifest
from api.version import get_version

logger = logging.getLogger("api")

_API_DIR = Path(__file__).resolve().parent
_warned_latest = False  # the ':latest' fallback warning is logged once per process


class Settings(BaseSettings):
    """Typed application settings, populated from the environment / .env."""

    # Load .env from the repo root AND from api/ (next to .env.example), by absolute
    # path so it works regardless of the working directory uvicorn is launched from.
    # Later files win; real environment variables still override both.
    model_config = SettingsConfigDict(
        env_prefix="MORPH_API_",
        env_file=(_API_DIR.parent / ".env", _API_DIR / ".env"),
        extra="ignore",
    )

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
    #: Explicit image override (env). ``None`` = use the release manifest (see ``resolved_*``).
    nnunet_image: str | None = None
    torsion_image: str | None = None
    docker_shm_size: str = "32G"

    #: Parent directory for per-job scratch dirs (``None`` = system temp). Created on use.
    job_tmp_dir: Path | None = None

    # --- image encoding --------------------------------------------------------
    #: Worker-side matplotlib pool size for slice PNG rendering (None = os default).
    encode_pool_size: int | None = None

    # --- auth / CORS -----------------------------------------------------------
    #: Accepted ``X-API-Key`` values (machine clients, e.g. the Orthanc plugin).
    #: NoDecode lets env vars be comma-separated strings (handled by the validator below).
    api_keys: Annotated[list[str], NoDecode] = []
    #: Enforce authentication (a user token OR an API key) on protected endpoints.
    #: Independent of ``api_keys`` so user login can be required without any API key.
    auth_required: bool = False
    #: Secret used to sign session tokens. If unset, a random one is generated and
    #: persisted to ``storage_dir/.secret_key`` (see runtime.get_secret_key) so tokens
    #: survive restarts in dev; set ``MORPH_API_SECRET_KEY`` explicitly in production.
    secret_key: str | None = None
    #: Lifetime of an issued login token, in seconds (default 7 days).
    session_ttl_seconds: int = 7 * 24 * 3600
    #: Allowed CORS origins. Use explicit origins in production, not ['*'].
    cors_allow_origins: Annotated[list[str], NoDecode] = ["*"]

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

    def _resolve_image(self, key: str, override: str | None) -> str:
        """Env override, else manifest entry for the running version, else ``:latest`` (warns once)."""
        global _warned_latest
        if override:
            return override
        _, entry = release_manifest.images_for(get_version(), release_manifest.load_manifest())
        if entry.get(key):
            return entry[key]
        repo = {"nnunet": "swestfechtel/nnunet_torsion", "torsion": "swestfechtel/torsion"}[key]
        if not _warned_latest:
            _warned_latest = True
            logger.warning("No manifest entry for %s image at version %s; falling back to %s:latest",
                           key, get_version(), repo)
        return f"{repo}:latest"

    @property
    def resolved_nnunet_image(self) -> str:
        """nnUNet image: ``MORPH_API_NNUNET_IMAGE`` if set, else the release manifest, else ``:latest``."""
        return self._resolve_image("nnunet", self.nnunet_image)

    @property
    def resolved_torsion_image(self) -> str:
        """Torsion image: ``MORPH_API_TORSION_IMAGE`` if set, else the release manifest, else ``:latest``."""
        return self._resolve_image("torsion", self.torsion_image)

    def job_tempdir_parent(self) -> str | None:
        """Create (if needed) and return ``job_tmp_dir`` for ``TemporaryDirectory(dir=...)``; ``None`` if unset."""
        if self.job_tmp_dir is None:
            return None
        self.job_tmp_dir.mkdir(parents=True, exist_ok=True)
        return str(self.job_tmp_dir)

    @property
    def incoming_dir(self) -> Path:
        """Staging directory for in-flight Orthanc DICOM instances."""
        return self.storage_dir / "_incoming"

    @property
    def auth_enabled(self) -> bool:
        """Whether protected endpoints require authentication (token or API key)."""
        return self.auth_required or len(self.api_keys) > 0

    def examination_dir(self, examination_id: str) -> Path:
        """Per-examination storage directory (``storage_dir/{id}``)."""
        return self.storage_dir / examination_id


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()

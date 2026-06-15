"""Process-wide runtime dependencies (settings, DB engine, store).

Both the API process and the RQ worker resolve their dependencies through these
cached accessors, so a job function only needs JSON-serializable ids — it rebuilds
the engine/store from settings. Tests point everything at temp dirs by setting
``MORPH_API_*`` env vars and calling :func:`reset` to clear the caches.
"""
from functools import lru_cache

from api.db.engine import init_db, make_engine
from api.settings import Settings, get_settings
from api.storage.store import Store


@lru_cache
def get_engine():
    """The shared SQLAlchemy engine (tables ensured)."""
    engine = make_engine(get_settings().resolved_database_url)
    init_db(engine)
    return engine


@lru_cache
def get_store() -> Store:
    """The shared file-based image store."""
    return Store(get_settings().storage_dir)


def reset() -> None:
    """Clear all cached dependencies (used by tests after changing env)."""
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_store.cache_clear()


def current_settings() -> Settings:
    return get_settings()

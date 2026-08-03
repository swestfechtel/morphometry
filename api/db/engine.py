"""SQLite engine creation and session management.

WAL journal mode + a busy timeout let the API and the worker process both write
to the same SQLite file safely at this (low) volume. Swapping to Postgres later
is a URL change (SQLModel/SQLAlchemy underneath).
"""
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, Session, create_engine

from api.db import models  # noqa: F401  (ensure tables are registered on metadata)


def make_engine(database_url: str) -> Engine:
    """Create a SQLite engine with WAL + busy_timeout and cross-thread access."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)

    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create all tables, then add any columns missing from an existing DB.

    ``create_all`` only creates absent tables — it never ALTERs an existing one — so
    a column added to a model (e.g. ``Examination.series``) would be missing on a
    pre-existing SQLite file and every read would fail. This lightweight, idempotent
    migration adds such columns. (Swap to Alembic if migrations get non-trivial.)
    """
    SQLModel.metadata.create_all(engine)
    _ensure_columns(engine)


# Columns added to models after the initial schema, and their SQLite type — added
# to an existing DB if absent (see init_db).
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "examinations": {"series": "JSON"},
}


def _ensure_columns(engine: Engine) -> None:
    if not engine.url.get_backend_name().startswith("sqlite"):
        return
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    for table, columns in _ADDED_COLUMNS.items():
        if table not in tables:
            continue
        existing = {col["name"] for col in inspector.get_columns(table)}
        with engine.begin() as conn:
            for name, sql_type in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Provide a transactional session scope (commit on success, rollback on error)."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

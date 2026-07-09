"""Session SQLAlchemy et création des tables."""

from __future__ import annotations

import os
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from email_analyzer.db.models import Base

_database_url: str | None = None
engine = None
SessionLocal = None


def normalize_database_url(raw: str) -> str:
    """Normalise DATABASE_URL pour SQLAlchemy + client PostgreSQL.

    - Ré-encode user/password (RFC 3986) pour les caractères non ASCII dans l’URI.
    - Force le dialecte ``postgresql+psycopg`` (psycopg v3) : sous Windows, un
      serveur PostgreSQL en locale CP1252 peut renvoyer des messages d’erreur
      non UTF-8 ; psycopg2 les décode alors en UTF-8 et lève UnicodeDecodeError
      (ex. octet 0xe9). psycopg v3 gère ce cas (voir psycopg2 #1816).
    """
    raw = raw.strip()
    if not raw:
        return raw
    try:
        u = make_url(raw)
        if u.drivername in ("postgresql", "postgresql+psycopg2"):
            u = u.set(drivername="postgresql+psycopg")
        return u.render_as_string(hide_password=False)
    except Exception:
        return raw


def _get_database_url() -> str | None:
    raw = (os.environ.get("DATABASE_URL") or "").strip() or None
    if not raw:
        return None
    return normalize_database_url(raw)


def init_db() -> None:
    """Initialise engine et SessionLocal si DATABASE_URL est défini."""
    global engine, SessionLocal, _database_url
    url = _get_database_url()
    if not url:
        engine = None
        SessionLocal = None
        _database_url = None
        return
    if url == _database_url and engine is not None:
        return
    _database_url = url
    os.environ.setdefault("PGCLIENTENCODING", "UTF8")
    engine = create_engine(url, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    init_db()
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL non configuré")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_optional() -> Generator[Optional[Session], None, None]:
    """Session DB si DATABASE_URL est défini, sinon None (mode legacy sans SaaS)."""
    init_db()
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    """Crée les tables (dev/tests ; préférer Alembic en prod)."""
    init_db()
    if engine is None:
        raise RuntimeError("DATABASE_URL non configuré")
    Base.metadata.create_all(bind=engine)

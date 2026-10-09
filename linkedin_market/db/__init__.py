"""Persistance PostgreSQL 18 de linkedin_market (SQLAlchemy 2, psycopg 3, Alembic).

Helpers de dates communs : tout nouveau code écrit de l'UTC avec fuseau (``utcnow``) et l'API expose les
horodatages en ISO 8601 UTC suffixé ``Z``, à la seconde (``utc_iso``, contrat C-1 / C-2).
"""

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Instant courant en UTC, avec fuseau."""
    return datetime.now(UTC)


def utc_iso(value: datetime | None) -> str | None:
    """``2026-10-08T12:32:05Z`` (C-1) ; ``None`` reste ``None`` (C-2).

    Une date naïve (fichiers ``[ACTUEL]``) est interprétée comme heure locale de la machine, comme à l'import JSON.
    """
    if value is None:
        return None
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

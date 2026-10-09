"""Sessions de plateforme (``platform_sessions``) : cookies chiffrés Fernet, valables 72 h (I-15)."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import PlatformSession


def get(session: Session, user_id: UUID, source_code: str) -> PlatformSession | None:
    return session.execute(
        select(PlatformSession).where(PlatformSession.user_id == user_id, PlatformSession.source_code == source_code)
    ).scalar_one_or_none()


def save(
    session: Session,
    user_id: UUID,
    source_code: str,
    cookies_encrypted: bytes,
    last_login_at: datetime,
    expires_at: datetime,
) -> None:
    """Écrase la session précédente (une par plateforme)."""
    values = {"cookies_encrypted": cookies_encrypted, "last_login_at": last_login_at, "expires_at": expires_at}
    statement = insert(PlatformSession).values(user_id=user_id, source_code=source_code, **values)
    session.execute(statement.on_conflict_do_update(index_elements=["user_id", "source_code"], set_=values))


def remove(session: Session, user_id: UUID, source_code: str) -> None:
    session.execute(
        delete(PlatformSession).where(PlatformSession.user_id == user_id, PlatformSession.source_code == source_code)
    )

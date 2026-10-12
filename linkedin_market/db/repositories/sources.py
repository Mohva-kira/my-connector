"""Sources (``sources``, données de référence) et consentements aux risques (``source_consents``, I-16)."""

from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import Source, SourceConsent


def all_sources(session: Session) -> list[Source]:
    """Sources de référence, triées par ``code`` (ordre stable, données posées par ``0001_initial``)."""
    return list(session.execute(select(Source).order_by(Source.code)).scalars())


def get(session: Session, code: str) -> Source | None:
    return session.get(Source, code)


def active_consent(session: Session, user_id: UUID, code: str) -> SourceConsent | None:
    """Consentement actif (``revoked_at IS NULL``) de l'utilisateur pour cette source, ou ``None``."""
    return session.execute(
        select(SourceConsent).where(
            SourceConsent.user_id == user_id,
            SourceConsent.source_code == code,
            SourceConsent.revoked_at.is_(None),
        )
    ).scalar_one_or_none()


def accept(session: Session, user_id: UUID, code: str, risk_version: str) -> SourceConsent:
    """Révoque le consentement actif éventuel puis insère le nouveau (``backend/03`` § 5.2, une transaction)."""
    session.execute(
        update(SourceConsent)
        .where(
            SourceConsent.user_id == user_id,
            SourceConsent.source_code == code,
            SourceConsent.revoked_at.is_(None),
        )
        .values(revoked_at=func.now())
    )
    return session.execute(
        insert(SourceConsent)
        .values(user_id=user_id, source_code=code, risk_version=risk_version)
        .returning(SourceConsent)
    ).scalar_one()


def revoke(session: Session, user_id: UUID, code: str) -> bool:
    """Révoque le consentement actif ; ``True`` si une ligne a été révoquée, ``False`` si aucune n'était active."""
    result = session.execute(
        update(SourceConsent)
        .where(
            SourceConsent.user_id == user_id,
            SourceConsent.source_code == code,
            SourceConsent.revoked_at.is_(None),
        )
        .values(revoked_at=func.now())
    )
    return result.rowcount > 0

"""Modèles de matching versionnés (``matching_models``) : un seul actif par utilisateur."""

from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..models import MatchingModelRecord, UserSettings


def active_model(session: Session, user_id: UUID) -> MatchingModelRecord | None:
    return session.execute(
        select(MatchingModelRecord).where(MatchingModelRecord.user_id == user_id, MatchingModelRecord.is_active)
    ).scalar_one_or_none()


def activate_new_version(session: Session, user_id: UUID, values: dict[str, Any]) -> int:
    """Désactive le modèle courant et insère la version suivante, active (``backend/03`` § 5.2). Renvoie la version.

    La ligne ``user_settings`` est verrouillée pour sérialiser deux activations concurrentes.
    """
    session.execute(select(UserSettings.user_id).where(UserSettings.user_id == user_id).with_for_update())
    version = (
        session.execute(
            select(func.coalesce(func.max(MatchingModelRecord.version), 0)).where(
                MatchingModelRecord.user_id == user_id
            )
        ).scalar_one()
        + 1
    )
    session.execute(
        update(MatchingModelRecord)
        .where(MatchingModelRecord.user_id == user_id, MatchingModelRecord.is_active)
        .values(is_active=False)
    )
    session.add(MatchingModelRecord(user_id=user_id, version=version, is_active=True, **values))
    session.flush()
    return version

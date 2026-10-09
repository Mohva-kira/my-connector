"""Réglages (``user_settings``) et réponses mémorisées (``memorized_answers``)."""

from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import MemorizedAnswer, UserSettings

COLUMNS = (
    "keyword",
    "location",
    "remote_only",
    "caution",
    "max_items",
    "profile_source",
    "interpreter",
    "apply_min_score",
    "apply_daily_max",
    "message_daily_max",
    "contact_email",
    "contact_phone",
    "signature",
    "linkedin_skill_sync",
)


def _row(session: Session, user_id: UUID, *, lock: bool = False) -> UserSettings:
    session.execute(insert(UserSettings).values(user_id=user_id).on_conflict_do_nothing(index_elements=["user_id"]))
    query = select(UserSettings).where(UserSettings.user_id == user_id)
    if lock:
        query = query.with_for_update()
    return session.execute(query).scalar_one()


def get_settings(session: Session, user_id: UUID, *, lock: bool = False) -> dict[str, Any]:
    """Réglages (colonnes de ``user_settings``) ; ``lock`` pose ``FOR UPDATE`` sur la ligne."""
    row = _row(session, user_id, lock=lock)
    return {name: getattr(row, name) for name in COLUMNS}


def update_settings(session: Session, user_id: UUID, values: dict[str, Any]) -> None:
    """Met à jour les colonnes connues présentes dans ``values`` (les autres clés sont ignorées)."""
    known = {k: v for k, v in values.items() if k in COLUMNS}
    _row(session, user_id)
    if known:
        session.execute(update(UserSettings).where(UserSettings.user_id == user_id).values(**known))


def get_answers(session: Session, user_id: UUID) -> dict[str, str]:
    rows = session.execute(
        select(MemorizedAnswer.question_key, MemorizedAnswer.answer)
        .where(MemorizedAnswer.user_id == user_id)
        .order_by(MemorizedAnswer.question_key)
    )
    return {key: answer for key, answer in rows}


def get_answer(session: Session, user_id: UUID, question_key: str) -> str | None:
    return session.execute(
        select(MemorizedAnswer.answer).where(
            MemorizedAnswer.user_id == user_id, MemorizedAnswer.question_key == question_key
        )
    ).scalar_one_or_none()


def set_answer(session: Session, user_id: UUID, question_key: str, answer: str) -> None:
    statement = insert(MemorizedAnswer).values(user_id=user_id, question_key=question_key, answer=answer)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["user_id", "question_key"], set_={"answer": answer, "updated_at": func.now()}
        )
    )


def replace_answers(session: Session, user_id: UUID, answers: dict[str, str]) -> None:
    """Remplace l'ensemble des réponses mémorisées (``PUT /api/settings`` avec ``answers``)."""
    session.execute(delete(MemorizedAnswer).where(MemorizedAnswer.user_id == user_id))
    for key, answer in answers.items():
        set_answer(session, user_id, key, answer)

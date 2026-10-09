"""Tâches de fond (``tasks``) : insérées à la soumission, mises à jour à chaque changement d'état."""

from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import Task

COLUMNS = (
    "kind",
    "title",
    "status",
    "browser",
    "params",
    "result",
    "error",
    "needs_input",
    "meta",
    "logs",
    "created_at",
    "started_at",
    "finished_at",
)


def insert_task(session: Session, user_id: UUID, task_id: UUID, values: dict[str, Any]) -> None:
    session.execute(
        insert(Task)
        .values(id=task_id, user_id=user_id, **{k: v for k, v in values.items() if k in COLUMNS})
        .on_conflict_do_nothing(index_elements=["id"])
    )


def update_task(session: Session, user_id: UUID, task_id: UUID, values: dict[str, Any]) -> None:
    session.execute(
        update(Task)
        .where(Task.user_id == user_id, Task.id == task_id)
        .values(**{k: v for k, v in values.items() if k in COLUMNS})
    )


def recent(session: Session, user_id: UUID, limit: int) -> list[Task]:
    """Dernières tâches, de la plus ancienne à la plus récente."""
    rows = session.execute(
        select(Task).where(Task.user_id == user_id).order_by(Task.created_at.desc(), Task.id.desc()).limit(limit)
    ).scalars()
    return list(reversed(rows.all()))

"""Journal d'activité (``activity_events``) : monitoring, plafonds quotidiens, blocages."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import ActivityEvent


def record(
    session: Session, user_id: UUID, kind: str, data: dict[str, Any], at: datetime | None = None
) -> ActivityEvent:
    values: dict[str, Any] = {"user_id": user_id, "kind": kind, "data": data}
    if at is not None:
        values["at"] = at
    return session.execute(insert(ActivityEvent).values(**values).returning(ActivityEvent)).scalar_one()


def latest(session: Session, user_id: UUID, limit: int | None = None) -> list[ActivityEvent]:
    """Derniers événements, du plus ancien au plus récent."""
    query = (
        select(ActivityEvent)
        .where(ActivityEvent.user_id == user_id)
        .order_by(ActivityEvent.at.desc(), ActivityEvent.id.desc())
    )
    if limit:
        query = query.limit(limit)
    return list(reversed(session.execute(query).scalars().all()))


def between(session: Session, user_id: UUID, start: datetime, end: datetime) -> list[ActivityEvent]:
    return list(
        session.execute(
            select(ActivityEvent)
            .where(ActivityEvent.user_id == user_id, ActivityEvent.at >= start, ActivityEvent.at < end)
            .order_by(ActivityEvent.at, ActivityEvent.id)
        ).scalars()
    )


def count_between(session: Session, user_id: UUID, kinds: tuple[str, ...], start: datetime, end: datetime) -> int:
    return session.execute(
        select(func.count())
        .select_from(ActivityEvent)
        .where(
            ActivityEvent.user_id == user_id,
            ActivityEvent.kind.in_(kinds),
            ActivityEvent.at >= start,
            ActivityEvent.at < end,
        )
    ).scalar_one()


def last_block_event(session: Session, user_id: UUID, source: str = "linkedin") -> ActivityEvent | None:
    """Dernier ``blocked`` ou ``unblocked`` de cette source (I-06 ; repos par source, vague 3).

    ``data.source`` absent (événements antérieurs à la vague 3) vaut ``linkedin`` (compatibilité).
    """
    source_expr = func.coalesce(ActivityEvent.data["source"].astext, "linkedin")
    return session.execute(
        select(ActivityEvent)
        .where(
            ActivityEvent.user_id == user_id, ActivityEvent.kind.in_(("blocked", "unblocked")), source_expr == source
        )
        .order_by(ActivityEvent.at.desc(), ActivityEvent.id.desc())
        .limit(1)
    ).scalar_one_or_none()

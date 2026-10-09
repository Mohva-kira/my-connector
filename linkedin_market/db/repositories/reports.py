"""Rapports d'analyse (``reports``), immuables (I-02) : ce dépôt n'expose aucune mise à jour."""

import re
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import Report


def next_slug(session: Session, user_id: UUID, base: str) -> str:
    """``base`` s'il est libre, sinon ``base-2``, ``base-3``… (C-8, E-1)."""
    taken = set(
        session.execute(
            select(Report.slug).where(Report.user_id == user_id, (Report.slug == base) | Report.slug.like(f"{base}-%"))
        ).scalars()
    )
    if base not in taken:
        return base
    suffix = re.compile(rf"^{re.escape(base)}-(\d+)$")
    numbers = [int(m.group(1)) for slug in taken if (m := suffix.match(slug))]
    return f"{base}-{max([1, *numbers]) + 1}"


def insert_report(
    session: Session,
    user_id: UUID,
    *,
    slug: str,
    keyword: str,
    summary: dict[str, Any],
    interpretation: str,
    interpreter_used: str,
    html_path: str,
    created_at: Any,
    task_id: UUID | None = None,
) -> bool:
    """Insère le rapport ; ``False`` si le slug est déjà pris (course entre deux écritures)."""
    statement = (
        insert(Report)
        .values(
            user_id=user_id,
            task_id=task_id,
            slug=slug,
            keyword=keyword,
            summary=summary,
            interpretation=interpretation,
            interpreter_used=interpreter_used,
            html_path=html_path,
            created_at=created_at,
        )
        .on_conflict_do_nothing(constraint="uq_reports_user_id_slug")
        .returning(Report.id)
    )
    return session.execute(statement).first() is not None


def list_reports(session: Session, user_id: UUID) -> list[dict[str, Any]]:
    """Rapports du plus récent au plus ancien : ``slug``, ``keyword``, ``created_at``, ``jobs``, ``html_path``."""
    jobs = func.coalesce(func.jsonb_array_length(Report.summary["top_jobs"]), 0)
    rows = session.execute(
        select(Report.slug, Report.keyword, Report.created_at, jobs, Report.html_path)
        .where(Report.user_id == user_id)
        .order_by(Report.created_at.desc(), Report.slug.desc())
    )
    return [
        {"slug": slug, "keyword": keyword, "created_at": created_at, "jobs": count, "html_path": html_path}
        for slug, keyword, created_at, count, html_path in rows
    ]


def get_report(session: Session, user_id: UUID, slug: str) -> Report | None:
    return session.execute(select(Report).where(Report.user_id == user_id, Report.slug == slug)).scalar_one_or_none()

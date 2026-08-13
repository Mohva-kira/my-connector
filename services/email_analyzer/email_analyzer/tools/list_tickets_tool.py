"""ListTicketsTool — tickets/réclamations client détectés dans les sujets
d'emails (motif ``[Banque_Plateforme_Numéro]``, voir
``email_analyzer/subject_tickets.py``), regroupés par projet et par libellé de
plateforme. Read-only, no LLM call — même discipline que
``list_meetings_tool.py``/``search_emails_tool.py`` (agrégation SQL
déterministe, pas de re-dérivation par le LLM de ce qui est déjà compté ici,
voir llm.py::TicketPatternItem)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from email_analyzer.db.models import Email, Project


class ListTicketsTool:
    name = "list_tickets"
    description = (
        "Liste les tickets/réclamations détectés dans les sujets d'emails (motif "
        "[Banque_Plateforme_NuméroTicket]), regroupés par plateforme, avec le "
        "nombre de tickets distincts, la dernière réception et des exemples de "
        "sujets — optionnellement filtrés par projet."
    )

    def __init__(self, db: Session) -> None:
        self._db = db

    def run(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        if project_id is not None:
            project = (
                self._db.query(Project)
                .filter(Project.id == project_id, Project.tenant_id == tenant_id)
                .first()
            )
            if project is None:
                raise HTTPException(status_code=404, detail="Projet introuvable")

        query = (
            self._db.query(
                Email.project_id,
                Project.name,
                Email.ticket_label,
                func.count(Email.id),
                func.count(func.distinct(Email.ticket_number)),
                func.max(Email.received_at),
            )
            .join(Project, Email.project_id == Project.id)
            .filter(Project.tenant_id == tenant_id, Email.ticket_label.isnot(None))
        )
        if project_id is not None:
            query = query.filter(Project.id == project_id)

        rows = (
            query.group_by(Email.project_id, Project.name, Email.ticket_label)
            .order_by(func.count(Email.id).desc())
            .limit(limit)
            .all()
        )

        results: List[Dict[str, Any]] = []
        for pid, project_name, ticket_label, email_count, distinct_ticket_count, last_received_at in rows:
            sample_subjects = [
                subj
                for (subj,) in (
                    self._db.query(Email.subject)
                    .filter(Email.project_id == pid, Email.ticket_label == ticket_label)
                    .order_by(Email.received_at.desc())
                    .limit(3)
                    .all()
                )
                if subj
            ]
            results.append(
                {
                    "project_id": str(pid),
                    "project_name": project_name,
                    "ticket_label": ticket_label,
                    "email_count": email_count,
                    "distinct_ticket_count": distinct_ticket_count,
                    "last_received_at": last_received_at.isoformat() if last_received_at else None,
                    "sample_subjects": sample_subjects,
                }
            )
        return results

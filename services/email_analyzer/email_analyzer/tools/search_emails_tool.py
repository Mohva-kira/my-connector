"""SearchEmailsTool — emails of a project: without a keyword, the top ones
ranked by importance score (same query already used by
api/routers/projects.py's project-detail endpoint — reused here, not
reimplemented) ; with a keyword, the ones whose subject or body match it
(ILIKE, no full-text index — matches the idiom already used for documents,
see document_repository.py), most recent first, with a short snippet around
the match. Read-only, no LLM call.

No cross-project search: `project_id` stays required — a tenant-wide
free-text email search is a distinct, larger feature (no bound, no
per-project relevance ranking) and is not what this tool does."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from email_analyzer.db.models import Email, Project
from email_analyzer.text_snippets import extract_snippet


class SearchEmailsTool:
    name = "search_emails"
    description = (
        "Retourne les emails d'un projet : sans mot-clé, les plus importants "
        "(triés par score d'importance puis date de réception) ; avec un "
        "mot-clé, ceux dont le sujet ou le corps contiennent ce mot-clé (triés "
        "par date, avec un extrait du passage trouvé). Ne retourne jamais le "
        "corps complet."
    )

    def __init__(self, db: Session) -> None:
        self._db = db

    def run(
        self,
        tenant_id: uuid.UUID,
        project_id: uuid.UUID,
        keyword: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        project = (
            self._db.query(Project)
            .filter(Project.id == project_id, Project.tenant_id == tenant_id)
            .first()
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Projet introuvable")

        query = self._db.query(Email).filter(Email.project_id == project.id)
        if keyword:
            needle = f"%{keyword.strip()}%"
            emails = (
                query.filter(Email.subject.ilike(needle) | Email.body_encrypted.ilike(needle))
                .order_by(Email.received_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "email_id": str(e.id),
                    "subject": e.subject,
                    "received_at": e.received_at.isoformat() if e.received_at else None,
                    "importance_score": e.importance_score,
                    "tags": e.tags,
                    "body_snippet": extract_snippet(e.body_encrypted, keyword),
                }
                for e in emails
            ]

        emails = (
            query.order_by(Email.importance_score.desc().nullslast(), Email.received_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "email_id": str(e.id),
                "subject": e.subject,
                "received_at": e.received_at.isoformat() if e.received_at else None,
                "importance_score": e.importance_score,
                "tags": e.tags,
            }
            for e in emails
        ]

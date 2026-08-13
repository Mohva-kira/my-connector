"""SearchProjectSummariesTool — historique des résumés archivés d'un projet
(ProjectSummaryHistory, alimenté par le hook d'archivage dans
analysis_tasks.py à chaque régénération Fast-Track). Read-only, no LLM call
— même patron que SearchEmailsTool/SummarizeProjectTool."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from email_analyzer.db.models import Project, ProjectSummaryHistory
from email_analyzer.text_snippets import extract_snippet


class SearchProjectSummariesTool:
    name = "search_project_summaries"
    description = (
        "Recherche dans l'historique des résumés archivés d'un projet (versions "
        "précédentes, avant chaque régénération) — utile pour savoir où en était "
        "un projet à une date passée. Ne couvre que les résumés régénérés depuis "
        "la mise en place de cet historique : les résumés antérieurs déjà écrasés "
        "ne sont pas récupérables."
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

        query = self._db.query(ProjectSummaryHistory).filter(
            ProjectSummaryHistory.project_id == project.id
        )
        if keyword:
            query = query.filter(ProjectSummaryHistory.content.ilike(f"%{keyword}%"))

        rows = query.order_by(ProjectSummaryHistory.created_at.desc()).limit(limit).all()
        return [
            {
                "archived_at": row.created_at.isoformat() if row.created_at else None,
                "sentiment": row.sentiment,
                "llm_risk_level": row.llm_risk_level,
                "snippet": extract_snippet(row.content, keyword),
            }
            for row in rows
        ]

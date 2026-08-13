"""SummarizeProjectTool — return the current persisted summary of one
project (ProjectSummary, already produced by the Fast-Track pipeline).
Read-only, no LLM call — this tool does not generate a new summary, it
retrieves the existing one (regenerating is the Fast-Track's job,
api/routers/projects.py::refresh)."""

from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import HTTPException
from sqlalchemy.orm import Session

from email_analyzer.db.models import Project, ProjectSummary


class SummarizeProjectTool:
    name = "summarize_project"
    description = (
        "Retourne le résumé courant d'un projet (décisions, risques, sentiment, "
        "actions en attente) déjà calculé par le Fast-Track. Ne relance pas "
        "d'analyse — si aucun résumé n'existe encore, le dit explicitement."
    )

    def __init__(self, db: Session) -> None:
        self._db = db

    def run(self, tenant_id: uuid.UUID, project_id: uuid.UUID) -> Dict[str, Any]:
        project = (
            self._db.query(Project)
            .filter(Project.id == project_id, Project.tenant_id == tenant_id)
            .first()
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Projet introuvable")

        summary = (
            self._db.query(ProjectSummary).filter(ProjectSummary.project_id == project.id).first()
        )
        if summary is None:
            return {
                "project_id": str(project.id),
                "name": project.name,
                "summary": None,
                "sentiment": None,
                "message": "Aucun résumé disponible pour ce projet pour le moment.",
            }
        return {
            "project_id": str(project.id),
            "name": project.name,
            "summary": summary.content,
            "sentiment": summary.sentiment,
            "risk_level": summary.llm_risk_level,
            # Thèmes récurrents à travers les tickets/réclamations client (voir
            # llm.py::ProjectSummaryLLM.recurring_issues) — l'énumération/comptage
            # exact des tickets reste le rôle de list_tickets, pas de celui-ci.
            "recurring_issues": (summary.structured_content or {}).get("recurring_issues") or [],
        }

"""DraftReplyTool — generates a reply draft for a project from its
persisted, SaaS ProjectSummary (structured_content) instead of the legacy
in-memory `analysis` dict (`api/main.py::draft`, IMAP-only, no tenant).

No LLM call: `email_analyzer.templates.generate_response_draft` is a
deterministic template picker (routine/alerte/kickoff) already used by
the legacy endpoint — reused as-is, not reimplemented. This tool's only
job is the adapter: map the SaaS-persisted shape (ProjectSummary,
ProjectSummaryLLM.structured_content, top Email rows) onto the legacy
French-keyed `analysis` dict the template functions expect
(`évaluation_risque`, `analyse_sentiment`, `emails_critiques`,
`résumé_assistant`), so the same battle-tested heuristics apply to real
persisted data instead of duplicating the whole template engine."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from email_analyzer.db.models import Email, Project, ProjectSummary
from email_analyzer.templates import generate_response_draft

_SENTIMENT_MAP = {
    "positif": "Positive",
    "awaiting_feedback": "Neutre",
    "under_tension": "Négative",
}


def _tendance_from(summary: ProjectSummary, structured: Dict[str, Any]) -> str:
    trend = structured.get("sentiment_trend")
    if trend:
        lowered = str(trend).lower()
        if "posit" in lowered:
            return "Positive"
        if "neg" in lowered or "nég" in lowered:
            return "Négative"
        return "Neutre"
    return _SENTIMENT_MAP.get(summary.sentiment or "", "Neutre")


class DraftReplyTool:
    name = "draft_reply"
    description = (
        "Génère un brouillon de réponse (routine/alerte/kickoff) pour un projet, "
        "à partir de son résumé et de ses risques déjà persistés — jamais envoyé "
        "automatiquement, toujours à relire."
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
            raise HTTPException(
                status_code=422, detail="Aucun résumé disponible pour ce projet pour le moment."
            )

        analysis = self._build_legacy_analysis(summary)
        return generate_response_draft(project.name, analysis)

    def _build_legacy_analysis(self, summary: ProjectSummary) -> Dict[str, Any]:
        structured = summary.structured_content or {}
        risques: List[Dict[str, Any]] = structured.get("risques") or []
        next_steps: List[Dict[str, Any]] = structured.get("next_steps") or []

        recommandation: Optional[str] = None
        if next_steps and next_steps[0].get("description"):
            recommandation = next_steps[0]["description"]

        top_emails = (
            self._db.query(Email)
            .filter(Email.project_id == summary.project_id)
            .order_by(Email.importance_score.desc().nullslast(), Email.received_at.desc())
            .limit(3)
            .all()
        )

        return {
            "évaluation_risque": {
                "niveau_risque": summary.llm_risk_level or "INDETERMINÉ",
                "recommandation": recommandation,
                "facteurs_risque": [r.get("description") for r in risques if r.get("description")],
            },
            "analyse_sentiment": {"tendance": _tendance_from(summary, structured)},
            "emails_critiques": [{"subject": e.subject} for e in top_emails],
            "résumé_assistant": {"texte": summary.content},
        }

"""Data access for the Daily Brief (Personal Executive AI Assistant pivot,
Phase 10). Extracted from api/routers/brief.py's GET /api/brief, which
had these queries inline — same pattern as ConversationRepository/
MemoryRepository/KnowledgeRepository. No new tables, no new queries:
reads Project/SuggestedAction/Email/ProjectSummary exactly as before."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from sqlalchemy.orm import Session

from email_analyzer.db.models import Email, Project, ProjectSummary, SuggestedAction

# Seuil au-delà duquel un email compte comme "échange important" dans le Brief
# (Email.importance_score, 0-100 — voir ai_intelligent.score_email_importance).
IMPORTANT_EMAIL_THRESHOLD = 70


class BriefRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def count_new_projects(self, tenant_id: uuid.UUID, since: datetime) -> int:
        return (
            self._db.query(Project)
            .filter(Project.tenant_id == tenant_id, Project.created_at >= since)
            .count()
        )

    def list_pending_actions_with_project(
        self, tenant_id: uuid.UUID
    ) -> List[Tuple[SuggestedAction, Project]]:
        return (
            self._db.query(SuggestedAction, Project)
            .join(Project, SuggestedAction.project_id == Project.id)
            .filter(Project.tenant_id == tenant_id, SuggestedAction.status == "pending")
            .order_by(SuggestedAction.deadline.asc().nullslast(), SuggestedAction.created_at.desc())
            .all()
        )

    def count_important_emails(self, tenant_id: uuid.UUID, since: datetime) -> int:
        return (
            self._db.query(Email)
            .filter(
                Email.tenant_id == tenant_id,
                Email.importance_score >= IMPORTANT_EMAIL_THRESHOLD,
                Email.received_at >= since,
            )
            .count()
        )

    def count_at_risk_projects(self, tenant_id: uuid.UUID) -> int:
        return (
            self._db.query(ProjectSummary)
            .join(Project, ProjectSummary.project_id == Project.id)
            .filter(
                Project.tenant_id == tenant_id,
                (ProjectSummary.sentiment == "under_tension")
                | (ProjectSummary.llm_risk_level == "CRITIQUE"),
            )
            .count()
        )

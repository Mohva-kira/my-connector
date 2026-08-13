"""Data access for the unified Timeline (Personal Executive AI Assistant
pivot, Phase 12). Extracted from api/routers/brief.py's GET /api/timeline,
which had the email/summary queries inline (same pattern as
BriefRepository, Unit 52) — extended with SuggestedAction and Appointment
so the timeline actually merges "Emails, Actions, Summaries, Future
Meetings" as the pivot plan asks, not just the first two."""

from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from email_analyzer.db.models import Appointment, Email, Project, ProjectSummary, SuggestedAction


class TimelineRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def list_email_events(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID], limit: int
    ) -> List[Tuple[Email, Project]]:
        query = (
            self._db.query(Email, Project)
            .join(Project, Email.project_id == Project.id)
            .filter(Email.tenant_id == tenant_id, Email.received_at.isnot(None))
        )
        if project_id is not None:
            query = query.filter(Project.id == project_id)
        return query.order_by(Email.received_at.desc()).limit(limit).all()

    def list_summary_events(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID]
    ) -> List[Tuple[ProjectSummary, Project]]:
        query = (
            self._db.query(ProjectSummary, Project)
            .join(Project, ProjectSummary.project_id == Project.id)
            .filter(Project.tenant_id == tenant_id)
        )
        if project_id is not None:
            query = query.filter(Project.id == project_id)
        return query.all()

    def list_action_events(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID]
    ) -> List[Tuple[SuggestedAction, Project]]:
        query = (
            self._db.query(SuggestedAction, Project)
            .join(Project, SuggestedAction.project_id == Project.id)
            .filter(Project.tenant_id == tenant_id)
        )
        if project_id is not None:
            query = query.filter(Project.id == project_id)
        return query.all()

    def list_appointment_events(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID]
    ) -> List[Tuple[Appointment, Project]]:
        query = (
            self._db.query(Appointment, Project)
            .join(Project, Appointment.project_id == Project.id)
            .filter(Appointment.tenant_id == tenant_id)
        )
        if project_id is not None:
            query = query.filter(Project.id == project_id)
        return query.all()

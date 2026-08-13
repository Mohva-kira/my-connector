"""Unified Timeline foundation (Personal Executive AI Assistant pivot,
Phase 12). Merges Emails, suggested Actions, Summary updates and
Appointments (meetings) into one chronologically sorted list — extracted
from api/routers/brief.py's GET /api/timeline (Emails+Summaries only)
and extended, not rewritten, per the pivot plan ("Merge Emails, Actions,
Summaries, Future Meetings. Display everything chronologically.").

No new data: all four event types are read from tables that already
exist (Email, SuggestedAction, ProjectSummary, Appointment) and were
already produced by the Fast-Track/agenda pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from email_analyzer.repositories.timeline_repository import TimelineRepository


@dataclass
class TimelineEvent:
    type: str  # "email" | "summary_update" | "action_suggested" | "meeting"
    project_id: uuid.UUID
    project_name: str
    at: datetime
    label: str


class TimelineService:
    def __init__(self, db: Session) -> None:
        self._repo = TimelineRepository(db)

    def build_timeline(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID] = None, limit: int = 50
    ) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []

        for email, project in self._repo.list_email_events(tenant_id, project_id, limit):
            events.append(
                TimelineEvent(
                    type="email",
                    project_id=project.id,
                    project_name=project.name,
                    at=email.received_at,
                    label=email.subject or "(sans objet)",
                )
            )

        for summary, project in self._repo.list_summary_events(tenant_id, project_id):
            events.append(
                TimelineEvent(
                    type="summary_update",
                    project_id=project.id,
                    project_name=project.name,
                    at=summary.updated_at,
                    label="Résumé mis à jour",
                )
            )

        for action, project in self._repo.list_action_events(tenant_id, project_id):
            events.append(
                TimelineEvent(
                    type="action_suggested",
                    project_id=project.id,
                    project_name=project.name,
                    at=action.created_at,
                    label=action.description,
                )
            )

        for appointment, project in self._repo.list_appointment_events(tenant_id, project_id):
            events.append(
                TimelineEvent(
                    type="meeting",
                    project_id=project.id,
                    project_name=project.name,
                    at=appointment.scheduled_at,
                    label=appointment.description,
                )
            )

        events.sort(key=lambda e: e.at, reverse=True)
        return events[:limit]

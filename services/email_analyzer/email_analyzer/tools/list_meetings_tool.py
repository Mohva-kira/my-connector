"""ListMeetingsTool — upcoming appointments (meetings) for a tenant,
optionally scoped to a project. Read-only, no LLM call.

Reuses the exact query already used by api/routers/agenda.py::get_agenda
(status != cancelled, scheduled_at >= now, ascending) rather than
reinventing "what counts as upcoming" — same reasoning as SearchEmailsTool
(Phase 8) reusing projects.py's top-emails query."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from email_analyzer.db.models import Appointment, AppointmentStatus, Project


class ListMeetingsTool:
    name = "list_meetings"
    description = (
        "Liste les prochains rendez-vous/réunions (non annulés, à venir), "
        "optionnellement filtrés par projet, triés du plus proche au plus lointain."
    )

    def __init__(self, db: Session) -> None:
        self._db = db

    def run(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        query = (
            self._db.query(Appointment, Project)
            .join(Project, Appointment.project_id == Project.id)
            .filter(
                Appointment.tenant_id == tenant_id,
                Appointment.status != AppointmentStatus.cancelled.value,
                Appointment.scheduled_at >= now,
            )
        )
        if project_id is not None:
            query = query.filter(Project.id == project_id)

        rows = query.order_by(Appointment.scheduled_at.asc()).limit(limit).all()
        return [
            {
                "appointment_id": str(appt.id),
                "project_id": str(project.id),
                "project_name": project.name,
                "description": appt.description,
                "scheduled_at": appt.scheduled_at.isoformat(),
                "participants": appt.participants or [],
                "status": appt.status,
            }
            for appt, project in rows
        ]

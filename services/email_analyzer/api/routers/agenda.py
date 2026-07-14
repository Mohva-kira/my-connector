"""Agenda IA : rendez-vous extraits des emails, et projets nécessitant un
prochain retour (probable_next_contact_*, voir analysis_tasks.py::
_run_fasttrack_sync et llm.ProbableContactItem).

Agrège des données déjà persistées par le Fast-Track/cron — aucun appel LLM
synchrone ici (ai-workflow-rules.md §4). Le rafraîchissement à la demande
(`POST /api/agenda/refresh`) délègue au worker arq comme le reste du Fast-
Track (job_id + polling générique `GET /api/analyze/{job_id}`).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from email_analyzer.db.models import (
    Appointment,
    AppointmentStatus,
    Project,
    ProjectStatus,
    ProjectSummary,
)
from email_analyzer.db.session import get_db
from email_analyzer.jobs import create_job, enqueue
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled

router = APIRouter(prefix="/api/agenda", tags=["agenda"])


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class AppointmentOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    project_name: str
    description: str
    scheduled_at: datetime
    participants: List[str]
    status: str


class UpdateAppointmentStatusBody(BaseModel):
    status: str


class AgendaProjectOut(BaseModel):
    project_id: uuid.UUID
    project_name: str
    sentiment: Optional[str]
    llm_risk_level: Optional[str]
    probable_next_contact_date: Optional[datetime]
    probable_next_contact_reason: Optional[str]
    probable_next_contact_confidence: Optional[str]
    last_processed_email_timestamp: Optional[datetime]


class AgendaOut(BaseModel):
    appointments: List[AppointmentOut]
    awaiting_projects: List[AgendaProjectOut]
    at_risk_projects: List[AgendaProjectOut]
    # Plus ancienne préparation IA parmi les projets retournés ci-dessus — sert
    # à afficher "dernière préparation IA : il y a Xh" côté frontend ; None si
    # aucun projet en attente/en rouge n'a encore été préparé.
    agenda_updated_at: Optional[datetime]


def _agenda_project_out(project: Project, summary: ProjectSummary) -> AgendaProjectOut:
    return AgendaProjectOut(
        project_id=project.id,
        project_name=project.name,
        sentiment=summary.sentiment,
        llm_risk_level=summary.llm_risk_level,
        probable_next_contact_date=summary.probable_next_contact_date,
        probable_next_contact_reason=summary.probable_next_contact_reason,
        probable_next_contact_confidence=summary.probable_next_contact_confidence,
        last_processed_email_timestamp=summary.last_processed_email_timestamp,
    )


@router.get("", response_model=AgendaOut)
def get_agenda(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> AgendaOut:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    now = datetime.now(timezone.utc)

    appointment_rows = (
        db.query(Appointment, Project)
        .join(Project, Appointment.project_id == Project.id)
        .filter(
            Appointment.tenant_id == tenant.id,
            Appointment.status != AppointmentStatus.cancelled.value,
            Appointment.scheduled_at >= now,
        )
        .order_by(Appointment.scheduled_at.asc())
        .all()
    )
    appointments = [
        AppointmentOut(
            id=appt.id,
            project_id=project.id,
            project_name=project.name,
            description=appt.description,
            scheduled_at=appt.scheduled_at,
            participants=appt.participants or [],
            status=appt.status,
        )
        for appt, project in appointment_rows
    ]

    base_query = (
        db.query(Project, ProjectSummary)
        .join(ProjectSummary, ProjectSummary.project_id == Project.id)
        .filter(Project.tenant_id == tenant.id, Project.status == ProjectStatus.active.value)
    )

    awaiting_rows = base_query.filter(ProjectSummary.sentiment == "awaiting_feedback").order_by(
        ProjectSummary.probable_next_contact_date.asc().nullslast()
    ).all()
    at_risk_rows = base_query.filter(
        or_(
            ProjectSummary.sentiment == "under_tension",
            ProjectSummary.llm_risk_level == "CRITIQUE",
        )
    ).order_by(ProjectSummary.probable_next_contact_date.asc().nullslast()).all()

    awaiting_projects = [_agenda_project_out(p, s) for p, s in awaiting_rows]
    at_risk_projects = [_agenda_project_out(p, s) for p, s in at_risk_rows]

    agenda_timestamps = [
        s.agenda_updated_at for _, s in (awaiting_rows + at_risk_rows) if s.agenda_updated_at
    ]
    agenda_updated_at = min(agenda_timestamps) if agenda_timestamps else None

    return AgendaOut(
        appointments=appointments,
        awaiting_projects=awaiting_projects,
        at_risk_projects=at_risk_projects,
        agenda_updated_at=agenda_updated_at,
    )


@router.patch("/appointments/{appointment_id}", response_model=AppointmentOut)
def update_appointment_status(
    appointment_id: uuid.UUID,
    body: UpdateAppointmentStatusBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> AppointmentOut:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    valid_statuses = {s.value for s in AppointmentStatus}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=422, detail=f"status invalide (attendu : {sorted(valid_statuses)})"
        )
    row = (
        db.query(Appointment, Project)
        .join(Project, Appointment.project_id == Project.id)
        .filter(Appointment.id == appointment_id, Appointment.tenant_id == tenant.id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Rendez-vous introuvable")
    appointment, project = row
    appointment.status = body.status
    db.commit()
    db.refresh(appointment)
    return AppointmentOut(
        id=appointment.id,
        project_id=project.id,
        project_name=project.name,
        description=appointment.description,
        scheduled_at=appointment.scheduled_at,
        participants=appointment.participants or [],
        status=appointment.status,
    )


class RefreshAgendaJobOut(BaseModel):
    job_id: str
    status: str


@router.post("/refresh", response_model=RefreshAgendaJobOut)
def refresh_agenda(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> RefreshAgendaJobOut:
    """Rafraîchissement Agenda à la demande, en plus du cron périodique (voir
    analysis_tasks.run_agenda_refresh) — même contrat générique de polling que
    le reste du Fast-Track (`GET /api/analyze/{job_id}`)."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    job_id = create_job(tenant_id=str(tenant.id))
    enqueue(job_id, "run_agenda_refresh_for_tenant", str(tenant.id))
    return RefreshAgendaJobOut(job_id=job_id, status="pending")

"""Brief quotidien et timeline portefeuille (tous projets du tenant).

Agrège des données déjà persistées par le Fast-Track/cron
(`analysis_tasks.py::_run_fasttrack_sync`) — aucune nouvelle table, aucun
appel LLM synchrone ici (voir `ai-workflow-rules.md` §4 : pas d'I/O lourd
dans le thread API).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.routers.actions import ActionOut, action_out
from email_analyzer.db.models import Email, Project, ProjectSummary, SuggestedAction
from email_analyzer.db.session import get_db
from email_analyzer.jobs import create_job, enqueue
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled

router = APIRouter(tags=["brief"])

# Seuil au-delà duquel un email compte comme "échange important" dans le Brief
# (Email.importance_score, 0-100 — voir ai_intelligent.score_email_importance).
_IMPORTANT_EMAIL_THRESHOLD = 70

# Fenêtre "échéances cette semaine" (jours) et de repli pour un tout premier
# login (previous_login_at absent — pas de session antérieure à comparer).
_UPCOMING_DEADLINE_DAYS = 7
_FIRST_VISIT_FALLBACK_DAYS = 7


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class BriefCounts(BaseModel):
    new_projects: int
    pending_actions: int
    upcoming_deadlines: int
    important_emails: int
    at_risk_projects: int


class BriefResponse(BaseModel):
    # Référence utilisée pour les compteurs ci-dessous (User.previous_login_at) ;
    # None si aucune session précédente (premier login), auquel cas une fenêtre
    # de repli de _FIRST_VISIT_FALLBACK_DAYS jours est utilisée à la place.
    since: Optional[datetime]
    counts: BriefCounts
    recommended_actions: List[ActionOut]


@router.get("/api/brief", response_model=BriefResponse)
def get_brief(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> BriefResponse:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    since = user.previous_login_at or (
        datetime.now(timezone.utc) - timedelta(days=_FIRST_VISIT_FALLBACK_DAYS)
    )

    new_projects = (
        db.query(Project)
        .filter(Project.tenant_id == tenant.id, Project.created_at >= since)
        .count()
    )

    pending_actions_rows = (
        db.query(SuggestedAction, Project)
        .join(Project, SuggestedAction.project_id == Project.id)
        .filter(Project.tenant_id == tenant.id, SuggestedAction.status == "pending")
        .order_by(SuggestedAction.deadline.asc().nullslast(), SuggestedAction.created_at.desc())
        .all()
    )
    deadline_cutoff = datetime.now(timezone.utc) + timedelta(days=_UPCOMING_DEADLINE_DAYS)
    upcoming_deadlines = sum(
        1 for a, _ in pending_actions_rows if a.deadline is not None and a.deadline <= deadline_cutoff
    )

    important_emails = (
        db.query(Email)
        .filter(
            Email.tenant_id == tenant.id,
            Email.importance_score >= _IMPORTANT_EMAIL_THRESHOLD,
            Email.received_at >= since,
        )
        .count()
    )

    at_risk_projects = (
        db.query(ProjectSummary)
        .join(Project, ProjectSummary.project_id == Project.id)
        .filter(
            Project.tenant_id == tenant.id,
            (ProjectSummary.sentiment == "under_tension") | (ProjectSummary.llm_risk_level == "CRITIQUE"),
        )
        .count()
    )

    return BriefResponse(
        since=user.previous_login_at,
        counts=BriefCounts(
            new_projects=new_projects,
            pending_actions=len(pending_actions_rows),
            upcoming_deadlines=upcoming_deadlines,
            important_emails=important_emails,
            at_risk_projects=at_risk_projects,
        ),
        recommended_actions=[action_out(a, p) for a, p in pending_actions_rows[:5]],
    )


class DiscoverProjectsBody(BaseModel):
    # 90 jours (~3 mois) : valeur envoyée par le bouton "Analyser" du Brief.
    # Bornée comme force_days (refresh_project) par sécurité, pas de sélecteur
    # de plage côté frontend pour ce bouton (contrairement à AnalyzeRangeModal).
    days: int = Field(90, ge=1, le=365)


class DiscoverProjectsJobOut(BaseModel):
    job_id: str
    status: str


@router.post("/api/brief/discover-projects", response_model=DiscoverProjectsJobOut)
def discover_projects(
    body: DiscoverProjectsBody = DiscoverProjectsBody(),
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> DiscoverProjectsJobOut:
    """Lance en tâche de fond un scan complet de la boîte mail (sans filtre
    projet) sur ``body.days`` jours, regroupé par domaine expéditeur — voir
    ``analysis_tasks.run_domain_discovery``. Le client interroge ensuite
    ``GET /api/analyze/{job_id}`` (même contrat générique que ``/api/analyze``
    et le Fast-Track, aucune nouvelle route de polling)."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    job_id = create_job(tenant_id=str(tenant.id))
    enqueue(job_id, "run_domain_discovery", str(tenant.id), body.days)
    return DiscoverProjectsJobOut(job_id=job_id, status="pending")


class TimelineEventOut(BaseModel):
    type: str  # "email" | "summary_update"
    project_id: uuid.UUID
    project_name: str
    at: datetime
    label: str


@router.get("/api/timeline", response_model=List[TimelineEventOut])
def get_timeline(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    limit: int = Query(50, ge=1, le=200),
    project_id: Optional[uuid.UUID] = Query(None),
) -> List[TimelineEventOut]:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    email_query = (
        db.query(Email, Project)
        .join(Project, Email.project_id == Project.id)
        .filter(Email.tenant_id == tenant.id, Email.received_at.isnot(None))
    )
    summary_query = (
        db.query(ProjectSummary, Project)
        .join(Project, ProjectSummary.project_id == Project.id)
        .filter(Project.tenant_id == tenant.id)
    )
    if project_id is not None:
        email_query = email_query.filter(Project.id == project_id)
        summary_query = summary_query.filter(Project.id == project_id)

    events: List[TimelineEventOut] = []
    for email, project in email_query.order_by(Email.received_at.desc()).limit(limit).all():
        events.append(
            TimelineEventOut(
                type="email",
                project_id=project.id,
                project_name=project.name,
                at=email.received_at,
                label=email.subject or "(sans objet)",
            )
        )
    for summary, project in summary_query.all():
        events.append(
            TimelineEventOut(
                type="summary_update",
                project_id=project.id,
                project_name=project.name,
                at=summary.updated_at,
                label="Résumé mis à jour",
            )
        )

    events.sort(key=lambda e: e.at, reverse=True)
    return events[:limit]

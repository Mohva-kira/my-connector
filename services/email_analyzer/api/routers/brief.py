"""Brief quotidien et timeline portefeuille (tous projets du tenant).

Agrège des données déjà persistées par le Fast-Track/cron
(`analysis_tasks.py::_run_fasttrack_sync`) — aucune nouvelle table, aucun
appel LLM synchrone ici (voir `ai-workflow-rules.md` §4 : pas d'I/O lourd
dans le thread API).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.routers.actions import ActionOut, action_out
from email_analyzer.db.session import get_db
from email_analyzer.jobs import create_job, enqueue
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled
from email_analyzer.services.brief_service import BriefService
from email_analyzer.services.timeline_service import TimelineService

router = APIRouter(tags=["brief"])


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
    # de repli est utilisée à la place (BriefService.FIRST_VISIT_FALLBACK_DAYS).
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
    brief = BriefService(db).build_daily_brief(tenant.id, user.previous_login_at)

    return BriefResponse(
        since=brief.since,
        counts=BriefCounts(
            new_projects=brief.counts.new_projects,
            pending_actions=brief.counts.pending_actions,
            upcoming_deadlines=brief.counts.upcoming_deadlines,
            important_emails=brief.counts.important_emails,
            at_risk_projects=brief.counts.at_risk_projects,
        ),
        recommended_actions=[action_out(a, p) for a, p in brief.recommended_actions],
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


class DiscoverTicketsBody(BaseModel):
    # Même borne/valeur par défaut que DiscoverProjectsBody — pas de
    # sélecteur de plage dédié côté frontend pour ce bouton non plus.
    days: int = Field(90, ge=1, le=365)


@router.post("/api/brief/discover-tickets", response_model=DiscoverProjectsJobOut)
def discover_tickets(
    body: DiscoverTicketsBody = DiscoverTicketsBody(),
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> DiscoverProjectsJobOut:
    """Lance en tâche de fond un scan complet de la boîte mail (sans filtre
    projet) sur ``body.days`` jours, regroupé par clé de ticket détectée dans
    le sujet ([Banque_Plateforme_Numéro], voir subject_tickets.py) — voir
    ``analysis_tasks.run_subject_key_discovery``. Même contrat de polling
    générique que ``discover_projects`` ci-dessus (``GET /api/analyze/{job_id}``)."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    job_id = create_job(tenant_id=str(tenant.id))
    enqueue(job_id, "run_subject_key_discovery", str(tenant.id), body.days)
    return DiscoverProjectsJobOut(job_id=job_id, status="pending")


class TimelineEventOut(BaseModel):
    type: str  # "email" | "summary_update" | "action_suggested" | "meeting"
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
    events = TimelineService(db).build_timeline(tenant.id, project_id, limit)
    return [
        TimelineEventOut(
            type=e.type, project_id=e.project_id, project_name=e.project_name, at=e.at, label=e.label
        )
        for e in events
    ]

"""CRUD minimal pour les projets persistés (Unit 10/11).

Sans ces endpoints, aucun `Project` ne peut jamais exister en base : le
Fast-Track (`POST /api/projects/{id}/refresh`, `analysis_tasks.py`) et la
persistance Unit 10 restent construits mais inaccessibles. Ce module ne fait
que créer/lister/consulter des projets manuellement — le clustering
automatique à l'onboarding (`evolution-plan.md` Phase 1, Historical
Clustering Worker) reste une unité séparée, non implémentée ici.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from email_analyzer.db.models import Project, ProjectSummary, SuggestedAction
from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    rules_matrix: Optional[dict] = None


class SuggestedActionOut(BaseModel):
    id: uuid.UUID
    description: str
    deadline: Optional[datetime]
    status: str
    created_at: datetime


class ProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    status: str
    created_at: datetime
    updated_at: datetime
    summary_content: Optional[str]
    sentiment: Optional[str]
    last_processed_email_timestamp: Optional[datetime]
    # "Outstanding actions" sur la carte projet (architecture.md, Project Hub) :
    # nombre d'actions suggérées encore `pending` (ni complétées ni écartées).
    pending_actions_count: int


class ProjectDetailOut(ProjectOut):
    suggested_actions: List[SuggestedActionOut]


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


def _pending_actions_count(db: Session, project_id: uuid.UUID) -> int:
    return (
        db.query(SuggestedAction)
        .filter(SuggestedAction.project_id == project_id, SuggestedAction.status == "pending")
        .count()
    )


def _project_out(
    db: Session, project: Project, summary: Optional[ProjectSummary]
) -> ProjectOut:
    return ProjectOut(
        id=project.id,
        name=project.name,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
        summary_content=summary.content if summary else None,
        sentiment=summary.sentiment if summary else None,
        last_processed_email_timestamp=(
            summary.last_processed_email_timestamp if summary else None
        ),
        pending_actions_count=_pending_actions_count(db, project.id),
    )


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(
    body: CreateProjectBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ProjectOut:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    project = Project(tenant_id=tenant.id, name=body.name.strip(), rules_matrix=body.rules_matrix)
    db.add(project)
    db.commit()
    db.refresh(project)
    return _project_out(db, project, None)


@router.get("", response_model=List[ProjectOut])
def list_projects(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> List[ProjectOut]:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    projects = (
        db.query(Project)
        .filter(Project.tenant_id == tenant.id)
        .order_by(Project.created_at.desc())
        .all()
    )
    return [_project_out(db, p, p.summary) for p in projects]


@router.get("/{project_id}", response_model=ProjectDetailOut)
def get_project(
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ProjectDetailOut:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.tenant_id == tenant.id)
        .first()
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Projet introuvable")

    actions = (
        db.query(SuggestedAction)
        .filter(SuggestedAction.project_id == project.id)
        .order_by(SuggestedAction.created_at.desc())
        .all()
    )
    base = _project_out(db, project, project.summary)
    return ProjectDetailOut(
        **base.model_dump(),
        suggested_actions=[
            SuggestedActionOut(
                id=a.id,
                description=a.description,
                deadline=a.deadline,
                status=a.status,
                created_at=a.created_at,
            )
            for a in actions
        ],
    )

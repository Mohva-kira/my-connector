"""Objectives — foundation endpoints (Personal Executive AI Assistant
pivot, Phase 11). Contrôleur HTTP fin : toute la logique vit dans
ObjectiveService."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled
from email_analyzer.services.objective_service import ObjectiveProgress, ObjectiveService

router = APIRouter(prefix="/api/objectives", tags=["objectives"])


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class MilestoneOut(BaseModel):
    id: uuid.UUID
    title: str
    due_date: Optional[datetime]
    status: str


class ObjectiveOut(BaseModel):
    id: uuid.UUID
    project_id: Optional[uuid.UUID]
    title: str
    target_date: Optional[datetime]
    status: str
    blocking_reason: Optional[str]
    milestones: List[MilestoneOut]
    completed_count: int
    total_count: int
    percent: int


def _objective_out(progress: ObjectiveProgress) -> ObjectiveOut:
    o = progress.objective
    return ObjectiveOut(
        id=o.id,
        project_id=o.project_id,
        title=o.title,
        target_date=o.target_date,
        status=o.status,
        blocking_reason=o.blocking_reason,
        milestones=[
            MilestoneOut(id=m.id, title=m.title, due_date=m.due_date, status=m.status)
            for m in progress.milestones
        ],
        completed_count=progress.completed_count,
        total_count=progress.total_count,
        percent=progress.percent,
    )


class CreateObjectiveBody(BaseModel):
    title: str
    project_id: Optional[uuid.UUID] = None
    target_date: Optional[datetime] = None


class UpdateObjectiveStatusBody(BaseModel):
    status: str
    blocking_reason: Optional[str] = None


class CreateMilestoneBody(BaseModel):
    title: str
    due_date: Optional[datetime] = None


class UpdateMilestoneStatusBody(BaseModel):
    status: str


@router.get("", response_model=List[ObjectiveOut])
def list_objectives(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    project_id: Optional[uuid.UUID] = Query(None),
) -> List[ObjectiveOut]:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    return [_objective_out(p) for p in ObjectiveService(db).list_objectives(tenant.id, project_id)]


@router.post("", response_model=ObjectiveOut)
def create_objective(
    body: CreateObjectiveBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ObjectiveOut:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    service = ObjectiveService(db)
    objective = service.create_objective(tenant.id, body.title, body.project_id, body.target_date)
    return _objective_out(service.get_objective(tenant.id, objective.id))


@router.patch("/{objective_id}", response_model=ObjectiveOut)
def update_objective_status(
    objective_id: uuid.UUID,
    body: UpdateObjectiveStatusBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ObjectiveOut:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    progress = ObjectiveService(db).update_status(
        tenant.id, objective_id, body.status, body.blocking_reason
    )
    return _objective_out(progress)


@router.post("/{objective_id}/milestones", response_model=ObjectiveOut)
def add_milestone(
    objective_id: uuid.UUID,
    body: CreateMilestoneBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ObjectiveOut:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    progress = ObjectiveService(db).add_milestone(tenant.id, objective_id, body.title, body.due_date)
    return _objective_out(progress)


@router.patch("/{objective_id}/milestones/{milestone_id}", response_model=ObjectiveOut)
def update_milestone_status(
    objective_id: uuid.UUID,
    milestone_id: uuid.UUID,
    body: UpdateMilestoneStatusBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ObjectiveOut:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    progress = ObjectiveService(db).set_milestone_status(
        tenant.id, objective_id, milestone_id, body.status
    )
    return _objective_out(progress)

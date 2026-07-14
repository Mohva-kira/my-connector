"""Actions suggérées tous projets confondus (portefeuille du tenant).

`SuggestedAction` n'était accessible que nichée dans `GET /api/projects/{id}`
(`api/routers/projects.py`) — nécessaire pour les pages Agenda/Actions du
Brief (context/rfc frontend), qui doivent lister/traiter les actions à
travers tous les projets sans connaître leur project_id à l'avance.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from email_analyzer.db.models import Project, SuggestedAction, SuggestedActionStatus
from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled

router = APIRouter(prefix="/api/actions", tags=["actions"])


class ActionOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    project_name: str
    description: str
    deadline: Optional[datetime]
    status: str
    created_at: datetime
    # Détail pour la fiche "ouvrir une action" (frontend) : pourquoi elle est
    # recommandée, qui est concerné, conseil pour éviter la récidive. NULL
    # pour les actions créées avant cette colonne ou via le repli sans
    # extraction structurée (voir analysis_tasks.py).
    rationale: Optional[str]
    stakeholder: Optional[str]
    advice: Optional[str]


class UpdateActionStatusBody(BaseModel):
    status: str


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


def action_out(action: SuggestedAction, project: Project) -> ActionOut:
    return ActionOut(
        id=action.id,
        project_id=project.id,
        project_name=project.name,
        description=action.description,
        deadline=action.deadline,
        status=action.status,
        created_at=action.created_at,
        rationale=action.rationale,
        stakeholder=action.stakeholder,
        advice=action.advice,
    )


@router.get("", response_model=List[ActionOut])
def list_actions(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    status: Optional[str] = Query(None, description="Filtre : pending/completed/dismissed"),
) -> List[ActionOut]:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    query = (
        db.query(SuggestedAction, Project)
        .join(Project, SuggestedAction.project_id == Project.id)
        .filter(Project.tenant_id == tenant.id)
    )
    if status is not None:
        query = query.filter(SuggestedAction.status == status)
    rows = query.order_by(
        SuggestedAction.deadline.asc().nullslast(), SuggestedAction.created_at.desc()
    ).all()
    return [action_out(a, p) for a, p in rows]


@router.patch("/{action_id}", response_model=ActionOut)
def update_action_status(
    action_id: uuid.UUID,
    body: UpdateActionStatusBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ActionOut:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    valid_statuses = {s.value for s in SuggestedActionStatus}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=422, detail=f"status invalide (attendu : {sorted(valid_statuses)})"
        )
    row = (
        db.query(SuggestedAction, Project)
        .join(Project, SuggestedAction.project_id == Project.id)
        .filter(SuggestedAction.id == action_id, Project.tenant_id == tenant.id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Action introuvable")
    action, project = row
    action.status = body.status
    db.commit()
    db.refresh(action)
    return action_out(action, project)

"""Knowledge Layer — foundation endpoints (Personal Executive AI Assistant
pivot, Phase 6). Contrôleur HTTP fin : toute la logique vit dans
KnowledgeService."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled
from email_analyzer.services.knowledge_service import KnowledgeService

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class ContactOut(BaseModel):
    id: uuid.UUID
    name: str
    organization_id: Optional[uuid.UUID]
    created_at: datetime


class OrganizationOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime


class SyncResponse(BaseModel):
    contacts: List[ContactOut]


class BulkSyncResponse(BaseModel):
    synced_projects: int
    failed_projects: int
    total_projects: int


@router.get("/contacts", response_model=List[ContactOut])
def list_contacts(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    project_id: Optional[uuid.UUID] = Query(None),
    limit: int = Query(200, ge=1, le=500),
) -> List[ContactOut]:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    service = KnowledgeService(db)
    contacts = (
        service.list_project_contacts(tenant.id, project_id)
        if project_id is not None
        else service.list_contacts(tenant.id, limit)
    )
    return [ContactOut.model_validate(c, from_attributes=True) for c in contacts]


@router.get("/organizations", response_model=List[OrganizationOut])
def list_organizations(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    limit: int = Query(200, ge=1, le=500),
) -> List[OrganizationOut]:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    orgs = KnowledgeService(db).list_organizations(tenant.id, limit)
    return [OrganizationOut.model_validate(o, from_attributes=True) for o in orgs]


@router.post("/sync", response_model=BulkSyncResponse)
def sync_all_knowledge(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> BulkSyncResponse:
    """Resynchronise le Knowledge Layer pour tous les projets actifs du
    tenant courant (bouton "Synchroniser" du chat, portée portefeuille —
    contrairement à /projects/{project_id}/sync qui est scopé à un seul
    projet)."""
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    counts = KnowledgeService(db).sync_all_for_tenant(tenant.id)
    return BulkSyncResponse(**counts)


@router.post("/projects/{project_id}/sync", response_model=SyncResponse)
def sync_project_knowledge(
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> SyncResponse:
    """Relit le dernier résumé structuré du projet (déjà produit par le
    Fast-Track) et upsert les contacts/organisations correspondants. Aucun
    appel LLM ici — voir KnowledgeService."""
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    contacts = KnowledgeService(db).sync_from_project_summary(tenant.id, project_id)
    return SyncResponse(contacts=[ContactOut.model_validate(c, from_attributes=True) for c in contacts])

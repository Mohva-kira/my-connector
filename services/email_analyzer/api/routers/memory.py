"""Memory Engine — foundation endpoints (Personal Executive AI Assistant
pivot, Phase 5). Contrôleur HTTP fin : toute la logique vit dans
MemoryService/ConversationService."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled
from email_analyzer.services.conversation_service import ConversationService
from email_analyzer.services.memory_service import MemoryService

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class MemoryOut(BaseModel):
    id: uuid.UUID
    content: str
    category: str
    source: str
    confidence: float
    created_at: datetime


class ExtractResponse(BaseModel):
    created: List[MemoryOut]


@router.get("", response_model=List[MemoryOut])
def list_memories(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    limit: int = Query(200, ge=1, le=500),
) -> List[MemoryOut]:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    memories = MemoryService(db).list_memories(tenant.id, user.id, limit)
    return [MemoryOut.model_validate(m, from_attributes=True) for m in memories]


@router.post("/extract", response_model=ExtractResponse)
def extract_memories(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    turns_limit: int = Query(20, ge=1, le=100, description="Nombre de derniers tours de chat à analyser"),
) -> ExtractResponse:
    """Déclenche explicitement une extraction de faits à partir des derniers
    tours de l'assistant portefeuille. Pas d'appel automatique après chaque
    message : évite un coût/latence LLM caché sur `/api/assistant/chat`."""
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)

    history = ConversationService(db).list_history_for_active_conversation(
        tenant.id, user.id, limit=turns_limit
    )
    if not history:
        raise HTTPException(status_code=422, detail="Aucun historique de conversation à analyser")

    turns = [{"role": m.role, "content": m.content} for m in history]
    created = MemoryService(db).extract_and_store(tenant.id, user.id, turns)
    return ExtractResponse(
        created=[MemoryOut.model_validate(m, from_attributes=True) for m in created]
    )


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> dict:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    deleted = MemoryService(db).delete_memory(tenant.id, user.id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Souvenir introuvable")
    return {"ok": True}

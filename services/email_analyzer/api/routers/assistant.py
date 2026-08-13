"""Assistant permanent (portefeuille) — point d'entrée principal du Brief.

Contrairement à `/api/chat` (api/main.py, scoped à un seul projet, contexte
= corpus IMAP brut, historique éphémère côté frontend), cet assistant répond
sur l'ensemble des projets du tenant et garde sa mémoire entre les sessions
(Conversation + AssistantMessage, db/models.py), avec plusieurs
conversations possibles par tenant/utilisateur (bouton "nouvelle
conversation" + panneau d'historique côté frontend). Depuis l'unification
Agent/Conversation, c'est aussi la seule surface de chat portefeuille : elle
absorbe les capacités de l'ancien `/api/agent/query` (retiré) via le
tool-calling natif du LLM (voir ConversationService.stream_message).

Ce routeur est un contrôleur HTTP fin : la logique (sélection de
fournisseur LLM, construction du contexte portefeuille, boucle agentique,
persistance) vit dans ConversationService
(email_analyzer/services/conversation_service.py). La réponse au chat est
un flux SSE (`text/event-stream`, voir api/sse.py) plutôt qu'un JSON unique
— première route de streaming du repo.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Iterator, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi import Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.sse import format_sse_event
from email_analyzer.config import DEFAULT_GEMINI_MODEL
from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled
from email_analyzer.services.conversation_service import AssistantProvider, ConversationService

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class AssistantMessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime


class ChatBody(BaseModel):
    message: str
    # Optionnel : sans préférence explicite du client, le serveur choisit le
    # premier fournisseur dont la clé API est configurée (voir
    # ConversationService._select_provider) plutôt que de faire porter ce
    # choix technique à l'utilisateur sur l'assistant permanent (contrairement
    # au formulaire d'analyse classique).
    assistant_provider: Optional[AssistantProvider] = None
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = DEFAULT_GEMINI_MODEL


@router.get("/conversations", response_model=List[ConversationOut])
def list_conversations(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> List[ConversationOut]:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    rows = ConversationService(db).list_conversations(tenant.id, user.id)
    return [
        ConversationOut(
            id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at
        )
        for c in rows
    ]


@router.post("/conversations", response_model=ConversationOut)
def create_conversation(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ConversationOut:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    c = ConversationService(db).create_conversation(tenant.id, user.id)
    return ConversationOut(id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at)


@router.get("/conversations/{conversation_id}/messages", response_model=List[AssistantMessageOut])
def list_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    limit: int = Query(100, ge=1, le=500),
) -> List[AssistantMessageOut]:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    rows = ConversationService(db).list_history(tenant.id, user.id, conversation_id, limit)
    return [
        AssistantMessageOut(role=m.role, content=m.content, created_at=m.created_at) for m in rows
    ]


@router.post("/conversations/{conversation_id}/chat")
def chat(
    conversation_id: uuid.UUID,
    body: ChatBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> StreamingResponse:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    events: Iterator[dict] = ConversationService(db).stream_message(
        tenant_id=tenant.id,
        user_id=user.id,
        conversation_id=conversation_id,
        message=body.message,
        preferred_provider=body.assistant_provider,
        openai_model=body.openai_model,
        gemini_model=body.gemini_model,
    )

    # `stream_message` est un générateur : rien ne s'exécute avant sa
    # première itération. On l'amorce ici, dans le corps synchrone de la
    # route, pour que la validation immédiate (message vide, conversation
    # introuvable, fournisseur non configuré) lève encore une vraie
    # HTTPException avec le bon status code — si on laissait ce premier
    # `next()` se produire dans `body_iter` ci-dessous, StreamingResponse
    # aurait déjà envoyé les entêtes HTTP 200 au client, trop tard pour
    # renvoyer un 404/422/503.
    try:
        first_event = next(events)
    except StopIteration:
        first_event = None

    def body_iter() -> Iterator[str]:
        if first_event is not None:
            yield format_sse_event(first_event["type"], first_event)
        for ev in events:
            yield format_sse_event(ev["type"], ev)

    return StreamingResponse(body_iter(), media_type="text/event-stream")

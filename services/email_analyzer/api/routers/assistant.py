"""Assistant permanent (portefeuille) — point d'entrée principal du Brief.

Contrairement à `/api/chat` (api/main.py, scoped à un seul projet, contexte
= corpus IMAP brut, historique éphémère côté frontend), cet assistant répond
sur l'ensemble des projets du tenant et garde sa mémoire entre les sessions
(AssistantMessage, db/models.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from email_analyzer.config import DEFAULT_GEMINI_MODEL
from email_analyzer.db.models import AssistantMessage, Project, ProjectSummary, SuggestedAction
from email_analyzer.db.session import get_db
from email_analyzer.llm import (
    build_portfolio_chat_system_prompt,
    build_portfolio_context,
    get_gemini_api_key,
    get_openai_api_key,
    portfolio_assistant_chat_gemini,
    portfolio_assistant_chat_openai,
)
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

AssistantProvider = Literal["openai", "gemini"]


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class AssistantMessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime


class ChatBody(BaseModel):
    message: str
    # Optionnel : sans préférence explicite du client, le serveur choisit le
    # premier fournisseur dont la clé API est configurée (voir _select_provider)
    # plutôt que de faire porter ce choix technique à l'utilisateur sur
    # l'assistant permanent (contrairement au formulaire d'analyse classique).
    assistant_provider: Optional[AssistantProvider] = None
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = DEFAULT_GEMINI_MODEL


class ChatResponse(BaseModel):
    message: str


def _select_provider(preferred: Optional[AssistantProvider]) -> AssistantProvider:
    if preferred == "openai" and get_openai_api_key():
        return "openai"
    if preferred == "gemini" and get_gemini_api_key():
        return "gemini"
    if get_openai_api_key():
        return "openai"
    if get_gemini_api_key():
        return "gemini"
    raise HTTPException(
        status_code=503,
        detail="Aucun fournisseur LLM configuré (OPENAI_API_KEY / GEMINI_API_KEY).",
    )


def _pending_action_descriptions(db: Session, project_id: uuid.UUID) -> List[str]:
    rows = (
        db.query(SuggestedAction)
        .filter(SuggestedAction.project_id == project_id, SuggestedAction.status == "pending")
        .order_by(SuggestedAction.deadline.asc().nullslast(), SuggestedAction.created_at.desc())
        .all()
    )
    return [a.description for a in rows]


def _portfolio_projects_context(db: Session, tenant_id: uuid.UUID) -> str:
    projects = db.query(Project).filter(Project.tenant_id == tenant_id).all()
    rows = []
    for p in projects:
        summary = db.query(ProjectSummary).filter(ProjectSummary.project_id == p.id).first()
        rows.append(
            {
                "name": p.name,
                "sentiment": summary.sentiment if summary else None,
                "summary_content": summary.content if summary else None,
                "pending_action_descriptions": _pending_action_descriptions(db, p.id),
            }
        )
    return build_portfolio_context(rows)


@router.get("/messages", response_model=List[AssistantMessageOut])
def list_messages(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    limit: int = Query(100, ge=1, le=500),
) -> List[AssistantMessageOut]:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    rows = (
        db.query(AssistantMessage)
        .filter(AssistantMessage.tenant_id == tenant.id, AssistantMessage.user_id == user.id)
        .order_by(AssistantMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return [
        AssistantMessageOut(role=m.role, content=m.content, created_at=m.created_at) for m in rows
    ]


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ChatResponse:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message vide")

    prior = (
        db.query(AssistantMessage)
        .filter(AssistantMessage.tenant_id == tenant.id, AssistantMessage.user_id == user.id)
        .order_by(AssistantMessage.created_at.asc())
        .all()
    )
    thread = [{"role": m.role, "content": m.content} for m in prior] + [
        {"role": "user", "content": message}
    ]

    system_prompt = build_portfolio_chat_system_prompt(_portfolio_projects_context(db, tenant.id))
    provider = _select_provider(body.assistant_provider)
    if provider == "gemini":
        result = portfolio_assistant_chat_gemini(
            system_prompt, thread, body.gemini_model, get_gemini_api_key()
        )
    else:
        result = portfolio_assistant_chat_openai(
            system_prompt, thread, body.openai_model, get_openai_api_key()
        )

    if result.get("erreur") or not result.get("message"):
        raise HTTPException(status_code=502, detail=result.get("erreur") or "Réponse vide")

    db.add(AssistantMessage(tenant_id=tenant.id, user_id=user.id, role="user", content=message))
    db.add(
        AssistantMessage(
            tenant_id=tenant.id, user_id=user.id, role="assistant", content=result["message"]
        )
    )
    db.commit()

    return ChatResponse(message=result["message"])

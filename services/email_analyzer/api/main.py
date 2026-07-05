"""
API HTTP pour l'analyse d'emails (interface web) — mode legacy ou SaaS multi-tenant.

Lancement (venv activé). Recommandé — sans PYTHONPATH :

  cd services\\email_analyzer
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

Depuis la racine du dépôt (CMD) : set PYTHONPATH=services\\email_analyzer puis la même commande uvicorn.
Sous PowerShell, "set" ne définit pas PYTHONPATH : utiliser $env:PYTHONPATH = "services\\email_analyzer" avant uvicorn.

Si DATABASE_URL est défini : authentification JWT et isolation par organisation.
Sinon : comportement d'origine (identifiants IMAP uniquement via .env).
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

_REPO_ROOT = _SERVICE_ROOT.parent.parent
from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_SERVICE_ROOT / ".env")

from email_analyzer.analyzer import EmailProcessor
from email_analyzer.config import (
    CHAT_MAX_OUTPUT_TOKENS,
    DEFAULT_GEMINI_MODEL,
    VALID_ASSISTANT_PROVIDERS,
    VALID_PERIODS,
)
from email_analyzer.db.session import get_db_optional
from email_analyzer.llm import (
    get_gemini_api_key,
    get_openai_api_key,
    project_assistant_chat_gemini,
    project_assistant_chat_openai,
)
from email_analyzer.saas_logic import (
    assert_can_run_analysis,
    authenticate_bearer,
    processor_from_tenant,
    record_analysis_usage,
    saas_enabled,
)
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.routers import auth as auth_router
from api.routers import billing as billing_router
from api.routers import oauth as oauth_router
from api.routers import tenants as tenants_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from email_analyzer.db.session import SessionLocal, init_db
    from email_analyzer.seed_plans import seed_plans_if_empty

    init_db()
    if SessionLocal is not None:
        db = SessionLocal()
        try:
            seed_plans_if_empty(db)
        finally:
            db.close()
    yield


def _imap_configured() -> bool:
    user = os.environ.get("IMAP_USER") or os.environ.get("EMAIL")
    pwd = os.environ.get("IMAP_PASSWORD")
    return bool(user and str(user).strip() and pwd and str(pwd).strip())


def _cors_origins() -> List[str]:
    base = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ]
    extra = (os.environ.get("FRONTEND_ORIGIN") or "").strip()
    if extra:
        base.append(extra)
    return base


app = FastAPI(title="Email Analyzer API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(tenants_router.router)
app.include_router(billing_router.router)
app.include_router(oauth_router.router)


class AnalyzeRequest(BaseModel):
    project: str = Field(..., min_length=1, description="Filtre nom de projet")
    period: Optional[str] = Field(
        None,
        description="today | yesterday | 3 | 7 | 11",
    )
    days: int = Field(30, ge=1, le=365, description="Fenêtre IMAP si period absent")
    assistant_provider: Literal["openai", "gemini", "none"] = "openai"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = DEFAULT_GEMINI_MODEL


class DraftRequest(BaseModel):
    project_name: str = Field(..., min_length=1)
    analysis: Dict[str, Any] = Field(
        ...,
        description="Bloc analyse d'un projet (sortie generate_intelligent_summary pour une clé)",
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    project_name: str = Field(..., min_length=1)
    period: Optional[str] = Field(
        None,
        description="Même logique que /api/analyze (today | yesterday | 3 | 7 | 11) ; sinon fenêtre days",
    )
    days: int = Field(30, ge=1, le=365, description="Fenêtre IMAP si period absent")
    messages: List[ChatMessage] = Field(..., min_length=1)
    assistant_provider: Literal["openai", "gemini"]
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = DEFAULT_GEMINI_MODEL


class ChatResponse(BaseModel):
    message: str


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "saas_enabled": saas_enabled(),
        "imap_configured": _imap_configured(),
    }


@app.post("/api/analyze")
def analyze(
    body: AnalyzeRequest,
    authorization: Optional[str] = Header(None),
    db: Optional[Session] = Depends(get_db_optional),
) -> Dict[str, Any]:
    if body.period is not None and body.period not in VALID_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Période invalide: {body.period}. Valeurs: {sorted(VALID_PERIODS)}",
        )
    if body.assistant_provider not in VALID_ASSISTANT_PROVIDERS:
        raise HTTPException(status_code=400, detail="assistant_provider invalide")

    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_SERVICE_ROOT / ".env")

    if db is None:
        proc = EmailProcessor(load_env=False)
        result = proc.process_latest_emails(
            body.project.strip(),
            period=body.period,
            days=body.days,
            assistant_provider=body.assistant_provider,
            openai_model=body.openai_model,
            gemini_model=body.gemini_model,
        )
        if isinstance(result, dict) and result.get("_error"):
            raise HTTPException(status_code=502, detail=str(result["_error"]))
        return result

    _, tenant, _ = authenticate_bearer(db, authorization)
    assert_can_run_analysis(db, tenant)
    proc = processor_from_tenant(tenant)
    result = proc.process_latest_emails(
        body.project.strip(),
        period=body.period,
        days=body.days,
        assistant_provider=body.assistant_provider,
        openai_model=body.openai_model,
        gemini_model=body.gemini_model,
    )
    if isinstance(result, dict) and result.get("_error"):
        db.rollback()
        raise HTTPException(status_code=502, detail=str(result["_error"]))
    record_analysis_usage(db, tenant)
    db.commit()
    return result


@app.post("/api/draft")
def draft(
    body: DraftRequest,
    authorization: Optional[str] = Header(None),
    db: Optional[Session] = Depends(get_db_optional),
) -> Dict[str, Any]:
    if db is None:
        proc = EmailProcessor(load_env=False)
        return proc.generate_response_draft(body.analysis, project_name=body.project_name)

    _, tenant, _ = authenticate_bearer(db, authorization)
    assert_can_run_analysis(db, tenant)
    proc = processor_from_tenant(tenant)
    return proc.generate_response_draft(body.analysis, project_name=body.project_name)


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    authorization: Optional[str] = Header(None),
    db: Optional[Session] = Depends(get_db_optional),
) -> ChatResponse:
    if body.period is not None and body.period not in VALID_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Période invalide: {body.period}. Valeurs: {sorted(VALID_PERIODS)}",
        )
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_SERVICE_ROOT / ".env")

    if db is None:
        proc = EmailProcessor(load_env=False)
    else:
        _, tenant, _ = authenticate_bearer(db, authorization)
        assert_can_run_analysis(db, tenant)
        proc = processor_from_tenant(tenant)

    msgs = [{"role": m.role, "content": m.content} for m in body.messages]

    recent_emails = proc.fetch_last_n_emails_for_chat(
        body.project_name.strip(),
        period=body.period,
        days=body.days,
        n=10,
    )
    if not recent_emails:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucun email trouvé pour ce filtre sur la période IMAP — "
                "impossible d'alimenter le mode conversationnel."
            ),
        )

    if body.assistant_provider == "openai":
        raw = project_assistant_chat_openai(
            body.project_name.strip(),
            recent_emails,
            msgs,
            model=body.openai_model,
            api_key=get_openai_api_key(),
            max_tokens=CHAT_MAX_OUTPUT_TOKENS,
        )
    else:
        raw = project_assistant_chat_gemini(
            body.project_name.strip(),
            recent_emails,
            msgs,
            model=body.gemini_model,
            api_key=get_gemini_api_key(),
            max_tokens=CHAT_MAX_OUTPUT_TOKENS,
        )

    err = raw.get("erreur")
    if err:
        raise HTTPException(status_code=400, detail=str(err))
    msg = raw.get("message")
    if not msg:
        raise HTTPException(status_code=502, detail="Réponse assistant vide")
    return ChatResponse(message=msg)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Email Analyzer API — GET /api/health ou POST /api/analyze"}

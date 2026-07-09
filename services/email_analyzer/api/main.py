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
import uuid
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
from email_analyzer.jobs import create_job, enqueue, get_job
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.routers import auth as auth_router
from api.routers import billing as billing_router
from api.routers import oauth as oauth_router
from api.routers import projects as projects_router
from api.routers import tenants as tenants_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from email_analyzer.ai_intelligent import get_shared_analyzer
    from email_analyzer.db.session import SessionLocal, init_db
    from email_analyzer.seed_plans import seed_plans_if_empty

    init_db()
    if SessionLocal is not None:
        db = SessionLocal()
        try:
            seed_plans_if_empty(db)
        finally:
            db.close()

    # Pré-charge les modèles ML une seule fois au démarrage : évite un cold-load
    # (dizaines de secondes) sur la première analyse, cause de timeouts 504.
    get_shared_analyzer()
    yield


def _imap_configured() -> bool:
    user = os.environ.get("IMAP_USER") or os.environ.get("EMAIL")
    pwd = os.environ.get("IMAP_PASSWORD")
    return bool(user and str(user).strip() and pwd and str(pwd).strip())


app = FastAPI(title="Email Analyzer API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(tenants_router.router)
app.include_router(billing_router.router)
app.include_router(oauth_router.router)
app.include_router(projects_router.router)


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


@app.post("/api/analyze", status_code=202)
def analyze(
    body: AnalyzeRequest,
    authorization: Optional[str] = Header(None),
    db: Optional[Session] = Depends(get_db_optional),
) -> Dict[str, Any]:
    """Lance l'analyse en tâche de fond et renvoie un job_id (202).

    Le travail lourd (IMAP + ML + LLM) dépasse le timeout de passerelle : on le
    met en file d'attente auprès du worker arq (process séparé, voir
    ``email_analyzer/analysis_tasks.py``) et le client interroge
    ``GET /api/analyze/{job_id}``.
    """
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
        job_id = create_job(tenant_id=None)
        enqueue(
            job_id,
            "run_analysis_legacy",
            body.project,
            body.period,
            body.days,
            body.assistant_provider,
            body.openai_model,
            body.gemini_model,
        )
        return {"job_id": job_id, "status": "pending"}

    # Auth + quota sont vérifiés de façon synchrone (rapide) : rejet immédiat si
    # non autorisé, avant de créer le job.
    _, tenant, _ = authenticate_bearer(db, authorization)
    assert_can_run_analysis(db, tenant)
    tenant_id = tenant.id
    job_id = create_job(tenant_id=str(tenant_id))
    enqueue(
        job_id,
        "run_analysis_saas",
        str(tenant_id),
        body.project,
        body.period,
        body.days,
        body.assistant_provider,
        body.openai_model,
        body.gemini_model,
    )
    return {"job_id": job_id, "status": "pending"}


@app.post("/api/projects/{project_id}/refresh", status_code=202)
def refresh_project(
    project_id: uuid.UUID,
    authorization: Optional[str] = Header(None),
    db: Optional[Session] = Depends(get_db_optional),
) -> Dict[str, Any]:
    """Fast-Track : régénère le résumé d'un projet à partir des seuls emails
    reçus depuis son dernier rafraîchissement (architecture.md, Process 2).

    Un projet est toujours scopé à un tenant (Unit 10) : cet endpoint
    nécessite le mode SaaS (DATABASE_URL défini), contrairement à
    ``/api/analyze`` qui reste utilisable en mode legacy.
    """
    if db is None:
        raise HTTPException(
            status_code=400, detail="Fast-Track nécessite le mode SaaS (DATABASE_URL)"
        )

    _, tenant, _ = authenticate_bearer(db, authorization)

    from email_analyzer.db.models import Project

    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.tenant_id == tenant.id)
        .first()
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Projet introuvable")

    job_id = create_job(tenant_id=str(tenant.id))
    enqueue(job_id, "run_fasttrack_refresh", str(tenant.id), str(project.id))
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/analyze/{job_id}")
def analyze_status(
    job_id: str,
    authorization: Optional[str] = Header(None),
    db: Optional[Session] = Depends(get_db_optional),
) -> Dict[str, Any]:
    """Retourne l'état d'un job d'analyse (pending | running | done | error)."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job introuvable ou expiré")

    # Mode SaaS : n'exposer un job qu'au tenant qui l'a lancé.
    if db is not None:
        _, tenant, _ = authenticate_bearer(db, authorization)
        if job.get("tenant_id") != str(tenant.id):
            raise HTTPException(status_code=404, detail="Job introuvable ou expiré")

    return {
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
        "progress": {"processed": job.get("processed", 0), "total": job.get("total", 0)},
        "partial": job.get("partial"),
    }


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


# --- Service du frontend build (mono-origine) -------------------------------
# Après `vite build`, le proxy dev de Vite n'existe plus. On sert donc le
# dossier `frontend/dist` directement depuis FastAPI : un seul uvicorn sur le
# port 8000 répond à la fois aux routes /api/* et à la SPA (URLs relatives).
_FRONTEND_DIST = _SERVICE_ROOT / "frontend" / "dist"
_FRONTEND_INDEX = _FRONTEND_DIST / "index.html"


if _FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(_FRONTEND_INDEX)

    # Fallback SPA : toute route non-API et non-fichier renvoie index.html
    # (routage côté client par react-router). Enregistré en dernier pour ne
    # jamais masquer /api/* ni /assets/*.
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_INDEX)

else:

    @app.get("/")
    def root() -> Dict[str, str]:
        return {"message": "Email Analyzer API — GET /api/health ou POST /api/analyze"}

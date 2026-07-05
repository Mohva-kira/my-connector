"""Tâches exécutées par le worker arq (process séparé du serveur uvicorn).

Avant migration, ces fonctions (``_run_analysis_legacy`` / ``_run_analysis_saas``)
vivaient dans ``api/main.py`` et étaient soumises à un ``ThreadPoolExecutor`` du
même process via ``jobs.submit``. Elles sont déplacées ici pour être
importables par le worker arq, lancé dans un process indépendant :

    arq email_analyzer.analysis_tasks.WorkerSettings

Lancer ce worker est désormais obligatoire — sans lui, les jobs mis en file
par ``jobs.enqueue`` restent ``pending`` indéfiniment.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from arq.connections import RedisSettings

from email_analyzer.config import redis_url
from email_analyzer.jobs import (
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_RUNNING,
    report_progress,
    set_status,
)


def _run_legacy_sync(
    job_id: str,
    project: str,
    period: Optional[str],
    days: int,
    assistant_provider: str,
    openai_model: str,
    gemini_model: str,
) -> Dict[str, Any]:
    """Analyse en mode legacy (identifiants IMAP via .env, sans DB)."""
    from email_analyzer.analyzer import EmailProcessor

    proc = EmailProcessor(load_env=False)
    on_batch = lambda processed, total, partial: report_progress(job_id, processed, total, partial)
    result = proc.process_latest_emails(
        project.strip(),
        period=period,
        days=days,
        assistant_provider=assistant_provider,
        openai_model=openai_model,
        gemini_model=gemini_model,
        on_batch=on_batch,
    )
    if isinstance(result, dict) and result.get("_error"):
        raise RuntimeError(str(result["_error"]))
    return result


def _run_saas_sync(
    job_id: str,
    tenant_id: str,
    project: str,
    period: Optional[str],
    days: int,
    assistant_provider: str,
    openai_model: str,
    gemini_model: str,
) -> Dict[str, Any]:
    """Analyse en mode SaaS : session DB dédiée (le worker n'a pas la session HTTP
    d'origine), usage comptabilisé une fois l'analyse terminée avec succès."""
    from email_analyzer.db.models import Tenant
    from email_analyzer.db.session import SessionLocal
    from email_analyzer.saas_logic import processor_from_tenant, record_analysis_usage

    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL non configuré")
    db = SessionLocal()
    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if tenant is None:
            raise RuntimeError("Organisation introuvable")
        proc = processor_from_tenant(tenant)
        on_batch = lambda processed, total, partial: report_progress(job_id, processed, total, partial)
        result = proc.process_latest_emails(
            project.strip(),
            period=period,
            days=days,
            assistant_provider=assistant_provider,
            openai_model=openai_model,
            gemini_model=gemini_model,
            on_batch=on_batch,
        )
        if isinstance(result, dict) and result.get("_error"):
            db.rollback()
            raise RuntimeError(str(result["_error"]))
        record_analysis_usage(db, tenant)
        db.commit()
        return result
    finally:
        db.close()


async def run_analysis_legacy(
    ctx: Dict[str, Any],
    job_id: str,
    project: str,
    period: Optional[str],
    days: int,
    assistant_provider: str,
    openai_model: str,
    gemini_model: str,
) -> None:
    set_status(job_id, STATUS_RUNNING)
    try:
        result = await asyncio.to_thread(
            _run_legacy_sync, job_id, project, period, days, assistant_provider, openai_model, gemini_model
        )
        set_status(job_id, STATUS_DONE, result=result)
    except Exception as exc:  # noqa: BLE001 — frontière du worker : rien ne doit remonter
        set_status(job_id, STATUS_ERROR, error=str(exc))


async def run_analysis_saas(
    ctx: Dict[str, Any],
    job_id: str,
    tenant_id: str,
    project: str,
    period: Optional[str],
    days: int,
    assistant_provider: str,
    openai_model: str,
    gemini_model: str,
) -> None:
    set_status(job_id, STATUS_RUNNING)
    try:
        result = await asyncio.to_thread(
            _run_saas_sync,
            job_id,
            tenant_id,
            project,
            period,
            days,
            assistant_provider,
            openai_model,
            gemini_model,
        )
        set_status(job_id, STATUS_DONE, result=result)
    except Exception as exc:  # noqa: BLE001 — frontière du worker : rien ne doit remonter
        set_status(job_id, STATUS_ERROR, error=str(exc))


async def on_startup(ctx: Dict[str, Any]) -> None:
    """Pré-charge l'analyseur ML partagé une fois par process worker (même logique
    que le lifespan FastAPI côté API — évite un cold-load sur le premier job)."""
    from email_analyzer.ai_intelligent import get_shared_analyzer

    get_shared_analyzer()


class WorkerSettings:
    functions = [run_analysis_legacy, run_analysis_saas]
    on_startup = on_startup
    redis_settings = RedisSettings.from_dsn(redis_url())
    # Un job à la fois par process worker : reproduit la limite de concurrence
    # de l'ancien ThreadPoolExecutor(max_workers=2) en lançant 2 process arq
    # (voir README / nginx-api.conf), plutôt qu'un seul process à forte concurrence.
    max_jobs = 2

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
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from arq import func
from arq.connections import RedisSettings
from arq.cron import cron
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Le worker arq est lancé comme process indépendant (voir docstring module) : il
# n'hérite pas du chargement de .env fait par api/main.py. Sans cet appel,
# DATABASE_URL/REDIS_URL restent absents de l'environnement du worker même
# quand ils sont bien définis dans .env, et `init_db()` (appelé dans
# `on_startup`) échoue avec "DATABASE_URL non configuré".
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SERVICE_ROOT.parent.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_SERVICE_ROOT / ".env")

from email_analyzer.config import redis_url
from email_analyzer.jobs import (
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_RUNNING,
    report_progress,
    set_status,
)

# Mapping du niveau de risque (ai_intelligent.calculate_risk_score) vers le
# sentiment stocké sur ProjectSummary (architecture.md : on_track / under_tension
# / awaiting_feedback). "MODÉRÉ" et l'absence de signal (INDETERMINÉ / clé
# manquante) retombent sur "awaiting_feedback", ni alarmant ni faussement positif.
_SENTIMENT_BY_RISK_LEVEL = {
    "FAIBLE": "on_track",
    "MODÉRÉ": "awaiting_feedback",
    "CRITIQUE": "under_tension",
}


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
        # Commit isolé et immédiat : un rafraîchissement de token OAuth réussi
        # côté fournisseur ne doit pas être perdu si l'analyse échoue plus loin
        # (voir email_analyzer/analyzer.py, _fetch_gmail_project_data/
        # _fetch_outlook_project_data).
        if proc.last_gmail_token_refresh:
            _persist_oauth_token_refresh(proc.gmail_connection, proc.last_gmail_token_refresh)
            db.commit()
        if proc.last_outlook_token_refresh:
            _persist_oauth_token_refresh(proc.outlook_connection, proc.last_outlook_token_refresh)
            db.commit()
        if isinstance(result, dict) and result.get("_error"):
            db.rollback()
            raise RuntimeError(str(result["_error"]))
        record_analysis_usage(db, tenant)
        db.commit()
        return result
    finally:
        db.close()


def _persist_oauth_token_refresh(connection: Any, token_refresh: Dict[str, Any]) -> None:
    """Persiste un access token OAuth rafraîchi (Gmail ou Outlook — voir
    gmail_oauth._get_valid_access_token / outlook_oauth._get_valid_access_token),
    même logique de chiffrement que le callback OAuth initial (api/routers/oauth.py).
    Outlook peut aussi renvoyer un nouveau refresh_token (rotation) ; Gmail non."""
    from email_analyzer.encryption import encrypt_secret

    connection.access_token_encrypted = encrypt_secret(token_refresh["access_token"])
    if token_refresh.get("expiry"):
        connection.token_expiry = token_refresh["expiry"]
    if token_refresh.get("refresh_token"):
        connection.refresh_token_encrypted = encrypt_secret(token_refresh["refresh_token"])


def _run_fasttrack_sync(job_id: str, tenant_id: str, project_id: str) -> Dict[str, Any]:
    """Fast-Track : régénère le résumé d'UN projet à partir du seul delta
    d'emails reçus depuis son dernier rafraîchissement (architecture.md,
    Process 2), et persiste le résultat (Unit 10 : Project/Email/ProjectSummary/
    SuggestedAction)."""
    from email_analyzer.ai_intelligent import get_shared_analyzer
    from email_analyzer.classification import ProjectRules, derive_tags, score_project_relevance
    from email_analyzer.db.models import (
        Email,
        Project,
        ProjectSummary,
        RecipientStatus,
        SuggestedAction,
        SuggestedActionStatus,
        Tenant,
    )
    from email_analyzer.db.session import SessionLocal
    from email_analyzer.period import parse_email_datetime
    from email_analyzer.saas_logic import processor_from_tenant

    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL non configuré")

    db = SessionLocal()
    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if tenant is None:
            raise RuntimeError("Organisation introuvable")
        project = (
            db.query(Project)
            .filter(Project.id == project_id, Project.tenant_id == tenant_id)
            .first()
        )
        if project is None:
            raise RuntimeError("Projet introuvable")

        summary_row = (
            db.query(ProjectSummary).filter(ProjectSummary.project_id == project.id).first()
        )
        since = summary_row.last_processed_email_timestamp if summary_row else None

        proc = processor_from_tenant(tenant)
        delta = proc.process_delta(project.name, since, rules_matrix=project.rules_matrix)
        if isinstance(delta, dict) and delta.get("_error"):
            raise RuntimeError(str(delta["_error"]))

        new_emails = delta.get("emails") or []
        refreshed_at = datetime.now(timezone.utc)
        tenant_email = (tenant.imap_user or "").lower()

        # Niveau de risque déjà calculé par le même appel LLM (delta["summary"]),
        # utilisé comme plancher du score d'importance par email — voir
        # ai_intelligent.score_email_importance. Extrait avant la boucle : le
        # même niveau de risque (projet) s'applique à tous les emails du delta.
        risk_level_for_scoring = (
            (delta.get("summary") or {}).get("évaluation_risque") or {}
        ).get("niveau_risque")
        shared_analyzer = get_shared_analyzer()
        # Rejoue les mêmes règles que le matching IMAP (project_mail.py) pour
        # persister un score de confiance auditable par email — recalcul pur,
        # sans I/O supplémentaire (déjà en mémoire à ce stade).
        project_rules = ProjectRules.from_dict(project.rules_matrix)

        persisted = 0
        for email_content in new_emails:
            message_id = (email_content.get("message_id") or "").strip()
            if not message_id:
                # Pas de Message-ID exploitable : impossible de dédupliquer cet
                # email de façon fiable, on ne le persiste pas (il reste compté
                # dans le résumé ci-dessous, seule sa persistance est sautée).
                continue
            exists = (
                db.query(Email.id)
                .filter(Email.tenant_id == tenant.id, Email.external_id == message_id)
                .first()
            )
            if exists:
                continue
            to_field = (email_content.get("to") or "").lower()
            recipient_status = (
                RecipientStatus.cc.value
                if tenant_email and tenant_email not in to_field
                else RecipientStatus.direct_to.value
            )
            importance_score = shared_analyzer.score_email_importance(
                email_content, recipient_status=recipient_status, niveau_risque=risk_level_for_scoring
            )
            relevance = score_project_relevance(email_content, project.name, project_rules)
            tags = derive_tags(email_content, importance_score=importance_score)
            db.add(
                Email(
                    tenant_id=tenant.id,
                    project_id=project.id,
                    external_id=message_id,
                    recipient_status=recipient_status,
                    subject=email_content.get("subject"),
                    body_encrypted=email_content.get("body"),
                    received_at=parse_email_datetime(email_content.get("date")),
                    importance_score=importance_score,
                    tags=tags,
                    classification_score=relevance.score,
                )
            )
            persisted += 1

        if summary_row is None:
            summary_row = ProjectSummary(project_id=project.id)
            db.add(summary_row)

        content = summary_row.content
        sentiment = summary_row.sentiment
        if new_emails:
            summary_block = delta.get("summary") or {}
            content = (
                (summary_block.get("résumé_assistant") or {}).get("texte")
                or summary_block.get("résumé_automatique")
                or content
            )
            risk = summary_block.get("évaluation_risque") or {}
            risk_level = risk.get("niveau_risque")
            sentiment = _SENTIMENT_BY_RISK_LEVEL.get(risk_level, "awaiting_feedback")
            summary_row.content = content
            summary_row.sentiment = sentiment

            db.query(SuggestedAction).filter(SuggestedAction.project_id == project.id).delete()
            recommendation = risk.get("recommandation")
            if recommendation and risk_level not in (None, "FAIBLE"):
                db.add(
                    SuggestedAction(
                        project_id=project.id,
                        description=recommendation,
                        status=SuggestedActionStatus.pending.value,
                    )
                )

        summary_row.last_processed_email_timestamp = refreshed_at
        db.commit()

        return {
            "project_id": str(project.id),
            "new_emails": len(new_emails),
            "persisted_emails": persisted,
            "sentiment": sentiment,
            "content": content,
            "last_processed_email_timestamp": refreshed_at.isoformat(),
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def run_fasttrack_refresh(
    ctx: Dict[str, Any],
    job_id: str,
    tenant_id: str,
    project_id: str,
) -> None:
    set_status(job_id, STATUS_RUNNING)
    try:
        result = await asyncio.to_thread(_run_fasttrack_sync, job_id, tenant_id, project_id)
        set_status(job_id, STATUS_DONE, result=result)
    except Exception as exc:  # noqa: BLE001 — frontière du worker : rien ne doit remonter
        set_status(job_id, STATUS_ERROR, error=str(exc))


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
    que le lifespan FastAPI côté API — évite un cold-load sur le premier job).

    Initialise aussi ``SessionLocal`` (``init_db``) : le worker arq tourne dans
    un process séparé de l'API (c'est le but de la migration Unit 9), donc le
    ``init_db()`` appelé par le lifespan FastAPI ne s'applique pas ici — sans
    cet appel, ``SessionLocal`` reste `None` pour toute la durée de vie du
    worker et chaque tâche touchant la DB (``run_analysis_saas``,
    ``run_fasttrack_refresh``) échoue avec "DATABASE_URL non configuré" même
    quand la variable d'environnement est bien définie.
    """
    from email_analyzer.ai_intelligent import get_shared_analyzer
    from email_analyzer.db.session import init_db

    init_db()
    get_shared_analyzer()


async def run_scheduled_sync(ctx: Dict[str, Any]) -> None:
    """Tâche cron arq (2x/jour, voir `WorkerSettings.cron_jobs`) : régénère
    chaque projet actif via la même logique Fast-Track que le rafraîchissement
    manuel (`_run_fasttrack_sync`, delta depuis `last_processed_email_timestamp`)
    — pas de logique de sync dupliquée. Remplace la mention historique
    APScheduler/Celery de `progress-tracker.md` : la stack réelle est `arq`.

    Un projet sans source exploitable (ni IMAP ni OAuth Gmail/Outlook — cas des
    tenants Gmail/Outlook-only tant que `process_delta` reste IMAP-only, voir
    Unit 15/17) échoue proprement via `_run_fasttrack_sync` (`_error` ->
    `RuntimeError`) et est loggé sans interrompre le traitement des projets
    suivants — même discipline que `_run_saas_sync`/`_run_fasttrack_sync`
    (jamais de `except Exception: pass` générique)."""
    from email_analyzer.db.models import Project, ProjectStatus
    from email_analyzer.db.session import SessionLocal

    if SessionLocal is None:
        logger.error("Sync planifiée annulée : DATABASE_URL non configuré")
        return

    db = SessionLocal()
    try:
        rows = (
            db.query(Project.id, Project.tenant_id)
            .filter(Project.status == ProjectStatus.active.value)
            .all()
        )
    finally:
        db.close()

    logger.info("Sync planifiée : %s projet(s) actif(s) à traiter", len(rows))
    succeeded = 0
    for project_id, tenant_id in rows:
        try:
            await asyncio.to_thread(
                _run_fasttrack_sync, f"scheduled:{project_id}", str(tenant_id), str(project_id)
            )
            succeeded += 1
        except Exception:
            logger.exception(
                "Sync planifiée : échec pour project_id=%s tenant_id=%s", project_id, tenant_id
            )
    logger.info("Sync planifiée terminée : %s/%s projet(s) rafraîchi(s)", succeeded, len(rows))


class WorkerSettings:
    # run_analysis_legacy/run_analysis_saas ne sont pas idempotents : un retry
    # après timeout refait entièrement le fetch IMAP + les résumés LLM au lieu
    # de reprendre où le job précédent s'est arrêté (aucun état intermédiaire
    # persisté). Le max_tries par défaut d'arq (5) transforme un job lent en
    # boucle de timeouts qui rejoue le même travail coûteux ; max_tries=1 fait
    # échouer proprement une fois plutôt que de la thrasher.
    functions = [
        func(run_analysis_legacy, max_tries=1),
        func(run_analysis_saas, max_tries=1),
        run_fasttrack_refresh,
    ]
    # Sync planifiée 2x/jour (7h/19h, heure du process worker — cf.
    # project-overview.md "briefing matin/soir") ; max_tries=1 car
    # run_scheduled_sync gère déjà ses échecs par projet en interne, un retry
    # global ne ferait que retraiter des projets déjà réussis.
    cron_jobs = [cron(run_scheduled_sync, hour={7, 19}, minute=0, max_tries=1)]
    on_startup = on_startup
    redis_settings = RedisSettings.from_dsn(redis_url())
    # Un job à la fois par process worker : reproduit la limite de concurrence
    # de l'ancien ThreadPoolExecutor(max_workers=2) en lançant 2 process arq
    # (voir README / nginx-api.conf), plutôt qu'un seul process à forte concurrence.
    max_jobs = 2

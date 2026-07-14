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

from email_analyzer.config import agenda_refresh_cron_hours, redis_url
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


def _parse_iso_date(date_iso: Optional[str]) -> Optional[datetime]:
    """Parse une date ISO ("YYYY-MM-DD" ou datetime complet) renvoyée par le
    LLM (llm.DeadlineItem.date_iso) en datetime tz-aware pour
    SuggestedAction.deadline. Le LLM n'est pas garanti de renvoyer un format
    valide — une valeur non parsable devient ``None`` plutôt qu'une exception
    qui interromprait toute la persistance du Fast-Track."""
    if not date_iso:
        return None
    try:
        parsed = datetime.fromisoformat(date_iso)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _run_legacy_sync(
    job_id: str,
    project: Optional[str],
    period: Optional[str],
    days: int,
    assistant_provider: str,
    openai_model: str,
    gemini_model: str,
) -> Dict[str, Any]:
    """Analyse en mode legacy (identifiants IMAP via .env, sans DB). ``project``
    vide/``None`` déclenche le mode "sans filtre" (voir `EmailProcessor.
    process_latest_emails`)."""
    from email_analyzer.analyzer import EmailProcessor

    proc = EmailProcessor(load_env=False)
    on_batch = lambda processed, total, partial: report_progress(job_id, processed, total, partial)
    project_clean = project.strip() if project else None
    result = proc.process_latest_emails(
        project_clean or None,
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
    project: Optional[str],
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
        project_clean = project.strip() if project else None
        result = proc.process_latest_emails(
            project_clean or None,
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


def _existing_domain_project_map(db: Any, tenant_id: str) -> Dict[str, Dict[str, str]]:
    """Construit domaine minuscule -> {id, name} à partir des
    ``rules_matrix.sender_domains`` des projets actifs du tenant — signal
    utilisé pour annoter la découverte (voir ``_run_domain_discovery_sync``)
    afin que le frontend propose une mise à jour plutôt qu'une création en
    double pour un domaine déjà suivi."""
    from email_analyzer.db.models import Project, ProjectStatus

    mapping: Dict[str, Dict[str, str]] = {}
    projects = (
        db.query(Project)
        .filter(Project.tenant_id == tenant_id, Project.status == ProjectStatus.active.value)
        .all()
    )
    for project in projects:
        domains = (project.rules_matrix or {}).get("sender_domains") or []
        for domain in domains:
            normalized = str(domain).strip().lower()
            if normalized:
                mapping[normalized] = {"id": str(project.id), "name": project.name}
    return mapping


def _run_domain_discovery_sync(job_id: str, tenant_id: str, days: int) -> Dict[str, Any]:
    """Découverte de domaines expéditeurs (bouton "Analyser" du Brief) : scanne
    toute la boîte mail sans filtre projet et regroupe par domaine, hors le
    domaine interne de l'entreprise (mediasoftci.net) et celui du tenant lui-même
    — aucune persistance, aucun appel LLM (voir email_analyzer/analyzer.py::
    discover_sender_domains). ``job_id`` alimente ``report_progress`` via
    ``on_batch`` : un scan sans filtre sur 90 jours peut porter sur plusieurs
    milliers d'emails (mesuré en conditions réelles, voir progress-tracker.md),
    sans progression le job resterait plusieurs minutes sans aucun signal côté
    client.

    Chaque domaine du résultat est annoté ``existing_project_id``/
    ``existing_project_name`` (``None`` si aucun projet actif du tenant ne
    couvre déjà ce domaine) — ``discover_sender_domains`` reste une fonction
    pure côté ``analyzer.py`` (pas d'accès DB), l'annotation se fait ici où la
    session DB est déjà ouverte. Permet au frontend
    (``DiscoverProjectsModal.tsx``) de proposer une mise à jour (Fast-Track)
    au lieu d'une création en double pour un domaine déjà suivi."""
    from email_analyzer.db.models import Tenant
    from email_analyzer.db.session import SessionLocal
    from email_analyzer.saas_logic import processor_from_tenant

    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL non configuré")
    db = SessionLocal()
    try:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if tenant is None:
            raise RuntimeError("Organisation introuvable")
        proc = processor_from_tenant(tenant)
        exclude_domains = {"mediasoftci.net"}
        tenant_email = (tenant.imap_user or "").strip().lower()
        if "@" in tenant_email:
            exclude_domains.add(tenant_email.rsplit("@", 1)[-1])
        on_batch = lambda processed, total, partial: report_progress(job_id, processed, total, partial)
        result = proc.discover_sender_domains(
            days_back=days, exclude_domains=exclude_domains, on_batch=on_batch
        )
        # Même discipline que _run_saas_sync : un rafraîchissement de token OAuth
        # réussi ne doit pas être perdu si la découverte échoue plus loin.
        if proc.last_gmail_token_refresh:
            _persist_oauth_token_refresh(proc.gmail_connection, proc.last_gmail_token_refresh)
            db.commit()
        if proc.last_outlook_token_refresh:
            _persist_oauth_token_refresh(proc.outlook_connection, proc.last_outlook_token_refresh)
            db.commit()
        if isinstance(result, dict) and result.get("_error"):
            raise RuntimeError(str(result["_error"]))
        if isinstance(result, dict) and result.get("domains"):
            domain_map = _existing_domain_project_map(db, tenant_id)
            for entry in result["domains"]:
                match = domain_map.get(str(entry.get("domain", "")).strip().lower())
                entry["existing_project_id"] = match["id"] if match else None
                entry["existing_project_name"] = match["name"] if match else None
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


def _run_fasttrack_sync(
    job_id: str,
    tenant_id: str,
    project_id: str,
    fallback_days: int = 30,
    force_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Fast-Track : régénère le résumé d'UN projet à partir du seul delta
    d'emails reçus depuis son dernier rafraîchissement (architecture.md,
    Process 2), et persiste le résultat (Unit 10 : Project/Email/ProjectSummary/
    SuggestedAction).

    ``fallback_days`` : fenêtre utilisée uniquement si le projet n'a jamais été
    synchronisé (``since is None``) — 30 par défaut, 60 pour le cron
    (``run_scheduled_sync``), 20 pour l'auto-sync au login
    (``saas_logic.trigger_login_auto_sync``).

    ``force_days`` : si fourni, ignore ``last_processed_email_timestamp`` et
    force un rescan complet sur cette fenêtre — utilisé par le bouton
    "chercher sur plus de jours" du frontend quand un premier sync ne trouve
    rien (voir ``api/main.py::refresh_project``).
    """
    from email_analyzer.ai_intelligent import get_shared_analyzer
    from email_analyzer.classification import ProjectRules, derive_tags, score_project_relevance
    from email_analyzer.db.models import (
        Appointment,
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
    from sqlalchemy.dialects.postgresql import insert as pg_insert

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
        if force_days is not None:
            since = None
            effective_fallback_days = force_days
        else:
            since = summary_row.last_processed_email_timestamp if summary_row else None
            effective_fallback_days = fallback_days

        proc = processor_from_tenant(tenant)
        delta = proc.process_delta(
            project.name,
            since,
            rules_matrix=project.rules_matrix,
            fallback_days=effective_fallback_days,
            # Persisté ci-dessous (ProjectSummary.structured_content) : seul
            # ce chemin (Fast-Track/cron) écrit en base, contrairement à
            # /api/analyze — voir analyzer.py::process_delta.
            include_structured=True,
        )
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

        known_external_ids = {
            row[0]
            for row in db.query(Email.external_id).filter(Email.tenant_id == tenant.id).all()
        }
        email_rows: list[dict] = []
        seen_external_ids: set = set()
        for email_content in new_emails:
            message_id = (email_content.get("message_id") or "").strip()
            if not message_id:
                # Pas de Message-ID exploitable : impossible de dédupliquer cet
                # email de façon fiable, on ne le persiste pas (il reste compté
                # dans le résumé ci-dessous, seule sa persistance est sautée).
                continue
            if message_id in known_external_ids or message_id in seen_external_ids:
                # Déjà en base, ou déjà vu dans ce même delta (process_delta peut
                # renvoyer le même message plusieurs fois s'il correspond à
                # plusieurs dossiers IMAP scannés) — évite un doublon dans le
                # batch d'insertion ci-dessous.
                continue
            seen_external_ids.add(message_id)
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
            email_rows.append(
                dict(
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

        persisted = 0
        if email_rows:
            # ON CONFLICT DO NOTHING plutôt qu'un simple check-then-insert : un
            # autre sync concurrent du même projet (cron 2x/jour vs. rafraîchissement
            # manuel/login) peut insérer le même (tenant_id, external_id) entre notre
            # SELECT de dédup ci-dessus et ce commit — sans ceci, le second des deux
            # échoue toute la transaction sur uq_email_tenant_external_id.
            insert_stmt = pg_insert(Email.__table__).values(email_rows).on_conflict_do_nothing(
                index_elements=["tenant_id", "external_id"]
            ).returning(Email.__table__.c.id)
            persisted = len(db.execute(insert_stmt).fetchall())

        if summary_row is None:
            summary_row = ProjectSummary(project_id=project.id)
            db.add(summary_row)

        content = summary_row.content
        sentiment = summary_row.sentiment
        tags_reference: list = []
        if new_emails:
            summary_block = delta.get("summary") or {}
            # "résumé_automatique" est lui-même un dict ({"résumé_automatique": <texte>,
            # "emails_analysés": ..., "méthode": ...}, voir ai_intelligent.generate_auto_summary/
            # generate_basic_summary) — même double-déballage que project_mail.py/templates.py,
            # sinon `content` (colonne Text) reçoit un dict et le commit échoue.
            auto_summary_block = summary_block.get("résumé_automatique")
            auto_summary_text = (
                auto_summary_block.get("résumé_automatique")
                if isinstance(auto_summary_block, dict)
                else None
            )
            content = (
                (summary_block.get("résumé_assistant") or {}).get("texte")
                or auto_summary_text
                or content
            )
            tags_reference = summary_block.get("tags_reference") or []
            risk = summary_block.get("évaluation_risque") or {}
            risk_level = risk.get("niveau_risque")
            sentiment = _SENTIMENT_BY_RISK_LEVEL.get(risk_level, "awaiting_feedback")
            summary_row.content = content
            summary_row.sentiment = sentiment

            # Sortie LLM structurée (llm.ProjectSummaryLLM, rfc-email-pipeline-v2.md
            # §11) : additive, jamais bloquante — une erreur d'extraction (clé
            # "_structured_erreur") laisse structured_content à None plutôt que
            # d'interrompre la persistance du reste du résumé.
            structured_content = summary_block.get("structured_content")
            summary_row.structured_content = structured_content
            summary_row.llm_risk_level = (
                (structured_content or {}).get("llm_risk_level")
            )

            db.query(SuggestedAction).filter(SuggestedAction.project_id == project.id).delete()
            next_steps = (structured_content or {}).get("next_steps") or []
            deadline_items = (structured_content or {}).get("deadlines") or []
            if next_steps or deadline_items:
                # Une ligne SuggestedAction par prochaine étape/échéance
                # structurée (llm.ProjectSummaryLLM) plutôt que l'ancienne
                # recommandation unique dérivée du seul niveau de risque.
                for step in next_steps:
                    description = (step.get("description") or "").strip()
                    if not description:
                        continue
                    db.add(
                        SuggestedAction(
                            project_id=project.id,
                            description=description,
                            status=SuggestedActionStatus.pending.value,
                            rationale=step.get("raison"),
                            stakeholder=step.get("responsable"),
                            advice=step.get("conseil_prevention"),
                        )
                    )
                for item in deadline_items:
                    description = (item.get("description") or "").strip()
                    if not description:
                        continue
                    db.add(
                        SuggestedAction(
                            project_id=project.id,
                            description=description,
                            deadline=_parse_iso_date(item.get("date_iso")),
                            status=SuggestedActionStatus.pending.value,
                            rationale=item.get("raison"),
                            stakeholder=item.get("responsable"),
                            advice=item.get("conseil_prevention"),
                        )
                    )
            else:
                # Repli : extraction structurée absente ou en échec (voir
                # summary_block["_structured_erreur"]) — conserve l'ancienne
                # recommandation dérivée des règles plutôt que de ne créer
                # aucune action.
                recommendation = risk.get("recommandation")
                if recommendation and risk_level not in (None, "FAIBLE"):
                    db.add(
                        SuggestedAction(
                            project_id=project.id,
                            description=recommendation,
                            status=SuggestedActionStatus.pending.value,
                        )
                    )

            # Rendez-vous extraits par l'IA (llm.AppointmentItem) : même
            # discipline idempotente que SuggestedAction ci-dessus (wipe +
            # rebuild à chaque resynchronisation, pas d'historique de rendez-
            # vous annulés/modifiés à préserver).
            db.query(Appointment).filter(Appointment.project_id == project.id).delete()
            for appt in (structured_content or {}).get("appointments") or []:
                description = (appt.get("description") or "").strip()
                scheduled_at = _parse_iso_date(appt.get("date_iso"))
                if not description or scheduled_at is None:
                    # Un rendez-vous sans date exploitable n'est pas
                    # affichable dans l'Agenda — on ne le persiste pas plutôt
                    # que d'inventer une date.
                    continue
                db.add(
                    Appointment(
                        tenant_id=tenant.id,
                        project_id=project.id,
                        description=description,
                        scheduled_at=scheduled_at,
                        participants=appt.get("participants") or [],
                    )
                )

            # Date de retour probable (llm.ProbableContactItem) — n'existe que
            # si le LLM a trouvé un signal réel dans les emails (voir prompt) ;
            # remis à None sinon plutôt que de garder une estimation périmée
            # d'un précédent rafraîchissement.
            probable_contact = (structured_content or {}).get("probable_next_contact")
            if probable_contact:
                summary_row.probable_next_contact_date = _parse_iso_date(
                    probable_contact.get("date_iso")
                )
                summary_row.probable_next_contact_reason = probable_contact.get("reason")
                summary_row.probable_next_contact_confidence = probable_contact.get("confidence")
            else:
                summary_row.probable_next_contact_date = None
                summary_row.probable_next_contact_reason = None
                summary_row.probable_next_contact_confidence = None
            summary_row.agenda_updated_at = refreshed_at

        summary_row.last_processed_email_timestamp = refreshed_at
        db.commit()

        return {
            "project_id": str(project.id),
            "new_emails": len(new_emails),
            "persisted_emails": persisted,
            "sentiment": sentiment,
            "content": content,
            "structured_content": summary_row.structured_content,
            "tags_reference": tags_reference,
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
    fallback_days: int = 30,
    force_days: Optional[int] = None,
) -> None:
    set_status(job_id, STATUS_RUNNING)
    try:
        result = await asyncio.to_thread(
            _run_fasttrack_sync, job_id, tenant_id, project_id, fallback_days, force_days
        )
        set_status(job_id, STATUS_DONE, result=result)
    except (Exception, asyncio.CancelledError) as exc:
        # CancelledError (job_timeout arq ou worker interrompu) n'est pas une
        # sous-classe d'Exception depuis Python 3.8 : sans ce cas explicite,
        # le job restait figé à "running" dans Redis pour toujours (aucun
        # signal jamais renvoyé au polling frontend) — voir
        # progress-tracker.md, jobs bloqués observés en conditions réelles.
        set_status(job_id, STATUS_ERROR, error=str(exc) or "Analyse interrompue (délai dépassé ou worker redémarré).")
        raise


async def run_analysis_legacy(
    ctx: Dict[str, Any],
    job_id: str,
    project: Optional[str],
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
    except (Exception, asyncio.CancelledError) as exc:
        # Voir run_fasttrack_refresh ci-dessus : CancelledError doit être
        # capturée explicitement, sinon le job reste "running" pour toujours.
        set_status(job_id, STATUS_ERROR, error=str(exc) or "Analyse interrompue (délai dépassé ou worker redémarré).")
        raise


async def run_analysis_saas(
    ctx: Dict[str, Any],
    job_id: str,
    tenant_id: str,
    project: Optional[str],
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
    except (Exception, asyncio.CancelledError) as exc:
        # Voir run_fasttrack_refresh ci-dessus : CancelledError doit être
        # capturée explicitement, sinon le job reste "running" pour toujours.
        set_status(job_id, STATUS_ERROR, error=str(exc) or "Analyse interrompue (délai dépassé ou worker redémarré).")
        raise


async def run_domain_discovery(
    ctx: Dict[str, Any],
    job_id: str,
    tenant_id: str,
    days: int,
) -> None:
    set_status(job_id, STATUS_RUNNING)
    try:
        result = await asyncio.to_thread(_run_domain_discovery_sync, job_id, tenant_id, days)
        set_status(job_id, STATUS_DONE, result=result)
    except (Exception, asyncio.CancelledError) as exc:
        # Voir run_fasttrack_refresh ci-dessus : CancelledError doit être
        # capturée explicitement, sinon le job reste "running" pour toujours.
        set_status(job_id, STATUS_ERROR, error=str(exc) or "Analyse interrompue (délai dépassé ou worker redémarré).")
        raise


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
            # fallback_days=60 : repli plus généreux que le défaut (30) pour les
            # projets jamais synchronisés — voir CLAUDE.md / progress-tracker.md,
            # le cron tournait souvent sans rien trouver avec une fenêtre trop
            # courte. N'affecte pas les projets déjà synchronisés (delta normal).
            await asyncio.to_thread(
                _run_fasttrack_sync,
                f"scheduled:{project_id}",
                str(tenant_id),
                str(project_id),
                60,
            )
            succeeded += 1
        except Exception:
            logger.exception(
                "Sync planifiée : échec pour project_id=%s tenant_id=%s", project_id, tenant_id
            )
    logger.info("Sync planifiée terminée : %s/%s projet(s) rafraîchi(s)", succeeded, len(rows))


def _agenda_relevant_project_rows(db: Any) -> list:
    """Projets actifs "en attente"/"en rouge" (même prédicat que
    api/routers/brief.py::at_risk_projects pour la partie risque) — portée
    volontairement réduite par rapport à `run_scheduled_sync` : c'est un sous-
    ensemble bon marché à rafraîchir plus souvent, pas un remplacement de la
    sync générale 2x/jour."""
    from email_analyzer.db.models import Project, ProjectStatus, ProjectSummary
    from sqlalchemy import or_

    return (
        db.query(Project.id, Project.tenant_id)
        .join(ProjectSummary, ProjectSummary.project_id == Project.id)
        .filter(
            Project.status == ProjectStatus.active.value,
            or_(
                ProjectSummary.sentiment.in_(["awaiting_feedback", "under_tension"]),
                ProjectSummary.llm_risk_level == "CRITIQUE",
            ),
        )
        .all()
    )


async def run_agenda_refresh(ctx: Dict[str, Any]) -> None:
    """Tâche cron arq dédiée à l'Agenda (voir `WorkerSettings.cron_jobs`,
    cadence `config.agenda_refresh_cron_hours()`) : régénère uniquement les
    projets "en attente"/"en rouge" via la même logique Fast-Track que
    `run_scheduled_sync` (`_run_fasttrack_sync`) — c'est ce qui alimente
    `Appointment` et `ProjectSummary.probable_next_contact_*`/
    `agenda_updated_at` pour `GET /api/agenda`. Même discipline d'échec par
    projet, sans bloquer le lot."""
    from email_analyzer.db.session import SessionLocal

    if SessionLocal is None:
        logger.error("Rafraîchissement Agenda annulé : DATABASE_URL non configuré")
        return

    db = SessionLocal()
    try:
        rows = _agenda_relevant_project_rows(db)
    finally:
        db.close()

    logger.info("Rafraîchissement Agenda : %s projet(s) concerné(s)", len(rows))
    succeeded = 0
    for project_id, tenant_id in rows:
        try:
            await asyncio.to_thread(
                _run_fasttrack_sync,
                f"agenda:{project_id}",
                str(tenant_id),
                str(project_id),
                30,
            )
            succeeded += 1
        except Exception:
            logger.exception(
                "Rafraîchissement Agenda : échec pour project_id=%s tenant_id=%s",
                project_id,
                tenant_id,
            )
    logger.info("Rafraîchissement Agenda terminé : %s/%s projet(s)", succeeded, len(rows))


def _run_agenda_refresh_for_tenant_sync(job_id: str, tenant_id: str) -> Dict[str, Any]:
    """Variante à la demande de `run_agenda_refresh`, scopée à un seul tenant
    (bouton "Rafraîchir" de l'Agenda, `POST /api/agenda/refresh`) — même
    sélection de projets, mais synchrone/bloquante comme les autres tâches
    `_run_*_sync` (exécutée via `asyncio.to_thread` par la tâche arq)."""
    from email_analyzer.db.models import Project
    from email_analyzer.db.session import SessionLocal

    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL non configuré")

    db = SessionLocal()
    try:
        rows = [
            (pid, tid)
            for pid, tid in _agenda_relevant_project_rows(db)
            if str(tid) == str(tenant_id)
        ]
    finally:
        db.close()

    refreshed = 0
    failed = 0
    for project_id, tid in rows:
        try:
            _run_fasttrack_sync(f"{job_id}:{project_id}", str(tid), str(project_id), 30)
            refreshed += 1
        except Exception:
            logger.exception(
                "Rafraîchissement Agenda (tenant %s) : échec pour project_id=%s", tenant_id, project_id
            )
            failed += 1
    return {"refreshed_projects": refreshed, "failed_projects": failed, "total_projects": len(rows)}


async def run_agenda_refresh_for_tenant(ctx: Dict[str, Any], job_id: str, tenant_id: str) -> None:
    set_status(job_id, STATUS_RUNNING)
    try:
        result = await asyncio.to_thread(_run_agenda_refresh_for_tenant_sync, job_id, tenant_id)
        set_status(job_id, STATUS_DONE, result=result)
    except (Exception, asyncio.CancelledError) as exc:
        # Voir run_fasttrack_refresh ci-dessus : CancelledError doit être
        # capturée explicitement, sinon le job reste "running" pour toujours.
        set_status(job_id, STATUS_ERROR, error=str(exc) or "Rafraîchissement Agenda interrompu.")
        raise


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
        # Timeout dédié (au lieu d'hériter des 1800s globaux) : même en lecture
        # headers-only (voir project_mail.py::_process_one_email_id), un scan
        # sans filtre sur une grosse boîte peut porter sur plusieurs milliers
        # d'emails. 7200s est un filet de sécurité, pas le temps attendu.
        func(run_domain_discovery, max_tries=1, timeout=7200),
        func(run_agenda_refresh_for_tenant, max_tries=1),
    ]
    # Sync planifiée 2x/jour (7h/19h, heure du process worker — cf.
    # project-overview.md "briefing matin/soir") ; max_tries=1 car
    # run_scheduled_sync gère déjà ses échecs par projet en interne, un retry
    # global ne ferait que retraiter des projets déjà réussis.
    # run_agenda_refresh : cadence dédiée et configurable (config.py::
    # agenda_refresh_cron_hours(), défaut toutes les 2h en heures ouvrées) —
    # même discipline d'échec par projet, portée réduite aux projets "en
    # attente"/"en rouge" (voir _agenda_relevant_project_rows).
    cron_jobs = [
        cron(run_scheduled_sync, hour={7, 19}, minute=0, max_tries=1),
        cron(run_agenda_refresh, hour=agenda_refresh_cron_hours(), minute=0, max_tries=1),
    ]
    on_startup = on_startup
    redis_settings = RedisSettings.from_dsn(redis_url())
    # Le défaut arq (300s) est trop court pour run_analysis_saas/legacy dès que
    # `days` couvre une grosse fenêtre (beaucoup d'emails -> beaucoup d'appels
    # LLM synchrones dans process_latest_emails) : le job se fait tuer par
    # TimeoutError avant la fin plutôt que d'aboutir. max_tries=1 fait déjà
    # échouer proprement au lieu de rejouer le travail ; il faut aussi laisser
    # assez de temps pour qu'un job normalement lent puisse réussir du premier
    # coup.
    job_timeout = 1800
    # Un job à la fois par process worker : reproduit la limite de concurrence
    # de l'ancien ThreadPoolExecutor(max_workers=2) en lançant 2 process arq
    # (voir README / nginx-api.conf), plutôt qu'un seul process à forte concurrence.
    max_jobs = 2

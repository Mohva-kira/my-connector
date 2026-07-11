"""Logique métier SaaS : processeur email par tenant, quotas, contexte requête."""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from datetime import timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from fastapi import HTTPException
from jose import JWTError
from sqlalchemy import and_
from sqlalchemy.orm import Session

from email_analyzer.analyzer import EmailProcessor
from email_analyzer.auth_jwt import decode_token, parse_user_tenant_ids
from email_analyzer.encryption import decrypt_secret

if TYPE_CHECKING:
    from email_analyzer.db.models import Membership, Plan, Subscription, Tenant, User


def saas_enabled() -> bool:
    return bool((os.environ.get("DATABASE_URL") or "").strip())


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")[:72]
    return s or "org"


def unique_tenant_slug(db: Session, base: str) -> str:
    from email_analyzer.db.models import Tenant

    slug = slugify(base)[:80]
    if not db.query(Tenant).filter(Tenant.slug == slug).first():
        return slug
    return f"{slug}-{uuid.uuid4().hex[:8]}"


def trial_analyses_limit() -> int:
    return int(os.environ.get("TRIAL_ANALYSES_LIMIT", "5"))


def year_month_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def authenticate_bearer_user_only(db: Session, authorization: Optional[str]) -> "User":
    from email_analyzer.db.models import User

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentification requise")
    token = authorization[7:].strip()
    try:
        payload = decode_token(token)
        user_id, _ = parse_user_tenant_ids(payload)
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Jeton invalide ou expiré") from e
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")
    return user


def authenticate_bearer(db: Session, authorization: Optional[str]) -> Tuple["User", "Tenant", "Membership"]:
    from email_analyzer.db.models import Membership, Tenant, User

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentification requise")
    token = authorization[7:].strip()
    try:
        payload = decode_token(token)
        user_id, tenant_id = parse_user_tenant_ids(payload)
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Jeton invalide ou expiré") from e

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=403, detail="Organisation introuvable")
    m = (
        db.query(Membership)
        .filter(and_(Membership.user_id == user.id, Membership.tenant_id == tenant.id))
        .first()
    )
    if not m:
        raise HTTPException(status_code=403, detail="Accès à cette organisation refusé")
    return user, tenant, m


def processor_from_tenant(tenant: "Tenant") -> EmailProcessor:
    cache_dir = (os.environ.get("EMAIL_ANALYZER_CACHE_DIR") or ".").strip()
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tenant_{tenant.id}.json")
    max_deep_emails = int(os.environ.get("EMAIL_ANALYZER_MAX_DEEP_EMAILS", "20"))

    if tenant.imap_password_encrypted:
        try:
            pwd = decrypt_secret(tenant.imap_password_encrypted)
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail="Configuration IMAP : déchiffrement impossible (ENCRYPTION_KEY)",
            ) from e
        port = tenant.imap_port if tenant.imap_port is not None else 993
        return EmailProcessor(
            email_address=tenant.imap_user or "",
            password=pwd,
            imap_server=tenant.imap_host or os.environ.get("IMAP_HOST", "mail.mediasoftci.net"),
            port=port,
            cache_file=cache_path,
            max_deep_emails=max_deep_emails,
            load_env=False,
            imap_folder=tenant.imap_folder,
            imap_use_ssl=tenant.imap_use_ssl,
            use_env_fallback=False,
        )

    # Pas d'IMAP configuré pour ce tenant : bascule sur Gmail ou Outlook OAuth
    # si une connexion existe (Open Question #1 de progress-tracker.md ;
    # EmailProcessor donne la priorité à Gmail si les deux sont connectés).
    # IMAP garde la priorité s'il est configuré (branche ci-dessus).
    connections = tenant.oauth_connections or []
    gmail_conn = next((c for c in connections if c.provider == "gmail"), None)
    outlook_conn = next((c for c in connections if c.provider == "outlook"), None)
    return EmailProcessor(
        cache_file=cache_path,
        max_deep_emails=max_deep_emails,
        load_env=False,
        use_env_fallback=False,
        gmail_connection=gmail_conn,
        outlook_connection=outlook_conn,
    )


def get_active_subscription_plan(
    db: Session, tenant_id: uuid.UUID
) -> Optional[Tuple["Subscription", "Plan"]]:
    from email_analyzer.db.models import Plan, Subscription

    row = (
        db.query(Subscription, Plan)
        .join(Plan, Subscription.plan_id == Plan.id)
        .filter(
            Subscription.tenant_id == tenant_id,
            Subscription.status == "active",
        )
        .order_by(Subscription.id.desc())
        .first()
    )
    return row


def assert_can_run_analysis(db: Session, tenant: "Tenant") -> None:
    from email_analyzer.db.models import TenantStatus, UsageCounter

    if tenant.status == TenantStatus.cancelled.value:
        raise HTTPException(status_code=402, detail="Organisation résiliée.")
    if tenant.status == TenantStatus.past_due.value:
        raise HTTPException(status_code=402, detail="Paiement en retard — renouvelez votre abonnement.")

    sub_plan = get_active_subscription_plan(db, tenant.id)
    if sub_plan:
        _, plan = sub_plan
        if plan.quota_analyses_per_month is not None:
            ym = year_month_now()
            uc = (
                db.query(UsageCounter)
                .filter(
                    UsageCounter.tenant_id == tenant.id,
                    UsageCounter.year_month == ym,
                )
                .first()
            )
            used = uc.analysis_count if uc else 0
            if used >= plan.quota_analyses_per_month:
                raise HTTPException(
                    status_code=402,
                    detail=f"Quota mensuel d'analyses atteint ({plan.quota_analyses_per_month}).",
                )
        return

    if tenant.status == TenantStatus.trial.value:
        if tenant.trial_analyses_used >= trial_analyses_limit():
            raise HTTPException(
                status_code=402,
                detail="Essai gratuit épuisé — souscrivez à un plan pour continuer.",
            )
        return

    raise HTTPException(status_code=402, detail="Abonnement actif requis.")


def record_analysis_usage(db: Session, tenant: "Tenant") -> None:
    from email_analyzer.db.models import TenantStatus, UsageCounter

    sub_plan = get_active_subscription_plan(db, tenant.id)
    if sub_plan:
        _, plan = sub_plan
        if plan.quota_analyses_per_month is None:
            return
        ym = year_month_now()
        uc = (
            db.query(UsageCounter)
            .filter(
                UsageCounter.tenant_id == tenant.id,
                UsageCounter.year_month == ym,
            )
            .first()
        )
        if not uc:
            uc = UsageCounter(
                tenant_id=tenant.id,
                year_month=ym,
                analysis_count=0,
            )
            db.add(uc)
        uc.analysis_count += 1
        db.flush()
        return

    if tenant.status == TenantStatus.trial.value:
        tenant.trial_analyses_used += 1
        db.flush()


def period_end_for_plan(plan_interval: str) -> datetime:
    now = datetime.now(timezone.utc)
    if plan_interval == "year":
        return now + timedelta(days=365)
    return now + timedelta(days=30)


# Fenêtre de repli (premier sync uniquement, voir analyzer.py::process_delta)
# utilisée par l'auto-sync déclenché au login — délibérément plus courte que le
# repli du cron (60j) : c'est un aperçu rapide à la connexion, le cron 2x/jour
# prend ensuite le relais avec une fenêtre plus large.
_LOGIN_AUTO_SYNC_FALLBACK_DAYS = 20


def trigger_login_auto_sync(db: Session, tenant_id: uuid.UUID) -> List[Dict[str, str]]:
    """Déclenche un Fast-Track sur tous les projets actifs d'un tenant au
    login, au plus 1x/24h (voir ``jobs.claim_daily_auto_sync``) — remplace le
    besoin pour l'utilisateur de retaper un filtre "Projet" à chaque
    connexion. Renvoie la liste des jobs déclenchés (``[]`` si le throttle est
    déjà consommé ou si le tenant n'a aucun projet actif), pour que le
    frontend puisse suivre leur progression (voir ``api/routers/auth.py::login``)."""
    from email_analyzer.jobs import claim_daily_auto_sync, create_job, enqueue

    if not claim_daily_auto_sync(str(tenant_id)):
        return []

    from email_analyzer.db.models import Project, ProjectStatus

    projects = (
        db.query(Project)
        .filter(Project.tenant_id == tenant_id, Project.status == ProjectStatus.active.value)
        .all()
    )
    triggered: List[Dict[str, str]] = []
    for project in projects:
        job_id = create_job(tenant_id=str(tenant_id))
        enqueue(
            job_id,
            "run_fasttrack_refresh",
            str(tenant_id),
            str(project.id),
            _LOGIN_AUTO_SYNC_FALLBACK_DAYS,
        )
        triggered.append({"project_id": str(project.id), "job_id": job_id})
    return triggered

"""Paramètres organisation / IMAP."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from email_analyzer.db.models import Tenant
from email_analyzer.db.session import get_db
from email_analyzer.encryption import encrypt_secret
from email_analyzer.auth_jwt import create_access_token
from email_analyzer.saas_logic import (
    authenticate_bearer,
    authenticate_bearer_user_only,
    saas_enabled,
    unique_tenant_slug,
)

router = APIRouter(prefix="/api/tenants", tags=["tenants"])


class ImapUpdateBody(BaseModel):
    imap_host: str = Field(..., min_length=1, max_length=255)
    imap_port: int = Field(993, ge=1, le=65535)
    imap_user: str = Field(..., min_length=1, max_length=320)
    imap_password: Optional[str] = Field(None, description="Nouveau mot de passe (optionnel si inchangé)")
    imap_folder: str = Field("INBOX", max_length=255)
    imap_use_ssl: bool = True


class CreateTenantBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class TenantOut(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    status: str
    imap_configured: bool


class TenantCreated(TenantOut):
    access_token: str


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


def _tenant_out(t: Tenant) -> TenantOut:
    ok = bool(t.imap_host and t.imap_user and t.imap_password_encrypted)
    return TenantOut(
        id=t.id,
        name=t.name,
        slug=t.slug,
        status=t.status,
        imap_configured=ok,
    )


@router.post("", response_model=TenantCreated)
def create_tenant(
    body: CreateTenantBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> TenantCreated:
    _require_saas()
    user = authenticate_bearer_user_only(db, authorization)
    slug = unique_tenant_slug(db, body.name)
    t = Tenant(name=body.name.strip(), slug=slug, status="trial")
    db.add(t)
    db.flush()
    from email_analyzer.db.models import Membership, MembershipRole

    db.add(
        Membership(
            user_id=user.id,
            tenant_id=t.id,
            role=MembershipRole.owner.value,
        )
    )
    db.commit()
    db.refresh(t)
    token = create_access_token(user_id=user.id, tenant_id=t.id, email=user.email)
    out = _tenant_out(t)
    return TenantCreated(**out.model_dump(), access_token=token)


@router.patch("/{tenant_id}/imap", response_model=TenantOut)
def update_imap(
    tenant_id: uuid.UUID,
    body: ImapUpdateBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> TenantOut:
    _require_saas()
    user, tenant, m = authenticate_bearer(db, authorization)
    if tenant.id != tenant_id:
        raise HTTPException(status_code=403, detail="Mauvaise organisation (utilisez le bon token ou X-Tenant)")
    if m.role not in ("owner",):
        raise HTTPException(status_code=403, detail="Seul le propriétaire peut modifier l'IMAP")

    tenant.imap_host = body.imap_host.strip()
    tenant.imap_port = body.imap_port
    tenant.imap_user = body.imap_user.strip()
    if body.imap_password and body.imap_password.strip():
        tenant.imap_password_encrypted = encrypt_secret(body.imap_password)
    elif not tenant.imap_password_encrypted:
        raise HTTPException(status_code=400, detail="Mot de passe IMAP requis pour la première configuration")
    tenant.imap_folder = (body.imap_folder or "INBOX").strip() or "INBOX"
    tenant.imap_use_ssl = body.imap_use_ssl
    db.commit()
    db.refresh(tenant)
    return _tenant_out(tenant)


@router.post("/{tenant_id}/imap/test")
def test_imap(
    tenant_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> dict:
    _require_saas()
    user, tenant, m = authenticate_bearer(db, authorization)
    if tenant.id != tenant_id:
        raise HTTPException(status_code=403, detail="Mauvaise organisation")
    if not tenant.imap_user or not tenant.imap_password_encrypted:
        raise HTTPException(status_code=400, detail="IMAP non configuré")
    from email_analyzer.saas_logic import processor_from_tenant

    proc = processor_from_tenant(tenant)
    from email_analyzer.project_mail import EmailProjectAnalyzer

    an = EmailProjectAnalyzer(
        proc.email_address,
        proc.password,
        proc.imap_server,
        proc.port,
        max_deep_emails=5,
        cache_file=proc.cache_file,
        imap_folder=proc.imap_folder,
        use_email_cache=False,
        prefer_ssl=proc.imap_use_ssl,
    )
    try:
        ok = an.connect()
    except Exception as e:
        logging.exception("IMAP test")
        raise HTTPException(status_code=400, detail=f"Erreur IMAP: {e!s}") from e
    finally:
        an.disconnect()
    if not ok:
        raise HTTPException(status_code=400, detail="Connexion IMAP refusée")
    return {"ok": True, "message": "Connexion IMAP réussie"}

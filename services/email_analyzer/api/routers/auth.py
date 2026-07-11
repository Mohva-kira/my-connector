"""Inscription, connexion, profil."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from email_analyzer.auth_jwt import create_access_token, hash_password, verify_password
from email_analyzer.db.models import Membership, MembershipRole, Tenant, TenantStatus, User
from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import (
    authenticate_bearer,
    saas_enabled,
    trigger_login_auto_sync,
    unique_tenant_slug,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterBody(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    organization_name: str = Field(..., min_length=1, max_length=200)


class LoginBody(BaseModel):
    email: EmailStr
    password: str
    tenant_id: Optional[uuid.UUID] = None


class SwitchTenantBody(BaseModel):
    tenant_id: uuid.UUID


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    # Jobs Fast-Track déclenchés automatiquement par ce login (voir
    # saas_logic.trigger_login_auto_sync) — vide hors du endpoint login (register/
    # switch) ou si le throttle 1x/24h est déjà consommé pour ce tenant.
    auto_sync_jobs: List[Dict[str, str]] = []


class TenantInfo(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    role: str
    status: str


class MeResponse(BaseModel):
    email: str
    active_tenant_id: uuid.UUID
    tenants: List[TenantInfo]
    # Référence "depuis votre dernière visite" pour le Brief (GET /api/brief) —
    # voir User.previous_login_at (db/models.py). None pour un tout premier
    # login (pas de session précédente à comparer).
    previous_login_at: Optional[datetime] = None


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé (DATABASE_URL manquant)")


@router.post("/register", response_model=TokenResponse)
def register(body: RegisterBody, db: Session = Depends(get_db)) -> TokenResponse:
    _require_saas()
    if db.query(User).filter(User.email == body.email.lower().strip()).first():
        raise HTTPException(status_code=400, detail="Cet email est déjà inscrit")

    user = User(
        email=body.email.lower().strip(),
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    db.flush()

    slug = unique_tenant_slug(db, body.organization_name)
    tenant = Tenant(
        name=body.organization_name.strip(),
        slug=slug,
        status=TenantStatus.trial.value,
    )
    db.add(tenant)
    db.flush()

    db.add(
        Membership(
            user_id=user.id,
            tenant_id=tenant.id,
            role=MembershipRole.owner.value,
        )
    )
    db.commit()
    db.refresh(user)
    db.refresh(tenant)

    token = create_access_token(user_id=user.id, tenant_id=tenant.id, email=user.email)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginBody, db: Session = Depends(get_db)) -> TokenResponse:
    _require_saas()
    user = db.query(User).filter(User.email == body.email.lower().strip()).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    if body.tenant_id is not None:
        m = (
            db.query(Membership)
            .filter(
                Membership.user_id == user.id,
                Membership.tenant_id == body.tenant_id,
            )
            .first()
        )
        if not m:
            raise HTTPException(status_code=403, detail="Organisation non accessible")
        tid = body.tenant_id
    else:
        first = (
            db.query(Membership, Tenant)
            .join(Tenant, Membership.tenant_id == Tenant.id)
            .filter(Membership.user_id == user.id)
            .order_by(Tenant.created_at.desc())
            .first()
        )
        if not first:
            raise HTTPException(status_code=400, detail="Aucune organisation")
        _, tenant = first
        tid = tenant.id

    # Décale la fenêtre "depuis votre dernière visite" (GET /api/brief) : la
    # valeur qui servait de référence pour la session qui vient de se terminer
    # devient previous_login_at pour la nouvelle session ; last_login_at avance
    # à maintenant. Voir User.previous_login_at (db/models.py).
    user.previous_login_at = user.last_login_at
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    auto_sync_jobs = trigger_login_auto_sync(db, tid)
    token = create_access_token(user_id=user.id, tenant_id=tid, email=user.email)
    return TokenResponse(access_token=token, auto_sync_jobs=auto_sync_jobs)


@router.post("/switch", response_model=TokenResponse)
def switch_tenant(
    body: SwitchTenantBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> TokenResponse:
    _require_saas()
    user, _, _ = authenticate_bearer(db, authorization)
    m = (
        db.query(Membership)
        .filter(
            Membership.user_id == user.id,
            Membership.tenant_id == body.tenant_id,
        )
        .first()
    )
    if not m:
        raise HTTPException(status_code=403, detail="Organisation non accessible")
    token = create_access_token(user_id=user.id, tenant_id=body.tenant_id, email=user.email)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=MeResponse)
def me(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> MeResponse:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    rows = (
        db.query(Membership, Tenant)
        .join(Tenant, Membership.tenant_id == Tenant.id)
        .filter(Membership.user_id == user.id)
        .all()
    )
    tenants: List[TenantInfo] = []
    for m, t in rows:
        tenants.append(
            TenantInfo(
                id=t.id,
                name=t.name,
                slug=t.slug,
                role=m.role,
                status=t.status,
            )
        )
    return MeResponse(
        email=user.email,
        active_tenant_id=tenant.id,
        tenants=tenants,
        previous_login_at=user.previous_login_at,
    )

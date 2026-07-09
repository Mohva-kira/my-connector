"""Gmail OAuth2 connection management endpoints."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from email_analyzer.auth_jwt import create_access_token, decode_token
from email_analyzer.db.models import OAuthConnection
from email_analyzer.db.session import get_db
from email_analyzer.encryption import decrypt_secret, encrypt_secret
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled
from jose import JWTError, jwt

router = APIRouter(prefix="/api/oauth", tags=["oauth"])

logger = logging.getLogger(__name__)

_STATE_ALGORITHM = "HS256"
_STATE_EXPIRY_SECONDS = 600  # 10 minutes


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé (DATABASE_URL manquant)")


def _jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("JWT_SECRET is required")
    return secret


def _gmail_redirect_uri() -> str:
    base = (os.environ.get("APP_PUBLIC_URL") or "http://127.0.0.1:8000").rstrip("/")
    return f"{base}/api/oauth/gmail/callback"


def _outlook_redirect_uri() -> str:
    base = (os.environ.get("APP_PUBLIC_URL") or "http://127.0.0.1:8000").rstrip("/")
    return f"{base}/api/oauth/outlook/callback"


def _frontend_origin() -> str:
    return (os.environ.get("FRONTEND_ORIGIN") or "http://localhost:5173").rstrip("/")


def _upsert_oauth_connection(
    db: Session,
    tenant_id: uuid.UUID,
    provider: str,
    email: str,
    tokens: Dict[str, Any],
) -> None:
    """Crée ou met à jour une `OAuthConnection` (upsert par tenant/provider/
    email) — partagé par gmail_callback et outlook_callback, seule la
    provenance des `tokens` diffère."""
    access_enc = encrypt_secret(tokens["access_token"])
    refresh_enc = encrypt_secret(tokens["refresh_token"]) if tokens.get("refresh_token") else None
    expiry: Optional[datetime] = tokens.get("expiry")

    existing = (
        db.query(OAuthConnection)
        .filter(
            OAuthConnection.tenant_id == tenant_id,
            OAuthConnection.provider == provider,
            OAuthConnection.email == email,
        )
        .first()
    )

    if existing:
        existing.access_token_encrypted = access_enc
        if refresh_enc:
            existing.refresh_token_encrypted = refresh_enc
        existing.token_expiry = expiry
        existing.scopes = tokens.get("scopes")
        existing.updated_at = datetime.now(timezone.utc)
    else:
        db.add(
            OAuthConnection(
                tenant_id=tenant_id,
                provider=provider,
                email=email,
                access_token_encrypted=access_enc,
                refresh_token_encrypted=refresh_enc,
                token_expiry=expiry,
                scopes=tokens.get("scopes"),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )

    db.commit()


def _create_state_token(tenant_id: uuid.UUID) -> str:
    payload = {
        "tenant_id": str(tenant_id),
        "exp": int(datetime.now(timezone.utc).timestamp()) + _STATE_EXPIRY_SECONDS,
        "purpose": "oauth_state",
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=_STATE_ALGORITHM)


def _decode_state_token(state: str) -> uuid.UUID:
    try:
        payload = jwt.decode(state, _jwt_secret(), algorithms=[_STATE_ALGORITHM])
        if payload.get("purpose") != "oauth_state":
            raise HTTPException(status_code=400, detail="State token invalide")
        return uuid.UUID(payload["tenant_id"])
    except JWTError as exc:
        raise HTTPException(status_code=400, detail="State expiré ou invalide") from exc


# ── Response schemas ──────────────────────────────────────────────────────────

class AuthorizeResponse(BaseModel):
    url: str


class OAuthConnectionOut(BaseModel):
    id: uuid.UUID
    provider: str
    email: str
    scopes: Optional[str]
    created_at: datetime
    updated_at: datetime


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/gmail/authorize", response_model=AuthorizeResponse)
def gmail_authorize(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> AuthorizeResponse:
    """Return the Google OAuth2 authorization URL for the current tenant."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    try:
        from email_analyzer.gmail_oauth import build_authorization_url
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    state = _create_state_token(tenant.id)
    redirect_uri = _gmail_redirect_uri()
    try:
        url = build_authorization_url(redirect_uri=redirect_uri, state=state)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AuthorizeResponse(url=url)


@router.get("/gmail/callback")
def gmail_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """
    Google redirects here after user grants permission.
    Exchanges the code for tokens and stores them encrypted in the DB.
    Redirects to the frontend with ?oauth=success or ?oauth=error.
    """
    frontend = _frontend_origin()
    error_redirect = RedirectResponse(url=f"{frontend}/settings?oauth=error")

    if error:
        logger.warning("Gmail OAuth callback error: %s", error)
        return error_redirect

    if not code or not state:
        return error_redirect

    try:
        tenant_id = _decode_state_token(state)
    except HTTPException:
        return error_redirect

    try:
        from email_analyzer.gmail_oauth import exchange_code_for_tokens, get_connected_email
    except ImportError:
        return error_redirect

    redirect_uri = _gmail_redirect_uri()
    try:
        tokens = exchange_code_for_tokens(code=code, redirect_uri=redirect_uri)
    except Exception:
        logger.exception("Gmail token exchange failed")
        return error_redirect

    try:
        gmail_email = get_connected_email(tokens["access_token"])
    except Exception:
        logger.exception("Could not retrieve Gmail user email")
        return error_redirect

    _upsert_oauth_connection(db, tenant_id, "gmail", gmail_email, tokens)
    logger.info("Gmail OAuth connection stored for tenant=%s email=%s", tenant_id, gmail_email)
    return RedirectResponse(url=f"{frontend}/settings?oauth=success&provider=gmail")


@router.get("/outlook/authorize", response_model=AuthorizeResponse)
def outlook_authorize(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> AuthorizeResponse:
    """Return the Microsoft identity platform authorization URL for the current tenant."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    try:
        from email_analyzer.outlook_oauth import build_authorization_url
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    state = _create_state_token(tenant.id)
    redirect_uri = _outlook_redirect_uri()
    try:
        url = build_authorization_url(redirect_uri=redirect_uri, state=state)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AuthorizeResponse(url=url)


@router.get("/outlook/callback")
def outlook_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """
    Microsoft redirects here after user grants permission. Same shape as
    gmail_callback (exchange code, store encrypted tokens, redirect to
    frontend with ?oauth=success|error&provider=outlook).
    """
    frontend = _frontend_origin()
    error_redirect = RedirectResponse(url=f"{frontend}/settings?oauth=error&provider=outlook")

    if error:
        logger.warning("Outlook OAuth callback error: %s", error)
        return error_redirect

    if not code or not state:
        return error_redirect

    try:
        tenant_id = _decode_state_token(state)
    except HTTPException:
        return error_redirect

    try:
        from email_analyzer.outlook_oauth import exchange_code_for_tokens, get_connected_email
    except ImportError:
        return error_redirect

    redirect_uri = _outlook_redirect_uri()
    try:
        tokens = exchange_code_for_tokens(code=code, redirect_uri=redirect_uri)
    except Exception:
        logger.exception("Outlook token exchange failed")
        return error_redirect

    try:
        outlook_email = get_connected_email(tokens["access_token"])
    except Exception:
        logger.exception("Could not retrieve Outlook user email")
        return error_redirect

    _upsert_oauth_connection(db, tenant_id, "outlook", outlook_email, tokens)
    logger.info("Outlook OAuth connection stored for tenant=%s email=%s", tenant_id, outlook_email)
    return RedirectResponse(url=f"{frontend}/settings?oauth=success&provider=outlook")


@router.get("/connections", response_model=List[OAuthConnectionOut])
def list_connections(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> List[OAuthConnectionOut]:
    """List all OAuth connections for the current tenant."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    rows = (
        db.query(OAuthConnection)
        .filter(OAuthConnection.tenant_id == tenant.id)
        .order_by(OAuthConnection.created_at)
        .all()
    )
    return [
        OAuthConnectionOut(
            id=r.id,
            provider=r.provider,
            email=r.email,
            scopes=r.scopes,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


@router.delete("/connections/{connection_id}", status_code=204)
def delete_connection(
    connection_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> None:
    """Revoke and delete a stored OAuth connection."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    conn = (
        db.query(OAuthConnection)
        .filter(
            OAuthConnection.id == connection_id,
            OAuthConnection.tenant_id == tenant.id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(status_code=404, detail="Connexion introuvable")

    # Best-effort token revocation (do not fail if it errors)
    if conn.provider == "gmail":
        try:
            access_token = decrypt_secret(conn.access_token_encrypted)
            import httpx
            httpx.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": access_token},
                timeout=5,
            )
        except Exception:
            logger.warning("Could not revoke Gmail token for connection %s", connection_id)

    db.delete(conn)
    db.commit()

"""Outlook OAuth2 integration (Microsoft identity platform + Graph API).

Miroir de `gmail_oauth.py`, mêmes clés de sortie normalisées
(`id/subject/from/to/date/body/normalized_text`) pour rester compatible avec
le pipeline de matching/scoring existant (`EmailProjectAnalyzer.
search_project_emails_from_list`). Implémenté en appels HTTP directs via
`httpx` (déjà une dépendance du projet) plutôt qu'avec `msal` : les
endpoints Microsoft Identity Platform v2.0 et Microsoft Graph sont de
l'OAuth2/OIDC et du REST standard, une dépendance supplémentaire n'est pas
justifiée pour ça.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from .gmail_oauth import _strip_html

logger = logging.getLogger(__name__)

OUTLOOK_SCOPES = [
    "openid",
    "profile",
    "offline_access",
    "https://graph.microsoft.com/User.Read",
    "https://graph.microsoft.com/Mail.Read",
]

_AUTHORIZE_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _outlook_client_config() -> Dict[str, str]:
    client_id = os.environ.get("OUTLOOK_CLIENT_ID", "").strip()
    client_secret = os.environ.get("OUTLOOK_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise ValueError(
            "OUTLOOK_CLIENT_ID and OUTLOOK_CLIENT_SECRET must be set to use Outlook OAuth."
        )
    return {"client_id": client_id, "client_secret": client_secret}


def build_authorization_url(redirect_uri: str, state: str) -> str:
    """Return the Microsoft identity platform authorization URL."""
    cfg = _outlook_client_config()
    params = {
        "client_id": cfg["client_id"],
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": " ".join(OUTLOOK_SCOPES),
        "state": state,
    }
    return f"{_AUTHORIZE_URL}?{urlencode(params)}"


def _token_expiry_from_response(data: Dict[str, Any]) -> Optional[datetime]:
    expires_in = data.get("expires_in")
    if expires_in is None:
        return None
    try:
        return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    except (TypeError, ValueError):
        return None


def exchange_code_for_tokens(code: str, redirect_uri: str) -> Dict[str, Any]:
    """
    Exchange the OAuth2 authorization code for access + refresh tokens.
    Returns dict with keys: access_token, refresh_token, expiry (datetime|None), scopes (str).
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required: pip install httpx") from exc

    cfg = _outlook_client_config()
    resp = httpx.post(
        _TOKEN_URL,
        data={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(OUTLOOK_SCOPES),
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token"),
        "expiry": _token_expiry_from_response(data),
        "scopes": data.get("scope", ""),
    }


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """
    Use the stored refresh token to get a new access token.
    Returns dict with keys: access_token, expiry (datetime|None), refresh_token
    (str|None — Microsoft may rotate the refresh token; persist it when present).
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required: pip install httpx") from exc

    cfg = _outlook_client_config()
    resp = httpx.post(
        _TOKEN_URL,
        data={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "scope": " ".join(OUTLOOK_SCOPES),
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "access_token": data["access_token"],
        "expiry": _token_expiry_from_response(data),
        "refresh_token": data.get("refresh_token"),
    }


def get_connected_email(access_token: str) -> str:
    """Return the mailbox address for this access token via Graph `/me`."""
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required: pip install httpx") from exc

    resp = httpx.get(
        f"{_GRAPH_BASE}/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    email = (data.get("mail") or data.get("userPrincipalName") or "").lower().strip()
    if not email:
        raise ValueError("Could not retrieve email from Microsoft Graph /me endpoint.")
    return email


def _normalize_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw Graph API message into the project's standard email dict."""
    from_field = ((msg.get("from") or {}).get("emailAddress") or {}).get("address", "")
    to_field = ", ".join(
        (r.get("emailAddress") or {}).get("address", "")
        for r in (msg.get("toRecipients") or [])
    )
    subject = msg.get("subject", "") or ""
    body_block = msg.get("body") or {}
    raw_body = body_block.get("content", "") or ""
    body = _strip_html(raw_body) if body_block.get("contentType") == "html" else raw_body
    normalized_text = f"{subject} {body}".lower()

    return {
        "id": msg.get("id", ""),
        "subject": subject,
        "from": from_field,
        "to": to_field,
        "date": msg.get("receivedDateTime", ""),
        "body": body,
        "normalized_text": normalized_text,
    }


def _get_valid_access_token(
    access_token_encrypted: str,
    refresh_token_encrypted: Optional[str],
    token_expiry: Optional[datetime],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Decrypt tokens, refresh if expired.
    Returns (access_token, refresh_result_or_None).
    """
    from email_analyzer.encryption import decrypt_secret

    access_token = decrypt_secret(access_token_encrypted)
    refresh_token = decrypt_secret(refresh_token_encrypted) if refresh_token_encrypted else None

    now = datetime.now(timezone.utc)
    expiry_aware = token_expiry.replace(tzinfo=timezone.utc) if token_expiry and token_expiry.tzinfo is None else token_expiry

    needs_refresh = (
        refresh_token is not None
        and expiry_aware is not None
        and expiry_aware <= now
    )
    if not needs_refresh:
        return access_token, None

    logger.info("Outlook access token expired, refreshing.")
    refreshed = refresh_access_token(refresh_token)
    return refreshed["access_token"], refreshed


def build_outlook_filter(days_back: int = 30) -> str:
    """OData `$filter` bornant la recherche à une fenêtre glissante (jours),
    équivalent du `newer_than:Nd` de `gmail_oauth.build_gmail_query`."""
    since = datetime.now(timezone.utc) - timedelta(days=max(1, days_back))
    return f"receivedDateTime ge {since.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def fetch_emails(
    access_token_encrypted: str,
    refresh_token_encrypted: Optional[str],
    token_expiry: Optional[datetime],
    query: str = "",
    max_results: int = 50,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Fetch emails from Microsoft Graph (`$filter` = `query`, one page — no
    pagination, same known limitation as `gmail_oauth.fetch_emails`).

    Returns:
        (emails, token_refresh) où token_refresh est non-None quand le token
        a été rafraîchi et doit être persisté par l'appelant.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required: pip install httpx") from exc

    access_token, token_refresh = _get_valid_access_token(
        access_token_encrypted, refresh_token_encrypted, token_expiry
    )

    params: Dict[str, Any] = {"$top": max_results, "$orderby": "receivedDateTime desc"}
    if query:
        params["$filter"] = query

    resp = httpx.get(
        f"{_GRAPH_BASE}/me/messages",
        params=params,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    messages = resp.json().get("value", [])
    emails = [_normalize_message(m) for m in messages]
    return emails, token_refresh

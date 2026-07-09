"""Gmail OAuth2 integration: authorization flow and email fetching via Gmail API."""

from __future__ import annotations

import base64
import logging
import os
import re
from datetime import datetime, timezone
from html import unescape
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]

_REQUIRED_LIBS_MESSAGE = (
    "Gmail OAuth requires: google-auth-oauthlib, google-auth-httplib2, google-api-python-client. "
    "Run: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
)


def _gmail_client_config() -> Dict[str, Any]:
    client_id = os.environ.get("GMAIL_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise ValueError(
            "GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET must be set to use Gmail OAuth."
        )
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


def build_authorization_url(redirect_uri: str, state: str) -> str:
    """Return Google OAuth2 authorization URL. State is a short-lived signed token (CSRF)."""
    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError as exc:
        raise ImportError(_REQUIRED_LIBS_MESSAGE) from exc

    flow = Flow.from_client_config(
        _gmail_client_config(), scopes=GMAIL_SCOPES, redirect_uri=redirect_uri
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=state,
        prompt="consent",
    )
    return auth_url


def exchange_code_for_tokens(code: str, redirect_uri: str) -> Dict[str, Any]:
    """
    Exchange the OAuth2 authorization code for access + refresh tokens.
    Returns dict with keys: access_token, refresh_token, expiry (datetime|None), scopes (str).
    """
    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError as exc:
        raise ImportError(_REQUIRED_LIBS_MESSAGE) from exc

    flow = Flow.from_client_config(
        _gmail_client_config(), scopes=GMAIL_SCOPES, redirect_uri=redirect_uri
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    return {
        "access_token": creds.token,
        "refresh_token": creds.refresh_token,
        "expiry": creds.expiry,
        "scopes": " ".join(sorted(creds.scopes or [])),
    }


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """
    Use the stored refresh token to get a new access token.
    Returns dict with keys: access_token, expiry (datetime|None).
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise ImportError(_REQUIRED_LIBS_MESSAGE) from exc

    cfg = _gmail_client_config()["web"]
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=cfg["token_uri"],
        client_id=cfg["client_id"],
        client_secret=cfg["client_secret"],
        scopes=GMAIL_SCOPES,
    )
    creds.refresh(Request())
    return {
        "access_token": creds.token,
        "expiry": creds.expiry,
    }


def get_connected_email(access_token: str) -> str:
    """Return the Gmail address for this access token via the userinfo endpoint."""
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required: pip install httpx") from exc

    resp = httpx.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    email = data.get("email", "")
    if not email:
        raise ValueError("Could not retrieve email from Google userinfo endpoint.")
    return email.lower().strip()


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    without_blocks = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    with_breaks = re.sub(r"(?i)<br\s*/?>", "\n", without_blocks)
    without_tags = re.sub(r"(?i)<[^>]+>", " ", with_breaks)
    return " ".join(unescape(without_tags).split())


def _decode_part_body(part: Dict[str, Any]) -> str:
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body_from_payload(payload: Dict[str, Any]) -> str:
    mime_type = payload.get("mimeType", "")
    parts = payload.get("parts", [])

    if mime_type == "text/plain":
        return _decode_part_body(payload)
    if mime_type == "text/html":
        return _strip_html(_decode_part_body(payload))

    plain, html = "", ""
    for part in parts:
        pt = part.get("mimeType", "")
        sub_parts = part.get("parts", [])
        if pt == "text/plain" and not plain:
            plain = _decode_part_body(part)
        elif pt == "text/html" and not html:
            html = _strip_html(_decode_part_body(part))
        elif pt.startswith("multipart/") and sub_parts:
            nested = _extract_body_from_payload(part)
            if nested and not plain:
                plain = nested

    return plain or html


def _header_value(headers: List[Dict[str, str]], name: str) -> str:
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _normalize_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw Gmail API message into the project's standard email dict."""
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    subject = _header_value(headers, "Subject")
    from_ = _header_value(headers, "From")
    to_ = _header_value(headers, "To")
    date = _header_value(headers, "Date")
    body = _extract_body_from_payload(payload)

    normalized_text = f"{subject} {body}".lower()

    return {
        "id": msg.get("id", ""),
        "subject": subject,
        "from": from_,
        "to": to_,
        "date": date,
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
    refresh_result contains new access_token + expiry when a refresh happened.
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

    logger.info("Gmail access token expired, refreshing.")
    refreshed = refresh_access_token(refresh_token)
    return refreshed["access_token"], refreshed


def fetch_emails(
    access_token_encrypted: str,
    refresh_token_encrypted: Optional[str],
    token_expiry: Optional[datetime],
    query: str = "",
    max_results: int = 50,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Fetch emails from Gmail API matching the query.

    Returns:
        (emails, token_refresh) where token_refresh is non-None when the access
        token was refreshed and the caller must persist the new values to the DB.
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(_REQUIRED_LIBS_MESSAGE) from exc

    access_token, token_refresh = _get_valid_access_token(
        access_token_encrypted, refresh_token_encrypted, token_expiry
    )

    cfg = _gmail_client_config()["web"]
    creds = Credentials(
        token=access_token,
        refresh_token=None,
        token_uri=cfg["token_uri"],
        client_id=cfg["client_id"],
        client_secret=cfg["client_secret"],
        scopes=GMAIL_SCOPES,
    )

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    gmail_users = service.users()

    list_kwargs: Dict[str, Any] = {
        "userId": "me",
        "maxResults": max_results,
    }
    if query:
        list_kwargs["q"] = query

    response = gmail_users.messages().list(**list_kwargs).execute()
    message_refs = response.get("messages", [])

    emails: List[Dict[str, Any]] = []
    for ref in message_refs:
        try:
            msg = (
                gmail_users.messages()
                .get(userId="me", id=ref["id"], format="full")
                .execute()
            )
            emails.append(_normalize_message(msg))
        except Exception:
            logger.exception("Failed to fetch Gmail message id=%s", ref.get("id"))

    return emails, token_refresh


def build_gmail_query(project_filter: str, days_back: int = 30) -> str:
    """Build a Gmail search query string for a project filter and time window."""
    parts: List[str] = []
    if project_filter.strip():
        escaped = project_filter.strip().replace('"', "")
        parts.append(f'"{escaped}"')
    if days_back > 0:
        parts.append(f"newer_than:{days_back}d")
    return " ".join(parts)

"""Client CinetPay : initialisation paiement, vérification HMAC, check transaction."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any, Dict, Optional

import httpx

CINETPAY_PAYMENT_URL = "https://api-checkout.cinetpay.com/v2/payment"
CINETPAY_CHECK_URL = "https://api-checkout.cinetpay.com/v2/payment/check"


def _cfg() -> tuple[str, str, str]:
    apikey = (os.environ.get("CINETPAY_API_KEY") or "").strip()
    site_id = (os.environ.get("CINETPAY_SITE_ID") or "").strip()
    secret = (os.environ.get("CINETPAY_SECRET_KEY") or "").strip()
    return apikey, site_id, secret


def public_notify_url() -> str:
    base = (os.environ.get("APP_PUBLIC_URL") or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("APP_PUBLIC_URL requis pour les URLs CinetPay (notify/return)")
    return f"{base}/api/billing/cinetpay/notify"


def public_return_url() -> str:
    base = (os.environ.get("APP_PUBLIC_URL") or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("APP_PUBLIC_URL requis")
    # Hash pour SPA Vite sans réécriture serveur
    return f"{base}/#/billing/return"


def round_amount_xof(amount: int) -> int:
    """Montant XOF : multiple de 5."""
    if amount <= 0:
        return 5
    return max(5, (amount // 5) * 5)


def initiate_payment(
    *,
    transaction_id: str,
    amount: int,
    currency: str,
    description: str,
    metadata: str,
    customer: Dict[str, str],
) -> Dict[str, Any]:
    apikey, site_id, _ = _cfg()
    if not apikey or not site_id:
        raise RuntimeError("CINETPAY_API_KEY et CINETPAY_SITE_ID requis")

    payload = {
        "apikey": apikey,
        "site_id": site_id,
        "transaction_id": transaction_id,
        "amount": amount,
        "currency": currency,
        "description": _sanitize_description(description),
        "notify_url": public_notify_url(),
        "return_url": public_return_url(),
        "channels": "ALL",
        "metadata": metadata,
        "lang": "FR",
        **customer,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "email-analyzer-saas/1.0",
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(CINETPAY_PAYMENT_URL, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()


def _sanitize_description(s: str) -> str:
    return re.sub(r"[#/,$_&]", " ", s)[:200] or "Paiement"


def check_transaction(transaction_id: str) -> Dict[str, Any]:
    apikey, site_id, _ = _cfg()
    if not apikey or not site_id:
        raise RuntimeError("CINETPAY_API_KEY et CINETPAY_SITE_ID requis")
    body = {
        "apikey": apikey,
        "site_id": site_id,
        "transaction_id": transaction_id,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "email-analyzer-saas/1.0",
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(CINETPAY_CHECK_URL, json=body, headers=headers)
        r.raise_for_status()
        return r.json()


def build_hmac_form_string(form: Dict[str, str]) -> str:
    keys = [
        "cpm_site_id",
        "cpm_trans_id",
        "cpm_trans_date",
        "cpm_amount",
        "cpm_currency",
        "signature",
        "payment_method",
        "cel_phone_num",
        "cpm_phone_prefixe",
        "cpm_language",
        "cpm_version",
        "cpm_payment_config",
        "cpm_page_action",
        "cpm_custom",
        "cpm_designation",
        "cpm_error_message",
    ]
    return "".join(str(form.get(k) or "") for k in keys)


def verify_notification_hmac(x_token: Optional[str], form: Dict[str, str]) -> bool:
    _, _, secret = _cfg()
    if not secret or not x_token:
        return False
    data = build_hmac_form_string(form)
    expected = hmac.new(
        secret.encode("utf-8"),
        data.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, x_token.strip())


def extract_payment_url(api_response: Dict[str, Any]) -> Optional[str]:
    if str(api_response.get("code")) != "201":
        return None
    data = api_response.get("data") or {}
    return data.get("payment_url")

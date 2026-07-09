"""Plans CinetPay, checkout, webhook IPN."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from email_analyzer.billing_cinetpay import (
    check_transaction,
    extract_payment_url,
    initiate_payment,
    round_amount_xof,
    verify_notification_hmac,
)
from email_analyzer.db.models import BillingEvent, Plan, Subscription, Tenant, TenantStatus
from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, period_end_for_plan, saas_enabled

router = APIRouter(tags=["billing"])


class PlanOut(BaseModel):
    id: uuid.UUID
    slug: str
    name: str
    price_amount: int
    currency: str
    interval: str
    quota_analyses_per_month: Optional[int]


class CheckoutBody(BaseModel):
    plan_slug: str = Field(..., min_length=1)


class CheckoutResponse(BaseModel):
    payment_url: str
    transaction_id: str


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


@router.get("/api/billing/plans", response_model=List[PlanOut])
def list_plans(db: Session = Depends(get_db)) -> List[PlanOut]:
    _require_saas()
    plans = db.query(Plan).order_by(Plan.slug).all()
    return [
        PlanOut(
            id=p.id,
            slug=p.slug,
            name=p.name,
            price_amount=p.price_amount,
            currency=p.currency,
            interval=p.interval,
            quota_analyses_per_month=p.quota_analyses_per_month,
        )
        for p in plans
    ]


@router.post("/api/billing/checkout", response_model=CheckoutResponse)
def checkout(
    body: CheckoutBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> CheckoutResponse:
    import os

    _require_saas()
    user, tenant, m = authenticate_bearer(db, authorization)
    if m.role != "owner":
        raise HTTPException(status_code=403, detail="Seul le propriétaire peut payer")

    plan = db.query(Plan).filter(Plan.slug == body.plan_slug.strip()).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan inconnu")

    amount = round_amount_xof(plan.price_amount)
    tx_id = uuid.uuid4().hex[:40]

    meta = json.dumps(
        {"tenant_id": str(tenant.id), "plan_id": str(plan.id)},
        separators=(",", ":"),
    )

    phone = (os.environ.get("CINETPAY_DEFAULT_CUSTOMER_PHONE") or "+2250500000000").strip()
    country = (os.environ.get("CINETPAY_DEFAULT_CUSTOMER_COUNTRY") or "CI").strip()[:2]
    customer = {
        "customer_id": str(user.id),
        "customer_name": (tenant.name[:64] or "Client"),
        "customer_surname": "Org",
        "customer_email": user.email,
        "customer_phone_number": phone,
        "customer_address": "BP",
        "customer_city": "Abidjan",
        "customer_country": country,
        "customer_state": country,
        "customer_zip_code": "00225",
    }

    db.add(
        BillingEvent(
            tenant_id=tenant.id,
            transaction_id=tx_id,
            amount=amount,
            currency=plan.currency,
            status="pending",
            raw_payload={"plan_slug": plan.slug},
        )
    )
    db.commit()

    try:
        api_resp = initiate_payment(
            transaction_id=tx_id,
            amount=amount,
            currency=plan.currency,
            description=f"Abonnement {plan.name}",
            metadata=meta,
            customer=customer,
        )
    except Exception as e:
        logging.exception("CinetPay initiate")
        raise HTTPException(status_code=502, detail=f"Paiement indisponible: {e!s}") from e

    url = extract_payment_url(api_resp)
    if not url:
        raise HTTPException(
            status_code=502,
            detail=api_resp.get("message") or api_resp.get("description") or "Réponse CinetPay invalide",
        )
    return CheckoutResponse(payment_url=url, transaction_id=tx_id)


@router.get("/api/billing/cinetpay/notify")
def cinetpay_notify_ping() -> Response:
    """Ping disponibilité (CinetPay vérifie en GET)."""
    return Response(status_code=200)


@router.post("/api/billing/cinetpay/notify")
async def cinetpay_notify(request: Request, db: Session = Depends(get_db)) -> Response:
    """IPN CinetPay : application/x-www-form-urlencoded."""
    _require_saas()
    try:
        form = await request.form()
    except Exception:
        return Response(status_code=400)

    form_dict: Dict[str, str] = {str(k): str(v) for k, v in form.items()}
    x_token = request.headers.get("x-token") or request.headers.get("X-Token")

    if not verify_notification_hmac(x_token, form_dict):
        logging.warning("CinetPay IPN HMAC invalide")
        return Response(status_code=403)

    trans_id = form_dict.get("cpm_trans_id") or ""
    if not trans_id:
        return Response(status_code=400)

    existing_done = (
        db.query(BillingEvent)
        .filter(BillingEvent.transaction_id == trans_id, BillingEvent.status == "accepted")
        .first()
    )
    if existing_done:
        return Response(status_code=200)

    try:
        chk = check_transaction(trans_id)
    except Exception as e:
        logging.exception("CinetPay check")
        return Response(status_code=502)

    raw_payload: Dict[str, Any] = dict(chk)
    data = chk.get("data") if isinstance(chk.get("data"), dict) else {}
    status = data.get("status") if isinstance(data, dict) else None
    code_ok = str(chk.get("code")) == "00"

    be = db.query(BillingEvent).filter(BillingEvent.transaction_id == trans_id).first()
    if be:
        be.raw_payload = {**(be.raw_payload or {}), "check": raw_payload}

    if code_ok and status == "ACCEPTED":
        meta_str = form_dict.get("cpm_custom") or ""
        try:
            meta = json.loads(meta_str) if meta_str.strip().startswith("{") else {}
        except json.JSONDecodeError:
            meta = {}
        tid_s = meta.get("tenant_id")
        pid_s = meta.get("plan_id")
        tenant = None
        plan = None
        if tid_s and pid_s:
            try:
                tenant_uuid = uuid.UUID(str(tid_s))
                plan_uuid = uuid.UUID(str(pid_s))
                tenant = db.query(Tenant).filter(Tenant.id == tenant_uuid).first()
                plan = db.query(Plan).filter(Plan.id == plan_uuid).first()
            except ValueError:
                tenant = None
                plan = None
        if (not tenant or not plan) and be:
            tenant = db.query(Tenant).filter(Tenant.id == be.tenant_id).first()
            slug = (be.raw_payload or {}).get("plan_slug") if isinstance(be.raw_payload, dict) else None
            if slug:
                plan = db.query(Plan).filter(Plan.slug == slug).first()
        if not tenant or not plan:
            logging.error("IPN sans tenant/plan résoluble (metadata=%s)", meta_str[:120])
            db.commit()
            return Response(status_code=200)

        tenant.status = TenantStatus.active.value

        for s in (
            db.query(Subscription)
            .filter(
                Subscription.tenant_id == tenant.id,
                Subscription.status == "active",
            )
            .all()
        ):
            s.status = "cancelled"

        sub = Subscription(
            tenant_id=tenant.id,
            plan_id=plan.id,
            status="active",
            current_period_end=period_end_for_plan(plan.interval),
            last_transaction_id=trans_id,
        )
        db.add(sub)

        if be:
            be.status = "accepted"
            be.amount = int(data.get("amount") or be.amount)
            be.currency = str(data.get("currency") or be.currency)
        else:
            db.add(
                BillingEvent(
                    tenant_id=tenant.id,
                    transaction_id=trans_id,
                    amount=int(float(str(data.get("amount") or "0"))),
                    currency=str(data.get("currency") or plan.currency),
                    status="accepted",
                    raw_payload={"check": raw_payload, "form": form_dict},
                )
            )

        db.commit()
        return Response(status_code=200)

    if status == "REFUSED" and be:
        be.status = "failed"
    db.commit()
    return Response(status_code=200)

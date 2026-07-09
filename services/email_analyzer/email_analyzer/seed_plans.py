"""Graine des plans par défaut (idempotent)."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from email_analyzer.db.models import Plan


def seed_plans_if_empty(db: Session) -> None:
    if db.query(Plan).first() is not None:
        return
    db.add(
        Plan(
            id=uuid.uuid4(),
            name="Pro Mensuel",
            slug="pro-month",
            price_amount=5000,
            currency="XOF",
            interval="month",
            quota_analyses_per_month=500,
        )
    )
    db.add(
        Plan(
            id=uuid.uuid4(),
            name="Pro Annuel",
            slug="pro-year",
            price_amount=50000,
            currency="XOF",
            interval="year",
            quota_analyses_per_month=6000,
        )
    )
    db.commit()

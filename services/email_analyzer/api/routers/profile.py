"""Contexte Personnel — profil et habitudes (domaine #1, "la base de
tout"). Contrôleur HTTP fin : toute la logique vit dans ProfileService."""

from __future__ import annotations

import uuid
from datetime import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled
from email_analyzer.services.profile_service import ProfileService

router = APIRouter(prefix="/api/profile", tags=["profile"])


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class FamilyMember(BaseModel):
    name: str
    relation: Optional[str] = None


class ProfileOut(BaseModel):
    display_name: Optional[str] = None
    languages: Optional[List[str]] = None
    timezone: Optional[str] = None
    location: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    family_members: Optional[List[FamilyMember]] = None
    favorite_contact_ids: Optional[List[uuid.UUID]] = None
    wake_time: Optional[time] = None
    sleep_time: Optional[time] = None
    work_start_time: Optional[time] = None
    work_end_time: Optional[time] = None
    sport_days: Optional[List[str]] = None
    focus_hours: Optional[str] = None


class ProfileUpdateBody(BaseModel):
    display_name: Optional[str] = None
    languages: Optional[List[str]] = None
    timezone: Optional[str] = None
    location: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    family_members: Optional[List[FamilyMember]] = None
    favorite_contact_ids: Optional[List[uuid.UUID]] = None
    wake_time: Optional[time] = None
    sleep_time: Optional[time] = None
    work_start_time: Optional[time] = None
    work_end_time: Optional[time] = None
    sport_days: Optional[List[str]] = None
    focus_hours: Optional[str] = None


@router.get("", response_model=ProfileOut)
def get_profile(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ProfileOut:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    profile = ProfileService(db).get_profile(tenant.id, user.id)
    if profile is None:
        return ProfileOut()
    return ProfileOut.model_validate(profile, from_attributes=True)


@router.patch("", response_model=ProfileOut)
def update_profile(
    body: ProfileUpdateBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ProfileOut:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    fields = body.model_dump(exclude_unset=True)
    profile = ProfileService(db).update_profile(tenant.id, user.id, fields)
    return ProfileOut.model_validate(profile, from_attributes=True)

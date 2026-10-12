"""Sources et consentements aux risques (I-16, vague 3) : ``backend/01`` § 5.8 et § 5.8.1."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ... import consent

router = APIRouter()


@router.get("/sources")
def list_sources() -> list[dict]:
    return consent.list_public()


class ConsentBody(BaseModel):
    risk_version: str
    accepted: bool = False


@router.post("/sources/{code}/consent", status_code=201)
def accept_consent(code: str, body: ConsentBody) -> dict:
    try:
        return consent.accept(code, risk_version=body.risk_version, accepted=body.accepted)
    except consent.SourceNotFound:
        raise HTTPException(404, "Source inconnue.") from None
    except consent.ConsentNotApplicable:
        raise HTTPException(422, "Cette source ne demande pas de consentement.") from None
    except consent.SourceUnavailable as exc:
        raise HTTPException(409, f"La source « {exc} » n'est pas encore disponible.") from None
    except consent.ConsentNotAccepted:
        raise HTTPException(422, "Accepte explicitement les risques (accepted: true).") from None
    except consent.RiskVersionStale:
        raise HTTPException(409, "Le texte des risques a changé : relis la nouvelle version.") from None


@router.delete("/sources/{code}/consent", status_code=204)
def revoke_consent(code: str) -> None:
    try:
        consent.revoke(code)
    except consent.SourceNotFound:
        raise HTTPException(404, "Source inconnue.") from None
    except consent.NoActiveConsent:
        raise HTTPException(404, "Aucun consentement actif pour cette source.") from None

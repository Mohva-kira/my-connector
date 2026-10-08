"""Conditions communes aux actions qui pilotent le navigateur (lancées depuis les routeurs)."""

from __future__ import annotations

from fastapi import HTTPException

from .. import activity, config


def require_browser_ready(quota_kinds: tuple[str, ...] = (), limit: int = 0, label: str = "") -> None:
    """Identifiants présents, plafond quotidien non atteint, aucun blocage LinkedIn récent."""
    email, password = config.linkedin_credentials()
    if not (email and password):
        raise HTTPException(409, "Identifiants LinkedIn absents : renseigne LINKEDIN_EMAIL et LINKEDIN_PASSWORD dans le fichier .env.")
    try:
        if quota_kinds:
            activity.check_quota(quota_kinds, limit, label)
        activity.guard_not_blocked()
    except activity.QuotaExceeded as exc:
        raise HTTPException(429, str(exc)) from exc
    except activity.BlockedByLinkedIn as exc:
        raise HTTPException(409, str(exc)) from exc

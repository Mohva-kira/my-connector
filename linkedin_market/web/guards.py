"""Conditions communes aux actions qui pilotent le navigateur ou consomment un plafond (lancées depuis les routeurs)."""

from __future__ import annotations

import math
from datetime import datetime, time, timedelta

from fastapi import HTTPException

from .. import activity, config


def _now() -> datetime:
    """Instant courant avec le fuseau local de la machine (remplaçable dans les tests : horloge figée)."""
    return datetime.now().astimezone()


def seconds_until_local_midnight(now: datetime | None = None) -> int:
    """Secondes jusqu'au prochain minuit, heure locale de la machine (début du jour suivant des plafonds) ; ≥ 1."""
    now = (now or _now()).astimezone()
    midnight = datetime.combine(now.date() + timedelta(days=1), time.min).astimezone()
    return max(1, math.ceil((midnight - now).total_seconds()))


def quota_exceeded(exc: activity.QuotaExceeded) -> HTTPException:
    """``429`` d'un plafond quotidien : ``detail`` inchangé + ``Retry-After`` jusqu'à minuit local (G-2)."""
    return HTTPException(429, str(exc), headers={"Retry-After": str(seconds_until_local_midnight())})


def require_credentials() -> None:
    email, password = config.linkedin_credentials()
    if not (email and password):
        raise HTTPException(
            409, "Identifiants LinkedIn absents : renseigne LINKEDIN_EMAIL et LINKEDIN_PASSWORD dans le fichier .env."
        )


def require_not_blocked(source: str = "linkedin") -> None:
    """Repos de 6 h propre à ``source`` (I-06 ; un blocage LinkedIn ne bloque pas une autre source, vague 3)."""
    try:
        activity.guard_not_blocked(source)
    except activity.BlockedByLinkedIn as exc:
        raise HTTPException(409, str(exc)) from exc


def require_browser_ready(
    quota_kinds: tuple[str, ...] = (), limit: int = 0, label: str = "", *, source: str = "linkedin"
) -> None:
    """Identifiants présents, plafond quotidien non atteint, aucun blocage récent de ``source``."""
    require_credentials()
    if quota_kinds:
        try:
            activity.check_quota(quota_kinds, limit, label)
        except activity.QuotaExceeded as exc:
            raise quota_exceeded(exc) from exc
    require_not_blocked(source)

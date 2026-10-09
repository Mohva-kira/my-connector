"""Journal d'activité (``activity_events``) : sert au monitoring et aux plafonds quotidiens.

Un événement = ``{at, kind, ...données}``. Types : ``run``, ``apply``, ``message_email``, ``message_linkedin``,
``profile_skill``, ``blocked``, ``unblocked``, ``error``. Les horodatages sont en UTC (``…Z``, C-1) ; les jours
(plafonds quotidiens, graphique) sont ceux du fuseau **local** de la machine.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, time, timedelta

from .db import utc_iso, utcnow
from .db.models import ActivityEvent
from .db.repositories import activity as activity_repo
from .db.repositories import user_transaction


def _public(event: ActivityEvent) -> dict:
    return {**event.data, "at": utc_iso(event.at), "kind": event.kind}


def record(kind: str, **data) -> dict:
    payload = json.loads(json.dumps(data, ensure_ascii=False, default=str))
    with user_transaction() as (s, uid):
        return _public(activity_repo.record(s, uid, kind, payload))


def load(limit: int | None = None) -> list[dict]:
    """Derniers événements, du plus ancien au plus récent."""
    with user_transaction() as (s, uid):
        return [_public(e) for e in activity_repo.latest(s, uid, limit)]


def _local_midnight(day: date) -> datetime:
    """Minuit local du jour ``day``, avec fuseau (gère les changements d'heure)."""
    return datetime.combine(day, time.min).astimezone()


def _local_today() -> date:
    return datetime.now().astimezone().date()


def count_today(*kinds: str, today: date | None = None) -> int:
    """Événements des types donnés depuis minuit local (plafonds quotidiens, I-07)."""
    day = today or _local_today()
    with user_transaction() as (s, uid):
        return activity_repo.count_between(
            s, uid, kinds, _local_midnight(day), _local_midnight(day + timedelta(days=1))
        )


def per_day(days: int = 14, today: date | None = None) -> list[dict]:
    """Événements par jour local sur les ``days`` derniers jours (jours vides inclus)."""
    end = today or _local_today()
    start = end - timedelta(days=days - 1)
    buckets: dict[str, Counter] = {(start + timedelta(days=i)).isoformat(): Counter() for i in range(days)}
    with user_transaction() as (s, uid):
        events = activity_repo.between(s, uid, _local_midnight(start), _local_midnight(end + timedelta(days=1)))
        for event in events:
            day = event.at.astimezone().date().isoformat()
            if day in buckets:
                buckets[day][event.kind] += 1
    return [{"date": day, **counts} for day, counts in buckets.items()]


class QuotaExceeded(Exception):
    """Plafond quotidien atteint (candidatures ou messages)."""

    def __init__(self, label: str, limit: int):
        super().__init__(
            f"Plafond quotidien atteint ({limit} {label} par jour) : réessaie demain ou relève-le dans les paramètres."
        )
        self.limit = limit


def check_quota(kinds: tuple[str, ...], limit: int, label: str) -> None:
    if count_today(*kinds) >= limit:
        raise QuotaExceeded(label, limit)


class BlockedByLinkedIn(RuntimeError):
    """LinkedIn a interrompu la navigation récemment : on évite d'insister (risque de restriction du compte)."""


BLOCK_COOLDOWN_HOURS = 6


def _block(hours: float, now: datetime | None) -> tuple[datetime, datetime, str] | None:
    now = now or utcnow()
    if now.tzinfo is None:
        now = now.astimezone()
    with user_transaction() as (s, uid):
        event = activity_repo.last_block_event(s, uid)
        if event is None or event.kind != "blocked":
            return None
        at, where = event.at, str(event.data.get("where", ""))
    until = at + timedelta(hours=hours)
    return (at, until, where) if now < until else None


def block_status(hours: float = BLOCK_COOLDOWN_HOURS, now: datetime | None = None) -> dict | None:
    """Dernier blocage encore dans la période de repos (et non levé depuis), sinon None (I-06)."""
    found = _block(hours, now)
    if found is None:
        return None
    at, until, where = found
    return {"at": utc_iso(at), "where": where, "until": utc_iso(until)}


def guard_not_blocked() -> None:
    found = _block(BLOCK_COOLDOWN_HOURS, None)
    if found:
        at, until, where = found
        raise BlockedByLinkedIn(
            f"LinkedIn a interrompu la navigation le {at.astimezone():%Y-%m-%d %H:%M} ({where or 'navigation'}). "
            f"Laisse passer du temps (jusqu'à {until.astimezone():%Y-%m-%d %H:%M}) ou lève le blocage dans le "
            "tableau de bord après avoir vérifié ton compte dans un navigateur normal."
        )

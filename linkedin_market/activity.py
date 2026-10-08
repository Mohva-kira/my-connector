"""Journal d'activité (``activity.jsonl``) : sert au monitoring et aux plafonds quotidiens.

Un événement = une ligne JSON ``{at, kind, ...}``. Types utilisés : ``apply`` (candidature
soumise), ``message_email``, ``message_linkedin``, ``run`` (collecte/analyse), ``error``.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import date, datetime, timedelta

from . import config

_LOCK = threading.Lock()


def record(kind: str, **data) -> dict:
    event = {"at": datetime.now().isoformat(timespec="seconds"), "kind": kind, **data}
    config.ensure_data_dirs()
    with _LOCK, config.ACTIVITY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return event


def load(limit: int | None = None) -> list[dict]:
    if not config.ACTIVITY_FILE.exists():
        return []
    events = []
    with config.ACTIVITY_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events[-limit:] if limit else events


def count_today(*kinds: str, today: date | None = None) -> int:
    prefix = (today or date.today()).isoformat()
    return sum(1 for e in load() if e.get("kind") in kinds and str(e.get("at", "")).startswith(prefix))


def per_day(days: int = 14, today: date | None = None) -> list[dict]:
    """Événements par jour sur les ``days`` derniers jours (jours vides inclus)."""
    end = today or date.today()
    start = end - timedelta(days=days - 1)
    buckets: dict[str, Counter] = {(start + timedelta(days=i)).isoformat(): Counter() for i in range(days)}
    for event in load():
        day = str(event.get("at", ""))[:10]
        if day in buckets:
            buckets[day][event.get("kind", "?")] += 1
    return [{"date": day, **counts} for day, counts in buckets.items()]


class QuotaExceeded(Exception):
    """Plafond quotidien atteint (candidatures ou messages)."""

    def __init__(self, label: str, limit: int):
        super().__init__(f"Plafond quotidien atteint ({limit} {label} par jour) : réessaie demain ou relève-le dans les paramètres.")
        self.limit = limit


def check_quota(kinds: tuple[str, ...], limit: int, label: str) -> None:
    if count_today(*kinds) >= limit:
        raise QuotaExceeded(label, limit)


class BlockedByLinkedIn(RuntimeError):
    """LinkedIn a interrompu la navigation récemment : on évite d'insister (risque de restriction du compte)."""


BLOCK_COOLDOWN_HOURS = 6


def block_status(hours: float = BLOCK_COOLDOWN_HOURS, now: datetime | None = None) -> dict | None:
    """Dernier blocage encore dans la période de repos (et non levé depuis), sinon None."""
    now = now or datetime.now()
    last_block = None
    for event in load():
        if event.get("kind") == "blocked":
            last_block = event
        elif event.get("kind") == "unblocked":
            last_block = None
    if not last_block:
        return None
    try:
        at = datetime.fromisoformat(last_block["at"])
    except (KeyError, ValueError):
        return None
    if now - at > timedelta(hours=hours):
        return None
    return {"at": last_block["at"], "where": last_block.get("where", ""), "until": (at + timedelta(hours=hours)).isoformat(timespec="minutes")}


def guard_not_blocked() -> None:
    blocked = block_status()
    if blocked:
        raise BlockedByLinkedIn(
            f"LinkedIn a interrompu la navigation le {blocked['at'][:16].replace('T', ' ')} ({blocked['where'] or 'navigation'}). "
            f"Laisse passer du temps (jusqu'à {blocked['until'].replace('T', ' ')}) ou lève le blocage dans le tableau de bord "
            "après avoir vérifié ton compte dans un navigateur normal.")

"""Fenêtres temporelles et filtrage des emails par période."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from email.utils import parsedate_to_datetime

from .config import VALID_PERIODS


def parse_email_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Parse l'en-tête Date RFC 2822 ; retourne None si impossible."""
    if not date_str or not str(date_str).strip():
        return None
    try:
        return parsedate_to_datetime(str(date_str).strip())
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_datetime_naive_local(dt: datetime) -> datetime:
    """Aligne les datetimes avec fuseau sur l'heure locale naive pour comparaison."""
    if dt.tzinfo is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


def get_period_bounds(period: str, now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Bornes [début, fin] en heure locale (naive).
    Pour N jours (3/7/11) : du début du jour (aujourd'hui - (N-1)) jusqu'à maintenant.
    """
    if period not in VALID_PERIODS:
        raise ValueError(f"Période invalide: {period}. Valeurs: {sorted(VALID_PERIODS)}")
    now = now or datetime.now()
    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "today":
        return start_today, now
    if period == "yesterday":
        start_y = start_today - timedelta(days=1)
        end_y = start_today - timedelta(microseconds=1)
        return start_y, end_y
    n = int(period)
    start = start_today - timedelta(days=n - 1)
    return start, now


def imap_days_back_for_period(period: str) -> int:
    """Nombre de jours à soustraire pour la date IMAP SINCE (approximation minimale)."""
    if period not in VALID_PERIODS:
        return 30
    if period == "today":
        return 0
    if period == "yesterday":
        return 2
    n = int(period)
    return max(0, n - 1)


def filter_emails_by_period(emails: List[Dict], period: str) -> List[Dict]:
    """Filtre les emails dont la date parsée tombe dans la fenêtre de période."""
    start, end = get_period_bounds(period)
    filtered: List[Dict] = []
    for e in emails:
        parsed = parse_email_datetime(e.get("date"))
        if parsed is None:
            continue
        p = normalize_datetime_naive_local(parsed)
        if start <= p <= end:
            filtered.append(e)
    return filtered

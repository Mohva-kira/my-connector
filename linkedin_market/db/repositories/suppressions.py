"""``suppression_list`` (I-13) : empreintes des destinataires opposés à tout contact, jamais l'adresse en clair.

Source unique de l'empreinte (``backend/01`` § 5.7.1 V4-5) :
- email : ``sha256(email.strip().lower())`` ;
- profil LinkedIn : forme canonique ``https://www.linkedin.com/in/<slug>`` (minuscules, sous-domaine pays remplacé
  par ``www.``, sans paramètres, fragment ni « / » final), puis ``sha256`` ;
- ni l'un ni l'autre : pas d'empreinte (jamais opposé).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ... import logmask
from .. import models

# Même expression que ``outreach._PROFILE_URL`` (le module de persistance n'importe pas Selenium) ; le slug est capturé.
_PROFILE = re.compile(r"^https://(?:[a-z]{2,3}\.)?linkedin\.com/in/([^/?#\s]+)")

OPPOSED_MESSAGE = "Ce destinataire a demandé à ne plus être contacté."
OPPOSED_CONTACT_MESSAGE = "Ce contact s'est opposé à tout contact : ses coordonnées ne peuvent plus être saisies."


class OpposedContact(ValueError):
    """Une coordonnée saisie appartient à un contact opposé (traduite en ``422`` par le routeur)."""


def canonical_profile_url(url: str) -> str | None:
    """Forme canonique d'une URL de profil LinkedIn, ou ``None`` si ce n'en est pas une."""
    found = _PROFILE.match((url or "").strip())
    return f"https://www.linkedin.com/in/{found.group(1).lower()}" if found else None


def fingerprint(value: str | None) -> str | None:
    """Empreinte SHA-256 (hexadécimale) d'un email ou d'une URL de profil ; ``None`` pour toute autre valeur."""
    text = (value or "").strip()
    if not text:
        return None
    if logmask.EMAIL_ADDRESS.match(text):
        normalized = text.lower()
    else:
        normalized = canonical_profile_url(text)
        if normalized is None:
            return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def fingerprints(*values: str | None) -> set[str]:
    """Empreintes des valeurs qui en ont une (les autres sont ignorées)."""
    return {h for h in (fingerprint(v) for v in values) if h is not None}


def any_suppressed(session: Session, user_id: UUID, hashes: Iterable[str]) -> bool:
    """Au moins une de ces empreintes figure dans la liste d'opposition de l'utilisateur."""
    wanted = set(hashes)
    if not wanted:
        return False
    return (
        session.execute(
            select(func.count())
            .select_from(models.SuppressionEntry)
            .where(models.SuppressionEntry.user_id == user_id, models.SuppressionEntry.recipient_hash.in_(wanted))
        ).scalar_one()
        > 0
    )


def add(session: Session, user_id: UUID, hashes: Iterable[str], reason: str) -> None:
    """Enregistre les empreintes ; idempotent, le ``reason`` d'origine d'une empreinte déjà connue est conservé."""
    rows = [{"user_id": user_id, "recipient_hash": h, "reason": reason} for h in sorted(set(hashes))]
    if rows:
        session.execute(
            insert(models.SuppressionEntry)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["user_id", "recipient_hash"])
        )

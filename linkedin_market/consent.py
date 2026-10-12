"""Consentement aux risques d'une source en mode ``navigateur`` (I-16, ``source_consents``, vague 3).

Couche domaine : aucune dépendance à FastAPI ni à Selenium. ``require()`` est la revérification posée au début
de toute tâche qui pilote le navigateur (``backend/02`` § 1.2) ; les routeurs appellent aussi ``require()`` avant
de créer la tâche, pour répondre ``428`` immédiatement plutôt que de la laisser échouer plus tard.
"""

from __future__ import annotations

import os

from . import activity
from .db import utc_iso
from .db.models import Source, SourceConsent
from .db.repositories import sources as sources_repo
from .db.repositories import user_transaction


class SourceNotFound(LookupError):
    """Code de source inconnu (``sources.code``)."""


class ConsentNotApplicable(ValueError):
    """La source n'est pas en mode ``navigateur`` : aucune popup de risques ne s'applique."""


class SourceUnavailable(RuntimeError):
    """Source pas encore activée dans cette version (``sources.enabled = false``) ; porte le libellé de la source."""


class ConsentNotAccepted(ValueError):
    """``accepted`` doit valoir ``true`` (acceptation explicite, I-16)."""


class RiskVersionStale(RuntimeError):
    """``risk_version`` du corps différente de la version courante : relire le nouveau texte avant d'accepter."""


class NoActiveConsent(LookupError):
    """Aucun consentement actif à révoquer pour cette source."""


class SourceConsentRequired(RuntimeError):
    """Action navigateur sans consentement actif à la version courante (I-16, HTTP 428 côté API)."""

    def __init__(self, source_code: str, label: str, risk_version: str) -> None:
        super().__init__(
            f"Accepte d'abord les risques liés à {label} (version {risk_version}) avant toute action "
            "dans le navigateur."
        )
        self.source_code = source_code
        self.label = label
        self.risk_version = risk_version


def _france_travail_configured() -> bool:
    return bool(os.environ.get("FRANCE_TRAVAIL_CLIENT_ID")) and bool(os.environ.get("FRANCE_TRAVAIL_CLIENT_SECRET"))


def _public(source: Source, consent_row: SourceConsent | None) -> dict:
    return {
        "code": source.code,
        "label": source.label,
        "access_mode": source.access_mode,
        "enabled": source.enabled,
        "risk_version": source.risk_version,
        "consent": (
            {"risk_version": consent_row.risk_version, "accepted_at": utc_iso(consent_row.accepted_at)}
            if consent_row is not None
            else None
        ),
        "consent_required": source.access_mode == "navigateur"
        and (consent_row is None or consent_row.risk_version != source.risk_version),
    }


def list_public() -> list[dict]:
    """Forme de ``GET /api/sources`` (``backend/01`` § 5.8 et § 5.8.1) : consentement, disponibilité, repos."""
    with user_transaction() as (s, uid):
        rows = [(src, sources_repo.active_consent(s, uid, src.code)) for src in sources_repo.all_sources(s)]
    results = []
    for source, consent_row in rows:
        row = _public(source, consent_row)
        if source.code == "france_travail":
            row["configured"] = _france_travail_configured()
        row["blocked"] = activity.block_status(source=source.code)
        results.append(row)
    return results


def get_public(code: str) -> dict:
    """Élément de ``GET /api/sources`` pour une seule source (après acceptation ou révocation)."""
    with user_transaction() as (s, uid):
        source = sources_repo.get(s, code)
        if source is None:
            raise SourceNotFound(code)
        consent_row = sources_repo.active_consent(s, uid, code)
        row = _public(source, consent_row)
    if code == "france_travail":
        row["configured"] = _france_travail_configured()
    row["blocked"] = activity.block_status(source=code)
    return row


def ensure_source_exists(code: str) -> None:
    with user_transaction() as (s, _uid):
        if sources_repo.get(s, code) is None:
            raise SourceNotFound(code)


def accept(code: str, *, risk_version: str, accepted: bool) -> dict:
    """Accepte les risques d'une source (``POST /api/sources/{code}/consent``) : contrôles dans l'ordre du contrat."""
    with user_transaction() as (s, uid):
        source = sources_repo.get(s, code)
        if source is None:
            raise SourceNotFound(code)
        if source.access_mode != "navigateur":
            raise ConsentNotApplicable(code)
        if not source.enabled:
            raise SourceUnavailable(source.label)
        if not accepted:
            raise ConsentNotAccepted(code)
        if risk_version != source.risk_version:
            raise RiskVersionStale(code)
        sources_repo.accept(s, uid, code, risk_version)
    return get_public(code)


def revoke(code: str) -> None:
    """Révoque le consentement actif (``DELETE /api/sources/{code}/consent``)."""
    with user_transaction() as (s, uid):
        source = sources_repo.get(s, code)
        if source is None:
            raise SourceNotFound(code)
        if not sources_repo.revoke(s, uid, code):
            raise NoActiveConsent(code)


def require(source_code: str) -> None:
    """Lève ``SourceConsentRequired`` si ``source_code`` est en mode ``navigateur`` sans consentement à jour (I-16).

    Revérifié au début de chaque tâche qui pilote le navigateur (le consentement a pu être révoqué entre la
    création de la tâche et son démarrage) et appelé une première fois par le routeur, avant de créer la tâche.
    Une source inconnue ou hors mode ``navigateur`` ne requiert aucun consentement : silencieuse dans ce cas (la
    validation de son existence relève d'un autre contrôle, en amont).
    """
    with user_transaction() as (s, uid):
        source = sources_repo.get(s, source_code)
        if source is None or source.access_mode != "navigateur":
            return
        active = sources_repo.active_consent(s, uid, source_code)
        if active is not None and active.risk_version == source.risk_version:
            return
        label, risk_version = source.label, source.risk_version
    raise SourceConsentRequired(source_code, label, risk_version)

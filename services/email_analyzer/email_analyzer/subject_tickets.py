"""Détection du motif de ticket entre crochets dans un sujet d'email
(``[<libellé banque/plateforme><séparateur><numéro>]``, ex.
``[BDM TOGO_BDM MOBILE_12002]``, ``[BMS MALI-BMSKIBARU-12005]``) — module pur
(aucun accès DB/IMAP/LLM), même discipline que ``classification.py``.

Aucun format fermé de banque/plateforme n'est supposé (voir
context/progress-tracker.md) : le motif reconnu est générique — un libellé
libre suivi d'un séparateur ``_``/``-`` puis d'un numéro de ticket, entre
crochets, en tête de sujet (après suppression des préfixes Re:/Fwd:/Tr:
répétés). Toute présence de ce motif signale une réclamation client, quel
que soit le contenu du corps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# En dessous de 3 chiffres, trop de faux positifs plausibles (numérotation de
# version "[v2]", copie "[1]", etc.) — les numéros de ticket observés sont
# systématiquement à 5 chiffres mais on garde une marge.
_MIN_TICKET_DIGITS = 3

_SUBJECT_TICKET_PATTERN = re.compile(
    r"^\s*\[\s*(?P<label>[^\[\]]+?)\s*[_\-]\s*(?P<ticket>\d{%d,})\s*\]" % _MIN_TICKET_DIGITS
)

_REPLY_PREFIX_PATTERN = re.compile(r"^\s*(re|tr|r[ée]p|fwd?)\s*:\s*", re.IGNORECASE)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SubjectTicketKey:
    """Résultat de ``parse_subject_ticket_key`` pour un sujet donné."""

    raw_label: str
    normalized_label: str
    ticket_number: str
    full_match: str


def _strip_reply_prefixes(subject: str) -> str:
    text = subject
    while True:
        stripped = _REPLY_PREFIX_PATTERN.sub("", text)
        if stripped == text:
            return text
        text = stripped


def parse_subject_ticket_key(subject: Optional[str]) -> Optional[SubjectTicketKey]:
    """Extrait la clé de ticket en tête d'un sujet, si présente.

    Renvoie ``None`` si le sujet est vide ou ne commence pas (après retrait
    des préfixes de réponse/transfert) par ``[libellé-ou-libellé_num]``.
    """
    if not subject:
        return None
    text = _strip_reply_prefixes(subject)
    match = _SUBJECT_TICKET_PATTERN.match(text)
    if not match:
        return None
    label = match.group("label").strip()
    if not label:
        return None
    return SubjectTicketKey(
        raw_label=label,
        normalized_label=_WHITESPACE_RE.sub(" ", label).upper(),
        ticket_number=match.group("ticket"),
        full_match=match.group(0).strip(),
    )

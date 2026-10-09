"""Masquage des données de tiers dans les journaux de tâche (E-12, I-15, RGPD art. 5(1)(c)).

Les journaux (``tasks.logs``, ``tasks.error``) sont persistés et affichés : on n'y écrit jamais l'adresse complète
d'un recruteur, son nom complet ni l'URL de son profil. L'information complète reste dans l'historique de la
candidature, ``outreach_messages`` et ``company_contacts`` (soumis à la rétention, I-14).

- adresse : ``jane.doe@acme.com`` -> ``j***@acme.com`` ;
- nom : ``Jane Doe`` -> ``J. D.`` ;
- profil : ``https://www.linkedin.com/in/jane-doe/`` -> ``linkedin.com/in/j***``.

Module pur (ni base, ni navigateur) : utilisable par le domaine, les intégrations et le gestionnaire de tâches.
"""

from __future__ import annotations

import re

# Adresse email acceptée comme destinataire et comme adresse de contact (sans « , ; < > » : pas d'injection
# d'en-tête ni de destinataires multiples). Source unique pour ``outreach`` et ``settings`` (G-12).
EMAIL_ADDRESS = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")

_EMAIL_IN_TEXT = re.compile(r"(?<![\w.%+-])([\w.%+-])[\w.%+-]*@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
_PROFILE_IN_TEXT = re.compile(r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/in/([^/?#\s\"'<>)]+)[^\s\"'<>)]*")


def mask_email(address: str) -> str:
    """``jane.doe@acme.com`` -> ``j***@acme.com`` ; une valeur sans « @ » est entièrement masquée."""
    address = (address or "").strip()
    local, at, domain = address.rpartition("@")
    if not at or not local:
        return "***" if address else ""
    return f"{local[0]}***@{domain}"


def mask_name(full_name: str) -> str:
    """``Jane Doe`` -> ``J. D.`` (initiales seulement) ; vide si le nom est vide."""
    parts = [p for p in re.split(r"[\s\-]+", full_name or "") if p and p[0].isalnum()]
    return " ".join(f"{p[0].upper()}." for p in parts)


def mask_profile_url(url: str) -> str:
    """``https://www.linkedin.com/in/jane-doe/`` -> ``linkedin.com/in/j***`` ; autre URL : ``***``."""
    match = _PROFILE_IN_TEXT.search(url or "")
    if match is None:
        return "***" if (url or "").strip() else ""
    return f"linkedin.com/in/{match.group(1)[0]}***"


def mask_recipient(recipient: str) -> str:
    """Destinataire d'un message : adresse email ou URL de profil LinkedIn, masqué."""
    recipient = (recipient or "").strip()
    if "@" in recipient and "/" not in recipient:
        return mask_email(recipient)
    return mask_profile_url(recipient)


def mask_text(text: str) -> str:
    """Masque les adresses email et les URL de profils LinkedIn présentes dans un texte libre (filet de sécurité).

    Les noms ne sont pas détectables dans un texte libre : l'appelant les passe par ``mask_name`` avant d'écrire.
    """
    if not text:
        return text
    text = _PROFILE_IN_TEXT.sub(lambda m: f"linkedin.com/in/{m.group(1)[0]}***", text)
    return _EMAIL_IN_TEXT.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)

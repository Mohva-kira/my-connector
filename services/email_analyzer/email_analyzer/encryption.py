"""Chiffrement des secrets tenant (mot de passe IMAP, etc.) avec Fernet."""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken


def _fernet() -> Fernet:
    raw = (os.environ.get("ENCRYPTION_KEY") or "").strip()
    if not raw:
        raise RuntimeError("ENCRYPTION_KEY manquant dans l'environnement")
    return Fernet(raw.encode("ascii"))


def encrypt_secret(plain: str) -> str:
    if plain is None:
        raise ValueError("plain ne peut pas être None")
    return _fernet().encrypt(plain.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> str:
    if not token:
        return ""
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise ValueError("Secret invalide ou clé ENCRYPTION_KEY incorrecte") from e

"""Chiffrement applicatif Fernet (I-15) : cookies de session stockés dans ``platform_sessions.cookies_encrypted``.

La clé ``LM_SECRET_KEY`` vient de l'environnement (``.env`` à la racine). Si elle est absente au premier
lancement, elle est générée et ajoutée au ``.env`` (droits ``0600``). Sous pytest, la clé doit être fournie par
l'environnement : la génération dans le vrai ``.env`` est refusée.
"""

import logging
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .. import config

log = logging.getLogger(__name__)

ENV_VAR = "LM_SECRET_KEY"
ENV_FILE = config.PROJECT_ROOT / ".env"


class SecretKeyError(RuntimeError):
    """Clé absente, invalide ou ne permettant pas de déchiffrer (message lisible)."""


def _under_pytest() -> bool:
    return "pytest" in sys.modules


def _append_to_env_file(env_file: Path, key: str) -> None:
    """Ajoute ``LM_SECRET_KEY=…`` au fichier (créé si besoin) et le restreint au propriétaire (``0600``)."""
    prefix = ""
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8")
        if content and not content.endswith("\n"):
            prefix = "\n"
    fd = os.open(env_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(f"{prefix}{ENV_VAR}={key}\n")
    env_file.chmod(0o600)


def secret_key(env_file: Path | None = None) -> str:
    """Clé Fernet courante ; générée et ajoutée à ``env_file`` (défaut : ``.env`` racine) si absente."""
    key = (os.environ.get(ENV_VAR) or "").strip()
    if key:
        return key
    target = Path(env_file) if env_file is not None else ENV_FILE
    if _under_pytest() and target.resolve() == ENV_FILE.resolve():
        raise SecretKeyError(f"{ENV_VAR} absente en test : définis-la dans l'environnement (jamais dans le .env).")
    key = Fernet.generate_key().decode("ascii")
    _append_to_env_file(target, key)
    os.environ[ENV_VAR] = key
    log.warning(
        "Clé %s générée et ajoutée à %s : sauvegarde ce fichier avec ta base, "
        "sans elle les cookies de session chiffrés sont illisibles.",
        ENV_VAR,
        target,
    )
    return key


def fernet() -> Fernet:
    """Instance Fernet sur ``LM_SECRET_KEY`` ; clé mal formée ⇒ ``SecretKeyError``."""
    try:
        return Fernet(secret_key().encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise SecretKeyError(
            f"{ENV_VAR} est invalide (44 caractères base64 attendus) : corrige-la dans le .env "
            "ou retire-la pour en générer une nouvelle (les cookies déjà chiffrés seront perdus)."
        ) from exc


def encrypt(data: bytes) -> bytes:
    """Chiffre ``data`` (jeton Fernet, à stocker en ``bytea``)."""
    return fernet().encrypt(data)


def decrypt(token: bytes) -> bytes:
    """Déchiffre un jeton Fernet ; clé différente ou jeton altéré ⇒ ``SecretKeyError``."""
    try:
        return fernet().decrypt(token)
    except InvalidToken as exc:
        raise SecretKeyError(
            f"Impossible de déchiffrer la session avec {ENV_VAR} actuelle : reconnecte-toi à la plateforme."
        ) from exc

"""JWT et hachage des mots de passe."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

import bcrypt
from jose import JWTError, jwt

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

# bcrypt ignore silencieusement tout octet au-delà du 72e ; tronquer
# explicitement documente ce comportement au lieu de le laisser implicite.
_BCRYPT_MAX_BYTES = 72


def _truncate_for_bcrypt(plain: str) -> bytes:
    return plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_truncate_for_bcrypt(plain), hashed.encode("utf-8"))


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(_truncate_for_bcrypt(plain), bcrypt.gensalt()).decode("utf-8")


def _secret() -> str:
    s = (os.environ.get("JWT_SECRET") or "").strip()
    if not s:
        raise RuntimeError("JWT_SECRET manquant")
    return s


def create_access_token(
    *,
    user_id: UUID,
    tenant_id: UUID,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode: dict[str, Any] = {
        "sub": str(user_id),
        "tid": str(tenant_id),
        "email": email,
        "exp": expire,
    }
    return jwt.encode(to_encode, _secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, _secret(), algorithms=[ALGORITHM])


def parse_user_tenant_ids(payload: dict[str, Any]) -> tuple[UUID, UUID]:
    try:
        uid = UUID(str(payload.get("sub")))
        tid = UUID(str(payload.get("tid")))
        return uid, tid
    except (ValueError, TypeError) as e:
        raise JWTError("Jeton invalide") from e

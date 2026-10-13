"""Import manuel d'une offre Indeed (R-6) : ``POST /api/items/import`` (``backend/01`` § 5.8.2).

Aucun appel réseau, ni API ni navigateur, ni téléchargement de l'adresse fournie : l'utilisateur colle le texte de
l'offre, l'adresse n'est conservée que pour mémoire.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from ... import settings, storage
from ...models import Item

router = APIRouter()

MAX_URL = 2000
TEXT_MIN, TEXT_MAX = 200, 20000
# Seuls paramètres d'adresse conservés : ceux qui identifient l'offre chez Indeed.
_KEPT_PARAMS = ("jk", "vjk")


def _within(value: str, maximum: int, message: str) -> str:
    value = value.strip()
    if len(value) > maximum:
        raise ValueError(message)
    return value


class ManualItem(BaseModel):
    """Offre Indeed collée par l'utilisateur."""

    source_code: Literal["indeed"]
    url: str
    title: str
    company: str = ""
    location: str = ""
    text: str
    keyword: str = ""

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = value.strip()
        message = "colle l'adresse https de l'offre Indeed."
        try:
            parts = urlsplit(value)
            host = (parts.hostname or "").lower()
            parts.port  # noqa: B018 - lève ValueError si le port est invalide
        except ValueError:
            raise ValueError(message) from None
        if (
            len(value) > MAX_URL
            or parts.scheme != "https"
            or not (host == "indeed.com" or host.endswith(".indeed.com"))
        ):
            raise ValueError(message)
        return value

    @field_validator("title")
    @classmethod
    def _title(cls, value: str) -> str:
        value = _within(value, 150, "renseigne le titre de l'offre (150 caractères maximum).")
        if not value:
            raise ValueError("renseigne le titre de l'offre (150 caractères maximum).")
        return value

    @field_validator("company")
    @classmethod
    def _company(cls, value: str) -> str:
        return _within(value, 80, "entreprise trop longue (80 caractères maximum).")

    @field_validator("location")
    @classmethod
    def _location(cls, value: str) -> str:
        return _within(value, 120, "lieu trop long (120 caractères maximum).")

    @field_validator("keyword")
    @classmethod
    def _keyword(cls, value: str) -> str:
        return _within(value, 80, "mot-clé trop long (80 caractères maximum).")

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < TEXT_MIN:
            raise ValueError("colle le texte complet de l'offre (200 caractères minimum).")
        if len(value) > TEXT_MAX:
            raise ValueError("texte trop long (20 000 caractères maximum).")
        return value


def normalize_url(url: str) -> str:
    """Adresse normalisée : schéma ``https``, hôte en minuscules, sans fragment, requête réduite à ``jk`` / ``vjk``."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    netloc = f"{host}:{parts.port}" if parts.port else host
    query = {k: v for k, v in sorted(parse_qsl(parts.query)) if k in _KEPT_PARAMS}
    return f"https://{netloc}{parts.path}" + (f"?{urlencode(query)}" if query else "")


def external_id_of(url: str) -> str:
    """``indeed-`` + 16 premiers caractères hexadécimaux du sha256 de l'adresse normalisée."""
    return "indeed-" + hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()[:16]


def _keyword(explicit: str) -> str:
    """Mot-clé du corps, sinon celui du dernier rapport, sinon celui des réglages ; ``422`` si aucun."""
    if explicit:
        return explicit
    reports = storage.list_reports()
    candidates = [reports[0]["keyword"] if reports else "", settings.load_settings()["keyword"]]
    for candidate in candidates:
        if (candidate or "").strip():
            return candidate.strip()
    raise HTTPException(422, "keyword : renseigne le mot-clé du poste visé (Paramètres ou ce formulaire).")


@router.post("/items/import", status_code=201)
def import_item(body: ManualItem) -> dict:
    """Ajoute une offre Indeed à l'historique ; scorée au prochain recalcul ou à la prochaine analyse."""
    keyword = _keyword(body.keyword)
    url = normalize_url(body.url)
    external_id = external_id_of(body.url)
    now = datetime.now(UTC)
    item = Item(
        kind="job",
        id=f"job:{external_id}",
        url=url,
        title=body.title,
        text=body.text,
        keyword=keyword,
        company=body.company,
        location=body.location,
        posted_at=now.date().isoformat(),
        scraped_at=now.isoformat(),
        source_code="indeed",
    )
    if not storage.import_item(item):
        raise HTTPException(409, "Cette offre est déjà dans l'historique.")
    return {
        "id": item.id,
        "source_code": "indeed",
        "title": item.title,
        "company": item.company,
        "location": item.location,
        "url": url,
        "keyword": keyword,
    }

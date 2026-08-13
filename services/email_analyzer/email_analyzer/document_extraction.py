"""Extraction de texte pour la recherche de documents (Contexte Documents,
domaine #7). v1 volontairement limité aux formats qu'on peut extraire de
façon fiable et vérifiable dans cette session : PDF (`pypdf`) et texte brut
(`text/plain`, `.md`, `.csv`). Word/Excel/PowerPoint/images ne sont pas
supportés ici — un document de ces formats reste stocké et listé, seulement
pas recherchable par contenu (voir Document.__doc__, db/models.py). Pas de
code spéculatif pour des formats non testés (ai-workflow-rules.md, "Ban
Speculative Coding")."""

from __future__ import annotations

import logging
from typing import Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)

_TEXT_CONTENT_TYPES = {"text/plain", "text/markdown", "text/csv"}
_MAX_EXTRACTED_CHARS = 200_000  # borne la taille stockée en base pour un document volumineux


def is_supported_attachment(content_type: Optional[str], filename: str) -> bool:
    """``True`` si ``extract_text`` sait traiter ce type (PDF ou texte brut) —
    utilisé par l'ingestion automatique des pièces jointes d'email
    (``project_mail.py``) pour ne capturer que ce qui sera réellement
    recherchable, plutôt que de stocker des images/logos de signature sans
    valeur pour l'analyse."""
    return _is_pdf(content_type, filename) or _is_plain_text(content_type, filename)


def extract_text(content_type: Optional[str], filename: str, data: bytes) -> Optional[str]:
    """Retourne le texte extrait, ou ``None`` si le format n'est pas pris
    en charge ou si l'extraction échoue (ex. PDF scanné sans couche texte,
    fichier corrompu) — jamais d'exception propagée à l'appelant : un échec
    d'extraction ne doit pas empêcher l'upload du document lui-même."""
    if _is_pdf(content_type, filename):
        return _extract_pdf_text(data)
    if _is_plain_text(content_type, filename):
        return _extract_plain_text(data)
    return None


def _is_pdf(content_type: Optional[str], filename: str) -> bool:
    return content_type == "application/pdf" or filename.lower().endswith(".pdf")


def _is_plain_text(content_type: Optional[str], filename: str) -> bool:
    return content_type in _TEXT_CONTENT_TYPES or filename.lower().endswith((".txt", ".md", ".csv"))


def _extract_pdf_text(data: bytes) -> Optional[str]:
    import io

    try:
        reader = PdfReader(io.BytesIO(data))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages_text).strip()
        return text[:_MAX_EXTRACTED_CHARS] if text else None
    except Exception as ex:  # noqa: BLE001 — extraction best-effort, jamais bloquant
        logger.warning("document_extraction pdf_error error=%s", ex)
        return None


def _extract_plain_text(data: bytes) -> Optional[str]:
    try:
        text = data.decode("utf-8", errors="replace").strip()
        return text[:_MAX_EXTRACTED_CHARS] if text else None
    except Exception as ex:  # noqa: BLE001
        logger.warning("document_extraction text_error error=%s", ex)
        return None

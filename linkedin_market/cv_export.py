"""Fichiers d'export du CV (PDF, DOCX) dans le dossier de candidature, nommés « Prénom-Nom-CV-Titre.<ext> ».

Un export est régénéré s'il manque ou s'il est plus ancien que ``cv.md`` ; changer de nom (réglage ``full_name``)
remplace l'ancien fichier du même format. Le nom de fichier est celui que voit le recruteur.
"""

from __future__ import annotations

from pathlib import Path

from . import cv_docx, cv_format, cv_pdf, nlp

FORMATS = {"pdf": cv_pdf.render_pdf, "docx": cv_docx.render_docx}
MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def ensure(folder: Path, markdown: str, fmt: str, *, title: str, full_name: str = "") -> Path:
    target = folder / f"{cv_format.file_stem(full_name, title)}.{fmt}"
    source = folder / "cv.md"
    if target.is_file() and (not source.is_file() or target.stat().st_mtime >= source.stat().st_mtime):
        return target
    for stale in folder.glob(f"*.{fmt}"):
        stale.unlink()
    render = FORMATS[fmt]
    target.write_bytes(
        render(
            markdown,
            title,
            author=full_name,
            keywords=nlp.extract_skills(markdown),
            lang=cv_format.language_of(markdown),
        )
    )
    return target


def remove(folder: Path) -> None:
    """Supprime les exports périmés (CV modifié) : ils sont régénérés à la demande."""
    for fmt in FORMATS:
        for path in folder.glob(f"*.{fmt}"):
            path.unlink()

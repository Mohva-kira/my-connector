"""Construction du profil utilisateur : fichier à la racine du projet ou LinkedIn.

Formats acceptés à la racine (voir ``config.PROFILE_FILE_CANDIDATES``) :
- ``profil.json`` : {"title", "skills", "years_experience", "seniority", "text"}
- ``profil.md`` / ``profil.txt`` : texte libre (CV, résumé)
- ``cv.pdf`` : texte extrait avec pypdf
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config, nlp, settings
from .models import Profile, SearchRequest

log = logging.getLogger(__name__)


def find_profile_file() -> Path | None:
    for name in config.PROFILE_FILE_CANDIDATES:
        path = config.PROJECT_ROOT / name
        if path.is_file():
            return path
    return None


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf non installé : impossible de lire %s.", path.name)
        return ""
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def from_file(path: Path) -> Profile:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        text = data.get("text") or ""
        skills = [nlp.normalize(s) for s in data.get("skills", [])]
        text = f"{data.get('title', '')}\n{' '.join(data.get('skills', []))}\n{text}"
        return Profile(
            text=text,
            origin=str(path),
            title=data.get("title", ""),
            skills=sorted(set(skills) | set(nlp.extract_skills(text))),
            years_experience=data.get("years_experience"),
            seniority=data.get("seniority", ""),
        )
    text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    return Profile(text=text, origin=str(path))


def from_text(text: str, origin: str) -> Profile:
    return Profile(text=text, origin=origin)


def finalize(profile: Profile, request: SearchRequest, use_overrides: bool = True) -> Profile:
    """Complète le profil par extraction et par les réponses de la saisie."""
    first_line = next((line.strip() for line in profile.text.splitlines() if line.strip()), "")
    profile.title = profile.title or first_line[:120]
    extracted = nlp.extract_skills(profile.text)
    extra = [nlp.normalize(s) for s in request.extra_skills]
    # Une compétence saisie qui correspond à une compétence connue prend son nom canonique.
    extra_canonical = {s for e in extra for s in nlp.extract_skills(e)} | {
        e for e in extra if not nlp.extract_skills(e)
    }
    profile.skills = sorted(set(profile.skills) | set(extracted) | extra_canonical)
    if request.years_experience is not None:
        profile.years_experience = request.years_experience
    elif profile.years_experience is None:
        profile.years_experience = nlp.extract_years(profile.text)
    profile.seniority = request.seniority or profile.seniority or nlp.extract_seniority(profile.text[:1500])
    if request.extra_skills:
        profile.text += "\n" + " ".join(request.extra_skills)
    if use_overrides:
        settings.apply_to_profile(profile)  # surcharges saisies dans l'interface web
    return profile

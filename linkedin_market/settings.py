"""Réglages de l'interface web : valeurs par défaut de recherche, plafonds, réponses mémorisées,
et surcharges du profil (compétences ajoutées/retirées).

Les surcharges ne réécrivent jamais ``profil.*`` : elles s'appliquent par-dessus à la
construction du profil (``profile.finalize``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import config, nlp

DEFAULT_SETTINGS: dict = {
    "keyword": "",
    "location": "",
    "remote_only": True,
    "caution": config.DEFAULT_CAUTION,
    "max_items": None,                 # None = selon la prudence
    "profile_source": "auto",          # auto | file | linkedin
    "interpreter": "claude",           # claude | local | none
    "apply_min_score": config.APPLY_MIN_SCORE,
    "apply_daily_max": 10,
    "message_daily_max": 15,
    "contact_email": "",               # email/téléphone saisis dans « Candidature simplifiée »
    "contact_phone": "",
    "signature": "",
    "linkedin_skill_sync": True,       # compétence ajoutée dans l'interface -> ajoutée au profil LinkedIn
    "answers": {},                     # question normalisée -> réponse mémorisée
}

DEFAULT_PROFILE_SETTINGS: dict = {
    "title": "",
    "years_experience": None,
    "seniority": "",
    "added_skills": [],
    "removed_skills": [],
}

_PROFILE_SOURCES = ("auto", "file", "linkedin")
_INTERPRETERS = ("claude", "local", "none")


class SettingsError(ValueError):
    """Valeur de réglage invalide (message lisible côté UI)."""


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: dict) -> None:
    config.ensure_data_dirs()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------- réglages
def load_settings() -> dict:
    return {**DEFAULT_SETTINGS, **{k: v for k, v in _read(config.SETTINGS_FILE).items() if k in DEFAULT_SETTINGS}}


def _int(value, name: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise SettingsError(f"{name} doit être un entier.") from None
    if not low <= number <= high:
        raise SettingsError(f"{name} doit être compris entre {low} et {high}.")
    return number


def validate_settings(update: dict) -> dict:
    """Valide un sous-ensemble de réglages ; retourne uniquement les clés connues, normalisées."""
    clean: dict = {}
    for key, value in update.items():
        if key not in DEFAULT_SETTINGS:
            continue
        if key == "caution":
            if value not in config.CAUTION_LEVELS:
                raise SettingsError(f"caution doit être l'un de : {', '.join(config.CAUTION_LEVELS)}.")
        elif key == "profile_source" and value not in _PROFILE_SOURCES:
            raise SettingsError(f"profile_source doit être l'un de : {', '.join(_PROFILE_SOURCES)}.")
        elif key == "interpreter" and value not in _INTERPRETERS:
            raise SettingsError(f"interpreter doit être l'un de : {', '.join(_INTERPRETERS)}.")
        elif key == "max_items":
            value = None if value in (None, "") else _int(value, key, 1, 200)
        elif key == "apply_min_score":
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise SettingsError("apply_min_score doit être un nombre.") from None
            if not 0 <= value <= 100:
                raise SettingsError("apply_min_score doit être compris entre 0 et 100.")
        elif key in ("apply_daily_max", "message_daily_max"):
            value = _int(value, key, 0, 200)
        elif key in ("remote_only", "linkedin_skill_sync"):
            value = bool(value)
        elif key == "answers":
            if not isinstance(value, dict):
                raise SettingsError("answers doit être un objet question -> réponse.")
            value = {nlp.normalize(str(q))[:200]: str(a)[:500] for q, a in value.items() if str(q).strip()}
        else:
            value = str(value or "").strip()
        clean[key] = value
    return clean


def save_settings(update: dict) -> dict:
    merged = {**load_settings(), **validate_settings(update)}
    _write(config.SETTINGS_FILE, merged)
    return merged


def remember_answer(question: str, answer: str) -> None:
    answers = dict(load_settings()["answers"])
    answers[nlp.normalize(question)[:200]] = str(answer)[:500]
    save_settings({"answers": answers})


def find_answer(question: str) -> str | None:
    return load_settings()["answers"].get(nlp.normalize(question)[:200])


# ------------------------------------------------------------ profil (UI)
def load_profile_settings() -> dict:
    data = _read(config.PROFILE_SETTINGS_FILE)
    merged = {**DEFAULT_PROFILE_SETTINGS, **{k: v for k, v in data.items() if k in DEFAULT_PROFILE_SETTINGS}}
    merged["added_skills"] = list(merged["added_skills"])
    merged["removed_skills"] = list(merged["removed_skills"])
    return merged


def _skill_list(values) -> list[str]:
    if not isinstance(values, list):
        raise SettingsError("La liste de compétences est invalide.")
    seen: list[str] = []
    for raw in values:
        skill = nlp.normalize(str(raw))
        if skill and len(skill) <= 60 and skill not in seen:
            seen.append(skill)
    return seen


def save_profile_settings(update: dict) -> dict:
    current = load_profile_settings()
    for key, value in update.items():
        if key not in DEFAULT_PROFILE_SETTINGS:
            continue
        if key in ("added_skills", "removed_skills"):
            value = _skill_list(value)
        elif key == "years_experience":
            value = None if value in (None, "") else _int(value, key, 0, 60)
        elif key == "seniority":
            value = str(value or "").strip().lower()
            if value and value not in nlp.SENIORITY_LEVELS:
                raise SettingsError(f"seniority doit être l'un de : {', '.join(nlp.SENIORITY_LEVELS)}.")
        else:
            value = str(value or "").strip()[:120]
        current[key] = value
    # Une compétence ne peut pas être à la fois ajoutée et retirée : le dernier choix gagne.
    current["removed_skills"] = [s for s in current["removed_skills"] if s not in current["added_skills"]]
    _write(config.PROFILE_SETTINGS_FILE, current)
    return current


def add_skill(skill: str) -> dict:
    name = nlp.normalize(skill)
    if not name:
        raise SettingsError("Compétence vide.")
    current = load_profile_settings()
    added = current["added_skills"] + ([name] if name not in current["added_skills"] else [])
    return save_profile_settings({"added_skills": added, "removed_skills": [s for s in current["removed_skills"] if s != name]})


def remove_skill(skill: str) -> dict:
    name = nlp.normalize(skill)
    current = load_profile_settings()
    removed = current["removed_skills"] + ([name] if name not in current["removed_skills"] else [])
    return save_profile_settings({"added_skills": [s for s in current["added_skills"] if s != name], "removed_skills": removed})


def apply_to_profile(profile) -> None:
    """Applique les surcharges à un ``Profile`` (modifié en place)."""
    overrides = load_profile_settings()
    skills = (set(profile.skills) | set(overrides["added_skills"])) - set(overrides["removed_skills"])
    profile.skills = sorted(skills)
    if overrides["title"]:
        profile.title = overrides["title"]
    if overrides["years_experience"] is not None:
        profile.years_experience = overrides["years_experience"]
    if overrides["seniority"]:
        profile.seniority = overrides["seniority"]

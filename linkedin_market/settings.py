"""Réglages de l'interface web : valeurs par défaut de recherche, plafonds, réponses mémorisées,
et surcharges du profil (compétences ajoutées/retirées).

Les surcharges ne réécrivent jamais ``profil.*`` : elles s'appliquent par-dessus à la
construction du profil (``profile.finalize``). Façade de persistance (D-1) : tables ``user_settings``,
``memorized_answers``, ``candidate_profiles`` et ``profile_skill_overrides``.
"""

from __future__ import annotations

import unicodedata
from zoneinfo import ZoneInfo

from . import config, logmask, nlp
from .db.repositories import profile as profile_repo
from .db.repositories import settings as settings_repo
from .db.repositories import user_transaction

DEFAULT_SETTINGS: dict = {
    "keyword": "",
    "location": "",
    "remote_only": True,
    "caution": config.DEFAULT_CAUTION,
    "max_items": None,  # None = plafond de la prudence
    "profile_source": "auto",  # auto | file | linkedin
    "interpreter": "claude",  # claude | local (« none » lu comme local, E-7)
    "apply_min_score": config.APPLY_MIN_SCORE,
    "apply_daily_max": 10,
    "message_daily_max": 15,
    "contact_email": "",  # email/téléphone saisis dans « Candidature simplifiée »
    "contact_phone": "",
    "signature": "",
    "home_location": "",  # lieu de l'utilisateur (prompts Claude, E-18) ; vide = non communiqué
    "timezone": "",  # fuseau IANA (ex. Europe/Paris) ; vide = non communiqué
    "linkedin_skill_sync": True,  # compétence ajoutée dans l'interface -> ajoutée au profil LinkedIn
    "answers": {},  # question normalisée -> réponse mémorisée
}

DEFAULT_PROFILE_SETTINGS: dict = {
    "title": "",
    "years_experience": None,
    "seniority": "",
    "added_skills": [],
    "removed_skills": [],
}

HOME_LOCATION_MAX = 80
_PROFILE_SOURCES = ("auto", "file", "linkedin")
_INTERPRETERS = ("claude", "local")  # « none » refusé à l'écriture (E-7, I-11)


class SettingsError(ValueError):
    """Valeur de réglage invalide (message lisible côté UI)."""


def read_interpreter(value: str | None) -> str:
    """Interprète effectif : ``none`` (ancienne valeur) est lu comme ``local`` (E-7)."""
    return "local" if value in (None, "", "none") else value


def max_items_cap(caution: str) -> int:
    return config.CAUTION_LEVELS[caution].max_items


def _question_key(question: str) -> str:
    return nlp.normalize(str(question))[:200]


# ----------------------------------------------------------------- réglages
def load_settings() -> dict:
    with user_transaction() as (s, uid):
        values = settings_repo.get_settings(s, uid)
        answers = settings_repo.get_answers(s, uid)
    values["interpreter"] = read_interpreter(values["interpreter"])
    return {**DEFAULT_SETTINGS, **values, "answers": answers}


def _int(value, name: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except TypeError, ValueError:
        raise SettingsError(f"{name} doit être un entier.") from None
    if not low <= number <= high:
        raise SettingsError(f"{name} doit être compris entre {low} et {high}.")
    return number


def _max_items(value) -> int:
    """Entier ≥ 1 (G-11) ; le plafond dépend de la prudence (``_check_max_items``)."""
    if isinstance(value, bool):
        raise SettingsError("max_items doit être un entier.")
    try:
        number = int(value)
    except TypeError, ValueError:
        raise SettingsError("max_items doit être un entier.") from None
    if number < 1:
        raise SettingsError("max_items doit être au moins 1.")
    return number


def _check_max_items(max_items: int | None, caution: str) -> None:
    """``max_items`` ≤ plafond de la prudence choisie (I-07, E-3)."""
    cap = max_items_cap(caution)
    if max_items is not None and max_items > cap:
        raise SettingsError(f"max_items ne peut pas dépasser {cap} en prudence « {caution} ».")


def _home_location(value) -> str:
    """Lieu : espaces de bord retirés, espaces internes réduits, ≤ 80, sans saut de ligne ni caractère de contrôle."""
    text = str(value or "")
    if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in text):
        raise SettingsError("home_location ne doit contenir ni saut de ligne ni caractère de contrôle.")
    text = " ".join(text.split())
    if len(text) > HOME_LOCATION_MAX:
        raise SettingsError(f"home_location ne peut pas dépasser {HOME_LOCATION_MAX} caractères.")
    return text


def _timezone(value) -> str:
    """Vide ou nom de fuseau IANA connu de ``zoneinfo``."""
    name = str(value or "").strip()
    if not name:
        return ""
    try:
        ZoneInfo(name)
    except Exception:  # noqa: BLE001 - ZoneInfoNotFoundError, ValueError (nom invalide), OSError (fichier)
        raise SettingsError("timezone doit être un fuseau horaire IANA valide (ex. Europe/Paris) ou vide.") from None
    return name


def validate_settings(update: dict, current: dict | None = None) -> dict:
    """Valide un sous-ensemble de réglages ; retourne uniquement les clés connues, normalisées.

    ``max_items`` est contrôlé contre la prudence de ``update`` ou, à défaut, celle de ``current`` (lue en base
    si absente).
    """
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
            value = None if value in (None, "") else _max_items(value)
        elif key == "apply_min_score":
            try:
                value = float(value)
            except TypeError, ValueError:
                raise SettingsError("apply_min_score doit être un nombre.") from None
            if not 0 <= value <= 100:
                raise SettingsError("apply_min_score doit être compris entre 0 et 100.")
        elif key in ("apply_daily_max", "message_daily_max"):
            value = _int(value, key, 0, 200)
        elif key in ("remote_only", "linkedin_skill_sync"):
            value = bool(value)
        elif key == "contact_email":
            value = str(value or "").strip()
            if value and not logmask.EMAIL_ADDRESS.match(value):
                raise SettingsError("contact_email doit être une adresse email valide.")
        elif key == "home_location":
            value = _home_location(value)
        elif key == "timezone":
            value = _timezone(value)
        elif key == "answers":
            if not isinstance(value, dict):
                raise SettingsError("answers doit être un objet question -> réponse.")
            value = {_question_key(q): str(a)[:500] for q, a in value.items() if str(q).strip()}
        else:
            value = str(value or "").strip()
        clean[key] = value
    if clean.get("max_items") is not None:
        caution = clean.get("caution") or (current if current is not None else load_settings())["caution"]
        _check_max_items(clean["max_items"], caution)
    return clean


def save_settings(update: dict) -> dict:
    """Mise à jour partielle ; ``answers`` remplace l'ensemble des réponses mémorisées."""
    with user_transaction() as (s, uid):
        current = settings_repo.get_settings(s, uid, lock=True)
        clean = validate_settings(update, current)
        merged = {**current, **clean}
        _check_max_items(merged["max_items"], merged["caution"])
        settings_repo.update_settings(s, uid, clean)
        if "answers" in clean:
            settings_repo.replace_answers(s, uid, clean["answers"])
    return load_settings()


def remember_answer(question: str, answer: str) -> None:
    with user_transaction() as (s, uid):
        settings_repo.set_answer(s, uid, _question_key(question), str(answer)[:500])


def find_answer(question: str) -> str | None:
    with user_transaction() as (s, uid):
        return settings_repo.get_answer(s, uid, _question_key(question))


# ------------------------------------------------------------ profil (UI)
def load_profile_settings() -> dict:
    with user_transaction() as (s, uid):
        return {**DEFAULT_PROFILE_SETTINGS, **profile_repo.get_overrides(s, uid)}


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
    with user_transaction() as (s, uid):
        profile_repo.save_overrides(s, uid, current)
    return current


def add_skill(skill: str) -> dict:
    name = nlp.normalize(skill)
    if not name:
        raise SettingsError("Compétence vide.")
    current = load_profile_settings()
    added = current["added_skills"] + ([name] if name not in current["added_skills"] else [])
    return save_profile_settings(
        {"added_skills": added, "removed_skills": [s for s in current["removed_skills"] if s != name]}
    )


def remove_skill(skill: str) -> dict:
    name = nlp.normalize(skill)
    current = load_profile_settings()
    removed = current["removed_skills"] + ([name] if name not in current["removed_skills"] else [])
    return save_profile_settings(
        {"added_skills": [s for s in current["added_skills"] if s != name], "removed_skills": removed}
    )


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

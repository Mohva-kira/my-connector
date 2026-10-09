"""Étapes du pipeline réutilisables par le CLI et l'interface web (aucune saisie, aucun ``sys.exit``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from . import activity, config, matching, storage, trends
from . import profile as profile_mod
from .models import Profile, SearchRequest

log = logging.getLogger("linkedin_market")


class ProfileMissingError(RuntimeError):
    """Aucun profil exploitable (ni fichier à la racine, ni profil LinkedIn en cache)."""


class Cancelled(RuntimeError):
    """Tâche annulée par l'utilisateur (vérifiée entre deux sources)."""


def scrape(
    request: SearchRequest,
    want_profile: bool,
    *,
    interactive: bool = True,
    on_manual_check: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_ready: Callable[[], None] | None = None,
) -> str:
    """Collecte LinkedIn et l'ajoute à l'historique ; retourne le texte du profil si demandé."""
    from .browser import create_driver
    from .scraper import LinkedInScraper
    from .session import login

    activity.guard_not_blocked()
    driver = create_driver()
    profile_text = ""
    try:
        login(driver, interactive=interactive, on_manual_check=on_manual_check)
        if on_ready:
            on_ready()  # connexion (et vérification manuelle éventuelle) terminée
        scraper = LinkedInScraper(driver, request)
        if want_profile:
            log.info("Lecture de ton profil LinkedIn...")
            profile_text = scraper.own_profile_text()
            if profile_text:
                storage.save_linkedin_profile_text(profile_text)
        for label, collect in (("offres", scraper.jobs), ("posts", scraper.posts), ("profils", scraper.people)):
            if should_stop and should_stop():
                raise Cancelled("Collecte annulée.")
            log.info("Collecte des %s...", label)
            items = collect()
            storage.append_items(items)
            log.info("%d %s enregistrés.", len(items), label)
            if scraper._blocked():
                activity.record("blocked", where="collecte")
                break
    finally:
        driver.quit()
    return profile_text


def build_profile(request: SearchRequest, linkedin_text: str = "", use_overrides: bool = True) -> Profile:
    path = profile_mod.find_profile_file()
    if not linkedin_text:
        linkedin_text = storage.load_linkedin_profile_text()
    if request.profile_source in ("file", "auto") and path:
        prof = profile_mod.from_file(path)
    elif linkedin_text:
        prof = profile_mod.from_text(linkedin_text, "linkedin")
    elif path:
        prof = profile_mod.from_file(path)
    else:
        raise ProfileMissingError(
            "Aucun profil : dépose profil.md / profil.json / cv.pdf à la racine du projet, "
            "ou importe ton profil LinkedIn (sans --no-scrape)."
        )
    return profile_mod.finalize(prof, request, use_overrides)


def analyse(request: SearchRequest, prof: Profile) -> tuple[dict, list]:
    items = storage.load_items(request.keyword)
    jobs = [i for i in items if i.kind == "job"]
    counts = {s: sum(1 for i in items if i.kind == s) for s in ("job", "post", "person")}
    log.info("Analyse de %d éléments (%s).", len(items), counts)

    model = matching.MatchingModel.load()
    features = matching.build_features(jobs, prof, request.keyword)
    scored = matching.score_jobs(features, model)
    demand = trends.skill_demand(items)
    rising = trends.rising_terms(items)
    weekly_skills = [r["term"] for r in rising.get("skills", []) if r["growth"] > 0][:3] or [
        d["skill"] for d in demand[:3]
    ]
    results = {
        "generated_at": datetime.now().isoformat(),
        "window_days": config.TREND_WINDOW_DAYS,
        "request": request.__dict__,
        "profile": {
            "origin": prof.origin,
            "title": prof.title,
            "skills": prof.skills,
            "years_experience": prof.years_experience,
            "seniority": prof.seniority,
        },
        "counts": counts,
        "top_terms": trends.top_terms(items),
        "skill_demand": demand,
        "rising": rising,
        "weekly": trends.weekly_series(items, weekly_skills),
        "gap": matching.skill_gap(scored, prof),
        "top_jobs": scored,
        "model": {
            "trained": model.trained,
            "n_labels": model.n_labels,
            "train_accuracy": model.train_accuracy,
            "importance": model.importance(),
            "loss_history": model.loss_history,
            "trained_at": model.trained_at,
        },
        "min_labels": matching.MIN_LABELS,
    }
    return results, scored


def summary_for_interpreter(results: dict) -> dict:
    """Agrégats seulement : aucun texte brut de post ou de profil."""
    return {
        "mot_cle": results["request"]["keyword"],
        "full_remote": results["request"]["remote_only"],
        "profil": results["profile"],
        "counts": results["counts"],
        "top_terms": results["top_terms"][:25],
        "skill_demand": results["skill_demand"][:25],
        "rising": results["rising"],
        "gap": {**results["gap"], "skills": results["gap"]["skills"][:25]},
        "top_jobs": [
            {k: j[k] for k in ("title", "company", "location", "score", "missing_skills", "criteria")}
            for j in results["top_jobs"][:10]
        ],
        "model": {k: results["model"][k] for k in ("trained", "n_labels", "train_accuracy", "importance")},
    }

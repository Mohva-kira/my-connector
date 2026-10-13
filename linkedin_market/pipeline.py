"""Étapes du pipeline réutilisables par le CLI et l'interface web (aucune saisie, aucun ``sys.exit``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime

from . import activity, config, consent, matching, sources, storage, trends
from . import profile as profile_mod
from .models import Profile, SearchRequest

log = logging.getLogger("linkedin_market")


class ProfileMissingError(RuntimeError):
    """Aucun profil exploitable (ni fichier à la racine, ni profil LinkedIn en cache)."""


class Cancelled(RuntimeError):
    """Collecte annulée par l'utilisateur (``should_stop`` testé avant chaque page, E-22).

    Interceptée par ``TaskManager._run`` : la tâche finit ``annulée`` (« Tâche annulée. »).
    """


@dataclass
class ScrapeOutcome:
    """Résultat d'une collecte : texte du profil LinkedIn (si demandé) et blocage éventuel (E-4, option A)."""

    profile_text: str = ""
    blocked: bool = False
    items: int = 0  # éléments enregistrés (offres, posts et profils)


# Statuts d'une source dans le résultat d'une collecte (``backend/01`` § 5.8.2).
STATUS_OK, STATUS_BLOCKED, STATUS_UNAVAILABLE, STATUS_CANCELLED = "ok", "bloquée", "indisponible", "annulée"


@dataclass
class SourceReport:
    """Issue de la collecte d'une source : ``{"code", "items", "status", "detail"}`` (``detail`` vide si ``ok``)."""

    code: str
    items: int = 0
    status: str = STATUS_OK
    detail: str = ""

    def public(self) -> dict:
        return {"code": self.code, "items": self.items, "status": self.status, "detail": self.detail}


@dataclass
class CollectOutcome:
    """Résultat de ``collect`` : profil LinkedIn, une entrée par source demandée, sources bloquées."""

    profile_text: str = ""
    sources: list[SourceReport] = field(default_factory=list)

    @property
    def blocked_sources(self) -> list[str]:
        return [r.code for r in self.sources if r.status == STATUS_BLOCKED]

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_sources)


def capped(request: SearchRequest) -> SearchRequest:
    """``max_items`` ramené au plafond de la prudence (I-07, E-3) : garde-fou après le contrôle 422 de l'API."""
    caution = config.CAUTION_LEVELS.get(request.caution, config.CAUTION_LEVELS[config.DEFAULT_CAUTION])
    if request.max_items <= caution.max_items:
        return request
    log.warning(
        "max_items ramené de %d à %d : plafond de la prudence « %s ».",
        request.max_items,
        caution.max_items,
        caution.name,
    )
    return replace(request, max_items=caution.max_items)


def scrape(
    request: SearchRequest,
    want_profile: bool,
    *,
    interactive: bool = True,
    on_manual_check: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_ready: Callable[[], None] | None = None,
) -> ScrapeOutcome:
    """Collecte LinkedIn ajoutée à l'historique ; texte du profil si demandé, et blocage éventuel.

    Blocage (checkpoint, authwall) : arrêt immédiat de la navigation, événement ``blocked`` (``where="collecte"``)
    et repos de 6 h (I-06) ; les éléments déjà collectés sont gardés et l'analyse porte sur eux (E-4, option A).
    Annulation (``should_stop``, testé avant chaque page) : les éléments déjà collectés sont enregistrés, puis
    ``Cancelled`` est levée. Consentement (I-16) revérifié ici : seul point d'entrée de la collecte LinkedIn, pour
    le CLI comme pour la tâche web (``consent.SourceConsentRequired`` si révoqué entre-temps).
    """
    from .browser import create_driver
    from .scraper import LinkedInScraper
    from .session import login

    request = capped(request)
    consent.require("linkedin")
    activity.guard_not_blocked()

    def check() -> None:
        if should_stop and should_stop():
            raise Cancelled("Collecte annulée.")

    check()
    driver = create_driver()
    outcome = ScrapeOutcome()
    try:
        login(driver, interactive=interactive, on_manual_check=on_manual_check)
        if on_ready:
            on_ready()  # connexion (et vérification manuelle éventuelle) terminée
        scraper = LinkedInScraper(driver, request, check=check)
        saved = 0
        if want_profile:
            log.info("Lecture de ton profil LinkedIn...")
            outcome.profile_text = scraper.own_profile_text()
            if outcome.profile_text:
                storage.save_linkedin_profile_text(outcome.profile_text)
        for label, collect in (("offres", scraper.jobs), ("posts", scraper.posts), ("profils", scraper.people)):
            if scraper.blocked:
                break
            check()
            log.info("Collecte des %s...", label)
            try:
                items = collect()
            except Cancelled:
                if scraper.partial:
                    storage.append_items(scraper.partial)
                    log.info("Collecte annulée : %d %s déjà lus enregistrés.", len(scraper.partial), label)
                raise
            storage.append_items(items)
            saved += len(items)
            log.info("%d %s enregistrés.", len(items), label)
            if scraper.blocked or scraper._blocked():
                break
        if scraper.blocked:
            activity.record("blocked", where="collecte", source="linkedin")
            outcome.blocked = True
        outcome.items = saved
    finally:
        driver.quit()
    return outcome


def guard_source_rest(source_code: str) -> None:
    """Repos de 6 h de ``source_code`` (I-06) : ``activity.BlockedByLinkedIn`` avec le nom de **cette** source.

    Pour LinkedIn, le message d'``activity.guard_not_blocked`` est conservé tel quel.
    """
    if source_code == "linkedin":
        activity.guard_not_blocked()
        return
    found = activity.block_status(source=source_code)
    if found is None:
        return
    label = sources.label_of(source_code)
    until = datetime.fromisoformat(found["until"].replace("Z", "+00:00")).astimezone()
    raise activity.BlockedByLinkedIn(
        f"{label} a interrompu la navigation ({found['where'] or 'navigation'}). Laisse passer du temps "
        f"(jusqu'à {until:%Y-%m-%d %H:%M}) ou lève le blocage de cette source dans Paramètres → Sources et "
        "consentements après avoir vérifié ton compte dans un navigateur normal."
    )


def collect(
    request: SearchRequest,
    source_codes: list[str],
    want_profile: bool,
    *,
    interactive: bool = True,
    on_manual_check: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_ready: Callable[[], None] | None = None,
) -> CollectOutcome:
    """Collecte les sources demandées, dans l'ordre, et ajoute les éléments à l'historique (vague 5).

    - Préconditions revérifiées ici avant toute collecte (I-16, I-06) : consentement et repos de 6 h de chaque source
      en mode navigateur (``consent.SourceConsentRequired`` / ``activity.BlockedByLinkedIn`` font échouer la tâche).
    - Une source bloquée (navigateur) s'arrête seule : événement ``blocked`` (``where="collecte"``, ``source``),
      repos de 6 h, aucun contournement ; les autres continuent. France Travail en échec réseau (``SourceFetchError``)
      ou sans identifiants est ignorée et journalisée (statut ``indisponible``).
    - Annulation : les éléments déjà lus sont enregistrés, puis ``Cancelled`` est relancée aussitôt.
    - ``want_profile`` ne concerne que LinkedIn. Malt, WTTJ et France Travail ne collectent que des offres (I-09, I-11).
    """
    request = capped(request)
    codes = list(dict.fromkeys(source_codes))
    for code in codes:
        consent.require(code)  # sans effet hors mode navigateur
        if code in ("linkedin", *sources.BROWSER_SOURCES):
            guard_source_rest(code)

    def check() -> None:
        if should_stop and should_stop():
            raise Cancelled("Collecte annulée.")

    outcome = CollectOutcome()
    if on_ready:
        on_ready()  # rien à attendre avant la première source : la tâche passe en cours
    for code in codes:
        check()
        if code == "linkedin":
            result = scrape(
                request,
                want_profile,
                interactive=interactive,
                on_manual_check=on_manual_check,
                should_stop=should_stop,
                on_ready=on_ready,
            )
            outcome.profile_text = result.profile_text
            report = SourceReport(code, result.items)
            if result.blocked:
                report.status = STATUS_BLOCKED
                report.detail = "LinkedIn a interrompu la navigation : repos de 6 h pour cette source."
        elif code == "france_travail":
            report = _collect_france_travail(request)
        elif code in sources.BROWSER_SOURCES:
            report = _collect_browser(code, request, check)
        else:
            report = SourceReport(code, 0, STATUS_UNAVAILABLE, "Cette source ne se collecte pas.")
        if report.status != STATUS_OK:
            log.warning("%s : %s", sources.label_of(code), report.detail)
        outcome.sources.append(report)
    return outcome


def _collect_france_travail(request: SearchRequest) -> SourceReport:
    """France Travail (API officielle) : source ignorée sans identifiants ou en échec, jamais d'arrêt des autres."""
    if not config.france_travail_configured():
        return SourceReport(
            "france_travail",
            0,
            STATUS_UNAVAILABLE,
            "Identifiants France Travail absents : renseigne FRANCE_TRAVAIL_CLIENT_ID et "
            "FRANCE_TRAVAIL_CLIENT_SECRET dans le fichier .env.",
        )
    log.info("Collecte des offres France Travail...")
    try:
        client = sources.france_travail_source()
        try:
            items = client.fetch(request)
        finally:
            client.close()
    except sources.SourceFetchError as exc:
        return SourceReport("france_travail", 0, STATUS_UNAVAILABLE, str(exc))
    storage.append_items(items)
    log.info("%d offres France Travail enregistrées.", len(items))
    return SourceReport("france_travail", len(items))


def _collect_browser(code: str, request: SearchRequest, check: Callable[[], None]) -> SourceReport:
    """Malt / WTTJ : un navigateur par source (fermé en ``finally``), lecture seule des offres."""
    from .browser import create_driver

    label = sources.label_of(code)
    driver = create_driver()
    try:
        source = sources.BROWSER_SOURCES[code](driver, request, check=check)
        log.info("Collecte des offres %s...", label)
        try:
            items = source.collect()
        except Cancelled:
            if source.partial:
                storage.append_items(source.partial)
                log.info("Collecte annulée : %d offres %s déjà lues enregistrées.", len(source.partial), label)
            raise
        storage.append_items(items)
        log.info("%d offres %s enregistrées.", len(items), label)
        if source.blocked:
            activity.record("blocked", where="collecte", source=code)
            return SourceReport(
                code,
                len(items),
                STATUS_BLOCKED,
                f"{label} a demandé une connexion ou une vérification : repos de 6 h pour cette source.",
            )
        return SourceReport(code, len(items))
    finally:
        driver.quit()


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

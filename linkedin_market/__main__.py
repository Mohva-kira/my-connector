"""Point d'entrée : ``python -m linkedin_market [options]``.

Sans option, un prompt de saisie recueille les paramètres. Exemples :

    python -m linkedin_market
    python -m linkedin_market --keyword "Data Engineer" --caution prudent
    python -m linkedin_market --keyword "Data Engineer" --no-scrape --label 10
    python -m linkedin_market --keyword "Data Engineer" --no-scrape --label 0 --apply
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import webbrowser
from typing import NoReturn

from . import (
    activity,
    application,
    config,
    consent,
    cv_format,
    interpreter,
    matching,
    report,
    retention,
    settings,
    storage,
)
from . import profile as profile_mod
from .db.bootstrap import prepare_database
from .models import Profile, SearchRequest
from .pipeline import (
    STATUS_BLOCKED,
    STATUS_OK,
    CollectOutcome,
    ProfileMissingError,
    SourceRefused,
    analyse,
    batch_scorer,
    build_profile,
    check_cli_sources,
    collect,
    summary_for_interpreter,
)
from .session import VerificationNotDoneError
from .sources import label_of

log = logging.getLogger("linkedin_market")


# ----------------------------------------------------------------- saisie
def _ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix} : ").strip()
    return answer or default


def _ask_yes(question: str, default: bool = True) -> bool:
    answer = _ask(question + (" (O/n)" if default else " (o/N)")).lower()
    return default if not answer else answer.startswith(("o", "y"))


def collect_request(args: argparse.Namespace) -> SearchRequest:
    interactive = not args.no_interactive
    if interactive:
        print("\n=== Analyse du marché LinkedIn (freelance, full remote) ===\n")

    keyword = args.keyword or (interactive and _ask("Mot-clé du poste visé (ex. « Data Engineer »)"))
    if not keyword:
        sys.exit("Un mot-clé de poste est obligatoire (--keyword).")

    remote = not args.onsite
    location = args.location
    caution = args.caution
    max_items = args.max_items
    profile_source = args.profile_source
    extra_skills: list[str] = args.skills or []
    years = args.years
    seniority = args.seniority or ""

    if interactive:
        if not args.onsite:
            remote = _ask_yes("Uniquement les offres en full remote ?", True)
        if location is None:
            location = _ask("Localisation des offres (vide = monde entier)", "")
        if caution is None:
            caution = _ask("Niveau de prudence anti-blocage : prudent / normal / agressif", config.DEFAULT_CAUTION)
        if max_items is None:
            default_max = config.CAUTION_LEVELS.get(caution, config.CAUTION_LEVELS["normal"]).max_items
            max_items = int(
                _ask("Nombre d'éléments par source (offres, posts, profils)", str(default_max)) or default_max
            )
        if profile_source == "auto":
            found = profile_mod.find_profile_file()
            label = f"fichier ({found.name}) / linkedin" if found else "fichier / linkedin (aucun fichier trouvé)"
            answer = _ask(f"Source du profil : {label}", "fichier" if found else "linkedin")
            profile_source = "file" if answer.lower().startswith("f") else "linkedin"
        if not extra_skills:
            raw = _ask("Compétences à ajouter à ton profil (séparées par des virgules, facultatif)", "")
            extra_skills = [s.strip() for s in raw.split(",") if s.strip()]
        if years is None:
            raw = _ask("Années d'expérience (facultatif)", "")
            years = int(raw) if raw.isdigit() else None
        if not seniority:
            seniority = _ask("Séniorité : junior / confirme / senior / lead (facultatif)", "")

    caution = caution if caution in config.CAUTION_LEVELS else config.DEFAULT_CAUTION
    return SearchRequest(
        keyword=keyword.strip(),
        location=(location or "").strip(),
        remote_only=remote,
        caution=caution,
        max_items=max_items or config.CAUTION_LEVELS[caution].max_items,
        profile_source=profile_source,
        extra_skills=extra_skills,
        years_experience=years,
        seniority=_parse_seniority(seniority),
    )


def _parse_seniority(raw: str) -> str:
    """« senior et lead » -> « lead » : le plus haut niveau reconnu."""
    levels = matching.nlp.SENIORITY_LEVELS
    found = [w for w in re.split(r"[^a-z]+", matching.nlp.normalize(raw)) if w in levels]
    return max(found, key=levels.get) if found else ""


# ------------------------------------------------------------- annotation
def label_jobs(scored: list[dict], count: int) -> int:
    """Demande « pertinente / pas pertinente » sur les offres les plus incertaines."""
    labels = storage.load_labels()
    to_label = matching.uncertain_jobs(scored, labels, count)
    if not to_label:
        return 0
    print(
        f"\nAnnotation de {len(to_label)} offres (p = pertinente, n = pas pertinente, Entrée = passer, q = arrêter)\n"
    )
    added = 0
    for index, job in enumerate(to_label, 1):
        print(application.format_card(job, index))
        answer = input("  Pertinente ? (p/n) ").strip().lower()
        print()
        if answer == "q":
            break
        if answer in ("p", "o", "y", "1"):
            labels[job["id"]] = 1
        elif answer in ("n", "0"):
            labels[job["id"]] = 0
        else:
            continue
        added += 1
    storage.save_labels(labels)
    return added


# ------------------------------------------------------------- candidature
def _parse_choice(raw: str, size: int) -> list[int]:
    """« 1,3-5 » ou « tous » -> indices (base 0) valides."""
    raw = raw.strip().lower()
    if raw in ("tous", "all", "*"):
        return list(range(size))
    picked: list[int] = []
    for part in re.split(r"[,\s]+", raw):
        bounds = part.split("-")
        if all(b.isdigit() for b in bounds) and 1 <= len(bounds) <= 2:
            lo, hi = int(bounds[0]), int(bounds[-1])
            picked += [i - 1 for i in range(lo, hi + 1) if 1 <= i <= size and i - 1 not in picked]
    return picked


def prepare_applications(
    request: SearchRequest, prof: Profile, results: dict, scored: list[dict], args: argparse.Namespace
) -> None:
    """Propose les offres pertinentes, prépare CV + message, ouvre l'offre, suit le statut."""
    tracked = application.load_applications()
    candidates = application.relevant_jobs(scored, storage.load_labels(), args.apply_min_score)
    hidden = max(len(candidates) - config.APPLY_MAX_LIST, 0)
    candidates = candidates[: config.APPLY_MAX_LIST]
    if not candidates:
        print(f"\nAucune offre pertinente (annotée « p » ou score >= {args.apply_min_score:.0f} %).")
        return
    print(f"\n=== Offres pertinentes ({len(candidates)}) ===\n")
    for index, job in enumerate(candidates, 1):
        print(application.format_card(job, index))
        if job["id"] in tracked:
            print(f"         Candidature : {tracked[job['id']]['status']} ({tracked[job['id']]['folder']})")
        print()
    if hidden:
        print(f"({hidden} autres offres au-dessus du seuil : relève --apply-min-score pour affiner.)\n")

    chosen = _parse_choice(
        _ask("Préparer une candidature pour quelles offres ? (ex. 1,3 · tous · Entrée = aucune)"), len(candidates)
    )
    items = {i.id: i for i in storage.load_items(request.keyword, window_days=3650)}
    prefs = settings.load_settings()
    names = application.known_contact_names()
    for idx in chosen:
        job = candidates[idx]
        item = items.get(job["id"])
        if item is None:
            log.warning("Texte de l'offre introuvable dans l'historique : %s", job["title"])
            continue
        print(f"\nPréparation : {job['title']} — {job['company']}...")
        app = application.generate(
            prof,
            item,
            job,
            results,
            args.interpreter,
            known_names=names,
            home_location=prefs["home_location"],
            timezone=prefs["timezone"],
            identity=cv_format.identity_from(prefs),
        )
        folder = application.write(app, item, job)
        print(
            f"CV ({'rédigé par Claude' if app.generator == 'claude' else 'assemblé localement'}) : {folder / 'cv.html'}"
        )
        if app.message:
            print("\n--- Message de candidature ---\n" + app.message + "\n------------------------------")
        if app.advice:
            print("Conseils :\n" + app.advice)
        if not args.no_open:
            webbrowser.open((folder / "cv.html").as_uri())
            webbrowser.open(job["url"])
        if _ask_yes("As-tu postulé à cette offre ?", False):
            application.set_status(job["id"], "postulée")
            storage.save_labels({**storage.load_labels(), job["id"]: 1})
            print("Statut : postulée.")


def _source_codes(raw: str) -> list[str]:
    """« linkedin, Malt » -> ``["linkedin", "malt"]`` (minuscules, sans doublon, ordre conservé)."""
    codes = list(dict.fromkeys(c.strip().lower() for c in raw.split(",") if c.strip()))
    if not codes:
        raise argparse.ArgumentTypeError("indique au moins un code de source (ex. linkedin,france_travail).")
    return codes


def print_collect_summary(outcome: CollectOutcome) -> None:
    """Récapitulatif par source en sortie standard (``pipeline.collect`` journalise déjà les échecs : pas de doublon).

    Un blocage survenu pendant la collecte n'arrête pas le CLI : le rapport est établi sur les données déjà
    collectées et le code de sortie reste 0 ; seuls les refus avant collecte sortent en code 1.
    """
    print("\nCollecte :")
    for row in outcome.sources:
        label = label_of(row.code)
        if row.status == STATUS_OK:
            print(f"  - {label} : {row.items} éléments enregistrés.")
        elif row.status == STATUS_BLOCKED:
            print(
                f"  - Collecte interrompue par {label} : rapport établi sur les données déjà collectées "
                f"({row.items} éléments enregistrés). Actions navigateur suspendues 6 h pour cette source ; "
                "vérifie ton compte dans un navigateur normal avant de relancer."
            )
        else:
            print(f"  - {label} ignorée : {row.detail}")


def _refuse(exc: Exception) -> NoReturn:
    """Quitte avec un code non nul et un message actionnable (P-3) ; relance ``exc`` si elle n'est pas un refus."""
    if isinstance(exc, consent.SourceConsentRequired):
        # P-3 (décidé le 2026-10-09) : message propre au CLI, distinct du détail HTTP 428 (backend/01 § 5.8.1).
        sys.exit(
            f"Accepte d'abord les risques liés à {exc.label} (version {exc.risk_version}) dans l'interface "
            "web : Paramètres → Sources et consentements."
        )
    if isinstance(exc, SourceRefused | activity.BlockedByLinkedIn | VerificationNotDoneError):
        # Vérification non terminée à la connexion : repos de 6 h déjà posé, navigateur fermé, aucune relance (I-06).
        sys.exit(str(exc))
    raise exc


def apply_update(args: argparse.Namespace) -> None:
    """``--update`` : reprend la recherche de la dernière analyse (mot-clé, lieu, prudence) sans poser de question."""
    reports = storage.list_reports()
    previous = storage.load_report(reports[0]["name"]) if reports else None
    if not previous:
        sys.exit("Aucune analyse à mettre à jour : lance d'abord une analyse avec --keyword.")
    saved = previous.get("request", {})
    args.keyword = args.keyword or saved.get("keyword")
    if args.location is None:
        args.location = saved.get("location", "")
    args.onsite = args.onsite or not saved.get("remote_only", True)
    args.caution = args.caution or saved.get("caution")
    args.max_items = args.max_items or saved.get("max_items")
    args.no_interactive = True
    args.no_scrape = False
    log.info("Mise à jour de « %s » : nouvelle collecte puis réanalyse.", args.keyword)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="linkedin_market", description=__doc__.split("\n")[0])
    p.add_argument("--keyword", "-k", help="mot-clé du poste visé")
    p.add_argument("--location", help="localisation des offres (défaut : monde entier)")
    p.add_argument("--onsite", action="store_true", help="ne pas filtrer sur le full remote")
    p.add_argument("--caution", choices=list(config.CAUTION_LEVELS), help="niveau anti-blocage")
    p.add_argument("--max-items", type=int, help="éléments par source (défaut : 100)")
    p.add_argument("--profile-source", choices=["auto", "file", "linkedin"], default="auto")
    p.add_argument(
        "--skills",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        help="compétences à ajouter, séparées par des virgules",
    )
    p.add_argument("--years", type=int, help="années d'expérience")
    p.add_argument("--seniority", choices=list(matching.nlp.SENIORITY_LEVELS))
    p.add_argument(
        "--sources",
        type=_source_codes,
        default=["linkedin"],
        metavar="CODES",
        help=(
            "sources à collecter, codes séparés par des virgules : linkedin, france_travail, malt, wttj "
            "(défaut : linkedin). Malt, WTTJ et LinkedIn exigent d'avoir accepté leurs risques dans l'interface "
            "web (Paramètres → Sources et consentements) ; France Travail exige FRANCE_TRAVAIL_CLIENT_ID et "
            "FRANCE_TRAVAIL_CLIENT_SECRET dans .env ; Indeed s'importe à la main depuis l'interface web. "
            "Une source bloquée est mise au repos 6 h. Sans effet avec --no-scrape."
        ),
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="mettre à jour : relance la collecte avec la recherche de la dernière analyse, sans question",
    )
    p.add_argument("--no-scrape", action="store_true", help="réanalyser l'historique sans collecter de source")
    p.add_argument("--label", type=int, default=None, help="nombre d'offres à annoter (0 = aucune)")
    p.add_argument("--interpreter", choices=list(config.INTERPRETERS), help="défaut : LINKEDIN_INTERPRETER ou claude")
    p.add_argument("--apply", action="store_true", help="aller directement à la préparation des candidatures")
    p.add_argument(
        "--apply-min-score",
        type=float,
        default=config.APPLY_MIN_SCORE,
        help="score minimal d'une offre non annotée pour être proposée (défaut : %(default)s)",
    )
    p.add_argument("--no-interactive", action="store_true", help="aucune question (pour une tâche planifiée)")
    p.add_argument("--no-open", action="store_true", help="ne pas ouvrir le rapport dans le navigateur")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    config.ensure_data_dirs()

    prepare_database()  # schéma, utilisateur, import des fichiers historiques ; base injoignable => arrêt
    try:
        retention.purge_current_user()  # purge de rétention (I-10, I-14), synchrone, une fois au démarrage
    except Exception as exc:  # noqa: BLE001 - une purge en échec ne doit pas empêcher l'analyse
        log.warning("Purge de rétention impossible : %s", exc.__class__.__name__)
    prefs = settings.load_settings()
    if args.update:
        apply_update(args)
    if not args.no_scrape:
        try:
            check_cli_sources(args.sources)  # avant toute saisie : refus immédiat, code de sortie non nul
        except (SourceRefused, consent.SourceConsentRequired, activity.BlockedByLinkedIn) as exc:
            _refuse(exc)
    request = collect_request(args)

    linkedin_text = ""
    if not args.no_scrape:
        want_profile = "linkedin" in args.sources and (
            request.profile_source == "linkedin"
            or (request.profile_source == "auto" and profile_mod.find_profile_file() is None)
        )
        try:
            outcome = collect(
                request,
                args.sources,
                want_profile,
                score_batch=batch_scorer(request, prefs["home_location"], prefs["timezone"]),
            )
        except (consent.SourceConsentRequired, activity.BlockedByLinkedIn, VerificationNotDoneError) as exc:
            _refuse(exc)  # révoqué ou mis au repos entre la vérification et la collecte
        linkedin_text = outcome.profile_text
        print_collect_summary(outcome)
    try:
        prof = build_profile(request, linkedin_text)
    except ProfileMissingError as exc:
        sys.exit(str(exc))
    log.info("Profil (%s) : %d compétences détectées.", prof.origin, len(prof.skills))

    results, scored = analyse(request, prof, prefs["home_location"], prefs["timezone"])

    label_count = args.label
    if label_count is None and not args.no_interactive and scored:
        label_count = 8 if _ask_yes("Annoter quelques offres pour entraîner le matching ?", True) else 0
    if label_count and scored:
        if label_jobs(scored, label_count):
            model = matching.train_from_labels(
                prof,
                request.keyword,
                storage.load_labels(),
                storage.load_labeled_items(),
                prefs["home_location"],
                prefs["timezone"],
            )
            if model:
                log.info(
                    "Modèle réentraîné sur %d annotations (précision %.0f %%).",
                    model.n_labels,
                    (model.train_accuracy or 0) * 100,
                )
                results, scored = analyse(request, prof, prefs["home_location"], prefs["timezone"])
            else:
                log.info(
                    "Pas encore assez d'annotations (min. %d, des deux types) pour entraîner.", matching.MIN_LABELS
                )

    text, used = interpreter.interpret(
        summary_for_interpreter(results), args.interpreter, prefs["home_location"], prefs["timezone"]
    )
    results["interpretation"] = text
    results["interpreter"] = used
    results["interpreter_model"] = config.anthropic_model() if used == "claude" else ""

    html_path, json_path = report.write(results)
    print(f"\nRapport : {html_path}\nNom du rapport : {json_path.name}")
    if not args.no_open and not args.no_interactive:
        webbrowser.open(html_path.as_uri())

    if (
        not args.no_interactive
        and scored
        and (args.apply or _ask_yes("Préparer des candidatures pour les offres pertinentes ?", True))
    ):
        prepare_applications(request, prof, results, scored, args)


if __name__ == "__main__":
    main()

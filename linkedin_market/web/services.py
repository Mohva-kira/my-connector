"""Logique métier de l'interface web : enchaîne les modules existants sans saisie interactive."""

from __future__ import annotations

import os
import re
import threading
from email.utils import parseaddr
from uuid import UUID

from .. import (
    activity,
    application,
    apply_bot,
    config,
    consent,
    cv_export,
    cv_format,
    interpreter,
    logmask,
    matching,
    outreach,
    pipeline,
    profile_sync,
    report,
    session,
    settings,
    storage,
)
from .. import (
    contacts as contacts_mod,
)
from ..db import utc_iso, utcnow
from ..db.repositories import applications as applications_repo
from ..db.repositories import outreach as outreach_repo
from ..db.repositories import settings as settings_repo
from ..db.repositories import sources as sources_repo
from ..db.repositories import user_transaction
from ..models import SearchRequest
from ..sources import label_of, source_of_ref
from .tasks import RUNNING, TaskContext


class NoAnalysisError(LookupError):
    """Aucune analyse disponible : lancer d'abord une collecte ou une analyse."""


# ------------------------------------------------------------ paramètres
def build_request(overrides: dict | None = None) -> SearchRequest:
    values = {**settings.load_settings(), **settings.validate_settings(overrides or {})}
    keyword = (values.get("keyword") or "").strip()
    if not keyword:
        raise ValueError("Renseigne le mot-clé du poste visé (Paramètres ou lancement d'analyse).")
    caution = values["caution"]
    cap = config.CAUTION_LEVELS[caution].max_items
    return SearchRequest(
        keyword=keyword,
        location=values["location"],
        remote_only=values["remote_only"],
        caution=caution,
        # Un max_items enregistré pour une prudence plus large est ramené au plafond de la prudence demandée (I-07).
        max_items=min(values["max_items"] or cap, cap),
        profile_source=values["profile_source"],
    )


def validate_run_sources(codes: list[str]) -> list[str]:
    """Valide ``RunBody.sources`` pour une collecte ; renvoie les sources demandées en mode ``navigateur``.

    Erreurs (``ValueError``, converties en ``422`` par le routeur, dans l'ordre rencontré, I-16/vague 3) :
    source inconnue, source ``import_manuel`` (Indeed s'importe à la main) puis source non encore activée.
    """
    with user_transaction() as (s, _uid):
        known = {source.code: source for source in sources_repo.all_sources(s)}
    navigate: list[str] = []
    for code in codes:
        source = known.get(code)
        if source is None:
            raise ValueError(f"sources : source inconnue « {code} ».")
        if source.access_mode == "import_manuel":
            raise ValueError("sources : Indeed s'importe à la main, elle ne se collecte pas.")
        if not source.enabled:
            raise ValueError(f"sources : la source « {source.label} » n'est pas encore disponible.")
        if source.access_mode == "navigateur":
            navigate.append(code)
    return navigate


# ------------------------------------------------------------- résultats
def latest_results(keyword: str | None = None) -> dict | None:
    for row in storage.list_reports():
        if keyword and row["keyword"].lower() != keyword.lower():
            continue
        data = storage.load_report(row["name"])
        if data:
            data["_report"] = row["name"]
            return data
    return None


def require_results() -> dict:
    results = latest_results()
    if results is None:
        raise NoAnalysisError("Aucune analyse disponible : lance d'abord une analyse.")
    return results


def find_job(results: dict, job_id: str) -> dict | None:
    return next((j for j in results.get("top_jobs", []) if j["id"] == job_id), None)


def tidy(text: str, limit: int, *, drop: bool = False) -> str:
    """Titre/entreprise lisibles : d'anciennes collectes y ont stocké du texte de page entier."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return "" if drop else text[: limit - 1].rstrip() + "…"


# Régions telles que LinkedIn les affiche (interface en français) à la fin de l'en-tête d'une offre.
_REGIONS = (
    "Europe, Moyen-Orient et Afrique",
    "Afrique subsaharienne",
    "Amérique du Nord",
    "Amérique latine",
    "Asie-Pacifique",
    "Union européenne",
    "Moyen-Orient",
    "Monde entier",
)
_POSTED = re.compile(r"(?:Republication )?il y a \d+ \w+")
_APPLICANTS = re.compile(r"(?:Plus de )?\d+ (?:candidatures|personnes ont cliqué sur Postuler)")


def repair_identity(job: dict) -> dict:
    """Titre, entreprise et lieu lisibles.

    D'anciennes collectes ont stocké l'entreprise comme titre et l'en-tête entier de la page comme entreprise :
    « Entreprise Titre Lieu · il y a 1 jour · Plus de 100 candidatures … ». On en retire aussi l'ancienneté de
    l'offre et le nombre de candidats, utiles pour juger de la concurrence.
    """
    title = " ".join((job.get("title") or "").split())
    company = " ".join((job.get("company") or "").split())
    if " · " not in company or not title or not company.startswith(title):
        return {**job, "title": tidy(title, 150), "company": tidy(company, 80, drop=True)}
    head, _, meta = company.partition(" · ")
    rest, location = head[len(title) :].strip(), ""
    for region in _REGIONS:
        if rest.endswith(region):
            rest, location = rest[: -len(region)].strip(), region
            break
    posted, applicants = _POSTED.search(meta), _APPLICANTS.search(meta)
    return {
        **job,
        "title": tidy(rest or title, 150),
        "company": tidy(title, 80, drop=True),
        "location": location,
        "posted_label": posted.group(0) if posted else "",
        "applicants": applicants.group(0) if applicants else "",
    }


def _overview_rows(jobs: list[dict], *, provisional: bool = False) -> list[dict]:
    labels = storage.load_labels()
    apps = application.load_applications()
    rows = []
    for job in jobs:
        entry = apps.get(job["id"])
        rows.append(
            {
                **repair_identity(job),
                # Un rapport antérieur à la vague 5 n'a pas de source_code (jamais réécrit, I-02) : LinkedIn.
                "source_code": job.get("source_code") or "linkedin",
                "label": labels.get(job["id"]),
                "application_status": entry["status"] if entry else None,
                **({"provisional": True} if provisional else {}),
            }
        )
    return rows


def jobs_overview(results: dict | None, live: dict | None = None) -> list[dict]:
    """Offres du dernier rapport ; ``live`` (offres déjà scorées par une collecte en cours) s'y ajoute."""
    rows = _overview_rows(results.get("top_jobs", [])) if results else []
    if live and live["rows"]:
        fresh = _overview_rows(list(live["rows"].values()), provisional=True)
        ids = {r["id"] for r in fresh}
        rows = sorted([r for r in rows if r["id"] not in ids] + fresh, key=lambda r: r["score"], reverse=True)
    return rows


# ------------------------------------------------- affichage progressif d'une collecte
_live_lock = threading.Lock()
_live: dict | None = None


def live_start(keyword: str) -> None:
    global _live
    with _live_lock:
        _live = {"keyword": keyword, "rows": {}}


def live_add(rows: list[dict]) -> None:
    """Offres scorées d'un lot, visibles dans ``GET /api/jobs`` avant la fin de l'analyse."""
    with _live_lock:
        if _live is not None:
            _live["rows"].update({row["id"]: row for row in rows})


def live_stop() -> None:
    global _live
    with _live_lock:
        _live = None


def live_snapshot() -> dict | None:
    with _live_lock:
        return {"keyword": _live["keyword"], "rows": dict(_live["rows"])} if _live else None


class RunProgress:
    """Avancement d'une analyse (étapes et lot en cours), publié dans ``meta.progress``."""

    def __init__(self, ctx: TaskContext, scrape: bool):
        self._ctx = ctx
        steps = [("collecte", "Collecte par lots")] if scrape else []
        steps += [("analyse", "Analyse et scoring"), ("interpretation", "Interprétation"), ("rapport", "Rapport")]
        self.steps = [{"key": k, "label": label, "state": "attente"} for k, label in steps]
        self.batch: dict | None = None
        self.scored = 0

    def step(self, key: str) -> None:
        """``key`` passe en cours ; les étapes qui le précèdent sont terminées."""
        reached = True
        for row in self.steps:
            if row["key"] == key:
                row["state"], reached = "en cours", False
            elif reached:
                row["state"] = "terminée"
        self.push()

    def finish(self) -> None:
        for row in self.steps:
            row["state"] = "terminée"
        self.batch = None
        self.push()

    def on_batch(self, state: dict) -> None:
        self.batch = state
        self.push()

    def push(self) -> None:
        self._ctx.set_progress({"steps": self.steps, "batch": self.batch, "scored": self.scored})


# ------------------------------------------------------------- exécution
def run_analysis(
    ctx: TaskContext,
    request: SearchRequest,
    *,
    scrape: bool,
    interpreter_name: str,
    sources: list[str] | None = None,
) -> dict:
    """Collecte (facultative) des ``sources`` demandées, puis analyse et rapport.

    Blocage d'une source pendant la collecte (E-4, option A) : sa navigation s'arrête net (I-06), les autres sources
    continuent, la tâche finit ``terminée`` avec un rapport établi sur les données déjà collectées et
    ``result.blocked = true`` (``blocked_sources`` liste les sources bloquées). Une source en échec réseau (France
    Travail) est ignorée et journalisée. Le profil LinkedIn n'est lu que si ``linkedin`` fait partie des sources.
    """
    codes = list(sources or ["linkedin"])
    progress = RunProgress(ctx, scrape)
    live_start(request.keyword)
    try:
        return _run_analysis(ctx, request, progress, scrape, interpreter_name, codes)
    finally:
        live_stop()  # le rapport (ou l'échec) prend le relais des offres provisoires


def _run_analysis(
    ctx: TaskContext,
    request: SearchRequest,
    progress: RunProgress,
    scrape: bool,
    interpreter_name: str,
    codes: list[str],
) -> dict:
    linkedin_text = ""
    reports: list[dict] = []
    blocked_sources: list[str] = []
    if scrape:
        progress.step("collecte")
        want_profile = "linkedin" in codes and (
            request.profile_source == "linkedin"
            or (request.profile_source == "auto" and pipeline.profile_mod.find_profile_file() is None)
        )
        prefs = settings.load_settings()

        def on_scored(rows: list[dict]) -> None:
            live_add(rows)
            progress.scored += len(rows)
            progress.push()

        outcome = pipeline.collect(
            request,
            codes,
            want_profile,
            score_batch=pipeline.batch_scorer(request, prefs["home_location"], prefs["timezone"], on_scored=on_scored),
            progress=ctx.log,
            on_state=progress.on_batch,
            interactive=False,
            on_manual_check=ctx.manual_check,
            should_stop=ctx.should_stop,
            on_ready=lambda: ctx.set_status(RUNNING),
        )
        linkedin_text, blocked_sources = outcome.profile_text, outcome.blocked_sources
        reports = [row.public() for row in outcome.sources]
        for row in outcome.sources:
            label = label_of(row.code)
            if row.status == pipeline.STATUS_BLOCKED:
                ctx.log(f"Collecte interrompue par {label} : rapport établi sur les données déjà collectées.")
            elif row.status != pipeline.STATUS_OK:
                ctx.log(f"{label} ignorée : {row.detail}")
    ctx.check()
    progress.step("analyse")
    prof = pipeline.build_profile(request, linkedin_text)
    ctx.log(f"Profil ({prof.origin}) : {len(prof.skills)} compétences.")
    prefs = settings.load_settings()
    results, scored = pipeline.analyse(request, prof, prefs["home_location"], prefs["timezone"])
    ctx.check()
    progress.step("interpretation")
    text, used = interpreter.interpret(
        pipeline.summary_for_interpreter(results), interpreter_name, prefs["home_location"], prefs["timezone"]
    )
    results.update(
        interpretation=text, interpreter=used, interpreter_model=config.anthropic_model() if used == "claude" else ""
    )
    progress.step("rapport")
    _, json_path = report.write(results)
    progress.finish()
    activity.record("run", keyword=request.keyword, scrape=scrape, jobs=len(scored))
    ctx.log(f"Rapport enregistré : {json_path.name}")
    blocked = bool(blocked_sources)
    result = {
        "report": json_path.stem,
        "jobs": len(scored),
        "counts": results["counts"],
        "blocked": blocked,
        "blocked_sources": blocked_sources,
        "sources": reports,
    }
    if blocked:
        result["blocked_where"] = "collecte"
    return result


def rescore(ctx: TaskContext) -> dict:
    """Recalcule les scores avec le profil actuel (compétences modifiées) sans retourner sur LinkedIn."""
    previous = require_results()
    request = SearchRequest(**previous["request"])
    prof = pipeline.build_profile(request)
    prefs = settings.load_settings()
    results, scored = pipeline.analyse(request, prof, prefs["home_location"], prefs["timezone"])
    for key in ("interpretation", "interpreter", "interpreter_model"):
        results[key] = previous.get(key, "")
    _, json_path = report.write(results)
    ctx.log(f"Scores recalculés ({len(scored)} offres) avec {len(prof.skills)} compétences.")
    return {"report": json_path.stem, "jobs": len(scored)}


def train_model(ctx: TaskContext) -> dict:
    previous = require_results()
    request = SearchRequest(**previous["request"])
    prof = pipeline.build_profile(request)
    prefs = settings.load_settings()
    model = matching.train_from_labels(
        prof,
        request.keyword,
        storage.load_labels(),
        storage.load_labeled_items(),
        prefs["home_location"],
        prefs["timezone"],
    )
    if model is None:
        raise ValueError(f"Pas encore assez d'annotations (min. {matching.MIN_LABELS}, des deux types) pour entraîner.")
    ctx.log(
        f"Modèle réentraîné sur {model.n_labels} annotations (précision {(model.train_accuracy or 0) * 100:.0f} %)."
    )
    return {**rescore(ctx), "n_labels": model.n_labels, "accuracy": model.train_accuracy}


def prepare_application(ctx: TaskContext, job_id: str) -> dict:
    results = require_results()
    job = find_job(results, job_id)
    if job is None:
        raise LookupError("Offre introuvable dans la dernière analyse.")
    item = storage.find_item(job_id)
    if item is None:
        raise LookupError("Texte de l'offre introuvable dans l'historique.")
    prof = pipeline.build_profile(SearchRequest(**results["request"]))
    ctx.log(f"Préparation : {job['title']} — {job.get('company', '')}")
    prefs = settings.load_settings()
    app = application.generate(
        prof,
        item,
        job,
        results,
        prefs["interpreter"],
        known_names=application.known_contact_names(),
        home_location=prefs["home_location"],
        timezone=prefs["timezone"],
        identity=cv_format.identity_from(prefs),
    )
    folder = application.write(app, item, job)
    return {"job_id": job_id, "folder": str(folder), "generator": app.generator}


def job_source_code(job_id: str) -> str:
    """Source d'une offre : préfixe de l'identifiant (``ft-``, ``malt-``…), sinon source de l'élément de l'historique.

    Une offre créée a minima par une candidature héritée porte ``linkedin`` par défaut : le préfixe fait foi.
    """
    code = source_of_ref(job_id)
    if code != "linkedin":
        return code
    item = storage.find_item(job_id)
    return item.source_code if item else "linkedin"


def linkedin_only_error(job_id: str, action: str) -> str | None:
    """Message du ``422`` si l'action LinkedIn (``apply`` ou ``contacts``) vise une offre d'une autre source."""
    code = job_source_code(job_id)
    if code == "linkedin":
        return None
    if action == "apply":
        return (
            "La candidature simplifiée n'existe que pour LinkedIn : "
            f"ouvre l'offre sur {label_of(code)} et postule toi-même."
        )
    return "La recherche de recruteurs n'existe que pour les offres LinkedIn."


def apply_job(ctx: TaskContext, job_id: str) -> dict:
    """Candidature simplifiée LinkedIn ; s'arrête à la relecture et attend la confirmation explicite."""
    from ..browser import create_driver

    entry = application.get_application(job_id)
    if error := linkedin_only_error(job_id, "apply"):
        raise RuntimeError(error)
    # Préconditions revérifiées au début de la tâche (I-07, I-16) : l'API a pu accepter d'autres candidatures, ou le
    # consentement a pu être révoqué, entre la création de la tâche et son démarrage.
    consent.require("linkedin")
    activity.check_quota_locked(("apply",), "apply_daily_max", "candidatures")
    activity.guard_not_blocked()
    cfg = settings.load_settings()
    folder = application.folder_of(entry)
    files = application.read_files(entry)
    pdf = cv_export.ensure(folder, files["cv"], "pdf", title=entry.get("title") or "CV", full_name=cfg["full_name"])
    profile = effective_profile()
    cfg = {**cfg, "years_hint": profile.get("years_experience") or ""}

    application.update_application(job_id, apply_state="en_cours")
    driver = create_driver()
    try:
        session.login(driver, interactive=False, on_manual_check=ctx.manual_check)
        ctx.set_status(RUNNING)
        bot = apply_bot.EasyApply(driver, ctx, cv_pdf=pdf, caution=config.CAUTION_LEVELS[cfg["caution"]], cfg=cfg)
        result = bot.run({"id": job_id, "title": entry.get("title", ""), "url": entry["url"]})
    except apply_bot.BlockedError:
        activity.record("blocked", where="candidature", job=job_id, source="linkedin")
        application.update_application(job_id, apply_state="échec")
        raise
    except BaseException:
        application.update_application(job_id, apply_state="échec")
        raise
    finally:
        driver.quit()
    return _record_apply_result(job_id, entry, result)


def _record_apply_result(job_id: str, entry: dict, result: apply_bot.ApplyResult) -> dict:
    title = entry.get("title", "")
    application.add_history(job_id, "apply", state=result.state, detail=result.detail)
    if result.state == "postulée":
        activity.record("apply", job=job_id, title=title)
        application.advance_status(job_id, "postulée")
        storage.save_labels({**storage.load_labels(), job_id: 1})
    elif result.state == "déjà_postulée":
        application.advance_status(job_id, "postulée")
    elif result.state == "externe":
        application.advance_status(job_id, "externe")
    application.update_application(job_id, apply_state="" if result.state in ("annulée", "externe") else result.state)
    return {"state": result.state, "detail": result.detail}


def lookup_contacts(ctx: TaskContext, job_id: str) -> dict:
    """Recruteurs affichés sur l'offre + leur email quand LinkedIn le montre."""
    from ..browser import create_driver

    entry = application.get_application(job_id)
    if error := linkedin_only_error(job_id, "contacts"):
        raise RuntimeError(error)
    consent.require("linkedin")
    activity.guard_not_blocked()
    caution = config.CAUTION_LEVELS[settings.load_settings()["caution"]]
    driver = create_driver()
    try:
        session.login(driver, interactive=False, on_manual_check=ctx.manual_check)
        ctx.set_status(RUNNING)
        found = contacts_mod.find_hiring_team(driver, entry["url"], caution, check=ctx.check, log_fn=ctx.log)
    except PermissionError as exc:
        activity.record("blocked", where="recherche de contacts", job=job_id, source="linkedin")
        raise RuntimeError(str(exc)) from exc
    finally:
        driver.quit()
    found = application.drop_opposed(found)  # un recruteur opposé est écarté en silence (I-13, V4-10 d)
    merged = contacts_mod.merge_contacts(entry["contacts"], found)
    application.update_application(job_id, contacts=merged)
    application.add_history(job_id, "contacts", found=len(found), emails=sum(1 for c in found if c["email"]))
    return {"found": len(found), "emails": sum(1 for c in found if c["email"])}


def sender_address() -> str:
    """Adresse nue qui reçoit les demandes de désinscription : ``SMTP_FROM``, sinon ``contact_email`` des réglages."""
    for candidate in (os.environ.get("SMTP_FROM") or "", settings.load_settings()["contact_email"]):
        address = parseaddr(candidate)[1].strip()
        if address:
            return address
    return "l'expéditeur de ce message"


def _footer(channel: str, job_title: str, origin: str, sender: str) -> str:
    """Pied imposé du message (I-12) : titre raccourci à 100 caractères (email) ou 60 (LinkedIn)."""
    title = tidy(job_title, 60 if channel == "linkedin" else 100)
    return outreach.compose_footer(channel, title, origin, sender)


def message_draft(job_id: str, to: str | None = None) -> dict:
    """Objet, corps (sans pied) et pied imposé exact d'un message à l'entreprise (``backend/01`` § 5.7.1, V4-6).

    Le canal se déduit de ``to`` ; absent, vide ou invalide (saisie en cours) : email, origine ``manuel``,
    ``opposed`` faux. L'opposition est signalée par ``opposed``, jamais par une erreur.
    """
    entry = application.get_application(job_id)
    files = application.read_files(entry)
    signature = settings.load_settings()["signature"]
    body = files["message"].strip()
    if signature and signature not in body:
        body += f"\n\n{signature}"
    to = (to or "").strip()
    channel = "linkedin" if outreach._PROFILE_URL.match(to) else "email"
    if channel == "email" and not outreach._EMAIL.match(to):
        to = ""
    origin, opposed = "manuel", False
    if to:
        with user_transaction() as (s, uid):
            found = applications_repo.get(s, uid, job_id)
            if found is None:
                raise application.ApplicationNotFound(job_id)
            contact = outreach_repo.find_contact(s, uid, found[0].id, to)
            origin = contact.source if contact else "manuel"
            opposed = outreach_repo.recipient_blocked(s, uid, to, contact.id if contact else None)
    title = entry.get("title", "")
    return {
        "subject": f"Candidature — {tidy(title, 100)}",
        "body": body,
        "footer": _footer(channel, title, origin, sender_address()),
        "origin": origin,
        "opposed": opposed,
    }


def reserve_message(
    job_id: str, *, channel: str, to: str, subject: str, body: str, attach_cv: bool
) -> tuple[UUID, str]:
    """Réserve une place du plafond de messages (``backend/03`` § 5.1, E-11) ; renvoie ``(message_id, pied)``.

    ``FOR UPDATE`` sur la candidature puis sur ``user_settings`` : deux envois simultanés (même vers des
    candidatures différentes) ne peuvent pas compter la même place. Toujours dans la transaction, dans cet ordre :
    destinataire opposé (``RecipientSuppressed``, I-13) puis plafond (``QuotaExceeded``) puis insertion ; une
    exception annule la transaction, sans rien écrire. Le serveur calcule l'origine et le pied imposé (I-12) et
    enregistre dans ``outreach_messages.body`` le texte réellement envoyé. Lève aussi ``ApplicationNotFound``.
    """
    start, end = activity.day_bounds()
    sender = sender_address()  # lu avant la transaction : une seule connexion tenue à la fois
    with user_transaction() as (s, uid):
        found = applications_repo.get(s, uid, job_id, lock=True)
        if found is None:
            raise application.ApplicationNotFound(job_id)
        application_id = found[0].id
        limit = settings_repo.get_settings(s, uid, lock=True)["message_daily_max"]
        contact = outreach_repo.find_contact(s, uid, application_id, to)
        if outreach_repo.recipient_blocked(s, uid, to, contact.id if contact else None):
            raise outreach.RecipientSuppressed
        if outreach_repo.count_day(s, uid, start, end) >= limit:
            raise activity.QuotaExceeded("messages", limit)
        footer = _footer(channel, found[1].title, contact.source if contact else "manuel", sender)
        message_id = outreach_repo.insert_confirmed(
            s,
            uid,
            application_id,
            {
                "channel": channel,
                "recipient": to,
                "subject": subject.strip() if channel == "email" else "",
                "body": outreach.assemble(body, footer),
                "cv_attached": attach_cv and channel == "email",
                "contact_id": contact.id if contact else None,
            },
        )
        return message_id, footer


def release_message(message_id: UUID) -> None:
    """Rend la place réservée quand la tâche d'envoi n'a finalement pas été créée (conflit, blocage)."""
    with user_transaction() as (s, uid):
        outreach_repo.release(s, uid, message_id)


def abort_message(message_id: UUID) -> None:
    """Tâche d'envoi terminée avant d'avoir démarré (annulée pendant l'attente du navigateur) : place rendue."""
    _message_outcome(message_id, error="Tâche annulée avant l'envoi.")


def message_count_today() -> int:
    """Messages du jour qui occupent le plafond (réservés ou envoyés)."""
    start, end = activity.day_bounds()
    with user_transaction() as (s, uid):
        return outreach_repo.count_day(s, uid, start, end)


def _confirm_reservation(message_id: UUID) -> None:
    """Début de tâche : réservation encore valide, destinataire non opposé (I-13), plafond non dépassé (I-07).

    Le plafond est relu sous verrou. Lève ``RecipientSuppressed`` ou ``QuotaExceeded`` : l'appelant marque la
    réservation ``échec`` (la place est rendue), sans aucun envoi.
    """
    start, end = activity.day_bounds()
    with user_transaction() as (s, uid):
        limit = settings_repo.get_settings(s, uid, lock=True)["message_daily_max"]
        row = outreach_repo.get(s, uid, message_id, lock=True)
        if row is None or row.status != "confirmé" or row.sent_at is not None:
            raise RuntimeError("Réservation d'envoi introuvable : relance l'envoi depuis la candidature.")
        blocked = outreach_repo.recipient_blocked(s, uid, row.recipient, row.contact_id)
        used = outreach_repo.count_day(s, uid, start, end)
    if blocked:
        raise outreach.RecipientSuppressed
    if used > limit:
        raise activity.QuotaExceeded("messages", limit)


def _message_outcome(message_id: UUID, *, sent_id: str | None = None, error: str = "", uncertain: bool = False) -> None:
    with user_transaction() as (s, uid):
        if uncertain:
            outreach_repo.note_uncertain(s, uid, message_id, error)
        elif error:
            outreach_repo.mark_failed(s, uid, message_id, error)
        else:
            outreach_repo.mark_sent(s, uid, message_id, sent_id)


def send_message(
    ctx: TaskContext,
    job_id: str,
    *,
    message_id: UUID,
    channel: str,
    to: str,
    subject: str,
    body: str,
    footer: str,
    attach_cv: bool,
) -> dict:
    """Envoie un message (email ou LinkedIn) déjà confirmé par l'utilisateur et réservé, puis le journalise.

    ``body`` est le texte saisi, sans pied ; ``footer`` le pied imposé calculé à la réservation (I-12), ajouté ici
    après le corps. Préconditions revérifiées ici (l'API les a déjà contrôlées) : réservation, destinataire non
    opposé et plafond, SMTP chiffré ou local (E-20), absence de blocage (LinkedIn). Un échec libère la place
    (``échec``) ; aucun renvoi automatique.
    """
    try:
        entry = application.get_application(job_id)
        _confirm_reservation(message_id)
        body = outreach.strip_footer(body, footer)
        if not body:
            raise outreach.OutreachError("Le message est vide.")
        cfg = settings.load_settings()
        sent_id: str | None = None
        if channel == "email":
            smtp = config.smtp_config()
            if smtp is None:
                raise outreach.OutreachError(
                    "SMTP non configuré : renseigne SMTP_HOST et SMTP_FROM dans le fichier .env."
                )
            insecure = config.smtp_security_error(smtp)
            if insecure:
                raise outreach.OutreachError(insecure)
            attachments = []
            if attach_cv:
                files = application.read_files(entry)
                attachments = [
                    cv_export.ensure(
                        application.folder_of(entry),
                        files["cv"],
                        "pdf",
                        title=entry.get("title") or "CV",
                        full_name=cfg["full_name"],
                    )
                ]
            ctx.check()
            ctx.log(f"Envoi de l'email à {logmask.mask_email(to)}…")
            sent, sent_id = outreach.deliver_email(
                to, subject, outreach.assemble(body, footer), attachments, smtp, unsubscribe=True
            )
        else:
            sent = _send_linkedin(ctx, job_id, to, body, footer, cfg)
    except outreach.UncertainDelivery as exc:
        _message_outcome(message_id, error=str(exc), uncertain=True)
        raise
    except Exception as exc:
        _message_outcome(message_id, error=logmask.mask_text(str(exc)) or exc.__class__.__name__)
        raise
    _message_outcome(message_id, sent_id=sent_id)
    # Journal d'activité exposé par /api/monitor : destinataire masqué (E-12) ; complet dans l'historique.
    activity.record(f"message_{channel}", job=job_id, to=logmask.mask_recipient(to), title=entry.get("title", ""))
    application.add_history(
        job_id,
        "message",
        channel=channel,
        to=to,
        extrait=body[:120],
        **({"mode": sent["mode"]} if "mode" in sent else {}),
    )
    application.advance_status(job_id, "contactée")
    return sent


def _send_linkedin(ctx: TaskContext, job_id: str, to: str, body: str, footer: str, cfg: dict) -> dict:
    from ..browser import create_driver

    consent.require("linkedin")
    activity.guard_not_blocked()
    driver = create_driver()
    try:
        session.login(driver, interactive=False, on_manual_check=ctx.manual_check)
        ctx.set_status(RUNNING)
        ctx.log(f"Envoi du message LinkedIn à {logmask.mask_profile_url(to)}…")
        sent = outreach.send_linkedin_message(
            driver, to, body, config.CAUTION_LEVELS[cfg["caution"]], footer=footer, check=ctx.check
        )
    except PermissionError as exc:
        activity.record("blocked", where="message LinkedIn", job=job_id, source="linkedin")
        raise RuntimeError(str(exc)) from exc
    finally:
        driver.quit()
    if sent.get("truncated"):
        ctx.log("La note de connexion dépassait 300 caractères : le corps a été tronqué, le pied est conservé.")
    return sent


def send_test_email(to: str) -> dict:
    return outreach.send_email(
        to, "Test my-connector", "Ceci est un email de test envoyé depuis l'interface linkedin_market."
    )


def import_linkedin_profile(ctx: TaskContext) -> dict:
    """Relit le profil LinkedIn (cache) pour mettre à jour les compétences détectées."""
    from ..browser import create_driver
    from ..scraper import LinkedInScraper

    consent.require("linkedin")
    activity.guard_not_blocked()
    request = build_request()
    driver = create_driver()
    try:
        session.login(driver, interactive=False, on_manual_check=ctx.manual_check)
        ctx.set_status(RUNNING)
        text = LinkedInScraper(driver, request).own_profile_text()
    finally:
        driver.quit()
    if not text:
        raise RuntimeError("Profil LinkedIn illisible (page inattendue ou accès bloqué).")
    storage.save_linkedin_profile_text(text)
    prof = pipeline.build_profile(request, text, use_overrides=False)
    ctx.log(f"Profil LinkedIn importé : {len(prof.skills)} compétences détectées.")
    return {"skills": prof.skills}


def sync_skill_to_linkedin(ctx: TaskContext, skill: str) -> dict:
    """Ajoute au profil LinkedIn une compétence que l'utilisateur vient d'ajouter dans l'interface."""
    from ..browser import create_driver

    consent.require("linkedin")
    activity.guard_not_blocked()
    caution = config.CAUTION_LEVELS[settings.load_settings()["caution"]]
    driver = create_driver()
    try:
        session.login(driver, interactive=False, on_manual_check=ctx.manual_check)
        ctx.set_status(RUNNING)
        result = profile_sync.add_skill(driver, skill, caution, check=ctx.check, log_fn=ctx.log)
    except PermissionError as exc:
        activity.record("blocked", where="profil LinkedIn (compétence)", source="linkedin")
        raise RuntimeError(str(exc)) from exc
    finally:
        driver.quit()
    if result["state"] == "ajoutée":
        activity.record("profile_skill", skill=result["skill"], linkedin_name=result["linkedin_name"])
    return result


# --------------------------------------------------------------- profil
def effective_profile(request: SearchRequest | None = None) -> dict:
    """Profil tel qu'utilisé pour le scoring, et sa base sans les surcharges de l'interface."""
    overrides = settings.load_profile_settings()
    try:
        request = request or _profile_request()
        effective = pipeline.build_profile(request)
        base = pipeline.build_profile(request, use_overrides=False)
    except pipeline.ProfileMissingError as exc:
        return {"available": False, "message": str(exc), "overrides": overrides, "skills": [], "base_skills": []}
    return {
        "available": True,
        "origin": effective.origin,
        "title": effective.title,
        "skills": effective.skills,
        "base_skills": base.skills,
        "years_experience": effective.years_experience,
        "seniority": effective.seniority,
        "overrides": overrides,
    }


def _profile_request() -> SearchRequest:
    values = settings.load_settings()
    return SearchRequest(keyword=values["keyword"] or "profil", profile_source=values["profile_source"])


def skill_suggestions(results: dict | None, owned: list[str]) -> dict:
    if not results:
        return {"missing": [], "rising": []}
    have = set(owned)
    missing = [
        {"skill": r["skill"], "weight": r["weight"]}
        for r in results.get("gap", {}).get("missing", [])
        if r["skill"] not in have
    ][:15]
    rising = [
        {"skill": r["term"], "growth": r["growth"]}
        for r in results.get("rising", {}).get("skills", [])
        if r["growth"] > 0 and r["term"] not in have
    ][:10]
    return {"missing": missing, "rising": rising}


# ------------------------------------------------------------ monitoring
def quotas() -> dict:
    values = settings.load_settings()
    applied = activity.count_today("apply")
    messages = message_count_today()
    return {
        "apply": {"used": applied, "limit": values["apply_daily_max"]},
        "messages": {"used": messages, "limit": values["message_daily_max"]},
    }


def monitor_snapshot(tasks) -> dict:
    apps = application.list_applications()
    by_status: dict[str, int] = {}
    for entry in apps:
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
    results = latest_results()
    labels = storage.load_labels()
    applied = sum(by_status.get(s, 0) for s in ("postulée", "entretien", "refusée"))
    funnel = [
        {"step": "Offres analysées", "count": len(results["top_jobs"]) if results else 0},
        {"step": "Jugées pertinentes", "count": sum(1 for v in labels.values() if v == 1)},
        {"step": "Candidatures préparées", "count": len(apps)},
        {"step": "Postulées", "count": applied},
        {"step": "Entretiens", "count": by_status.get("entretien", 0)},
    ]
    all_tasks = tasks.list(30)
    return {
        "generated_at": utc_iso(utcnow()),
        "by_status": by_status,
        "funnel": funnel,
        "quotas": quotas(),
        "activity": activity.per_day(14),
        "recent_events": list(reversed(activity.load(15))),
        "model": results["model"] if results else None,
        "last_analysis": (
            {
                "report": results["_report"],
                "generated_at": results["generated_at"],
                "keyword": results["request"]["keyword"],
                "counts": results["counts"],
            }
            if results
            else None
        ),
        "active_tasks": [t.public(with_logs=False) for t in tasks.active()],
        "recent_tasks": [t.public(with_logs=False) for t in all_tasks[:10]],
        "errors": [
            {"id": t.id, "title": t.title, "error": t.error, "at": t.finished_at}
            for t in all_tasks
            if t.status == "échec"
        ][:5],
    }


def status_snapshot(tasks) -> dict:
    values = settings.load_settings()
    age = session.session_age_hours()
    email, password = config.linkedin_credentials()
    return {
        "session": {
            "age_hours": round(age, 1) if age is not None else None,
            "ttl_hours": config.SESSION_TTL_HOURS,
            "valid": age is not None and age <= config.SESSION_TTL_HOURS,
        },
        "credentials_configured": bool(email and password),
        "smtp_configured": config.smtp_config() is not None,
        "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "interpreter": values["interpreter"],
        "browser": config.browser_name(),
        "headless": config.headless(),
        "profile_file": pipeline.profile_mod.find_profile_file() is not None,
        "has_analysis": latest_results() is not None,
        "quotas": quotas(),
        "active_tasks": len(tasks.active()),
        "blocked": activity.block_status(),
    }

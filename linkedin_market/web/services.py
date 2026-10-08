"""Logique métier de l'interface web : enchaîne les modules existants sans saisie interactive."""

from __future__ import annotations

import os
import re
from datetime import datetime

from .. import (activity, application, apply_bot, config, contacts as contacts_mod, cv_pdf, interpreter, matching, outreach,
                pipeline, profile_sync, report, session, settings, storage)
from ..models import SearchRequest
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
    return SearchRequest(
        keyword=keyword, location=values["location"], remote_only=values["remote_only"], caution=caution,
        max_items=values["max_items"] or config.CAUTION_LEVELS[caution].max_items,
        profile_source=values["profile_source"],
    )


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
_REGIONS = ("Europe, Moyen-Orient et Afrique", "Afrique subsaharienne", "Amérique du Nord", "Amérique latine",
            "Asie-Pacifique", "Union européenne", "Moyen-Orient", "Monde entier")
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
    rest, location = head[len(title):].strip(), ""
    for region in _REGIONS:
        if rest.endswith(region):
            rest, location = rest[: -len(region)].strip(), region
            break
    posted, applicants = _POSTED.search(meta), _APPLICANTS.search(meta)
    return {**job, "title": tidy(rest or title, 150), "company": tidy(title, 80, drop=True), "location": location,
            "posted_label": posted.group(0) if posted else "", "applicants": applicants.group(0) if applicants else ""}


def jobs_overview(results: dict) -> list[dict]:
    labels = storage.load_labels()
    apps = application.load_applications()
    rows = []
    for job in results.get("top_jobs", []):
        entry = apps.get(job["id"])
        rows.append({**repair_identity(job), "label": labels.get(job["id"]),
                     "application_status": entry["status"] if entry else None})
    return rows


# ------------------------------------------------------------- exécution
def run_analysis(ctx: TaskContext, request: SearchRequest, *, scrape: bool, interpreter_name: str) -> dict:
    linkedin_text = ""
    if scrape:
        want_profile = request.profile_source == "linkedin" or (
            request.profile_source == "auto" and pipeline.profile_mod.find_profile_file() is None)
        linkedin_text = pipeline.scrape(
            request, want_profile, interactive=False, on_manual_check=ctx.manual_check,
            should_stop=ctx.should_stop, on_ready=lambda: ctx.set_status(RUNNING))
    ctx.check()
    prof = pipeline.build_profile(request, linkedin_text)
    ctx.log(f"Profil ({prof.origin}) : {len(prof.skills)} compétences.")
    results, scored = pipeline.analyse(request, prof)
    ctx.check()
    text, used = interpreter.interpret(pipeline.summary_for_interpreter(results), interpreter_name)
    results.update(interpretation=text, interpreter=used,
                   interpreter_model=config.anthropic_model() if used == "claude" else "")
    _, json_path = report.write(results)
    activity.record("run", keyword=request.keyword, scrape=scrape, jobs=len(scored))
    ctx.log(f"Rapport enregistré : {json_path.name}")
    return {"report": json_path.stem, "jobs": len(scored), "counts": results["counts"]}


def rescore(ctx: TaskContext) -> dict:
    """Recalcule les scores avec le profil actuel (compétences modifiées) sans retourner sur LinkedIn."""
    previous = require_results()
    request = SearchRequest(**previous["request"])
    prof = pipeline.build_profile(request)
    results, scored = pipeline.analyse(request, prof)
    for key in ("interpretation", "interpreter", "interpreter_model"):
        results[key] = previous.get(key, "")
    _, json_path = report.write(results)
    ctx.log(f"Scores recalculés ({len(scored)} offres) avec {len(prof.skills)} compétences.")
    return {"report": json_path.stem, "jobs": len(scored)}


def train_model(ctx: TaskContext) -> dict:
    previous = require_results()
    request = SearchRequest(**previous["request"])
    prof = pipeline.build_profile(request)
    model = matching.train_from_labels(prof, request.keyword, storage.load_labels(), storage.load_labeled_items())
    if model is None:
        raise ValueError(f"Pas encore assez d'annotations (min. {matching.MIN_LABELS}, des deux types) pour entraîner.")
    ctx.log(f"Modèle réentraîné sur {model.n_labels} annotations (précision {(model.train_accuracy or 0) * 100:.0f} %).")
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
    generator = settings.load_settings()["interpreter"]
    app = application.generate(prof, item, job, results, generator)
    folder = application.write(app, item, job)
    return {"job_id": job_id, "folder": str(folder), "generator": app.generator}


def apply_job(ctx: TaskContext, job_id: str) -> dict:
    """Candidature simplifiée LinkedIn ; s'arrête à la relecture et attend la confirmation explicite."""
    from ..browser import create_driver

    entry = application.get_application(job_id)
    cfg = settings.load_settings()
    activity.check_quota(("apply",), cfg["apply_daily_max"], "candidatures")
    activity.guard_not_blocked()
    folder = application.folder_of(entry)
    files = application.read_files(entry)
    pdf = cv_pdf.ensure_cv_pdf(folder, files["cv"], entry.get("title") or "CV", config.CV_PDF_NAME)
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
        activity.record("blocked", where="candidature", job=job_id)
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
    application.add_history(job_id, "candidature", state=result.state, detail=result.detail)
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
    activity.guard_not_blocked()
    caution = config.CAUTION_LEVELS[settings.load_settings()["caution"]]
    driver = create_driver()
    try:
        session.login(driver, interactive=False, on_manual_check=ctx.manual_check)
        ctx.set_status(RUNNING)
        found = contacts_mod.find_hiring_team(driver, entry["url"], caution, check=ctx.check, log_fn=ctx.log)
    except PermissionError as exc:
        activity.record("blocked", where="recherche de contacts", job=job_id)
        raise RuntimeError(str(exc)) from exc
    finally:
        driver.quit()
    merged = contacts_mod.merge_contacts(entry["contacts"], found)
    application.update_application(job_id, contacts=merged)
    application.add_history(job_id, "contacts", found=len(found), emails=sum(1 for c in found if c["email"]))
    return {"found": len(found), "emails": sum(1 for c in found if c["email"])}


def message_draft(job_id: str) -> dict:
    """Objet et corps proposés pour un message à l'entreprise (modifiables avant l'envoi)."""
    entry = application.get_application(job_id)
    files = application.read_files(entry)
    signature = settings.load_settings()["signature"]
    body = files["message"].strip()
    if signature and signature not in body:
        body += f"\n\n{signature}"
    return {"subject": f"Candidature — {tidy(entry.get('title', ''), 100)}", "body": body}


def send_message(ctx: TaskContext, job_id: str, *, channel: str, to: str, subject: str, body: str, attach_cv: bool) -> dict:
    """Envoie un message (email ou LinkedIn) déjà confirmé par l'utilisateur, puis le journalise."""
    entry = application.get_application(job_id)
    cfg = settings.load_settings()
    activity.check_quota(("message_email", "message_linkedin"), cfg["message_daily_max"], "messages")
    if channel == "email":
        attachments = []
        if attach_cv:
            files = application.read_files(entry)
            attachments = [cv_pdf.ensure_cv_pdf(application.folder_of(entry), files["cv"], entry.get("title") or "CV", config.CV_PDF_NAME)]
        ctx.log(f"Envoi de l'email à {to}…")
        sent = outreach.send_email(to, subject, body, attachments)
    else:
        from ..browser import create_driver

        activity.guard_not_blocked()
        driver = create_driver()
        try:
            session.login(driver, interactive=False, on_manual_check=ctx.manual_check)
            ctx.set_status(RUNNING)
            ctx.log(f"Envoi du message LinkedIn à {to}…")
            sent = outreach.send_linkedin_message(driver, to, body, config.CAUTION_LEVELS[cfg["caution"]], check=ctx.check)
        except PermissionError as exc:
            activity.record("blocked", where="message LinkedIn", job=job_id)
            raise RuntimeError(str(exc)) from exc
        finally:
            driver.quit()
        if sent.get("truncated"):
            ctx.log("La note de connexion dépassait 300 caractères : elle a été tronquée.")
    activity.record(f"message_{channel}", job=job_id, to=to, title=entry.get("title", ""))
    application.add_history(job_id, "message", channel=channel, to=to, extrait=body[:120],
                            **({"mode": sent["mode"]} if "mode" in sent else {}))
    application.advance_status(job_id, "contactée")
    return sent


def send_test_email(to: str) -> dict:
    return outreach.send_email(to, "Test my-connector", "Ceci est un email de test envoyé depuis l'interface linkedin_market.")


def import_linkedin_profile(ctx: TaskContext) -> dict:
    """Relit le profil LinkedIn (cache) pour mettre à jour les compétences détectées."""
    from ..browser import create_driver
    from ..scraper import LinkedInScraper

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
    config.LINKEDIN_PROFILE_CACHE.write_text(text, encoding="utf-8")
    prof = pipeline.build_profile(request, text, use_overrides=False)
    ctx.log(f"Profil LinkedIn importé : {len(prof.skills)} compétences détectées.")
    return {"skills": prof.skills}


def sync_skill_to_linkedin(ctx: TaskContext, skill: str) -> dict:
    """Ajoute au profil LinkedIn une compétence que l'utilisateur vient d'ajouter dans l'interface."""
    from ..browser import create_driver

    activity.guard_not_blocked()
    caution = config.CAUTION_LEVELS[settings.load_settings()["caution"]]
    driver = create_driver()
    try:
        session.login(driver, interactive=False, on_manual_check=ctx.manual_check)
        ctx.set_status(RUNNING)
        result = profile_sync.add_skill(driver, skill, caution, check=ctx.check, log_fn=ctx.log)
    except PermissionError as exc:
        activity.record("blocked", where="profil LinkedIn (compétence)")
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
        "available": True, "origin": effective.origin, "title": effective.title, "skills": effective.skills,
        "base_skills": base.skills, "years_experience": effective.years_experience,
        "seniority": effective.seniority, "overrides": overrides,
    }


def _profile_request() -> SearchRequest:
    values = settings.load_settings()
    return SearchRequest(keyword=values["keyword"] or "profil", profile_source=values["profile_source"])


def skill_suggestions(results: dict | None, owned: list[str]) -> dict:
    if not results:
        return {"missing": [], "rising": []}
    have = set(owned)
    missing = [{"skill": r["skill"], "weight": r["weight"]} for r in results.get("gap", {}).get("missing", [])
               if r["skill"] not in have][:15]
    rising = [{"skill": r["term"], "growth": r["growth"]} for r in results.get("rising", {}).get("skills", [])
              if r["growth"] > 0 and r["term"] not in have][:10]
    return {"missing": missing, "rising": rising}


# ------------------------------------------------------------ monitoring
def quotas() -> dict:
    values = settings.load_settings()
    applied = activity.count_today("apply")
    messages = activity.count_today("message_email", "message_linkedin")
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
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "by_status": by_status, "funnel": funnel, "quotas": quotas(),
        "activity": activity.per_day(14),
        "recent_events": list(reversed(activity.load(15))),
        "model": results["model"] if results else None,
        "last_analysis": ({"report": results["_report"], "generated_at": results["generated_at"],
                           "keyword": results["request"]["keyword"], "counts": results["counts"]} if results else None),
        "active_tasks": [t.public(with_logs=False) for t in tasks.active()],
        "recent_tasks": [t.public(with_logs=False) for t in all_tasks[:10]],
        "errors": [{"id": t.id, "title": t.title, "error": t.error, "at": t.finished_at}
                   for t in all_tasks if t.status == "échec"][:5],
    }


def status_snapshot(tasks) -> dict:
    values = settings.load_settings()
    age = session.session_age_hours()
    email, password = config.linkedin_credentials()
    return {
        "session": {"age_hours": round(age, 1) if age is not None else None, "ttl_hours": config.SESSION_TTL_HOURS,
                    "valid": age is not None and age <= config.SESSION_TTL_HOURS},
        "credentials_configured": bool(email and password),
        "smtp_configured": config.smtp_config() is not None,
        "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "interpreter": values["interpreter"], "browser": config.browser_name(), "headless": config.headless(),
        "profile_file": pipeline.profile_mod.find_profile_file() is not None,
        "has_analysis": latest_results() is not None,
        "quotas": quotas(), "active_tasks": len(tasks.active()), "blocked": activity.block_status(),
    }


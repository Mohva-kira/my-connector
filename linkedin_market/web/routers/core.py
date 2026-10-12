"""État, réglages, profil et compétences, rapports, monitoring, tâches."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ... import activity, consent, nlp, settings, storage
from .. import services
from ..guards import require_browser_ready
from ..tasks import TaskManager

router = APIRouter()


def tasks_of(request: Request) -> TaskManager:
    return request.app.state.tasks


# ------------------------------------------------------------------ état
@router.get("/status")
def status(request: Request) -> dict:
    return services.status_snapshot(tasks_of(request))


class BlockClearBody(BaseModel):
    source: str = "linkedin"


@router.post("/block/clear")
def clear_block(body: BlockClearBody | None = None) -> dict:
    """Lève la période de repos d'une source après vérification du compte par l'utilisateur (défaut LinkedIn)."""
    code = body.source if body else "linkedin"
    try:
        consent.ensure_source_exists(code)
    except consent.SourceNotFound:
        raise HTTPException(404, "Source inconnue.") from None
    activity.record("unblocked", source=code)
    return {"ok": True}


@router.get("/monitor")
def monitor(request: Request) -> dict:
    return services.monitor_snapshot(tasks_of(request))


# ------------------------------------------------------------- réglages
@router.get("/settings")
def get_settings() -> dict:
    return settings.load_settings()


@router.put("/settings")
def put_settings(body: dict) -> dict:
    try:
        return settings.save_settings(body)
    except settings.SettingsError as exc:
        raise HTTPException(422, str(exc)) from exc


# ---------------------------------------------------- profil, compétences
class ProfileUpdate(BaseModel):
    title: str | None = None
    years_experience: int | None = None
    seniority: str | None = None
    clear_years: bool = False


class SkillBody(BaseModel):
    skill: str


@router.get("/profile")
def get_profile() -> dict:
    profile = services.effective_profile()
    results = services.latest_results()
    profile["suggestions"] = services.skill_suggestions(results, profile["skills"])
    # Les scores des offres reflètent le profil de la dernière analyse : à recalculer si les compétences ont changé.
    profile["scores_outdated"] = (
        bool(results)
        and profile["available"]
        and sorted(profile["skills"]) != sorted(results.get("profile", {}).get("skills", []))
    )
    return profile


@router.put("/profile")
def put_profile(body: ProfileUpdate) -> dict:
    update = body.model_dump(exclude_none=True, exclude={"clear_years"})
    if body.clear_years:
        update["years_experience"] = None
    try:
        settings.save_profile_settings(update)
    except settings.SettingsError as exc:
        raise HTTPException(422, str(exc)) from exc
    return get_profile()


@router.post("/profile/skills")
def add_skill(request: Request, body: SkillBody) -> dict:
    before = set(services.effective_profile()["skills"])
    try:
        settings.add_skill(body.skill)
    except settings.SettingsError as exc:
        raise HTTPException(422, str(exc)) from exc
    profile = get_profile()
    profile["linkedin_sync"] = _sync_skill(
        request, nlp.normalize(body.skill), new=nlp.normalize(body.skill) not in before
    )
    return profile


def _sync_skill(request: Request, skill: str, *, new: bool) -> dict:
    """Ajout au profil LinkedIn en tâche de fond ; un empêchement n'annule jamais l'ajout local."""
    if not settings.load_settings()["linkedin_skill_sync"]:
        return {"status": "off", "message": "Synchronisation LinkedIn désactivée dans les paramètres."}
    if not new:
        return {"status": "skipped", "message": "Compétence déjà dans ton profil : LinkedIn n'est pas modifié."}
    tasks = tasks_of(request)
    if tasks.has_active("profile_skill", skill):
        return {"status": "skipped", "message": "Ajout à LinkedIn déjà en cours pour cette compétence."}
    # Cas particulier (I-16) : cette route ne renvoie jamais 428 ; sans consentement, l'ajout local est conservé
    # mais la synchronisation LinkedIn est « unavailable », comme pour des identifiants absents.
    try:
        consent.require("linkedin")
        require_browser_ready()
    except consent.SourceConsentRequired as exc:
        return {"status": "unavailable", "message": f"Compétence ajoutée ici, mais pas sur LinkedIn : {exc}"}
    except HTTPException as exc:
        return {"status": "unavailable", "message": f"Compétence ajoutée ici, mais pas sur LinkedIn : {exc.detail}"}
    task = tasks.submit(
        "profile_skill",
        f"Ajouter « {skill} » au profil LinkedIn",
        lambda ctx: services.sync_skill_to_linkedin(ctx, skill),
        browser=True,
        meta={"key": skill},
    )
    return {"status": "queued", "task": task.public()}


@router.delete("/profile/skills/{skill:path}")
def remove_skill(skill: str) -> dict:
    settings.remove_skill(skill)
    return get_profile()


@router.post("/profile/reset-overrides")
def reset_overrides() -> dict:
    settings.save_profile_settings(
        {"added_skills": [], "removed_skills": [], "title": "", "years_experience": None, "seniority": ""}
    )
    return get_profile()


@router.post("/profile/import-linkedin", status_code=202)
def import_linkedin(request: Request) -> dict:
    tasks = tasks_of(request)
    if tasks.has_active("import_profile"):
        raise HTTPException(409, "Un import est déjà en cours.")
    consent.require("linkedin")
    require_browser_ready()
    task = tasks.submit("import_profile", "Importer le profil LinkedIn", services.import_linkedin_profile, browser=True)
    return task.public()


# -------------------------------------------------------------- rapports
@router.get("/reports")
def reports() -> list[dict]:
    return storage.list_reports()


@router.get("/reports/{name}")
def report_json(name: str) -> dict:
    data = storage.load_report(name)
    if data is None:
        raise HTTPException(404, "Rapport introuvable.")
    latest = next(iter(storage.list_reports()), None)
    # Offres lisibles + annotation et candidature ; les actions sur une offre visent le dernier rapport.
    return {**data, "top_jobs": services.jobs_overview(data), "is_latest": bool(latest and latest["name"] == name)}


@router.get("/reports/{name}/html")
def report_html(name: str):
    path = storage.report_html_path(name)
    if path is None:
        raise HTTPException(404, "Rapport introuvable.")
    return FileResponse(
        path, media_type="text/html", headers={"Content-Security-Policy": "sandbox allow-scripts allow-popups"}
    )


# ---------------------------------------------------------------- tâches
@router.get("/tasks")
def list_tasks(request: Request, limit: int = 30) -> list[dict]:
    return [t.public(with_logs=False) for t in tasks_of(request).list(min(limit, 100))]


@router.get("/tasks/{task_id}")
def get_task(request: Request, task_id: str) -> dict:
    task = tasks_of(request).get(task_id)
    if task is None:
        raise HTTPException(404, "Tâche introuvable.")
    return task.public()


@router.post("/tasks/{task_id}/cancel")
def cancel_task(request: Request, task_id: str) -> dict:
    if not tasks_of(request).cancel(task_id):
        raise HTTPException(409, "Tâche introuvable ou déjà terminée.")
    return {"ok": True}


class Answer(BaseModel):
    value: str
    remember: bool = True


@router.post("/tasks/{task_id}/answer")
def answer_task(request: Request, task_id: str, body: Answer) -> dict:
    tasks = tasks_of(request)
    task = tasks.get(task_id)
    if task is None or not task.needs_input:
        raise HTTPException(409, "Cette tâche n'attend pas de réponse.")
    # Les confirmations d'envoi (field « confirm_… ») ne sont jamais mémorisées comme une réponse de formulaire.
    if (
        body.remember
        and task.needs_input.get("question")
        and not str(task.needs_input.get("field", "")).startswith("confirm")
    ):
        settings.remember_answer(task.needs_input["question"], body.value)
    if not tasks.answer(task_id, body.value):
        raise HTTPException(409, "Cette tâche n'attend plus de réponse.")
    return {"ok": True}

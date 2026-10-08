"""Analyses (collecte), offres, annotations, entraînement."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ... import matching, nlp, settings, storage
from .. import services
from ..guards import require_browser_ready
from .core import tasks_of

router = APIRouter()


class RunBody(BaseModel):
    scrape: bool = True
    keyword: str | None = None
    location: str | None = None
    remote_only: bool | None = None
    caution: str | None = None
    max_items: int | None = None
    interpreter: str | None = None


@router.post("/runs", status_code=202)
def start_run(request: Request, body: RunBody) -> dict:
    tasks = tasks_of(request)
    if tasks.has_active("run"):
        raise HTTPException(409, "Une analyse est déjà en cours.")
    overrides = body.model_dump(exclude_none=True, exclude={"scrape", "interpreter"})
    try:
        search = services.build_request(overrides)
        interpreter_name = settings.validate_settings({"interpreter": body.interpreter or settings.load_settings()["interpreter"]})["interpreter"]
    except (ValueError, settings.SettingsError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if body.scrape:
        require_browser_ready()
        title = f"Collecte et analyse — {search.keyword}"
    else:
        title = f"Analyse de l'historique — {search.keyword}"
    task = tasks.submit(
        "run", title, lambda ctx: services.run_analysis(ctx, search, scrape=body.scrape, interpreter_name=interpreter_name),
        browser=body.scrape)
    return task.public()


@router.post("/runs/rescore", status_code=202)
def start_rescore(request: Request) -> dict:
    tasks = tasks_of(request)
    if tasks.has_active("rescore"):
        raise HTTPException(409, "Un recalcul est déjà en cours.")
    try:
        services.require_results()
    except services.NoAnalysisError as exc:
        raise HTTPException(409, str(exc)) from exc
    return tasks.submit("rescore", "Recalcul des scores", services.rescore).public()


@router.get("/jobs")
def list_jobs(min_score: float = 0, label: str | None = None, status: str | None = None, q: str | None = None) -> dict:
    results = services.latest_results()
    if results is None:
        return {"report": None, "jobs": []}
    rows = [r for r in services.jobs_overview(results) if r["score"] >= min_score]
    if label == "pertinente":
        rows = [r for r in rows if r["label"] == 1]
    elif label == "non_pertinente":
        rows = [r for r in rows if r["label"] == 0]
    elif label == "non_annotee":
        rows = [r for r in rows if r["label"] is None]
    if status == "aucune":
        rows = [r for r in rows if not r["application_status"]]
    elif status:
        rows = [r for r in rows if r["application_status"] == status]
    if q:
        needle = nlp.normalize(q)
        rows = [r for r in rows if needle in nlp.normalize(f"{r['title']} {r['company']} {r['location']}")]
    for row in rows:
        row.pop("summary", None)
    return {"report": results["_report"], "generated_at": results["generated_at"], "keyword": results["request"]["keyword"],
            "model": results["model"], "min_labels": results.get("min_labels", matching.MIN_LABELS), "jobs": rows}


@router.get("/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    try:
        results = services.require_results()
    except services.NoAnalysisError as exc:
        raise HTTPException(404, str(exc)) from exc
    job = services.find_job(results, job_id)
    if job is None:
        raise HTTPException(404, "Offre introuvable.")
    item = storage.find_item(job_id)
    row = next(r for r in services.jobs_overview(results) if r["id"] == job_id)
    return {**row, "text": (item.text[:8000] if item else ""), "owned_skills": results["profile"]["skills"]}


class LabelBody(BaseModel):
    value: int | None


@router.put("/jobs/{job_id}/label")
def set_label(job_id: str, body: LabelBody) -> dict:
    if body.value not in (0, 1, None):
        raise HTTPException(422, "value doit valoir 1, 0 ou null.")
    labels = storage.load_labels()
    if body.value is None:
        labels.pop(job_id, None)
    else:
        labels[job_id] = body.value
    storage.save_labels(labels)
    return {"id": job_id, "label": body.value, "n_labels": len(labels), "min_labels": matching.MIN_LABELS}


@router.post("/model/train", status_code=202)
def train(request: Request) -> dict:
    tasks = tasks_of(request)
    if tasks.has_active("train"):
        raise HTTPException(409, "Un entraînement est déjà en cours.")
    try:
        services.require_results()
    except services.NoAnalysisError as exc:
        raise HTTPException(409, str(exc)) from exc
    return tasks.submit("train", "Entraînement du modèle de matching", services.train_model).public()


@router.post("/jobs/{job_id}/prepare", status_code=202)
def prepare(request: Request, job_id: str) -> dict:
    tasks = tasks_of(request)
    if tasks.has_active("prepare", job_id):
        raise HTTPException(409, "La préparation de cette offre est déjà en cours.")
    try:
        results = services.require_results()
    except services.NoAnalysisError as exc:
        raise HTTPException(409, str(exc)) from exc
    job = services.find_job(results, job_id)
    if job is None:
        raise HTTPException(404, "Offre introuvable.")
    return tasks.submit("prepare", f"Préparer : {job['title']}", lambda ctx: services.prepare_application(ctx, job_id),
                        meta={"key": job_id, "job_id": job_id}).public()

"""Candidatures : suivi, édition du CV/message, contacts."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from typing import Literal

from pydantic import BaseModel, field_validator

from ... import activity, application, config, outreach, settings
from .. import services
from ..guards import require_browser_ready
from ..services import tidy
from .core import tasks_of

router = APIRouter()

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _entry(job_id: str) -> dict:
    try:
        return {"id": job_id, **application.get_application(job_id)}
    except application.ApplicationNotFound:
        raise HTTPException(404, "Aucune candidature préparée pour cette offre.") from None


def _clean(row: dict) -> dict:
    return {**row, "title": tidy(row.get("title", ""), 150), "company": tidy(row.get("company", ""), 80, drop=True)}


@router.get("/applications")
def list_applications() -> list[dict]:
    return [_clean(row) for row in application.list_applications()]


@router.get("/applications/{job_id}")
def get_application(job_id: str) -> dict:
    entry = _entry(job_id)
    try:
        files = application.read_files(entry)
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {**_clean(entry), **files, "statuses": list(application.STATUSES)}


class Contact(BaseModel):
    name: str = ""
    profile_url: str = ""
    email: str | None = None
    headline: str = ""
    source: str = "manuel"

    @field_validator("email")
    @classmethod
    def valid_email(cls, value):
        value = (value or "").strip()
        if value and not _EMAIL.match(value):
            raise ValueError("Adresse email invalide.")
        return value or None

    @field_validator("profile_url")
    @classmethod
    def linkedin_url(cls, value):
        value = (value or "").strip()
        if value and not re.match(r"^https://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+", value):
            raise ValueError("L'URL doit être un profil LinkedIn (https://www.linkedin.com/in/...).")
        return value


class ApplicationPatch(BaseModel):
    cv: str | None = None
    message: str | None = None
    advice: str | None = None
    status: str | None = None
    contacts: list[Contact] | None = None


@router.patch("/applications/{job_id}")
def patch_application(job_id: str, body: ApplicationPatch) -> dict:
    _entry(job_id)
    if body.status is not None:
        if body.status not in application.STATUSES:
            raise HTTPException(422, f"Statut inconnu. Valeurs : {', '.join(application.STATUSES)}.")
        application.set_status(job_id, body.status)
        application.add_history(job_id, "statut", status=body.status)
    if body.contacts is not None:
        application.update_application(job_id, contacts=[c.model_dump() for c in body.contacts])
    if any(v is not None for v in (body.cv, body.message, body.advice)):
        application.write_files(job_id, cv=body.cv, message=body.message, advice=body.advice)
    return get_application(job_id)


@router.get("/applications/{job_id}/cv.html")
def cv_html(job_id: str):
    path = application.folder_of(_entry(job_id)) / "cv.html"
    if not path.is_file():
        raise HTTPException(404, "CV introuvable.")
    # Sandbox : la page charge marked/DOMPurify depuis un CDN, elle ne doit pas toucher l'origine de l'API.
    return FileResponse(path, media_type="text/html",
                        headers={"Content-Security-Policy": "sandbox allow-scripts allow-modals allow-popups"})


@router.post("/applications/{job_id}/apply", status_code=202)
def apply(request: Request, job_id: str) -> dict:
    entry = _entry(job_id)
    tasks = tasks_of(request)
    if tasks.has_active("apply", job_id):
        raise HTTPException(409, "Une candidature est déjà en cours pour cette offre.")
    if entry["status"] in ("postulée", "entretien"):
        raise HTTPException(409, f"Cette candidature est déjà au statut « {entry['status']} ».")
    require_browser_ready(("apply",), settings.load_settings()["apply_daily_max"], "candidatures")
    title = tidy(entry.get("title", ""), 80)
    return tasks.submit("apply", f"Postuler : {title}", lambda ctx: services.apply_job(ctx, job_id), browser=True,
                        meta={"key": job_id, "job_id": job_id}).public()


@router.post("/applications/{job_id}/contacts/lookup", status_code=202)
def lookup_contacts(request: Request, job_id: str) -> dict:
    _entry(job_id)
    tasks = tasks_of(request)
    if tasks.has_active("contacts", job_id):
        raise HTTPException(409, "Une recherche de contacts est déjà en cours pour cette offre.")
    require_browser_ready()
    return tasks.submit("contacts", "Chercher les recruteurs de l'offre", lambda ctx: services.lookup_contacts(ctx, job_id),
                        browser=True, meta={"key": job_id, "job_id": job_id}).public()


@router.get("/applications/{job_id}/message-draft")
def message_draft(job_id: str) -> dict:
    _entry(job_id)
    return services.message_draft(job_id)


class MessageBody(BaseModel):
    channel: Literal["email", "linkedin"]
    to: str
    subject: str = ""
    body: str
    attach_cv: bool = True
    confirm: bool = False


@router.post("/applications/{job_id}/messages", status_code=202)
def send_message(request: Request, job_id: str, body: MessageBody) -> dict:
    """Envoie un message à l'entreprise. ``confirm: true`` est exigé : l'interface le pose après relecture."""
    _entry(job_id)
    if not body.confirm:
        raise HTTPException(422, "Confirmation explicite requise : relis le message puis confirme l'envoi.")
    limit = settings.load_settings()["message_daily_max"]
    if body.channel == "email":
        if config.smtp_config() is None:
            raise HTTPException(503, "SMTP non configuré : renseigne SMTP_HOST et SMTP_FROM dans le fichier .env.")
        try:
            outreach.validate_recipient(body.to)
        except outreach.OutreachError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not body.subject.strip() or "\n" in body.subject:
            raise HTTPException(422, "Objet requis, sur une seule ligne.")
        try:
            activity.check_quota(("message_email", "message_linkedin"), limit, "messages")
        except activity.QuotaExceeded as exc:
            raise HTTPException(429, str(exc)) from exc
    else:
        require_browser_ready(("message_email", "message_linkedin"), limit, "messages")
    tasks = tasks_of(request)
    if tasks.has_active("message", job_id):
        raise HTTPException(409, "Un envoi est déjà en cours pour cette candidature.")
    label = "email" if body.channel == "email" else "message LinkedIn"
    return tasks.submit(
        "message", f"Envoyer un {label}", lambda ctx: services.send_message(
            ctx, job_id, channel=body.channel, to=body.to.strip(), subject=body.subject, body=body.body, attach_cv=body.attach_cv),
        browser=body.channel == "linkedin", meta={"key": job_id, "job_id": job_id}).public()


class TestMail(BaseModel):
    to: str | None = None


@router.post("/smtp/test")
def smtp_test(body: TestMail) -> dict:
    """Email de test, uniquement vers soi-même (adresse de contact des paramètres ou expéditeur SMTP)."""
    smtp = config.smtp_config()
    if smtp is None:
        raise HTTPException(503, "SMTP non configuré : renseigne SMTP_HOST et SMTP_FROM dans le fichier .env.")
    own = {a for a in (settings.load_settings()["contact_email"], smtp["from"], smtp["user"]) if a}
    target = (body.to or settings.load_settings()["contact_email"] or smtp["from"]).strip()
    if target not in own:
        raise HTTPException(422, "Le test n'est autorisé que vers ton adresse de contact ou l'expéditeur SMTP.")
    try:
        return services.send_test_email(target)
    except outreach.OutreachError as exc:
        raise HTTPException(502, str(exc)) from exc

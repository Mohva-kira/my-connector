"""Contexte Documents (domaine #7, fichiers locaux) — contrôleur HTTP fin :
toute la logique vit dans DocumentService."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from email_analyzer.db.session import get_db
from email_analyzer.jobs import create_job, enqueue
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled
from email_analyzer.services.document_service import DocumentService

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: Optional[str]
    size_bytes: int
    project_id: Optional[uuid.UUID]
    has_extracted_text: bool
    # "upload" (dépôt manuel) ou "email" (pièce jointe ingérée automatiquement
    # par le Fast-Track, 2026-08-01) — `None` pour les lignes antérieures à
    # cette distinction, traitées comme "upload" côté frontend.
    source: Optional[str]
    created_at: datetime


def _to_out(d) -> DocumentOut:  # type: ignore[no-untyped-def]
    return DocumentOut(
        id=d.id,
        filename=d.filename,
        content_type=d.content_type,
        size_bytes=d.size_bytes,
        project_id=d.project_id,
        source=d.source,
        has_extracted_text=bool(d.extracted_text),
        created_at=d.created_at,
    )


@router.get("", response_model=List[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    project_id: Optional[uuid.UUID] = None,
    keyword: Optional[str] = None,
) -> List[DocumentOut]:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    docs = DocumentService(db).list_documents(tenant.id, project_id=project_id, keyword=keyword)
    return [_to_out(d) for d in docs]


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    project_id: Optional[uuid.UUID] = Form(None),
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> DocumentOut:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    data = await file.read()
    document = DocumentService(db).upload(
        tenant.id,
        user.id,
        filename=file.filename or "document",
        content_type=file.content_type,
        data=data,
        project_id=project_id,
    )
    return _to_out(document)


class BackfillJobOut(BaseModel):
    job_id: str
    status: str


@router.post("/backfill-from-emails", response_model=BackfillJobOut)
def backfill_from_emails(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> BackfillJobOut:
    """Rattrape rétroactivement les pièces jointes des emails DÉJÀ persistés
    (antérieurs à l'ingestion automatique du Fast-Track, 2026-08-01) en
    `Document` — tâche de fond arq
    (`analysis_tasks.run_email_attachment_backfill`), un seul passage IMAP
    sur tout l'historique connu en base (MIN(Email.received_at) des emails
    rattachés à un projet). Le client interroge `GET /api/analyze/{job_id}` —
    même contrat générique que `/api/brief/discover-projects`, aucune
    nouvelle route de polling. Sans effet de bord si relancé :
    `DocumentService.ingest_email_attachment` dédoublonne sur (tenant,
    projet, nom, taille)."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    job_id = create_job(tenant_id=str(tenant.id))
    enqueue(job_id, "run_email_attachment_backfill", str(tenant.id))
    return BackfillJobOut(job_id=job_id, status="pending")


@router.get("/{document_id}/download")
def download_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> Response:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    service = DocumentService(db)
    document = service.get_document(tenant.id, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document introuvable")
    data = service.read_file(tenant.id, document_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Fichier introuvable sur le disque")

    # Nom de fichier échappé (RFC 5987) : évite toute injection d'en-tête
    # via un nom de fichier utilisateur contenant des caractères de contrôle.
    safe_name = document.filename.replace("\r", "").replace("\n", "")
    encoded_name = quote(safe_name)
    return Response(
        content=data,
        media_type=document.content_type or "application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
    )


@router.delete("/{document_id}")
def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> dict:
    _require_saas()
    user, tenant, _ = authenticate_bearer(db, authorization)
    deleted = DocumentService(db).delete(tenant.id, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document introuvable")
    return {"ok": True}

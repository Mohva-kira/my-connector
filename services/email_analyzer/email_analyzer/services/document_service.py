"""Contexte Documents (domaine #7, fichiers locaux uniquement en v1 — voir
Document.__doc__, db/models.py). Le fichier vit sur disque, nommé par
`document_id` (jamais par le nom de fichier fourni par l'utilisateur) :
élimine tout risque de traversée de chemin sans avoir à valider/nettoyer un
nom de fichier arbitraire.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from email_analyzer.config import document_max_size_bytes, documents_storage_dir
from email_analyzer.db.models import Document, Project
from email_analyzer.document_extraction import extract_text
from email_analyzer.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, db: Session) -> None:
        self._db = db
        self._repo = DocumentRepository(db)

    def list_documents(
        self,
        tenant_id: uuid.UUID,
        project_id: Optional[uuid.UUID] = None,
        keyword: Optional[str] = None,
    ) -> List[Document]:
        return self._repo.list_for_tenant(tenant_id, project_id=project_id, keyword=keyword)

    def get_document(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> Optional[Document]:
        return self._repo.get(tenant_id, document_id)

    def upload(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str,
        content_type: Optional[str],
        data: bytes,
        project_id: Optional[uuid.UUID] = None,
    ) -> Document:
        max_size = document_max_size_bytes()
        if len(data) > max_size:
            raise HTTPException(
                status_code=413, detail=f"Fichier trop volumineux (max {max_size // (1024 * 1024)} Mo)"
            )
        if not data:
            raise HTTPException(status_code=422, detail="Fichier vide")

        if project_id is not None:
            project = (
                self._db.query(Project)
                .filter(Project.id == project_id, Project.tenant_id == tenant_id)
                .first()
            )
            if project is None:
                raise HTTPException(status_code=404, detail="Projet introuvable")

        return self._persist(tenant_id, user_id, project_id, filename, content_type, data, source="upload")

    def ingest_email_attachment(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
        filename: str,
        content_type: Optional[str],
        data: bytes,
    ) -> Optional[Document]:
        """Ingestion automatique d'une pièce jointe d'email (Fast-Track, voir
        analysis_tasks.py::_ingest_new_email_attachments) — appelée depuis un
        job arq, pas une requête HTTP : ne lève jamais (contrairement à
        `upload`), un fichier trop volumineux ou vide est simplement ignoré
        (`None`) plutôt que d'interrompre le reste du Fast-Track. Dédoublonne
        sur (tenant, projet, nom, taille) — voir
        `DocumentRepository.find_existing` — plutôt que de recréer le même
        document à chaque re-scan du même email."""
        max_size = document_max_size_bytes()
        if not data or len(data) > max_size:
            return None
        existing = self._repo.find_existing(tenant_id, project_id, filename, len(data))
        if existing is not None:
            return existing
        return self._persist(tenant_id, user_id, project_id, filename, content_type, data, source="email")

    def _persist(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        project_id: Optional[uuid.UUID],
        filename: str,
        content_type: Optional[str],
        data: bytes,
        source: str,
    ) -> Document:
        document_id = uuid.uuid4()
        extracted = extract_text(content_type, filename, data)

        path = self._storage_path(tenant_id, document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        try:
            return self._repo.create(
                document_id=document_id,
                tenant_id=tenant_id,
                user_id=user_id,
                project_id=project_id,
                filename=filename,
                content_type=content_type,
                size_bytes=len(data),
                extracted_text=extracted,
                source=source,
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def read_file(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> Optional[bytes]:
        path = self._storage_path(tenant_id, document_id)
        if not path.is_file():
            return None
        return path.read_bytes()

    def delete(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        deleted = self._repo.delete(tenant_id, document_id)
        if deleted:
            self._storage_path(tenant_id, document_id).unlink(missing_ok=True)
        return deleted

    def _storage_path(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> Path:
        return documents_storage_dir() / str(tenant_id) / str(document_id)

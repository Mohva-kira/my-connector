"""Data access for the `documents` table (Contexte Documents, domaine #7).

Same pattern as MemoryRepository/ObjectiveRepository: single place issuing
SQL for this concept, no query logic in the service or the router.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from email_analyzer.db.models import Document


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def list_for_tenant(
        self,
        tenant_id: uuid.UUID,
        project_id: Optional[uuid.UUID] = None,
        keyword: Optional[str] = None,
        limit: int = 200,
    ) -> List[Document]:
        query = self._db.query(Document).filter(Document.tenant_id == tenant_id)
        if project_id is not None:
            query = query.filter(Document.project_id == project_id)
        if keyword:
            needle = f"%{keyword.strip()}%"
            query = query.filter(
                Document.filename.ilike(needle) | Document.extracted_text.ilike(needle)
            )
        return query.order_by(Document.created_at.desc()).limit(limit).all()

    def get(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> Optional[Document]:
        return (
            self._db.query(Document)
            .filter(Document.id == document_id, Document.tenant_id == tenant_id)
            .first()
        )

    def create(
        self,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        project_id: Optional[uuid.UUID],
        filename: str,
        content_type: Optional[str],
        size_bytes: int,
        extracted_text: Optional[str],
        source: Optional[str] = None,
    ) -> Document:
        """``document_id`` fourni par l'appelant (pas le défaut du modèle) :
        DocumentService en a besoin *avant* l'insertion pour nommer le
        fichier sur disque du même id que la ligne qui le référence."""
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            project_id=project_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            extracted_text=extracted_text,
            source=source,
        )
        self._db.add(document)
        self._db.commit()
        self._db.refresh(document)
        return document

    def find_existing(
        self,
        tenant_id: uuid.UUID,
        project_id: Optional[uuid.UUID],
        filename: str,
        size_bytes: int,
    ) -> Optional[Document]:
        """Dédoublonnage best-effort pour l'ingestion automatique de pièces
        jointes (voir DocumentService.ingest_email_attachment) : (nom, taille)
        n'est pas une preuve cryptographique d'identité de contenu, mais
        suffit à éviter de recréer le même document à chaque re-scan d'un
        même email (ex. bouton "chercher sur plus de jours") sans introduire
        d'identifiant stable par pièce jointe (hors périmètre v1)."""
        return (
            self._db.query(Document)
            .filter(
                Document.tenant_id == tenant_id,
                Document.project_id == project_id,
                Document.filename == filename,
                Document.size_bytes == size_bytes,
            )
            .first()
        )

    def delete(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        document = self.get(tenant_id, document_id)
        if document is None:
            return False
        self._db.delete(document)
        self._db.commit()
        return True

"""SearchDocumentsTool — recherche parmi les documents locaux téléversés
(nom de fichier + texte extrait quand disponible), optionnellement filtrés
par projet. Read-only, no LLM call — thin wrapper over DocumentService,
même patron que SearchMemoryTool/SearchEmailsTool."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from email_analyzer.services.document_service import DocumentService
from email_analyzer.text_snippets import extract_snippet


class SearchDocumentsTool:
    name = "search_documents"
    description = (
        "Recherche parmi les documents locaux téléversés (contrats, PDF, notes, ...), "
        "par mot-clé dans le nom de fichier ou le contenu extrait, optionnellement "
        "filtrés par projet. Ne retourne pas le contenu complet, seulement les "
        "métadonnées et, quand un mot-clé matche dans le contenu, un extrait du "
        "passage trouvé."
    )

    def __init__(self, db: Session) -> None:
        self._service = DocumentService(db)

    def run(
        self,
        tenant_id: uuid.UUID,
        project_id: Optional[uuid.UUID] = None,
        keyword: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        documents = self._service.list_documents(tenant_id, project_id=project_id, keyword=keyword)
        results = []
        for d in documents:
            entry = {
                "document_id": str(d.id),
                "filename": d.filename,
                "content_type": d.content_type,
                "project_id": str(d.project_id) if d.project_id else None,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            if keyword:
                entry["snippet"] = extract_snippet(d.extracted_text, keyword)
            results.append(entry)
        return results

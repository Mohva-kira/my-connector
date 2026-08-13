"""Extraits de texte pour les résultats d'outils du chat assistant.

Module pur (aucun accès DB) — évite de renvoyer un corps d'email/document/
résumé entier au LLM (coût contexte, voir search_emails_tool.py,
search_documents_tool.py, search_project_summaries_tool.py) tout en donnant
un passage réellement pertinent plutôt qu'un simple "trouvé : oui/non".
"""

from __future__ import annotations

from typing import Optional


def extract_snippet(
    text: Optional[str], keyword: Optional[str], context_chars: int = 160
) -> Optional[str]:
    """Retourne un extrait de ``text`` centré sur la première occurrence
    (insensible à la casse) de ``keyword``, ou le début de ``text`` si
    ``keyword`` est absent ou ne s'y trouve pas (le mot-clé a alors matché
    via un autre champ, ex. le sujet d'un email ou le nom d'un document).
    ``None`` si ``text`` est ``None``."""
    if text is None:
        return None

    if keyword:
        index = text.lower().find(keyword.lower())
        if index != -1:
            start = max(0, index - context_chars)
            end = min(len(text), index + len(keyword) + context_chars)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(text) else ""
            return f"{prefix}{text[start:end].strip()}{suffix}"

    truncated = text[: 2 * context_chars]
    suffix = "…" if len(text) > len(truncated) else ""
    return f"{truncated.strip()}{suffix}"

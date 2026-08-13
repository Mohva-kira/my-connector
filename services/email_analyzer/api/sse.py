"""Server-Sent Events — première route de streaming de ce repo (voir
context/architecture.md §7). Un seul petit helper de formatage : pas un
framework de streaming générique, puisqu'une seule route
(POST /api/assistant/conversations/{id}/chat) en a besoin aujourd'hui."""

from __future__ import annotations

import json
from typing import Any, Dict


def format_sse_event(event: str, data: Dict[str, Any]) -> str:
    """Un bloc `event: <type>\\ndata: <json>\\n\\n` — le format `event:` texte
    permet au client de dispatcher sans reparser `data.type` en double."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

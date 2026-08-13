"""Conversation layer for the portfolio assistant.

Part of the Personal Executive AI Assistant pivot (see
context/project-overview.md, pillar "Conversation"). Orchestrates
multi-conversation history (Conversation + AssistantMessage,
ConversationRepository) on top of the existing LLM chat plumbing.

System-prompt/context building has moved to ContextService (Phase 7,
context_service.py) — this service no longer reads Project/ProjectSummary/
SuggestedAction directly, it asks ContextService for the unified prompt.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterator, List, Literal, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from email_analyzer.config import CHAT_INTERACTIVE_MAX_OUTPUT_TOKENS, chat_llm_timeout_seconds
from email_analyzer.db.models import AssistantMessage, Conversation
from email_analyzer.llm import (
    get_gemini_api_key,
    get_openai_api_key,
    select_assistant_provider,
    stream_portfolio_assistant_chat_gemini,
    stream_portfolio_assistant_chat_openai,
)
from email_analyzer.repositories.conversation_repository import ConversationRepository
from email_analyzer.services.context_service import ContextService
from email_analyzer.services.tool_executor import TOOL_SPECS, ToolExecutor

_TITLE_MAX_LEN = 60

logger = logging.getLogger(__name__)

AssistantProvider = Literal["openai", "gemini"]

# Historique borné envoyé au LLM à chaque tour de chat interactif — le
# défaut du repository (100) fait grossir le prompt (et la latence de
# génération) sans limite avec l'ancienneté de la conversation.
_CHAT_HISTORY_LIMIT = 20

# Borne le nombre d'aller-retours LLM<->tools au sein d'un même message
# utilisateur (boucle agentique, voir stream_message) — un backstop contre
# une latence qui s'emballerait si le modèle enchaînait les appels d'outils
# sans jamais conclure, pas une configurabilité demandée par un besoin réel.
_MAX_TOOL_ROUNDS = 4

# Détection best-effort d'un timeout côté fournisseur LLM (ni l'un ni
# l'autre SDK n'expose un champ dédié dans le message d'erreur générique
# renvoyé par portfolio_assistant_chat_*) — un timeout doit basculer sur le
# résumé de secours (ContextService.quick_fallback_summary), pas sur une
# erreur 502 brute.
_TIMEOUT_ERROR_HINTS = ("timeout", "timed out", "deadline")


def _looks_like_timeout(erreur: Optional[str]) -> bool:
    if not erreur:
        return False
    lowered = erreur.lower()
    return any(hint in lowered for hint in _TIMEOUT_ERROR_HINTS)


class ConversationService:
    """Orchestrates the portfolio assistant chat: context building,
    provider selection, LLM call and history persistence."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._repo = ConversationRepository(db)
        self._context = ContextService(db)

    def list_conversations(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> List[Conversation]:
        return self._repo.list_conversations(tenant_id, user_id)

    def create_conversation(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> Conversation:
        return self._repo.create_conversation(tenant_id, user_id)

    def _require_conversation(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation:
        conversation = self._repo.get_conversation(tenant_id, user_id, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation introuvable")
        return conversation

    def list_history(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> List[AssistantMessage]:
        self._require_conversation(tenant_id, user_id, conversation_id)
        return self._repo.list_messages(conversation_id, limit)

    def list_history_for_active_conversation(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int = 100
    ) -> List[AssistantMessage]:
        """Most-recently-active conversation's history — used by callers
        (e.g. memory extraction) that reason about "the current chat"
        rather than a specific conversation_id."""
        conversations = self._repo.list_conversations(tenant_id, user_id)
        if not conversations:
            return []
        return self._repo.list_messages(conversations[0].id, limit)

    def stream_message(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        message: str,
        preferred_provider: Optional[AssistantProvider],
        openai_model: str,
        gemini_model: str,
    ) -> Iterator[Dict[str, Any]]:
        """Boucle agentique du chat portefeuille : contrairement à l'ancien
        ``send_message`` (un seul appel LLM, retour synchrone, supprimé),
        le modèle peut désormais appeler des tools
        (``email_analyzer.services.tool_executor.ToolExecutor``) au fil de
        la réponse — le contexte déjà persisté (``ContextService``, Projects
        + Memory + Knowledge) reste fourni en premier à chaque tour ; les
        tools ne sont exécutés que si le LLM les demande lui-même
        (function-calling natif OpenAI/Gemini), jamais par une sélection
        déterministe par mots-clés (ancien
        ``email_analyzer/agent/planner.py``, supprimé).

        Toute validation (message vide, conversation introuvable,
        fournisseur non configuré) lève encore une vraie ``HTTPException``
        *avant* le premier ``yield`` — l'appelant
        (``api/routers/assistant.py``) n'a donc pas encore commencé à
        streamer de réponse HTTP à ce stade. Émet ensuite des événements
        ``{"type": "token"|"tool_call"|"tool_result"|"done"|"error", ...}``,
        consommés par l'endpoint SSE."""
        message = message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="Message vide")

        conversation = self._require_conversation(tenant_id, user_id, conversation_id)
        provider = self._select_provider(preferred_provider)

        logger.info(
            "conversation.stream_message start tenant=%s user=%s conversation=%s",
            tenant_id,
            user_id,
            conversation_id,
        )

        prior = self._repo.list_messages(conversation_id, _CHAT_HISTORY_LIMIT)
        turns: List[Dict[str, Any]] = [{"role": m.role, "content": m.content} for m in prior] + [
            {"role": "user", "content": message}
        ]

        system_prompt = self._context.build_portfolio_system_prompt(tenant_id, user_id)
        timeout = chat_llm_timeout_seconds()
        tool_executor = ToolExecutor(self._db)

        assistant_text: Optional[str] = None
        fallback_used = False

        for _round in range(_MAX_TOOL_ROUNDS):
            events = (
                stream_portfolio_assistant_chat_gemini(
                    system_prompt,
                    turns,
                    gemini_model,
                    get_gemini_api_key(),
                    tools=TOOL_SPECS,
                    max_tokens=CHAT_INTERACTIVE_MAX_OUTPUT_TOKENS,
                    timeout=timeout,
                )
                if provider == "gemini"
                else stream_portfolio_assistant_chat_openai(
                    system_prompt,
                    turns,
                    openai_model,
                    get_openai_api_key(),
                    tools=TOOL_SPECS,
                    max_tokens=CHAT_INTERACTIVE_MAX_OUTPUT_TOKENS,
                    timeout=timeout,
                )
            )

            round_outcome: Optional[str] = None
            # Accumulé localement en plus d'être re-émis au client : si une
            # vraie erreur survient après que du texte a déjà été streamé
            # (possible seulement depuis ce chantier, l'ancien chat était
            # tout-ou-rien), ce texte déjà vu par l'utilisateur ne doit pas
            # disparaître silencieusement d'une conversation qui, elle,
            # resterait vide.
            partial_text: List[str] = []

            for event in events:
                etype = event["type"]
                if etype == "token":
                    partial_text.append(event["delta"])
                    yield {"type": "token", "delta": event["delta"]}
                elif etype == "tool_calls":
                    round_outcome = "tool_calls"
                    calls = event["calls"]
                    turns.append({"role": "assistant", "content": None, "tool_calls": calls})
                    for call in calls:
                        yield {"type": "tool_call", "name": call["name"], "status": "start"}
                    results = {r["id"]: r for r in tool_executor.execute_calls(calls, tenant_id, user_id)}
                    for call in calls:
                        result = results.get(call["id"]) or {
                            "content": '{"error": "Pas de résultat"}',
                            "is_error": True,
                        }
                        turns.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": call["name"],
                                "content": result["content"],
                            }
                        )
                        yield {
                            "type": "tool_result",
                            "name": call["name"],
                            "status": "error" if result["is_error"] else "done",
                        }
                elif etype == "done":
                    round_outcome = "done"
                    assistant_text = event["message"]
                elif etype == "error":
                    round_outcome = "error"
                    erreur = event["erreur"]
                    if _looks_like_timeout(erreur):
                        logger.warning(
                            "conversation.stream_message llm_timeout_fallback tenant=%s user=%s error=%s",
                            tenant_id,
                            user_id,
                            erreur,
                        )
                        assistant_text = self._context.quick_fallback_summary(tenant_id)
                        fallback_used = True
                    elif partial_text:
                        logger.warning(
                            "conversation.stream_message partial_stream_error tenant=%s user=%s error=%s",
                            tenant_id,
                            user_id,
                            erreur,
                        )
                        assistant_text = (
                            "".join(partial_text).strip()
                            + "\n\n*(Réponse interrompue par une erreur de connexion.)*"
                        )
                    else:
                        logger.error(
                            "conversation.stream_message llm_error tenant=%s user=%s error=%s",
                            tenant_id,
                            user_id,
                            erreur,
                        )
                        yield {"type": "error", "erreur": erreur}
                        return

            if round_outcome in ("done", "error"):
                break
        else:
            # _MAX_TOOL_ROUNDS épuisé sans réponse finale — traité comme le
            # fallback timeout plutôt que de boucler indéfiniment.
            logger.warning(
                "conversation.stream_message max_tool_rounds_exhausted tenant=%s user=%s",
                tenant_id,
                user_id,
            )
            assistant_text = self._context.quick_fallback_summary(tenant_id)
            fallback_used = True

        self._repo.append_turn(tenant_id, user_id, conversation_id, message, assistant_text or "")
        if not prior:
            self._repo.rename_if_default(conversation, self._title_from_message(message))
        logger.info(
            "conversation.stream_message done tenant=%s user=%s conversation=%s fallback=%s",
            tenant_id,
            user_id,
            conversation_id,
            fallback_used,
        )
        yield {"type": "done", "message": assistant_text or ""}

    @staticmethod
    def _title_from_message(message: str) -> str:
        if len(message) <= _TITLE_MAX_LEN:
            return message
        return message[:_TITLE_MAX_LEN].rstrip() + "…"

    def _select_provider(self, preferred: Optional[AssistantProvider]) -> AssistantProvider:
        provider = select_assistant_provider(preferred)
        if provider is None:
            raise HTTPException(
                status_code=503,
                detail="Aucun fournisseur LLM configuré (OPENAI_API_KEY / GEMINI_API_KEY).",
            )
        return provider

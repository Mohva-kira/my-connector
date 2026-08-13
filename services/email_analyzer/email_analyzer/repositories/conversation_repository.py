"""Data access for the persistent portfolio assistant history.

Wraps the ``Conversation`` and ``AssistantMessage`` tables (db/models.py).
A tenant+user can have multiple conversations (new-conversation button +
history panel); each message belongs to exactly one. This repository is
the single place that issues SQL against these tables so the service
layer doesn't duplicate queries.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from email_analyzer.db.models import AssistantMessage, Conversation

DEFAULT_TITLE = "Nouvelle conversation"


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create_conversation(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, title: str = DEFAULT_TITLE
    ) -> Conversation:
        conversation = Conversation(tenant_id=tenant_id, user_id=user_id, title=title)
        self._db.add(conversation)
        self._db.commit()
        self._db.refresh(conversation)
        return conversation

    def list_conversations(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> List[Conversation]:
        return (
            self._db.query(Conversation)
            .filter(Conversation.tenant_id == tenant_id, Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    def get_conversation(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Optional[Conversation]:
        return (
            self._db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.tenant_id == tenant_id,
                Conversation.user_id == user_id,
            )
            .first()
        )

    def rename_if_default(self, conversation: Conversation, title: str) -> None:
        if conversation.title != DEFAULT_TITLE:
            return
        conversation.title = title
        self._db.add(conversation)
        self._db.commit()

    def list_messages(
        self, conversation_id: uuid.UUID, limit: int = 100
    ) -> List[AssistantMessage]:
        return (
            self._db.query(AssistantMessage)
            .filter(AssistantMessage.conversation_id == conversation_id)
            .order_by(AssistantMessage.created_at.asc())
            .limit(limit)
            .all()
        )

    def append_turn(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_content: str,
        assistant_content: str,
    ) -> None:
        self._db.add(
            AssistantMessage(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                role="user",
                content=user_content,
            )
        )
        self._db.add(
            AssistantMessage(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_content,
            )
        )
        self._db.query(Conversation).filter(Conversation.id == conversation_id).update(
            {"updated_at": datetime.utcnow()}
        )
        self._db.commit()

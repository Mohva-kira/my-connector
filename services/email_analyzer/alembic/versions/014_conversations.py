"""Add conversations table; scope assistant_messages by conversation.

Multi-conversation support for the portfolio assistant: until now
AssistantMessage was scoped only by (tenant_id, user_id), so every message
ever sent formed a single continuous thread per user (see the old
docstrings on AssistantMessage / ConversationRepository). This migration
introduces an explicit Conversation per thread and backfills existing
history into one conversation per (tenant_id, user_id) pair so no data is
lost.

Revision ID: 014_conversations
Revises: 013_objectives
Create Date: 2026-07-19

"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "014_conversations"
down_revision = "013_objectives"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_conversations_tenant_id", "conversations", ["tenant_id"])
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    op.add_column(
        "assistant_messages",
        sa.Column(
            "conversation_id", UUID(as_uuid=True), sa.ForeignKey("conversations.id"), nullable=True
        ),
    )

    # Backfill: one conversation per existing (tenant_id, user_id) pair,
    # covering every message sent before this migration.
    conn = op.get_bind()
    pairs = conn.execute(
        sa.text("SELECT DISTINCT tenant_id, user_id FROM assistant_messages")
    ).fetchall()
    for tenant_id, user_id in pairs:
        conversation_id = uuid.uuid4()
        conn.execute(
            sa.text(
                "INSERT INTO conversations (id, tenant_id, user_id, title, created_at, updated_at) "
                "SELECT :id, :tenant_id, :user_id, 'Conversation 1', MIN(created_at), MAX(created_at) "
                "FROM assistant_messages WHERE tenant_id = :tenant_id AND user_id = :user_id"
            ),
            {"id": conversation_id, "tenant_id": tenant_id, "user_id": user_id},
        )
        conn.execute(
            sa.text(
                "UPDATE assistant_messages SET conversation_id = :id "
                "WHERE tenant_id = :tenant_id AND user_id = :user_id"
            ),
            {"id": conversation_id, "tenant_id": tenant_id, "user_id": user_id},
        )

    op.alter_column("assistant_messages", "conversation_id", nullable=False)
    op.create_index(
        "ix_assistant_messages_conversation_id", "assistant_messages", ["conversation_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_assistant_messages_conversation_id", table_name="assistant_messages")
    op.drop_column("assistant_messages", "conversation_id")

    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_index("ix_conversations_tenant_id", table_name="conversations")
    op.drop_table("conversations")

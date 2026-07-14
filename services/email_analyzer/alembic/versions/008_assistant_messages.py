"""Add assistant_messages table.

Persists the portfolio assistant's conversation (see
email_analyzer/llm.py::build_portfolio_chat_system_prompt and
api/routers/assistant.py) — distinct from the existing per-project /api/chat,
which stays ephemeral (frontend-only state). One thread per (tenant, user)
pair; no separate "conversation" concept yet.

Revision ID: 008_assistant_messages
Revises: 007_user_login_timestamps
Create Date: 2026-07-12

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "008_assistant_messages"
down_revision = "007_user_login_timestamps"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant_messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_assistant_messages_tenant_id", "assistant_messages", ["tenant_id"]
    )
    op.create_index("ix_assistant_messages_user_id", "assistant_messages", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_assistant_messages_user_id", table_name="assistant_messages")
    op.drop_index("ix_assistant_messages_tenant_id", table_name="assistant_messages")
    op.drop_table("assistant_messages")

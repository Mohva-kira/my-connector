"""Add memories table.

Foundation of the Memory Engine pillar (Personal Executive AI Assistant
pivot, context/project-overview.md) — see
email_analyzer/services/memory_service.py. v1: one flat fact table per
(tenant, user), no vector search, no separate conversation/session concept
(source is always "conversation" for now, matching assistant_messages'
single continuous history per tenant/user pair).

Revision ID: 011_memory
Revises: 010_agenda
Create Date: 2026-07-17

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "011_memory"
down_revision = "010_agenda"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(16), nullable=False, server_default="fact"),
        sa.Column("source", sa.String(16), nullable=False, server_default="conversation"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memories_tenant_id", "memories", ["tenant_id"])
    op.create_index("ix_memories_user_id", "memories", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_memories_user_id", table_name="memories")
    op.drop_index("ix_memories_tenant_id", table_name="memories")
    op.drop_table("memories")

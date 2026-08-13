"""Add objectives, milestones tables.

Foundation of the Objectives pillar (Personal Executive AI Assistant
pivot, context/project-overview.md) — see
email_analyzer/services/objective_service.py. v1: no separate "Progress"
or "Blocking Issue" tables (progress computed on read, blocking_reason is
a plain column) — see Objective's docstring in db/models.py.

Revision ID: 013_objectives
Revises: 012_knowledge
Create Date: 2026-07-17

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "013_objectives"
down_revision = "012_knowledge"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "objectives",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("target_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="on_track"),
        sa.Column("blocking_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_objectives_tenant_id", "objectives", ["tenant_id"])
    op.create_index("ix_objectives_project_id", "objectives", ["project_id"])

    op.create_table(
        "milestones",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("objective_id", UUID(as_uuid=True), sa.ForeignKey("objectives.id"), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_milestones_objective_id", "milestones", ["objective_id"])


def downgrade() -> None:
    op.drop_index("ix_milestones_objective_id", table_name="milestones")
    op.drop_table("milestones")

    op.drop_index("ix_objectives_project_id", table_name="objectives")
    op.drop_index("ix_objectives_tenant_id", table_name="objectives")
    op.drop_table("objectives")

"""Add project_summary_history table.

Archives each ProjectSummary just before it gets overwritten in place by a
Fast-Track resync (see email_analyzer/analysis_tasks.py) — ProjectSummary
itself stays a single regenerated-in-place row (its own docstring), this is
the history that was missing. Populated going forward only: past summaries
already overwritten before this migration are not recoverable.

Revision ID: 019_project_summary_history
Revises: 018_document_source
Create Date: 2026-08-01

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "019_project_summary_history"
down_revision = "018_document_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_summary_history",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("structured_content", JSONB(), nullable=True),
        sa.Column("sentiment", sa.String(32), nullable=True),
        sa.Column("llm_risk_level", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_project_summary_history_project_id", "project_summary_history", ["project_id"])
    op.create_index("ix_project_summary_history_created_at", "project_summary_history", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_project_summary_history_created_at", table_name="project_summary_history")
    op.drop_index("ix_project_summary_history_project_id", table_name="project_summary_history")
    op.drop_table("project_summary_history")

"""Add structured_content, llm_risk_level, schema_version to project_summaries.

Structured LLM output (decisions/risques/deadlines/next_steps — see
email_analyzer/llm.py::ProjectSummaryLLM and context/rfc-email-pipeline-v2.md
§11), additive to the existing free-text `content` column. `llm_risk_level` is
stored separately from `sentiment` (derived from the deterministic
ai_intelligent.calculate_risk_score) and never merges the two — a disagreement
between them is a signal, not something to silently resolve. Nullable: rows
predate this extraction or the LLM call failed for that refresh (same
discipline as 004/005).

Revision ID: 006_structured_content
Revises: 005_email_tags_classification
Create Date: 2026-07-11

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "006_structured_content"
down_revision = "005_email_tags_classification"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("project_summaries", sa.Column("structured_content", JSONB(), nullable=True))
    op.add_column("project_summaries", sa.Column("llm_risk_level", sa.String(32), nullable=True))
    op.add_column(
        "project_summaries",
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("project_summaries", "schema_version")
    op.drop_column("project_summaries", "llm_risk_level")
    op.drop_column("project_summaries", "structured_content")

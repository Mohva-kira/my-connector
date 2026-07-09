"""Add importance_score to emails.

Per-email hybrid (rules + AI) importance score, 0-100, populated at
persistence time by the Fast-Track/scheduled sync pipeline (see
email_analyzer/ai_intelligent.py, score_email_importance and
progress-tracker.md Unit 19). Nullable: existing rows predate this column
and are left unscored rather than backfilled with a guess.

Revision ID: 004_email_importance_score
Revises: 003_projects
Create Date: 2026-07-09

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "004_email_importance_score"
down_revision = "003_projects"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("emails", sa.Column("importance_score", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("emails", "importance_score")

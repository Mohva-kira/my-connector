"""Add tags and classification_score to emails.

Rule-based tag taxonomy (Urgent/Bloquant/.../priority bucket) and a 0-100
multi-signal classification confidence score, populated at persistence time
by the Fast-Track/scheduled sync pipeline (see
email_analyzer/classification.py and progress-tracker.md). Nullable:
existing rows predate this classifier and are left untagged/unscored rather
than backfilled with a guess (same discipline as 004).

Note: classification_score reflects only the deterministic rules_matrix-based
signals (project name, keywords, sender email/domain, client/company names,
reference numbers) computed at persistence time — it does not include the
in-scan "known participants" reinforcement used during the IMAP scan itself,
which only affects which emails get selected, not the persisted score.

Revision ID: 005_email_tags_classification
Revises: 004_email_importance_score
Create Date: 2026-07-09

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "005_email_tags_classification"
down_revision = "004_email_importance_score"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("emails", sa.Column("tags", JSONB(), nullable=True))
    op.add_column("emails", sa.Column("classification_score", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("emails", "classification_score")
    op.drop_column("emails", "tags")

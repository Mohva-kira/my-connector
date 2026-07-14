"""Add rationale/stakeholder/advice columns to suggested_actions.

Backs the "detail" view of an action on the frontend Actions page: why it
matters, who is concerned, and advice to avoid the same situation again (see
email_analyzer/db/models.py::SuggestedAction). Nullable — actions created
before this migration, or created via the risk-recommendation fallback in
analysis_tasks.py, never populate these fields.

Revision ID: 009_suggested_action_detail
Revises: 008_assistant_messages
Create Date: 2026-07-13

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "009_suggested_action_detail"
down_revision = "008_assistant_messages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("suggested_actions", sa.Column("rationale", sa.Text(), nullable=True))
    op.add_column("suggested_actions", sa.Column("stakeholder", sa.String(200), nullable=True))
    op.add_column("suggested_actions", sa.Column("advice", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("suggested_actions", "advice")
    op.drop_column("suggested_actions", "stakeholder")
    op.drop_column("suggested_actions", "rationale")

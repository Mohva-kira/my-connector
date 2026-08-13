"""Add profiles table.

Foundation of the "Contexte Personnel" domain (Personal Executive AI
Assistant vision, context/project-overview.md — "Domaine #1 : la base de
tout"). One row per (tenant, user), unlike memories (many rows per pair) —
a profile is a singleton settings object the user edits directly, not an
LLM-extracted fact stream. Habits (wake/sleep/work hours, sport days) get
typed columns because their shape is fixed and known; preferences and
family members stay JSONB because those lists are open-ended by nature
(same distinction already applied to Project.rules_matrix/Email.tags).

Revision ID: 016_profile
Revises: 015_subject_tickets
Create Date: 2026-07-30

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "016_profile"
down_revision = "015_subject_tickets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("languages", ARRAY(sa.String()), nullable=True),
        sa.Column("timezone", sa.String(64), nullable=True),
        sa.Column("location", sa.String(255), nullable=True),
        sa.Column("preferences", JSONB(), nullable=True),
        sa.Column("family_members", JSONB(), nullable=True),
        sa.Column("favorite_contact_ids", ARRAY(UUID(as_uuid=True)), nullable=True),
        sa.Column("wake_time", sa.Time(), nullable=True),
        sa.Column("sleep_time", sa.Time(), nullable=True),
        sa.Column("work_start_time", sa.Time(), nullable=True),
        sa.Column("work_end_time", sa.Time(), nullable=True),
        sa.Column("sport_days", ARRAY(sa.String()), nullable=True),
        sa.Column("focus_hours", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("tenant_id", "user_id", name="uq_profiles_tenant_user"),
    )
    op.create_index("ix_profiles_tenant_id", "profiles", ["tenant_id"])
    op.create_index("ix_profiles_user_id", "profiles", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_profiles_user_id", table_name="profiles")
    op.drop_index("ix_profiles_tenant_id", table_name="profiles")
    op.drop_table("profiles")

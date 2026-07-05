"""Add projects, emails, project_summaries, suggested_actions tables.

Persistence layer required for the Fast-Track refresh (architecture.md,
Process 2) and the historical clustering worker: without a stored
`last_processed_email_timestamp` per project there is no delta to compute.

Revision ID: 003_projects
Revises: 002_gmail_oauth
Create Date: 2026-07-06

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "003_projects"
down_revision = "002_gmail_oauth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("rules_matrix", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_projects_tenant_id"), "projects", ["tenant_id"])

    op.create_table(
        "emails",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("external_id", sa.String(length=512), nullable=False),
        sa.Column("recipient_status", sa.String(length=16), nullable=False),
        sa.Column("subject", sa.Text(), nullable=True),
        sa.Column("body_encrypted", sa.Text(), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "external_id", name="uq_email_tenant_external_id"),
    )
    op.create_index(op.f("ix_emails_tenant_id"), "emails", ["tenant_id"])
    op.create_index(op.f("ix_emails_project_id"), "emails", ["project_id"])

    op.create_table(
        "project_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("sentiment", sa.String(length=32), nullable=True),
        sa.Column("last_processed_email_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", name="uq_project_summaries_project_id"),
    )
    op.create_index(op.f("ix_project_summaries_project_id"), "project_summaries", ["project_id"])

    op.create_table(
        "suggested_actions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_suggested_actions_project_id"), "suggested_actions", ["project_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_suggested_actions_project_id"), table_name="suggested_actions")
    op.drop_table("suggested_actions")

    op.drop_index(op.f("ix_project_summaries_project_id"), table_name="project_summaries")
    op.drop_table("project_summaries")

    op.drop_index(op.f("ix_emails_project_id"), table_name="emails")
    op.drop_index(op.f("ix_emails_tenant_id"), table_name="emails")
    op.drop_table("emails")

    op.drop_index(op.f("ix_projects_tenant_id"), table_name="projects")
    op.drop_table("projects")

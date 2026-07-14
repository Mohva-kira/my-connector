"""Add appointments table and probable-next-contact columns on project_summaries.

Backs the Agenda feature (see email_analyzer/db/models.py::Appointment and
::ProjectSummary): rendez-vous extraits des emails par l'IA, et une date de
retour probable pour les projets "en attente"/"en rouge" (sentiment
awaiting_feedback/under_tension ou llm_risk_level CRITIQUE). Tout est
nullable — aucune donnée existante n'est affectée.

Revision ID: 010_agenda
Revises: 009_suggested_action_detail
Create Date: 2026-07-13

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "010_agenda"
down_revision = "009_suggested_action_detail"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "appointments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("participants", JSONB(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="tentative"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_appointments_tenant_id", "appointments", ["tenant_id"])
    op.create_index("ix_appointments_project_id", "appointments", ["project_id"])
    op.create_index("ix_appointments_tenant_scheduled_at", "appointments", ["tenant_id", "scheduled_at"])

    op.add_column(
        "project_summaries",
        sa.Column("probable_next_contact_date", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "project_summaries",
        sa.Column("probable_next_contact_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "project_summaries",
        sa.Column("probable_next_contact_confidence", sa.String(32), nullable=True),
    )
    op.add_column(
        "project_summaries",
        sa.Column("agenda_updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("project_summaries", "agenda_updated_at")
    op.drop_column("project_summaries", "probable_next_contact_confidence")
    op.drop_column("project_summaries", "probable_next_contact_reason")
    op.drop_column("project_summaries", "probable_next_contact_date")

    op.drop_index("ix_appointments_tenant_scheduled_at", table_name="appointments")
    op.drop_index("ix_appointments_project_id", table_name="appointments")
    op.drop_index("ix_appointments_tenant_id", table_name="appointments")
    op.drop_table("appointments")

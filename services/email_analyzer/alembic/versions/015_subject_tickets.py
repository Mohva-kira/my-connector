"""Add ticket_label and ticket_number to emails.

Deterministic extraction of the bracketed ticket tag some subjects carry
(``[Banque_Plateforme_NuméroTicket]``, see email_analyzer/subject_tickets.py),
populated at persistence time by the Fast-Track pipeline alongside
tags/classification_score (see 005_email_tags_classification). Dedicated
columns rather than JSONB: project-level ticket enumeration/counting needs a
cheap GROUP BY, unlike the low-query-volume rules_matrix/structured_content
JSONB fields. Nullable — most emails carry no ticket tag.

Revision ID: 015_subject_tickets
Revises: 014_conversations
Create Date: 2026-07-28

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "015_subject_tickets"
down_revision = "014_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("emails", sa.Column("ticket_label", sa.String(255), nullable=True))
    op.add_column("emails", sa.Column("ticket_number", sa.String(64), nullable=True))
    op.create_index(
        "ix_emails_project_ticket_label", "emails", ["project_id", "ticket_label"]
    )


def downgrade() -> None:
    op.drop_index("ix_emails_project_ticket_label", table_name="emails")
    op.drop_column("emails", "ticket_number")
    op.drop_column("emails", "ticket_label")

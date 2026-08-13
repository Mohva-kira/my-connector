"""Add source to documents.

Distinguishes manual uploads ("upload", /documents page) from attachments
ingested automatically from emails ("email", Fast-Track sync) — see
email_analyzer/analysis_tasks.py::_ingest_new_email_attachments. Nullable,
no backfill: existing rows predate this column and are treated as "upload"
by the UI without guessing (same discipline as 004/005 on emails).

Revision ID: 018_document_source
Revises: 017_documents
Create Date: 2026-08-01

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "018_document_source"
down_revision = "017_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "source")

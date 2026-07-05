"""Add oauth_connections table for Gmail/Outlook OAuth2.

Revision ID: 002_gmail_oauth
Revises: 001_initial_saas
Create Date: 2026-05-04

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "002_gmail_oauth"
down_revision = "001_initial_saas"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "oauth_connections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("access_token_encrypted", sa.Text(), nullable=False),
        sa.Column("refresh_token_encrypted", sa.Text(), nullable=True),
        sa.Column("token_expiry", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scopes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id", "provider", "email", name="uq_oauth_tenant_provider_email"
        ),
    )
    op.create_index(op.f("ix_oauth_connections_tenant_id"), "oauth_connections", ["tenant_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_oauth_connections_tenant_id"), table_name="oauth_connections")
    op.drop_table("oauth_connections")

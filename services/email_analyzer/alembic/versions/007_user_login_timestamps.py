"""Add last_login_at and previous_login_at to users.

Powers the "depuis votre dernière visite" framing on the Brief page
(GET /api/brief). `previous_login_at` — not `last_login_at` — is the
reference point used there: it is only shifted forward on the *next* login
(see api/routers/auth.py::login), so it stays stable for the whole current
session instead of resetting on every page reload. Nullable: existing users
have no login history to backfill.

Revision ID: 007_user_login_timestamps
Revises: 006_structured_content
Create Date: 2026-07-11

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "007_user_login_timestamps"
down_revision = "006_structured_content"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("previous_login_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "previous_login_at")
    op.drop_column("users", "last_login_at")

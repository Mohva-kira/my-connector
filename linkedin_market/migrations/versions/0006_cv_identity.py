"""Coordonnées du CV au format ATS : ``user_settings.full_name`` et ``user_settings.linkedin_url``.

``TEXT NOT NULL DEFAULT ''`` (vide = omis de l'en-tête du CV).

Révision : 0006
Révise : 0005
Créée le : 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("user_settings", sa.Column("full_name", sa.Text(), server_default="", nullable=False))
    op.add_column("user_settings", sa.Column("linkedin_url", sa.Text(), server_default="", nullable=False))


def downgrade() -> None:
    op.drop_column("user_settings", "linkedin_url")
    op.drop_column("user_settings", "full_name")

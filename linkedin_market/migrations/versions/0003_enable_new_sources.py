"""Activation des nouvelles sources : France Travail, Malt, WTTJ et import Indeed (vague 5).

Révision : 0003
Révise : 0002
Créée le : 2026-10-13
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CODES = "('france_travail', 'malt', 'wttj', 'indeed')"


def upgrade() -> None:
    op.execute(f"UPDATE sources SET enabled = true WHERE code IN {_CODES}")


def downgrade() -> None:
    op.execute(f"UPDATE sources SET enabled = false WHERE code IN {_CODES}")

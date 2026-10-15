"""Plafond de ``max_items`` porté à 100 pour les trois niveaux de prudence.

Révision : 0005
Révise : 0004
Créée le : 2026-10-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAME = "ck_user_settings_max_items"
_TABLE = "user_settings"


def _replace(cap: str) -> None:
    op.drop_constraint(op.f(_NAME), _TABLE, type_="check")
    op.create_check_constraint(op.f(_NAME), _TABLE, f"max_items IS NULL OR max_items BETWEEN 1 AND {cap}")


def upgrade() -> None:
    _replace("CASE caution WHEN 'prudent' THEN 100 WHEN 'normal' THEN 100 WHEN 'agressif' THEN 100 END")


def downgrade() -> None:
    op.execute(
        "UPDATE user_settings SET max_items = CASE caution WHEN 'prudent' THEN 15 WHEN 'normal' THEN 30 ELSE 60 END "
        "WHERE max_items > CASE caution WHEN 'prudent' THEN 15 WHEN 'normal' THEN 30 ELSE 60 END"
    )
    _replace("CASE caution WHEN 'prudent' THEN 15 WHEN 'normal' THEN 30 WHEN 'agressif' THEN 60 END")

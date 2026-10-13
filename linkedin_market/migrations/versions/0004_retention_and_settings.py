"""Rétention et réglages de lieu (vague 6) : tâche ``purge``, ``home_location`` et ``timezone``.

- ``tasks.kind`` accepte ``purge`` (CHECK ``ck_tasks_kind`` remplacé) ;
- ``user_settings.home_location`` et ``user_settings.timezone`` : ``TEXT NOT NULL DEFAULT ''`` (E-18).

Révision : 0004
Révise : 0003
Créée le : 2026-10-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KINDS = "'run', 'rescore', 'train', 'prepare', 'apply', 'contacts', 'message', 'import_profile', 'profile_skill'"


def _replace_kind_check(kinds: str) -> None:
    op.drop_constraint(op.f("ck_tasks_kind"), "tasks", type_="check")
    op.create_check_constraint(op.f("ck_tasks_kind"), "tasks", f"kind IN ({kinds})")


def upgrade() -> None:
    op.add_column("user_settings", sa.Column("home_location", sa.Text(), server_default="", nullable=False))
    op.add_column("user_settings", sa.Column("timezone", sa.Text(), server_default="", nullable=False))
    _replace_kind_check(f"{_KINDS}, 'purge'")


def downgrade() -> None:
    op.execute("DELETE FROM tasks WHERE kind = 'purge'")
    _replace_kind_check(_KINDS)
    op.drop_column("user_settings", "timezone")
    op.drop_column("user_settings", "home_location")

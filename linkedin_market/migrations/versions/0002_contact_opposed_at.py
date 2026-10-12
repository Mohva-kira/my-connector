"""Opposition d'un contact d'entreprise : ``company_contacts.opposed_at`` (I-13, vague 4).

Révision : 0002
Révise : 0001
Créée le : 2026-10-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("company_contacts", sa.Column("opposed_at", sa.TIMESTAMP(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("company_contacts", "opposed_at")

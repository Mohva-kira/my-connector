"""Add organizations, contacts, project_contact_links tables.

Foundation of the Knowledge Layer pillar (Personal Executive AI Assistant
pivot, context/project-overview.md) — see
email_analyzer/services/knowledge_service.py. v1: no graph database, plain
relational tables, populated from data the LLM pipeline already extracts
(ProjectSummary.structured_content["participants"/"companies"]), not a new
extraction path.

Revision ID: 012_knowledge
Revises: 011_memory
Create Date: 2026-07-17

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "012_knowledge"
down_revision = "011_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("tenant_id", "name", name="uq_organization_tenant_name"),
    )
    op.create_index("ix_organizations_tenant_id", "organizations", ["tenant_id"])

    op.create_table(
        "contacts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("tenant_id", "name", name="uq_contact_tenant_name"),
    )
    op.create_index("ix_contacts_tenant_id", "contacts", ["tenant_id"])
    op.create_index("ix_contacts_organization_id", "contacts", ["organization_id"])

    op.create_table(
        "project_contact_links",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("contact_id", UUID(as_uuid=True), sa.ForeignKey("contacts.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "contact_id", name="uq_project_contact_link"),
    )
    op.create_index("ix_project_contact_links_project_id", "project_contact_links", ["project_id"])
    op.create_index("ix_project_contact_links_contact_id", "project_contact_links", ["contact_id"])


def downgrade() -> None:
    op.drop_index("ix_project_contact_links_contact_id", table_name="project_contact_links")
    op.drop_index("ix_project_contact_links_project_id", table_name="project_contact_links")
    op.drop_table("project_contact_links")

    op.drop_index("ix_contacts_organization_id", table_name="contacts")
    op.drop_index("ix_contacts_tenant_id", table_name="contacts")
    op.drop_table("contacts")

    op.drop_index("ix_organizations_tenant_id", table_name="organizations")
    op.drop_table("organizations")

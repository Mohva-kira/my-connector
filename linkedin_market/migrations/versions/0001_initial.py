"""Schéma initial PostgreSQL 18 : tables de l'ERD, sources de référence, immuabilité des rapports (I-02).

Révision : 0001
Révise : —
Créée le : 2026-10-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Sources de référence (backend/03 § 3.3). Seule linkedin est activée : les autres sources arrivent avec leur
# intégration. Un changement de texte de risques = une migration ``UPDATE sources SET risk_version = '…'`` (I-16).
SOURCES = [
    {"code": "linkedin", "label": "LinkedIn", "access_mode": "navigateur", "risk_version": "2026-10", "enabled": True},
    {"code": "malt", "label": "Malt", "access_mode": "navigateur", "risk_version": "2026-10", "enabled": False},
    {
        "code": "wttj",
        "label": "Welcome to the Jungle",
        "access_mode": "navigateur",
        "risk_version": "2026-10",
        "enabled": False,
    },
    {
        "code": "france_travail",
        "label": "France Travail",
        "access_mode": "api_officielle",
        "risk_version": None,
        "enabled": False,
    },
    {"code": "indeed", "label": "Indeed", "access_mode": "import_manuel", "risk_version": None, "enabled": False},
]

FK_INDEXES = [
    ("ix_application_events_application_id", "application_events", ["application_id"], None),
    ("ix_company_contacts_user_id_application_id", "company_contacts", ["user_id", "application_id"], None),
    ("ix_outreach_messages_application_id", "outreach_messages", ["application_id"], None),
    ("ix_reports_task_id", "reports", ["task_id"], "task_id IS NOT NULL"),
    ("ix_collected_items_task_id", "collected_items", ["task_id"], "task_id IS NOT NULL"),
    ("ix_labels_user_id", "labels", ["user_id"], None),
]

# Immuabilité des rapports (I-02). Seule exception : la mise à NULL de ``task_id`` par la clé étrangère
# ``ON DELETE SET NULL`` lors de la purge des tâches (12 mois), qui ne touche pas au contenu du rapport.
REPORTS_IMMUTABLE = """
CREATE FUNCTION reports_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.task_id IS NOT NULL AND NEW.task_id IS NULL
     AND (to_jsonb(NEW) - 'task_id') = (to_jsonb(OLD) - 'task_id') THEN
    RETURN NEW;
  END IF;
  RAISE EXCEPTION 'Un rapport est immuable (I-02).';
END $$;
CREATE TRIGGER trg_reports_immutable BEFORE UPDATE ON reports
  FOR EACH ROW EXECUTE FUNCTION reports_immutable();
"""


def upgrade() -> None:
    op.create_table(
        "sources",
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("access_mode", sa.Text(), nullable=False),
        sa.Column("risk_version", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.CheckConstraint(
            "access_mode IN ('api_officielle', 'navigateur', 'import_manuel')", name=op.f("ck_sources_access_mode")
        ),
        sa.PrimaryKeyConstraint("code", name=op.f("pk_sources")),
    )
    sources = sa.table(
        "sources",
        sa.column("code", sa.String),
        sa.column("label", sa.Text),
        sa.column("access_mode", sa.Text),
        sa.column("risk_version", sa.Text),
        sa.column("enabled", sa.Boolean),
    )
    op.bulk_insert(sources, SOURCES)
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.create_table(
        "activity_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "data", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "kind IN ('run', 'apply', 'message_email', 'message_linkedin', 'profile_skill', "
            "'blocked', 'unblocked', 'error')",
            name=op.f("ck_activity_events_kind"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_activity_events_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_activity_events")),
    )
    op.create_index(
        "ix_activity_events_user_id_kind_at",
        "activity_events",
        ["user_id", "kind", sa.literal_column("at DESC")],
        unique=False,
    )
    # Fichiers historiques importés (db/import_json.py) : relance idempotente si le renommage a été interrompu.
    op.create_table(
        "legacy_imports",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("imported_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_legacy_imports_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "file_name", "sha256", name=op.f("pk_legacy_imports")),
    )
    op.create_table(
        "candidate_profiles",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("origin", sa.Text(), server_default="", nullable=False),
        sa.Column("title", sa.Text(), server_default="", nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("skills", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'::text[]"), nullable=False),
        sa.Column("years_experience", sa.Integer(), nullable=True),
        sa.Column("seniority", sa.Text(), server_default="", nullable=False),
        sa.Column("synced_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "seniority IN ('junior', 'confirme', 'senior', 'lead', '')", name=op.f("ck_candidate_profiles_seniority")
        ),
        sa.CheckConstraint(
            "years_experience IS NULL OR years_experience BETWEEN 0 AND 60",
            name=op.f("ck_candidate_profiles_years_experience"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_candidate_profiles_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_candidate_profiles")),
        sa.UniqueConstraint("user_id", name=op.f("uq_candidate_profiles_user_id")),
    )
    op.create_table(
        "matching_models",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("feature_names", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("n_positive", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("n_negative", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("learned", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("trained_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_matching_models_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_matching_models")),
        sa.UniqueConstraint("user_id", "version", name=op.f("uq_matching_models_user_id_version")),
    )
    op.create_index(
        op.f("ix_matching_models_user_id"),
        "matching_models",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.create_table(
        "memorized_answers",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("question_key", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_memorized_answers_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "question_key", name=op.f("pk_memorized_answers")),
    )
    op.create_table(
        "platform_sessions",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("source_code", sa.String(length=32), nullable=False),
        sa.Column("cookies_encrypted", sa.LargeBinary(), nullable=False),
        sa.Column("last_login_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_code"], ["sources.code"], name=op.f("fk_platform_sessions_source_code_sources")
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_platform_sessions_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "source_code", name=op.f("pk_platform_sessions")),
    )
    op.create_table(
        "profile_skill_overrides",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("skill", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("action IN ('ajoutée', 'retirée')", name=op.f("ck_profile_skill_overrides_action")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_profile_skill_overrides_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "skill", name=op.f("pk_profile_skill_overrides")),
    )
    op.create_table(
        "source_consents",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("source_code", sa.String(length=32), nullable=False),
        sa.Column("risk_version", sa.Text(), nullable=False),
        sa.Column("accepted_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("revoked_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["source_code"], ["sources.code"], name=op.f("fk_source_consents_source_code_sources")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_source_consents_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_source_consents")),
    )
    op.create_index(
        op.f("ix_source_consents_user_id_source_code"),
        "source_consents",
        ["user_id", "source_code"],
        unique=True,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.create_table(
        "suppression_list",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("recipient_hash", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "reason IN ('désinscription', 'refus_oral', 'plainte')", name=op.f("ck_suppression_list_reason")
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_suppression_list_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_suppression_list")),
        sa.UniqueConstraint("user_id", "recipient_hash", name=op.f("uq_suppression_list_user_id_recipient_hash")),
    )
    op.create_table(
        "tasks",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), server_default="", nullable=False),
        sa.Column("status", sa.Text(), server_default="en_attente", nullable=False),
        sa.Column("browser", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "params", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("needs_input", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "meta", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("logs", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'::text[]"), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('run', 'rescore', 'train', 'prepare', 'apply', 'contacts', 'message', "
            "'import_profile', 'profile_skill')",
            name=op.f("ck_tasks_kind"),
        ),
        sa.CheckConstraint(
            "status IN ('en_attente', 'en_cours', 'attente_saisie', 'attente_verification_manuelle', "
            "'terminée', 'échec', 'annulée')",
            name=op.f("ck_tasks_status"),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_tasks_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tasks")),
    )
    op.create_index(
        op.f("ix_tasks_user_id_status"),
        "tasks",
        ["user_id", "status"],
        unique=False,
        postgresql_where=sa.text("status NOT IN ('terminée', 'échec', 'annulée')"),
    )
    op.create_table(
        "user_settings",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("keyword", sa.Text(), server_default="", nullable=False),
        sa.Column("location", sa.Text(), server_default="", nullable=False),
        sa.Column("remote_only", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("caution", sa.Text(), server_default="prudent", nullable=False),
        sa.Column("max_items", sa.Integer(), nullable=True),
        sa.Column("profile_source", sa.Text(), server_default="auto", nullable=False),
        sa.Column("interpreter", sa.Text(), server_default="claude", nullable=False),
        sa.Column(
            "apply_min_score",
            sa.Numeric(precision=4, scale=1, asdecimal=False),
            server_default=sa.text("70"),
            nullable=False,
        ),
        sa.Column("apply_daily_max", sa.Integer(), server_default=sa.text("10"), nullable=False),
        sa.Column("message_daily_max", sa.Integer(), server_default=sa.text("15"), nullable=False),
        sa.Column("contact_email", sa.Text(), server_default="", nullable=False),
        sa.Column("contact_phone", sa.Text(), server_default="", nullable=False),
        sa.Column("signature", sa.Text(), server_default="", nullable=False),
        sa.Column("linkedin_skill_sync", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("caution IN ('prudent', 'normal', 'agressif')", name=op.f("ck_user_settings_caution")),
        sa.CheckConstraint("interpreter IN ('claude', 'local')", name=op.f("ck_user_settings_interpreter")),
        sa.CheckConstraint(
            "max_items IS NULL OR max_items BETWEEN 1 AND "
            "CASE caution WHEN 'prudent' THEN 15 WHEN 'normal' THEN 30 WHEN 'agressif' THEN 60 END",
            name=op.f("ck_user_settings_max_items"),
        ),
        sa.CheckConstraint(
            "profile_source IN ('auto', 'file', 'linkedin')", name=op.f("ck_user_settings_profile_source")
        ),
        sa.CheckConstraint("apply_daily_max BETWEEN 0 AND 200", name=op.f("ck_user_settings_apply_daily_max")),
        sa.CheckConstraint("apply_min_score BETWEEN 0 AND 100", name=op.f("ck_user_settings_apply_min_score")),
        sa.CheckConstraint("message_daily_max BETWEEN 0 AND 200", name=op.f("ck_user_settings_message_daily_max")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_user_settings_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", name=op.f("pk_user_settings")),
    )
    op.create_table(
        "collected_items",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("source_code", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=True),
        sa.Column("url", sa.Text(), server_default="", nullable=False),
        sa.Column("title", sa.Text(), server_default="", nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("skills", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'::text[]"), nullable=False),
        sa.Column("keywords", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'::text[]"), nullable=False),
        sa.Column("company", sa.Text(), server_default="", nullable=False),
        sa.Column("location", sa.Text(), server_default="", nullable=False),
        sa.Column("posted_at", sa.Date(), nullable=True),
        sa.Column("collected_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("purge_after", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint("kind <> 'person' OR text IS NULL", name=op.f("ck_collected_items_person_without_text")),
        sa.CheckConstraint("kind IN ('job', 'post', 'person')", name=op.f("ck_collected_items_kind")),
        sa.ForeignKeyConstraint(["source_code"], ["sources.code"], name=op.f("fk_collected_items_source_code_sources")),
        sa.ForeignKeyConstraint(
            ["task_id"], ["tasks.id"], name=op.f("fk_collected_items_task_id_tasks"), ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_collected_items_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_collected_items")),
        sa.UniqueConstraint(
            "user_id",
            "source_code",
            "kind",
            "external_id",
            name=op.f("uq_collected_items_user_id_source_code_kind_external_id"),
        ),
    )
    op.create_index(
        op.f("ix_collected_items_purge_after"),
        "collected_items",
        ["purge_after"],
        unique=False,
        postgresql_where=sa.text("purge_after IS NOT NULL"),
    )
    # Fenêtre de 30 jours des tendances : ``keywords @> ARRAY[:kw]`` puis tri par date de collecte.
    op.create_index(
        "ix_collected_items_keywords", "collected_items", ["keywords"], unique=False, postgresql_using="gin"
    )
    op.create_index(
        "ix_collected_items_user_id_collected_at",
        "collected_items",
        ["user_id", sa.literal_column("collected_at DESC")],
        unique=False,
    )
    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=True),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("keyword", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "summary", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("interpretation", sa.Text(), server_default="", nullable=False),
        sa.Column("interpreter_used", sa.Text(), nullable=False),
        sa.Column("html_path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("interpreter_used IN ('claude', 'local')", name=op.f("ck_reports_interpreter_used")),
        sa.CheckConstraint(
            "left(html_path, 1) NOT IN ('/', '\\') AND html_path !~ '^[A-Za-z]:'",
            name=op.f("ck_reports_html_path_relative"),
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], name=op.f("fk_reports_task_id_tasks"), ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_reports_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
        sa.UniqueConstraint("user_id", "slug", name=op.f("uq_reports_user_id_slug")),
    )
    op.execute(REPORTS_IMMUTABLE)
    op.create_table(
        "applications",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("item_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.Text(), server_default="préparée", nullable=False),
        sa.Column("apply_state", sa.Text(), server_default="", nullable=False),
        sa.Column("score", sa.Numeric(precision=4, scale=1, asdecimal=False), nullable=False),
        sa.Column("generator", sa.Text(), nullable=False),
        sa.Column("folder_path", sa.Text(), nullable=False),
        sa.Column("prepared_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "apply_state IN ('', 'en_cours', 'postulée', 'déjà_postulée', 'incertaine', 'échec')",
            name=op.f("ck_applications_apply_state"),
        ),
        sa.CheckConstraint("generator IN ('claude', 'local')", name=op.f("ck_applications_generator")),
        sa.CheckConstraint(
            "left(folder_path, 1) NOT IN ('/', '\\') AND folder_path !~ '^[A-Za-z]:'",
            name=op.f("ck_applications_folder_path_relative"),
        ),
        sa.CheckConstraint(
            "status IN ('préparée', 'contactée', 'postulée', 'entretien', 'refusée', 'abandonnée', 'externe', 'échec')",
            name=op.f("ck_applications_status"),
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["collected_items.id"],
            name=op.f("fk_applications_item_id_collected_items"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_applications_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_applications")),
        sa.UniqueConstraint("user_id", "item_id", name=op.f("uq_applications_user_id_item_id")),
    )
    op.create_table(
        "labels",
        sa.Column("item_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("value", sa.SmallInteger(), nullable=False),
        sa.Column("labeled_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("value IN (0, 1)", name=op.f("ck_labels_value")),
        sa.ForeignKeyConstraint(
            ["item_id"], ["collected_items.id"], name=op.f("fk_labels_item_id_collected_items"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_labels_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("item_id", name=op.f("pk_labels")),
    )
    op.create_table(
        "application_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("application_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "details", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "kind IN ('status', 'apply', 'message', 'contacts', 'note')", name=op.f("ck_application_events_kind")
        ),
        sa.ForeignKeyConstraint(
            ["application_id"],
            ["applications.id"],
            name=op.f("fk_application_events_application_id_applications"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_application_events")),
    )
    op.create_table(
        "company_contacts",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("application_id", sa.Uuid(), nullable=False),
        sa.Column("full_name", sa.Text(), nullable=False),
        sa.Column("headline", sa.Text(), server_default="", nullable=False),
        sa.Column("profile_url", sa.Text(), server_default="", nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), server_default="manuel", nullable=False),
        sa.Column("legal_basis", sa.Text(), server_default="interet_legitime", nullable=False),
        sa.Column("collected_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_contacted_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "purge_after",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now() + interval '3 years'"),
            nullable=False,
        ),
        sa.CheckConstraint("legal_basis IN ('interet_legitime')", name=op.f("ck_company_contacts_legal_basis")),
        sa.CheckConstraint(
            "source IN ('linkedin_equipe_recrutement', 'manuel')", name=op.f("ck_company_contacts_source")
        ),
        sa.ForeignKeyConstraint(
            ["application_id"],
            ["applications.id"],
            name=op.f("fk_company_contacts_application_id_applications"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_company_contacts_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_company_contacts")),
    )
    op.create_table(
        "outreach_messages",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("application_id", sa.Uuid(), nullable=False),
        sa.Column("contact_id", sa.Uuid(), nullable=True),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("recipient", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), server_default="", nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("cv_attached", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("status", sa.Text(), server_default="confirmé", nullable=False),
        sa.Column("smtp_message_id", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("confirmed_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("sent_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint("channel IN ('email', 'linkedin')", name=op.f("ck_outreach_messages_channel")),
        sa.CheckConstraint(
            "status <> 'envoyé' OR sent_at IS NOT NULL", name=op.f("ck_outreach_messages_sent_has_date")
        ),
        sa.CheckConstraint("status IN ('confirmé', 'envoyé', 'échec')", name=op.f("ck_outreach_messages_status")),
        sa.ForeignKeyConstraint(
            ["application_id"],
            ["applications.id"],
            name=op.f("fk_outreach_messages_application_id_applications"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["contact_id"],
            ["company_contacts.id"],
            name=op.f("fk_outreach_messages_contact_id_company_contacts"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_outreach_messages_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outreach_messages")),
    )
    op.create_index(
        op.f("ix_outreach_messages_user_id_sent_at"), "outreach_messages", ["user_id", "sent_at"], unique=False
    )
    # Index des clés étrangères (jointures, cascades, purge des tâches).
    for name, table, columns, where in FK_INDEXES:
        op.create_index(op.f(name), table, columns, unique=False, postgresql_where=sa.text(where) if where else None)


def downgrade() -> None:
    for name, table, _, _ in reversed(FK_INDEXES):
        op.drop_index(op.f(name), table_name=table)
    op.drop_index(op.f("ix_outreach_messages_user_id_sent_at"), table_name="outreach_messages")
    op.drop_table("outreach_messages")
    op.drop_table("company_contacts")
    op.drop_table("application_events")
    op.drop_table("labels")
    op.drop_table("applications")
    op.execute("DROP TRIGGER IF EXISTS trg_reports_immutable ON reports")
    op.execute("DROP FUNCTION IF EXISTS reports_immutable()")
    op.drop_table("reports")
    op.drop_index("ix_collected_items_user_id_collected_at", table_name="collected_items")
    op.drop_index("ix_collected_items_keywords", table_name="collected_items", postgresql_using="gin")
    op.drop_index(
        op.f("ix_collected_items_purge_after"),
        table_name="collected_items",
        postgresql_where=sa.text("purge_after IS NOT NULL"),
    )
    op.drop_table("collected_items")
    op.drop_table("user_settings")
    op.drop_index(
        op.f("ix_tasks_user_id_status"),
        table_name="tasks",
        postgresql_where=sa.text("status NOT IN ('terminée', 'échec', 'annulée')"),
    )
    op.drop_table("tasks")
    op.drop_table("suppression_list")
    op.drop_index(
        op.f("ix_source_consents_user_id_source_code"),
        table_name="source_consents",
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.drop_table("source_consents")
    op.drop_table("profile_skill_overrides")
    op.drop_table("platform_sessions")
    op.drop_table("memorized_answers")
    op.drop_index(
        op.f("ix_matching_models_user_id"), table_name="matching_models", postgresql_where=sa.text("is_active")
    )
    op.drop_table("matching_models")
    op.drop_table("candidate_profiles")
    op.drop_table("legacy_imports")
    op.drop_index("ix_activity_events_user_id_kind_at", table_name="activity_events")
    op.drop_table("activity_events")
    op.drop_table("users")
    op.drop_table("sources")

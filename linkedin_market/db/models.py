"""Modèles SQLAlchemy 2 de la base PostgreSQL 18 (ERD : ``schemas-architecture/03-database-erd.md``).

Conventions (``backend/03`` § 2) :
- clés primaires ``uuid`` ``DEFAULT uuidv7()`` ; ``bigint`` identité pour les journaux volumineux ;
- ``user_id`` sur chaque table métier, en tête des index composites (I-01) ;
- dates ``timestamptz`` UTC ; énumérations en ``text`` + ``CHECK`` (valeurs françaises du code) ;
- chemins de fichiers (``html_path``, ``folder_path``) **relatifs à** ``DATA_DIR`` (D-5) ;
- noms de contraintes stables via ``NAMING`` (Alembic).
"""

from datetime import date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    CheckConstraint,
    Date,
    ForeignKey,
    Identity,
    Index,
    LargeBinary,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# --------------------------------------------------------------- énumérations
CAUTIONS = ("prudent", "normal", "agressif")
PROFILE_SOURCES = ("auto", "file", "linkedin")
INTERPRETERS = ("claude", "local")
SENIORITIES = ("junior", "confirme", "senior", "lead", "")
SKILL_ACTIONS = ("ajoutée", "retirée")
ACCESS_MODES = ("api_officielle", "navigateur", "import_manuel")
TASK_KINDS = ("run", "rescore", "train", "prepare", "apply", "contacts", "message", "import_profile", "profile_skill")
TASK_STATUSES = (
    "en_attente",
    "en_cours",
    "attente_saisie",
    "attente_verification_manuelle",
    "terminée",
    "échec",
    "annulée",
)
FINAL_TASK_STATUSES = ("terminée", "échec", "annulée")
ITEM_KINDS = ("job", "post", "person")
APPLICATION_STATUSES = ("préparée", "contactée", "postulée", "entretien", "refusée", "abandonnée", "externe", "échec")
APPLY_STATES = ("", "en_cours", "postulée", "déjà_postulée", "incertaine", "échec")
GENERATORS = ("claude", "local")
APPLICATION_EVENT_KINDS = ("status", "apply", "message", "contacts", "note")
CONTACT_SOURCES = ("linkedin_equipe_recrutement", "manuel")
LEGAL_BASES = ("interet_legitime",)
CHANNELS = ("email", "linkedin")
MESSAGE_STATUSES = ("confirmé", "envoyé", "échec")
SUPPRESSION_REASONS = ("désinscription", "refus_oral", "plainte")
ACTIVITY_KINDS = (
    "run",
    "apply",
    "message_email",
    "message_linkedin",
    "profile_skill",
    "blocked",
    "unblocked",
    "error",
)
# Plafond d'éléments par type selon la prudence (I-07, E-3) ; doit rester aligné sur ``config.CAUTION_LEVELS``.
CAUTION_MAX_ITEMS = {"prudent": 15, "normal": 30, "agressif": 60}


def _sql_list(values: tuple[str, ...]) -> str:
    """Liste SQL littérale ``('a', 'b')`` (valeurs internes, sans apostrophe)."""
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN {_sql_list(values)}"


def _relative_path(column: str) -> str:
    """Chemin relatif à ``DATA_DIR`` (D-5) : ni ``/``, ni ``\\``, ni lettre de lecteur en tête."""
    return f"left({column}, 1) NOT IN ('/', '\\') AND {column} !~ '^[A-Za-z]:'"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)
    type_annotation_map = {
        datetime: TIMESTAMP(timezone=True),  # tout datetime = timestamptz
        date: Date(),
        str: Text(),
        list[str]: ARRAY(Text()),
        dict[str, Any]: JSONB(),
        list[Any]: JSONB(),
    }


def uuid_pk() -> Mapped[UUID]:
    return mapped_column(primary_key=True, server_default=text("uuidv7()"))


def owner(*, primary_key: bool = False) -> Mapped[UUID]:
    return mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=primary_key)


def created() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


def updated() -> Mapped[datetime]:
    return mapped_column(server_default=func.now(), onupdate=func.now())


def empty_text() -> Mapped[str]:
    return mapped_column(server_default="")


def empty_array() -> Mapped[list[str]]:
    return mapped_column(server_default=text("'{}'::text[]"))


def empty_json() -> Mapped[dict[str, Any]]:
    return mapped_column(server_default=text("'{}'::jsonb"))


# --------------------------------------------------------------- utilisateur
class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = uuid_pk()
    email: Mapped[str] = mapped_column(unique=True)  # minuscule
    display_name: Mapped[str] = empty_text()
    created_at: Mapped[datetime] = created()
    updated_at: Mapped[datetime] = updated()


_MAX_ITEMS_CAP = "CASE caution " + " ".join(f"WHEN '{k}' THEN {v}" for k, v in CAUTION_MAX_ITEMS.items()) + " END"


class UserSettings(Base):
    __tablename__ = "user_settings"
    __table_args__ = (
        CheckConstraint(_in("caution", CAUTIONS), name="caution"),
        CheckConstraint(f"max_items IS NULL OR max_items BETWEEN 1 AND {_MAX_ITEMS_CAP}", name="max_items"),
        CheckConstraint(_in("profile_source", PROFILE_SOURCES), name="profile_source"),
        CheckConstraint(_in("interpreter", INTERPRETERS), name="interpreter"),
        CheckConstraint("apply_min_score BETWEEN 0 AND 100", name="apply_min_score"),
        CheckConstraint("apply_daily_max BETWEEN 0 AND 200", name="apply_daily_max"),
        CheckConstraint("message_daily_max BETWEEN 0 AND 200", name="message_daily_max"),
    )

    user_id: Mapped[UUID] = owner(primary_key=True)
    keyword: Mapped[str] = empty_text()
    location: Mapped[str] = empty_text()  # vide = monde entier
    remote_only: Mapped[bool] = mapped_column(server_default=text("true"))
    caution: Mapped[str] = mapped_column(server_default="prudent")
    max_items: Mapped[int | None]  # NULL = plafond de la prudence
    profile_source: Mapped[str] = mapped_column(server_default="auto")
    interpreter: Mapped[str] = mapped_column(server_default="claude")
    apply_min_score: Mapped[float] = mapped_column(Numeric(4, 1, asdecimal=False), server_default=text("70"))
    apply_daily_max: Mapped[int] = mapped_column(server_default=text("10"))
    message_daily_max: Mapped[int] = mapped_column(server_default=text("15"))
    contact_email: Mapped[str] = empty_text()
    contact_phone: Mapped[str] = empty_text()
    signature: Mapped[str] = empty_text()
    linkedin_skill_sync: Mapped[bool] = mapped_column(server_default=text("true"))
    updated_at: Mapped[datetime] = updated()


class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"
    __table_args__ = (
        CheckConstraint("years_experience IS NULL OR years_experience BETWEEN 0 AND 60", name="years_experience"),
        CheckConstraint(_in("seniority", SENIORITIES), name="seniority"),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    origin: Mapped[str] = empty_text()  # chemin du fichier ou ``linkedin``
    title: Mapped[str] = empty_text()  # surcharge utilisateur, vide = aucune
    text: Mapped[str | None]  # cache du profil LinkedIn importé
    skills: Mapped[list[str]] = empty_array()
    years_experience: Mapped[int | None]
    seniority: Mapped[str] = empty_text()
    synced_at: Mapped[datetime | None]
    updated_at: Mapped[datetime] = updated()


class ProfileSkillOverride(Base):
    __tablename__ = "profile_skill_overrides"
    __table_args__ = (CheckConstraint(_in("action", SKILL_ACTIONS), name="action"),)

    user_id: Mapped[UUID] = owner(primary_key=True)
    skill: Mapped[str] = mapped_column(primary_key=True)  # forme normalisée
    action: Mapped[str]
    created_at: Mapped[datetime] = created()


class MemorizedAnswer(Base):
    __tablename__ = "memorized_answers"

    user_id: Mapped[UUID] = owner(primary_key=True)
    question_key: Mapped[str] = mapped_column(primary_key=True)  # question normalisée
    answer: Mapped[str]
    updated_at: Mapped[datetime] = updated()


# --------------------------------------------------------------- sources
class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (CheckConstraint(_in("access_mode", ACCESS_MODES), name="access_mode"),)

    code: Mapped[str] = mapped_column(String(32), primary_key=True)
    label: Mapped[str]
    access_mode: Mapped[str]
    risk_version: Mapped[str | None]  # NULL si la source n'a pas de popup de risques
    enabled: Mapped[bool] = mapped_column(server_default=text("true"))


class SourceConsent(Base):
    __tablename__ = "source_consents"
    __table_args__ = (
        # Un seul consentement actif par source (I-16).
        Index(None, "user_id", "source_code", unique=True, postgresql_where=text("revoked_at IS NULL")),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    source_code: Mapped[str] = mapped_column(String(32), ForeignKey("sources.code"))
    risk_version: Mapped[str]
    accepted_at: Mapped[datetime] = created()
    revoked_at: Mapped[datetime | None]  # NULL = actif


class PlatformSession(Base):
    __tablename__ = "platform_sessions"

    user_id: Mapped[UUID] = owner(primary_key=True)
    source_code: Mapped[str] = mapped_column(String(32), ForeignKey("sources.code"), primary_key=True)
    cookies_encrypted: Mapped[bytes] = mapped_column(LargeBinary)  # Fernet(LM_SECRET_KEY), I-15
    last_login_at: Mapped[datetime]
    expires_at: Mapped[datetime]  # last_login_at + 72 h


# --------------------------------------------------------------- tâches et collecte
class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(_in("kind", TASK_KINDS), name="kind"),
        CheckConstraint(_in("status", TASK_STATUSES), name="status"),
        # Tâches actives : reprise au démarrage.
        Index(None, "user_id", "status", postgresql_where=text(f"status NOT IN {_sql_list(FINAL_TASK_STATUSES)}")),
    )

    id: Mapped[UUID] = uuid_pk()  # généré par Python (uuid.uuid7(), C-7) ; uuidv7() en secours
    user_id: Mapped[UUID] = owner()
    kind: Mapped[str]
    title: Mapped[str] = empty_text()
    status: Mapped[str] = mapped_column(server_default="en_attente")
    browser: Mapped[bool] = mapped_column(server_default=text("false"))
    params: Mapped[dict[str, Any]] = empty_json()
    result: Mapped[dict[str, Any] | None]
    error: Mapped[str | None]
    needs_input: Mapped[dict[str, Any] | None]  # question en attente
    meta: Mapped[dict[str, Any]] = empty_json()
    logs: Mapped[list[str]] = empty_array()  # 400 dernières lignes
    created_at: Mapped[datetime] = created()
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class CollectedItem(Base):
    __tablename__ = "collected_items"
    __table_args__ = (
        UniqueConstraint("user_id", "source_code", "kind", "external_id"),
        CheckConstraint(_in("kind", ITEM_KINDS), name="kind"),
        CheckConstraint("kind <> 'person' OR text IS NULL", name="person_without_text"),  # I-09
        Index(None, "purge_after", postgresql_where=text("purge_after IS NOT NULL")),  # I-10
        Index(None, "task_id", postgresql_where=text("task_id IS NOT NULL")),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    source_code: Mapped[str] = mapped_column(String(32), ForeignKey("sources.code"))
    kind: Mapped[str]
    external_id: Mapped[str]  # identifiant chez la source
    task_id: Mapped[UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"))
    url: Mapped[str] = empty_text()
    title: Mapped[str] = empty_text()
    text: Mapped[str | None]  # NULL pour person (I-09)
    skills: Mapped[list[str]] = empty_array()
    # Mots-clés sous lesquels l'élément a été collecté, normalisés ``lower(trim())`` et sans doublon : une offre
    # recollectée sous un second mot-clé apparaît dans les tendances des deux.
    keywords: Mapped[list[str]] = empty_array()
    company: Mapped[str] = empty_text()
    location: Mapped[str] = empty_text()
    posted_at: Mapped[date | None]
    collected_at: Mapped[datetime] = created()
    purge_after: Mapped[datetime | None]


# Fenêtre de 30 jours des tendances : ``keywords @> ARRAY[:kw]`` (GIN) puis tri par date de collecte.
Index("ix_collected_items_keywords", CollectedItem.keywords, postgresql_using="gin")
Index("ix_collected_items_user_id_collected_at", CollectedItem.user_id, CollectedItem.collected_at.desc())


class Label(Base):
    __tablename__ = "labels"
    __table_args__ = (CheckConstraint("value IN (0, 1)", name="value"), Index(None, "user_id"))

    item_id: Mapped[UUID] = mapped_column(ForeignKey("collected_items.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[UUID] = owner()
    value: Mapped[int] = mapped_column(SmallInteger)
    labeled_at: Mapped[datetime] = created()


class MatchingModelRecord(Base):
    """Version entraînée (ou a priori) du modèle de matching ; ``matching.MatchingModel`` reste la logique."""

    __tablename__ = "matching_models"
    __table_args__ = (
        UniqueConstraint("user_id", "version"),
        # Un seul modèle actif par utilisateur.
        Index(None, "user_id", unique=True, postgresql_where=text("is_active")),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    version: Mapped[int]
    weights: Mapped[dict[str, Any]]  # {weights (8 poids), bias, mean, std, train_accuracy, loss_history}
    feature_names: Mapped[list[Any]]
    n_positive: Mapped[int] = mapped_column(server_default=text("0"))
    n_negative: Mapped[int] = mapped_column(server_default=text("0"))
    learned: Mapped[bool] = mapped_column(server_default=text("false"))  # false = poids a priori
    is_active: Mapped[bool] = mapped_column(server_default=text("false"))
    trained_at: Mapped[datetime] = created()


class Report(Base):
    """Rapport d'analyse, immuable (I-02 : trigger ``trg_reports_immutable``)."""

    __tablename__ = "reports"
    __table_args__ = (
        UniqueConstraint("user_id", "slug"),  # D-5, E-1
        Index(None, "task_id", postgresql_where=text("task_id IS NOT NULL")),
        CheckConstraint(_in("interpreter_used", INTERPRETERS), name="interpreter_used"),
        CheckConstraint(_relative_path("html_path"), name="html_path_relative"),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    task_id: Mapped[UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"))
    slug: Mapped[str]  # mot-cle-AAAA-MM-JJ[-n] (date locale) ; les rapports importés gardent leur nom
    keyword: Mapped[str] = empty_text()
    summary: Mapped[dict[str, Any]] = empty_json()  # agrégats + offres scorées
    interpretation: Mapped[str] = empty_text()
    interpreter_used: Mapped[str]
    html_path: Mapped[str]  # relatif à DATA_DIR, ex. reports/x.html
    created_at: Mapped[datetime] = created()


# --------------------------------------------------------------- candidatures et prospection
class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint("user_id", "item_id"),
        CheckConstraint(_in("status", APPLICATION_STATUSES), name="status"),
        CheckConstraint(_in("apply_state", APPLY_STATES), name="apply_state"),
        CheckConstraint(_in("generator", GENERATORS), name="generator"),
        CheckConstraint(_relative_path("folder_path"), name="folder_path_relative"),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    item_id: Mapped[UUID] = mapped_column(ForeignKey("collected_items.id", ondelete="RESTRICT"))
    status: Mapped[str] = mapped_column(server_default="préparée")
    apply_state: Mapped[str] = empty_text()
    score: Mapped[float] = mapped_column(Numeric(4, 1, asdecimal=False))  # float en JSON (C-10)
    generator: Mapped[str]
    folder_path: Mapped[str]  # relatif à DATA_DIR
    prepared_at: Mapped[datetime] = created()
    updated_at: Mapped[datetime] = updated()

    events: Mapped[list[ApplicationEvent]] = relationship(
        back_populates="application", cascade="all, delete-orphan", order_by="ApplicationEvent.id"
    )


class ApplicationEvent(Base):
    __tablename__ = "application_events"
    __table_args__ = (CheckConstraint(_in("kind", APPLICATION_EVENT_KINDS), name="kind"), Index(None, "application_id"))

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    application_id: Mapped[UUID] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"))
    kind: Mapped[str]
    details: Mapped[dict[str, Any]] = empty_json()
    at: Mapped[datetime] = created()

    application: Mapped[Application] = relationship(back_populates="events")


class CompanyContact(Base):
    __tablename__ = "company_contacts"
    __table_args__ = (
        CheckConstraint(_in("source", CONTACT_SOURCES), name="source"),
        CheckConstraint(_in("legal_basis", LEGAL_BASES), name="legal_basis"),  # I-13
        Index(None, "user_id", "application_id"),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    application_id: Mapped[UUID] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"))
    full_name: Mapped[str]
    headline: Mapped[str] = empty_text()  # fonction affichée
    profile_url: Mapped[str] = empty_text()
    email: Mapped[str | None]  # NULL si non affiché par la source
    source: Mapped[str] = mapped_column(server_default="manuel")
    legal_basis: Mapped[str] = mapped_column(server_default="interet_legitime")
    collected_at: Mapped[datetime] = created()
    last_contacted_at: Mapped[datetime | None]
    # Dernier contact (ou collecte) + 3 ans (I-14) ; jamais NULL pour que la purge s'applique toujours.
    purge_after: Mapped[datetime] = mapped_column(server_default=text("now() + interval '3 years'"))


class OutreachMessage(Base):
    __tablename__ = "outreach_messages"
    __table_args__ = (
        CheckConstraint(_in("channel", CHANNELS), name="channel"),
        CheckConstraint(_in("status", MESSAGE_STATUSES), name="status"),
        CheckConstraint("status <> 'envoyé' OR sent_at IS NOT NULL", name="sent_has_date"),
        Index(None, "user_id", "sent_at"),  # plafond message_daily_max
        Index(None, "application_id"),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    application_id: Mapped[UUID] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"))
    contact_id: Mapped[UUID | None] = mapped_column(ForeignKey("company_contacts.id", ondelete="SET NULL"))
    channel: Mapped[str]
    recipient: Mapped[str]
    subject: Mapped[str] = empty_text()
    body: Mapped[str]
    cv_attached: Mapped[bool] = mapped_column(server_default=text("false"))
    status: Mapped[str] = mapped_column(server_default="confirmé")
    smtp_message_id: Mapped[str | None]
    error: Mapped[str | None]
    confirmed_at: Mapped[datetime] = created()
    sent_at: Mapped[datetime | None]


class SuppressionEntry(Base):
    """Entrée de ``suppression_list`` : opposition définitive (I-13), empreinte seulement."""

    __tablename__ = "suppression_list"
    __table_args__ = (
        UniqueConstraint("user_id", "recipient_hash"),
        CheckConstraint(_in("reason", SUPPRESSION_REASONS), name="reason"),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    recipient_hash: Mapped[str]  # SHA-256 de l'email ou de l'URL de profil normalisés
    reason: Mapped[str]
    created_at: Mapped[datetime] = created()


class ActivityEvent(Base):
    __tablename__ = "activity_events"
    __table_args__ = (CheckConstraint(_in("kind", ACTIVITY_KINDS), name="kind"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[UUID] = owner()
    kind: Mapped[str]
    data: Mapped[dict[str, Any]] = empty_json()
    at: Mapped[datetime] = created()


# Plafonds quotidiens, dernier blocage.
Index("ix_activity_events_user_id_kind_at", ActivityEvent.user_id, ActivityEvent.kind, ActivityEvent.at.desc())


class LegacyImport(Base):
    """Fichier historique de ``linkedin_market_data/`` déjà importé (``db/import_json.py``).

    Écrit dans la transaction de l'import : si le renommage ``.migrated`` est interrompu après le ``COMMIT``, la
    relance reconnaît le couple (nom relatif, empreinte) et se contente de renommer le fichier, sans réimport ni
    doublon. Un contenu identique sous un autre nom est un fichier distinct (deux rapports, par exemple).
    """

    __tablename__ = "legacy_imports"

    user_id: Mapped[UUID] = owner(primary_key=True)
    file_name: Mapped[str] = mapped_column(primary_key=True)  # chemin relatif à DATA_DIR (``reports/x.json``)
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)  # empreinte hexadécimale du contenu
    imported_at: Mapped[datetime] = created()

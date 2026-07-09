"""Modèles SQLAlchemy pour le SaaS multi-tenant."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TenantStatus(str, enum.Enum):
    trial = "trial"
    active = "active"
    past_due = "past_due"
    cancelled = "cancelled"


class MembershipRole(str, enum.Enum):
    owner = "owner"
    member = "member"


class PlanInterval(str, enum.Enum):
    month = "month"
    year = "year"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    memberships: Mapped[list["Membership"]] = relationship(back_populates="user")


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default=TenantStatus.trial.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # IMAP (mot de passe chiffré avec Fernet)
    imap_host: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    imap_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    imap_user: Mapped[Optional[str]] = mapped_column(String(320), nullable=True)
    imap_password_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    imap_folder: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    imap_use_ssl: Mapped[bool] = mapped_column(Boolean, default=True)
    trial_analyses_used: Mapped[int] = mapped_column(Integer, default=0)

    memberships: Mapped[list["Membership"]] = relationship(back_populates="tenant")
    subscriptions: Mapped[list["Subscription"]] = relationship(back_populates="tenant")
    oauth_connections: Mapped[list["OAuthConnection"]] = relationship(back_populates="tenant")
    projects: Mapped[list["Project"]] = relationship(back_populates="tenant")


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "tenant_id", name="uq_membership_user_tenant"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    role: Mapped[str] = mapped_column(String(32), default=MembershipRole.member.value)

    user: Mapped["User"] = relationship(back_populates="memberships")
    tenant: Mapped["Tenant"] = relationship(back_populates="memberships")


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    price_amount: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(10), default="XOF")
    interval: Mapped[str] = mapped_column(String(16))
    quota_analyses_per_month: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    subscriptions: Mapped[list["Subscription"]] = relationship(back_populates="plan")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    plan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("plans.id"))
    status: Mapped[str] = mapped_column(String(32), default="active")
    current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_transaction_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    tenant: Mapped["Tenant"] = relationship(back_populates="subscriptions")
    plan: Mapped["Plan"] = relationship(back_populates="subscriptions")


class BillingEvent(Base):
    __tablename__ = "billing_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    transaction_id: Mapped[str] = mapped_column(String(128), index=True)
    amount: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(64))
    raw_payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class UsageCounter(Base):
    """Compteur d'analyses par tenant et mois calendaire (YYYY-MM)."""

    __tablename__ = "usage_counters"
    __table_args__ = (UniqueConstraint("tenant_id", "year_month", name="uq_usage_tenant_month"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    year_month: Mapped[str] = mapped_column(String(7), index=True)
    analysis_count: Mapped[int] = mapped_column(Integer, default=0)


class OAuthConnection(Base):
    """OAuth2 connection for Gmail or Outlook per tenant."""

    __tablename__ = "oauth_connections"
    __table_args__ = (
        UniqueConstraint("tenant_id", "provider", "email", name="uq_oauth_tenant_provider_email"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))  # "gmail" | "outlook"
    email: Mapped[str] = mapped_column(String(320))
    access_token_encrypted: Mapped[str] = mapped_column(Text)
    refresh_token_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_expiry: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    scopes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    tenant: Mapped["Tenant"] = relationship(back_populates="oauth_connections")


class ProjectStatus(str, enum.Enum):
    active = "active"
    archived = "archived"


class RecipientStatus(str, enum.Enum):
    direct_to = "direct_to"
    cc = "cc"


class ProjectSentiment(str, enum.Enum):
    on_track = "on_track"
    under_tension = "under_tension"
    awaiting_feedback = "awaiting_feedback"


class SuggestedActionStatus(str, enum.Enum):
    pending = "pending"
    completed = "completed"
    dismissed = "dismissed"


class Project(Base):
    """Un projet/sujet regroupant des emails, au sens de project-overview.md.

    Base de la persistance nécessaire au Fast-Track (architecture.md, Process 2) :
    sans un `Project` et un `ProjectSummary.last_processed_email_timestamp`
    stockés, il n'y a pas de delta possible entre deux rafraîchissements.
    """

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(16), default=ProjectStatus.active.value)
    # Règles de rattachement d'un email au projet : mots-clés, adresses, domaines.
    rules_matrix: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    tenant: Mapped["Tenant"] = relationship(back_populates="projects")
    emails: Mapped[list["Email"]] = relationship(back_populates="project")
    summary: Mapped[Optional["ProjectSummary"]] = relationship(
        back_populates="project", uselist=False
    )
    suggested_actions: Mapped[list["SuggestedAction"]] = relationship(back_populates="project")


class Email(Base):
    """Email normalisé ingéré pour un tenant, éventuellement rattaché à un projet.

    `body_encrypted` est une colonne texte simple à ce stade — le chiffrement
    au repos (Fernet, comme les identifiants IMAP/OAuth) est une unité de
    travail séparée, pas encore câblée ici.
    """

    __tablename__ = "emails"
    __table_args__ = (
        UniqueConstraint("tenant_id", "external_id", name="uq_email_tenant_external_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    project_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True, index=True
    )
    # Identifiant du message côté fournisseur (Message-ID IMAP, id Gmail/Graph…) —
    # sert à la déduplication (code-standards.md §8 : dédup par (tenant_id, email_external_id)).
    external_id: Mapped[str] = mapped_column(String(512))
    recipient_status: Mapped[str] = mapped_column(String(16))
    subject: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    body_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Score hybride 0-100 (règles + AI, voir ai_intelligent.score_email_importance) ;
    # NULL pour les emails persistés avant cette colonne (non rétro-calculé).
    importance_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Tags métier + priorité dérivés par règles (voir email_analyzer.classification.
    # derive_tags) ; NULL pour les emails persistés avant cette colonne.
    tags: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    # Score de confiance 0-100 de la classification multi-critères vers ce
    # projet (email_analyzer.classification.score_project_relevance) ; reflète
    # uniquement les signaux déterministes issus de rules_matrix, pas le
    # renforcement "participants connus" propre au balayage IMAP (voir
    # migration 005). NULL pour les emails persistés avant cette colonne.
    classification_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    project: Mapped[Optional["Project"]] = relationship(back_populates="emails")


class ProjectSummary(Base):
    """Résumé courant d'un projet — une seule ligne par projet, régénérée en
    place par le Fast-Track ou la synchronisation batch (pas un historique)."""

    __tablename__ = "project_summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), unique=True, index=True
    )
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Curseur d'incrémentalité du Fast-Track (architecture.md, Process 2) : seuls
    # les emails reçus après cette date sont récupérés lors d'un rafraîchissement.
    last_processed_email_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    project: Mapped["Project"] = relationship(back_populates="summary")


class SuggestedAction(Base):
    """Action recommandée par l'IA pour un projet (jamais exécutée automatiquement —
    ai-workflow-rules.md §8)."""

    __tablename__ = "suggested_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    description: Mapped[str] = mapped_column(Text)
    deadline: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=SuggestedActionStatus.pending.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="suggested_actions")

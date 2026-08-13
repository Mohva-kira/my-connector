"""Modèles SQLAlchemy pour le SaaS multi-tenant."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, time
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
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
    # Horodatage du login le plus récent — mis à jour à chaque POST /api/auth/login.
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Horodatage du login précédent (avant celui ci-dessus) — c'est LUI qui sert
    # de référence "depuis votre dernière visite" (GET /api/brief) : il reste
    # stable pendant toute la session en cours (contrairement à last_login_at,
    # qui avance à chaque nouveau login), pour ne pas remettre le Brief à zéro
    # à chaque rechargement de page.
    previous_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

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


class AppointmentStatus(str, enum.Enum):
    tentative = "tentative"
    confirmed = "confirmed"
    cancelled = "cancelled"


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
    appointments: Mapped[list["Appointment"]] = relationship(back_populates="project")


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
    # Clé de ticket détectée dans le sujet ([Banque_Plateforme_Numéro], voir
    # email_analyzer.subject_tickets.parse_subject_ticket_key) et son numéro ;
    # NULL si le sujet ne porte pas ce motif. Colonnes dédiées (pas JSONB) :
    # l'énumération/comptage par projet (migration 015) a besoin d'un GROUP BY
    # SQL bon marché.
    ticket_label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ticket_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
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
    # Sortie LLM structurée (décisions/risques/échéances/prochaines étapes,
    # voir email_analyzer.llm.ProjectSummaryLLM et rfc-email-pipeline-v2.md §11) ;
    # additif à `content` (texte libre, inchangé) — NULL pour les résumés
    # générés avant l'ajout de cette colonne ou en cas d'échec d'extraction.
    structured_content: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Niveau de risque perçu par le LLM (rfc-email-pipeline-v2.md §11) — jamais
    # fusionné avec `sentiment` (dérivé des règles déterministes,
    # ai_intelligent.calculate_risk_score) : un désaccord entre les deux est un
    # signal, pas un conflit à résoudre silencieusement.
    llm_risk_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Version du schéma de `structured_content`, pour faire évoluer le contrat
    # JSON sans migration bloquante des résumés déjà persistés.
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    # Curseur d'incrémentalité du Fast-Track (architecture.md, Process 2) : seuls
    # les emails reçus après cette date sont récupérés lors d'un rafraîchissement.
    last_processed_email_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Date de retour probable estimée par le LLM (llm.ProbableContactItem) pour
    # un projet "en attente"/"en rouge" — jamais renseignée sans signal réel
    # dans le fil d'emails (voir prompt), NULL sinon (pas d'invention côté UI).
    probable_next_contact_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    probable_next_contact_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    probable_next_contact_confidence: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )
    # Horodatage de la dernière préparation IA de l'Agenda (rendez-vous +
    # date probable) pour ce projet — distinct de `updated_at` (mis à jour par
    # toute synchro Fast-Track, agenda ou non).
    agenda_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    project: Mapped["Project"] = relationship(back_populates="summary")


class ProjectSummaryHistory(Base):
    """Archive d'un ProjectSummary tel qu'il était juste avant d'être écrasé
    par une resynchronisation Fast-Track (analysis_tasks.py) — ProjectSummary
    reste une ligne unique régénérée en place (voir son docstring), ceci est
    l'historique qui en manquait jusqu'ici. Une ligne par régénération
    effective (new_emails non vide), pas par tick de sync sans changement.
    Pas de tenant_id (comme ProjectSummary) : l'isolation tenant repose sur la
    vérification Project.tenant_id == tenant_id faite avant toute requête
    sur cette table, jamais sur une colonne locale."""

    __tablename__ = "project_summary_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), index=True
    )
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    structured_content: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    llm_risk_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
    )

    project: Mapped["Project"] = relationship()


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
    # Détail pour la vue "ouvrir une action" (frontend ActionsPage) : pourquoi
    # elle est recommandée, qui est concerné, comment éviter la récidive.
    # Alimentés depuis llm.NextStepItem/DeadlineItem (raison/responsable/
    # conseil_prevention) quand l'extraction structurée les fournit — NULL
    # sinon (repli sur `recommandation`, ou actions créées avant cette
    # colonne) plutôt qu'une valeur inventée côté frontend.
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stakeholder: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    advice: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="suggested_actions")


class Appointment(Base):
    """Rendez-vous/réunion extrait par l'IA du contenu des emails d'un projet
    (voir llm.AppointmentItem) — jamais créé manuellement dans cette unité,
    seul le statut peut être ajusté par l'utilisateur (confirmé/annulé)."""

    __tablename__ = "appointments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    description: Mapped[str] = mapped_column(Text)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    participants: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=AppointmentStatus.tentative.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    project: Mapped["Project"] = relationship(back_populates="appointments")


class Conversation(Base):
    """Un fil de discussion avec l'assistant portefeuille. Un tenant+utilisateur
    peut avoir plusieurs conversations (bouton "nouvelle conversation" +
    panneau d'historique) ; chaque AssistantMessage appartient à exactement
    une conversation. `title` par défaut au premier message utilisateur
    (tronqué), voir ConversationService — pas de renommage manuel en v1."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class AssistantMessage(Base):
    """Tour de conversation avec l'assistant portefeuille (voir
    llm.build_portfolio_chat_system_prompt, api/routers/assistant.py).

    Persisté par conversation (Conversation ci-dessus) — contrairement au
    chat par projet (`/api/chat`, éphémère côté frontend), l'assistant
    permanent garde sa mémoire entre les sessions."""

    __tablename__ = "assistant_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class MemoryCategory(str, enum.Enum):
    preference = "preference"
    fact = "fact"
    context = "context"


class MemorySource(str, enum.Enum):
    conversation = "conversation"


class Memory(Base):
    """Fait durable retenu sur un utilisateur/tenant (préférence, fait,
    contexte projet) — pilier "Mémoire persistante" du pivot Personal
    Executive AI Assistant (context/project-overview.md).

    v1 volontairement simple (voir progress-tracker.md Unit 46) : pas de
    recherche vectorielle, pas de notion de "conversation" source distincte
    (source = "conversation", cf. AssistantMessage ci-dessus tant qu'il n'y
    a pas de session séparée). Extrait par
    email_analyzer.services.memory_service.MemoryService, jamais écrit
    directement par un routeur."""

    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(16), default=MemoryCategory.fact.value)
    source: Mapped[str] = mapped_column(String(16), default=MemorySource.conversation.value)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Profile(Base):
    """Contexte personnel de l'utilisateur — profil et habitudes, domaine
    #1 ("la base de tout") de la vision Personal Executive AI Assistant
    (context/project-overview.md). Unique par (tenant, user) : contrairement
    à Memory (plusieurs faits extraits), un profil est un objet de
    configuration unique que l'utilisateur édite directement dans les
    Paramètres — pas d'extraction LLM.

    Habitudes en colonnes typées (forme fixe et connue) ; préférences et
    membres de la famille en JSONB (listes ouvertes par nature, même
    logique que Project.rules_matrix/Email.tags)."""

    __tablename__ = "profiles"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id", name="uq_profiles_tenant_user"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    languages: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)
    timezone: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    preferences: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    family_members: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    favorite_contact_ids: Mapped[Optional[List[uuid.UUID]]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    wake_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    sleep_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    work_start_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    work_end_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    sport_days: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)
    focus_hours: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Organization(Base):
    """Entreprise/organisation mentionnée dans un ou plusieurs projets —
    pilier "Knowledge Layer" du pivot (context/project-overview.md).

    v1 volontairement simple : pas de graphe, une organisation = un nom
    unique par tenant, dédupliqué à l'exact (comme Memory). Alimentée par
    email_analyzer.services.knowledge_service.KnowledgeService à partir de
    `ProjectSummary.structured_content["companies"]` (déjà extrait par le
    LLM existant, cf. llm.ProjectSummaryLLM) — pas de nouvel appel LLM."""

    __tablename__ = "organizations"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_organization_tenant_name"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    contacts: Mapped[list["Contact"]] = relationship(back_populates="organization")


class Contact(Base):
    """Personne mentionnée dans un ou plusieurs projets (participant email).

    Alimentée depuis `ProjectSummary.structured_content["participants"]`
    (voir Organization ci-dessus pour le raisonnement) — un nom, pas encore
    une adresse email normalisée (Email.body_encrypted n'expose pas
    l'expéditeur en colonne dédiée aujourd'hui ; à revoir si ce besoin
    devient bloquant)."""

    __tablename__ = "contacts"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_contact_tenant_name"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    organization: Mapped[Optional["Organization"]] = relationship(back_populates="contacts")
    project_links: Mapped[list["ProjectContactLink"]] = relationship(back_populates="contact")


class ProjectContactLink(Base):
    """Relation "ce contact est lié à ce projet" — la brique "Project Links"
    du pivot. Table d'association simple (pas de graphe), une ligne par
    couple (project, contact)."""

    __tablename__ = "project_contact_links"
    __table_args__ = (
        UniqueConstraint("project_id", "contact_id", name="uq_project_contact_link"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    contact_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("contacts.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    project: Mapped["Project"] = relationship()
    contact: Mapped["Contact"] = relationship(back_populates="project_links")


class ObjectiveStatus(str, enum.Enum):
    on_track = "on_track"
    at_risk = "at_risk"
    blocked = "blocked"
    done = "done"


class MilestoneStatus(str, enum.Enum):
    pending = "pending"
    done = "done"


class Objective(Base):
    """Objectif de haut niveau (pilier "Objectives" du pivot,
    context/project-overview.md) — optionnellement rattaché à un `Project`
    existant ("use existing projects whenever possible", plan de pivot),
    sinon un objectif personnel du tenant sans projet associé.

    Pas de table "Progress" séparée : la progression est calculée à la
    volée (milestones terminés / total, voir
    ObjectiveService.compute_progress) plutôt que stockée et à
    resynchroniser. Pas de table "Blocking Issue" séparée non plus :
    `status=blocked` + `blocking_reason` suffisent en v1 — une raison de
    blocage n'est pas un objet indépendant tant qu'il n'y a qu'un blocage
    actif possible par objectif."""

    __tablename__ = "objectives"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    project_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    target_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=ObjectiveStatus.on_track.value)
    blocking_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    project: Mapped[Optional["Project"]] = relationship()
    milestones: Mapped[list["Milestone"]] = relationship(
        back_populates="objective", order_by="Milestone.due_date"
    )


class Milestone(Base):
    """Étape concrète d'un `Objective`. Une ligne = un jalon ; l'ensemble
    des jalons d'un objectif donne sa progression (voir Objective.__doc__)."""

    __tablename__ = "milestones"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    objective_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("objectives.id"), index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=MilestoneStatus.pending.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    objective: Mapped["Objective"] = relationship(back_populates="milestones")


class Document(Base):
    """Document local téléversé par l'utilisateur — domaine "Contexte
    Documents" (context/project-overview.md, domaine #7 : "la base de
    tout" #1 exceptée). v1 volontairement scopé aux fichiers locaux :
    Drive/OneDrive/Dropbox/NAS restent hors périmètre (bloqués sur choix de
    fournisseur + identifiants OAuth, voir project-overview.md).

    Le fichier lui-même vit sur disque (`document_service.storage_path_for`,
    nommé par `id` pour éviter tout risque de traversée de chemin depuis un
    nom de fichier utilisateur) ; cette ligne n'en est que les métadonnées +
    le texte extrait pour la recherche. `extracted_text` reste `NULL` pour
    les formats non pris en charge en v1 (Word/Excel/PowerPoint/images) —
    ces documents restent stockés et listés, seulement pas recherchables par
    contenu (voir `email_analyzer/document_extraction.py`)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    project_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True, index=True
    )
    filename: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # "upload" (dépôt manuel, /documents) ou "email" (pièce jointe ingérée
    # automatiquement par le Fast-Track, voir analysis_tasks.py). Nullable
    # (migration 018) : les lignes antérieures à cette colonne sont toutes
    # des dépôts manuels, traitées comme "upload" par l'UI sans backfill
    # (même discipline que 004/005 sur emails — pas de deviner une valeur).
    source: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    project: Mapped[Optional["Project"]] = relationship()

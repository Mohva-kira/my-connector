# Base de Données, Modélisation & Migrations

> **Statut** : §§ 1 à 4 et 8 `[ACTUEL]` depuis l'acceptation de la vague 1 (2026-10-09) ; § 5.1 (réservation de
> message) et § 5.2 (plafond `apply`) `[ACTUEL]` depuis l'acceptation de la vague 2 (2026-10-09) ; le reste de la
> généralisation « transaction par cas d'usage », § 6 (purge, vague 6) et la sauvegarde automatique du § 7
> restent `[CIBLE]`. Le schéma de référence est
> `schemas-architecture/03-database-erd.md` (tables, colonnes, contraintes, rétention, correspondance des
> fichiers). Ce document fixe **comment** l'implémenter.

## 1. Stack Base de Données

| Composant | Version | Notes |
| :--- | :--- | :--- |
| SGBD | **PostgreSQL 18** | Paquet système ou `docker compose up -d db` (`compose.yaml` du lot 1, publié sur `127.0.0.1:5432` uniquement). `uuidv7()` est natif depuis la 18. |
| ORM | **SQLAlchemy 2.1.x** (stable 2.1.4 au 2026-10-07), style déclaratif typé `Mapped[]` / `mapped_column()` | Borne `sqlalchemy>=2.0,<2.2` (lot 1). API 2.0 uniquement (`select()`, `Session.execute`), jamais `Query`. |
| Driver | **psycopg 3** (`psycopg[binary]`, stable 3.3.6) | URL `postgresql+psycopg://…` (`LM_DATABASE_URL`, décidé le 2026-10-08 : `DATABASE_URL` est déjà lu par email_analyzer dans le même `.env` racine). |
| Migrations | **Alembic** (stable 1.20.0) | Borne `alembic>=1.18,<2`. |
| Chiffrement applicatif | `cryptography` (Fernet) | Colonne `platform_sessions.cookies_encrypted` (cf. `04`). |
| Anciens fichiers | JSON/JSONL de `linkedin_market_data/` | Importés une fois au démarrage (`db/import_json.py`, § 3.4) puis renommés `*.migrated`. |

Moteur et sessions (`linkedin_market/db/session.py`) :

```python
# Moteur paresseux : créé au premier usage (le CLI sans base et les tests le pilotent), libéré par dispose_engine().
def get_engine() -> Engine:            # create_engine(config.database_url(), pool_pre_ping=True, pool_size=5, max_overflow=2)
def dispose_engine() -> None: ...
# config.database_url() lit LM_DATABASE_URL (message français si absente).
```

- `pool_size=5` ≥ `TaskManager(max_workers=4)` + requêtes HTTP. `[ACTUEL]` une transaction courte par appel de
  façade (`db.session.transaction()`, `db.repositories.user_transaction()` qui résout aussi `user_id`) ; une
  `Session` n'est jamais partagée entre threads. `[CIBLE]` dépendance FastAPI `get_db` (vague 2).
- Le CLI ouvre ses propres sessions via les mêmes services.
- Au démarrage (`db/bootstrap.py::prepare_database()`, appelé par le CLI et par `python -m linkedin_market.web`) :
  `alembic upgrade head` (programmatique), puis l'import JSON s'il reste des fichiers non suffixés `.migrated`,
  puis les tâches non finales de `tasks` passent à `échec` (« Tâche interrompue par l'arrêt de l'application. »).
  Base injoignable ⇒ arrêt avec un message français actionnable, mot de passe masqué ; **aucun repli sur les
  fichiers** (PostgreSQL est obligatoire après la vague 1).

### 1.1 Transition de la vague 1 (décision d'architecte, 2026-10-08)
`[ACTUEL]` Pour ne pas réécrire tous les appelants en une vague, les modules de persistance actuels (`storage.py`,
`application.py` (suivi), `activity.py`, `settings.py`, `session.py`, `report.py` (écriture), le chargement/
sauvegarde de `matching.MatchingModel`, `TaskManager`) **gardent leurs signatures publiques** et deviennent des
façades : chaque fonction ouvre une transaction courte (`with db.session.transaction() as s:`), résout
`user_id` (`db.users.current_user_id(s)`) et délègue à `db/repositories/*`. Aucune `Session` ne traverse un
thread ni une navigation Selenium. La dépendance FastAPI `get_db` reste `[CIBLE]` ; la transaction par cas d'usage
de la réservation de message (§ 5.1, E-11) et celle du plafond `apply` (§ 5.2) sont livrées en vague 2, en passant
par les façades existantes (`user_transaction()`), pas par une dépendance FastAPI dédiée. Autre exception dès la
vague 1 : `advance_status()` lit et écrit le statut dans **une** transaction avec `SELECT … FOR UPDATE` (I-03).

## 2. Conventions de Schéma

| Règle | Convention retenue | Écart avec le modèle générique |
| :--- | :--- | :--- |
| Noms de tables | snake_case, **noms de l'ERD du lot 1** (pluriel en général : `applications`, `tasks`, `collected_items` ; singulier assumé pour `user_settings` et `suppression_list`, termes du glossaire). | Le glossaire prime sur la règle « tout au pluriel ». |
| Clés primaires | `uuid` `DEFAULT uuidv7()` (ordonné dans le temps, index B-tree compact) ; `bigint GENERATED ALWAYS AS IDENTITY` pour les journaux volumineux (`activity_events`, `application_events`) ; clés naturelles composites pour les tables de liaison (`profile_skill_overrides`, `memorized_answers`, `platform_sessions`) et `sources.code`. | Pas d'UUIDv4. |
| Propriété | `user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE` sur chaque table métier (I-01), en **première** colonne des index composites. | Pas de RLS : une seule personne par base, l'isolation est la machine ; RLS ajouterait un rôle par connexion sans gain. |
| Dates | `timestamptz` UTC ; `created_at` / `updated_at` `NOT NULL DEFAULT now()` là où l'ERD les prévoit ; `updated_at` mis à jour par l'ORM (`onupdate=func.now()`), pas par trigger. Dates métier sans heure en `date` (`posted_at`). | Les tables append-only (`activity_events`, `reports`, `application_events`) n'ont pas d'`updated_at`. |
| Énumérations | `text` + `CHECK (col IN (…))`, valeurs françaises identiques au code. | Pas de type `ENUM` PostgreSQL (ajout de valeur sans `ALTER TYPE`). |
| JSON | `jsonb` pour les données sans contrainte relationnelle (paramètres de tâche, agrégats de rapport, poids). Jamais pour une donnée filtrée ou jointe. | — |
| Listes simples | `text[]` (`skills`). | — |
| Suppression | **Pas de soft-delete** (`deleted_at`) : la rétention RGPD exige une suppression physique (art. 5(1)(e)) ; l'opposition est conservée sous forme d'empreinte dans `suppression_list`. Seul `source_consents` garde l'historique via `revoked_at`. | Écart volontaire. |
| Nommage des contraintes | `MetaData(naming_convention=…)` (ci-dessous) : noms stables pour Alembic. | — |

```python
# linkedin_market/db/models.py
from datetime import datetime
from uuid import UUID

from sqlalchemy import (TIMESTAMP, CheckConstraint, ForeignKey, Index, MetaData, Numeric, String, Text,
                        UniqueConstraint, func, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)
    type_annotation_map = {datetime: TIMESTAMP(timezone=True)}  # tout datetime = timestamptz


def uuid_pk() -> Mapped[UUID]:
    return mapped_column(primary_key=True, server_default=text("uuidv7()"))


def owner() -> Mapped[UUID]:
    return mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=False)


APPLICATION_STATUSES = ("préparée", "contactée", "postulée", "entretien", "refusée", "abandonnée", "externe", "échec")


class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint("user_id", "item_id"),
        CheckConstraint(f"status IN {APPLICATION_STATUSES!r}", name="status"),
        CheckConstraint("apply_state IN ('', 'en_cours', 'postulée', 'déjà_postulée', 'incertaine', 'échec')",
                        name="apply_state"),
        CheckConstraint("generator IN ('claude', 'local')", name="generator"),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    item_id: Mapped[UUID] = mapped_column(ForeignKey("collected_items.id", ondelete="RESTRICT"))
    status: Mapped[str] = mapped_column(Text, server_default="préparée")
    apply_state: Mapped[str] = mapped_column(Text, server_default="")
    score: Mapped[float] = mapped_column(Numeric(4, 1))
    generator: Mapped[str] = mapped_column(Text)
    folder_path: Mapped[str] = mapped_column(Text)
    prepared_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    events: Mapped[list["ApplicationEvent"]] = relationship(back_populates="application", cascade="all, delete-orphan",
                                                            order_by="ApplicationEvent.id")


class OutreachMessage(Base):
    __tablename__ = "outreach_messages"
    __table_args__ = (
        CheckConstraint("channel IN ('email', 'linkedin')", name="channel"),
        CheckConstraint("status IN ('confirmé', 'envoyé', 'échec')", name="status"),
        CheckConstraint("status <> 'envoyé' OR sent_at IS NOT NULL", name="sent_has_date"),
        Index(None, "user_id", "sent_at"),            # plafond message_daily_max
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    application_id: Mapped[UUID] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"))
    contact_id: Mapped[UUID | None] = mapped_column(ForeignKey("company_contacts.id", ondelete="SET NULL"))
    channel: Mapped[str] = mapped_column(Text)
    recipient: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(Text, server_default="")
    body: Mapped[str] = mapped_column(Text)
    cv_attached: Mapped[bool] = mapped_column(server_default=text("false"))
    status: Mapped[str] = mapped_column(Text, server_default="confirmé")
    smtp_message_id: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    confirmed_at: Mapped[datetime] = mapped_column(server_default=func.now())
    sent_at: Mapped[datetime | None]


class SourceConsent(Base):
    __tablename__ = "source_consents"
    __table_args__ = (
        # Un seul consentement actif par source (I-16).
        Index(None, "user_id", "source_code", unique=True, postgresql_where=text("revoked_at IS NULL")),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = owner()
    source_code: Mapped[str] = mapped_column(String(32), ForeignKey("sources.code"))
    risk_version: Mapped[str] = mapped_column(Text)
    accepted_at: Mapped[datetime] = mapped_column(server_default=func.now())
    revoked_at: Mapped[datetime | None]
```

Les autres tables suivent le même gabarit ; les colonnes, `CHECK` et
index sont ceux de l'ERD § 1–2.

## 3. Politique des Migrations

### 3.1 Organisation
```
linkedin_market/migrations/
├── env.py                     # target_metadata = Base.metadata, compare_type=True, URL depuis LM_DATABASE_URL
├── script.py.mako
└── versions/
    ├── 0001_initial.py        # schéma complet + données de référence sources + trigger d'immuabilité
    └── 0002_<objet>.py        # évolutions de schéma suivantes
linkedin_market/db/import_json.py   # import idempotent des fichiers linkedin_market_data/ (pas une révision Alembic)
```
- Identifiants de révision **séquentiels sur 4 chiffres** (`alembic revision --rev-id 0002 -m "…"`,
  `file_template = %%(rev)s_%%(slug)s`) : lisibles, triés, sans conflit puisqu'un seul flux de développement.
- `alembic revision --autogenerate` sert de **brouillon** : toute migration est relue (les index partiels, `CHECK`
  et triggers ne sont pas tous détectés). La CI exécute `alembic upgrade head` puis `alembic check` (aucun écart
  modèles ↔ migrations).
- Les migrations sont appliquées **automatiquement au lancement** de l'application et du CLI. Elles doivent donc
  être sûres sans intervention : pas de question, pas de suppression de données sans sauvegarde préalable.
- `[ACTUEL]` `alembic check` passe sur `0001_initial` (modèles == migration) ; à relancer pour chaque révision.

### 3.2 Compatibilité
- Les utilisateurs mettent à jour par `git pull` et peuvent revenir à la version précédente : chaque migration
  suit le principe **expansion puis contraction** sur deux versions mineures (ajouter la colonne nullable et
  l'écrire, puis rendre `NOT NULL` / supprimer l'ancienne dans la version suivante).
- `downgrade()` est implémenté pour les migrations de schéma. L'import JSON n'a pas de retour arrière : les
  fichiers d'origine sont conservés en `*.migrated` (§ 3.4).
- `[CIBLE]` Avant toute migration qui supprime ou réécrit des données, l'application exécute un `pg_dump`
  automatique dans `linkedin_market_data/backups/avant-<rev>.dump` (§ 7). Aucune migration destructive à ce jour.

### 3.3 `0001_initial`
- Crée toutes les tables de l'ERD, contraintes et index (§ 2 de l'ERD).
- Insère les sources :

| `code` | `label` | `access_mode` | `risk_version` | `enabled` |
| :--- | :--- | :--- | :--- | :--- |
| `linkedin` | LinkedIn | `navigateur` | `2026-10` | `true` |
| `malt` | Malt | `navigateur` | `2026-10` | `false` |
| `wttj` | Welcome to the Jungle | `navigateur` | `2026-10` | `false` |
| `france_travail` | France Travail | `api_officielle` | `NULL` | `false` |
| `indeed` | Indeed | `import_manuel` | `NULL` | `false` |

  Une source passe à `enabled = true` par la révision qui livre son intégration (vague 5).

  Un changement de texte de risques = une migration `UPDATE sources SET risk_version = '…'` (I-16).
- Crée le trigger d'immuabilité (I-02) :

```python
op.execute("""
CREATE FUNCTION reports_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  -- Seule exception : task_id non nul -> NULL (FK ON DELETE SET NULL lors de la purge des tâches à 12 mois),
  -- toutes les autres colonnes identiques. Tout autre UPDATE lève.
  IF OLD.task_id IS NOT NULL AND NEW.task_id IS NULL
     AND (to_jsonb(NEW) - 'task_id') = (to_jsonb(OLD) - 'task_id') THEN
    RETURN NEW;
  END IF;
  RAISE EXCEPTION 'Un rapport est immuable (I-02).';
END $$;
CREATE TRIGGER trg_reports_immutable BEFORE UPDATE ON reports
  FOR EACH ROW EXECUTE FUNCTION reports_immutable();
""")
```

### 3.3 bis Précisions de schéma validées (vague 1, 2026-10-08)
- `CHECK` sur **tous** les champs énumérés, y compris `tasks.kind` : un nouveau type de tâche (ex. `purge`, vague 6)
  exige une révision `0002_…` **et** l'entrée au glossaire avant le code.
- `CHECK` « chemin relatif » sur `reports.html_path` et `applications.folder_path` (ni `/` initial, ni `..`).
- Textes `NOT NULL DEFAULT ''` sauf `collected_items.text`, `company_contacts.email`, `outreach_messages.error`,
  `outreach_messages.smtp_message_id`, `tasks.error` (NULL = absent). `tasks.params` / `tasks.meta` `NOT NULL
  DEFAULT '{}'` ; `tasks.id DEFAULT uuidv7()` en secours (Python fournit `uuid.uuid7()`).
- `company_contacts.purge_after NOT NULL DEFAULT now() + interval '3 years'` (recalculé à chaque contact).
- Index sur les clés étrangères parcourues par les jointures, cascades et `SET NULL` :
  `application_events (application_id)`, `company_contacts (user_id, application_id)`,
  `outreach_messages (application_id)`, `reports (task_id) WHERE task_id IS NOT NULL`,
  `collected_items (task_id) WHERE task_id IS NOT NULL`, `labels (user_id)`.
- Utilisateur unique : `ensure_user(email)` seulement quand `users` est vide (premier lancement / import) ; ensuite
  `current_user_id()`, qui lève si la table contient zéro ou plusieurs lignes (une instance = un utilisateur). Un
  changement de `LINKEDIN_EMAIL` ne crée pas de second utilisateur.
- Classe ORM de `matching_models` : `MatchingModelRecord` (évite la collision avec `matching.MatchingModel`).
  `weights` (jsonb) = `{"weights": [8 floats], "bias", "mean": [8], "std": [8], "train_accuracy", "loss_history"}` ;
  `feature_names` (jsonb) = liste des 8 critères ; une version dont `feature_names` diffère de `matching.FEATURES`
  est ignorée (poids a priori).
- `collected_items.keywords text[]` (remplace `keyword`) : mots-clés de collecte normalisés `lower(trim())`, sans
  doublon, dans l'ordre d'apparition ; à la recollecte (`ON CONFLICT … DO UPDATE`), les nouveaux mots-clés sont
  **ajoutés** aux anciens. Index `GIN (keywords)` + `(user_id, collected_at DESC)`. Le GIN n'a pas `user_id` en tête
  (il faudrait l'extension `btree_gin`) : écart assumé, une instance = un utilisateur ; chaque requête garde son
  filtre `user_id` (I-01). `Item.keyword` (domaine) = mot-clé **demandé** à la lecture (`storage.load_items`).
- `legacy_imports (user_id, file_name, sha256, imported_at)`, `PRIMARY KEY (user_id, file_name, sha256)` : nom
  relatif et empreinte de chaque fichier historique importé (§ 3.4). Décidé par l'utilisateur le 2026-10-09
  (option B) : un contenu identique sous un autre nom est un fichier distinct (ex. deux `reports/*.json` identiques
  ⇒ deux rapports).
- `updated_at` peut être `NULL` (jamais modifié) là où C-2 l'expose ainsi (`01` § 1.1).
- `compose.yaml` avec `healthcheck` (`pg_isready`) ; lancement depuis la racine :
  `docker compose --env-file .env -f linkedin_market/compose.yaml up -d db` (le `.env` est à la racine du dépôt).

### 3.4 Import JSON (`db/import_json.py`, idempotent)
Ce n'est **pas** une révision Alembic (décision d'architecte, vague 1) : une révision ne s'exécute qu'une fois et
tourne dans la transaction d'Alembic, ce qui empêche le renommage après `COMMIT` et la reprise de fichiers
ajoutés plus tard. `import_legacy_files(data_dir) -> dict[str, int]` (nombre de lignes importées par table) est
appelé par `prepare_database()` après `upgrade head`. Règles de l'ERD § 4, précisées :
1. Crée l'utilisateur unique (`users`, email = `LINKEDIN_EMAIL` en minuscules, sinon `local@localhost`) s'il
   n'existe pas, et sa ligne `user_settings` (valeurs par défaut).
2. Lit chaque fichier présent **et non suffixé** `.migrated` et calcule son empreinte SHA-256. Un fichier dont
   le couple (nom relatif à `DATA_DIR`, empreinte) figure déjà dans `legacy_imports` (renommage interrompu après un `COMMIT` précédent) n'est **pas**
   réimporté : il est seulement renommé (compteur `already_imported_files`). Tout le reste est inséré dans **une
   seule transaction**, qui inscrit aussi chaque fichier importé dans `legacy_imports (user_id, file_name relatif à
   DATA_DIR, sha256)` ; chaque `INSERT` de données utilise `ON CONFLICT DO NOTHING`. `labels.json` et
   `applications.json` sont relus même déjà importés (protection de la purge, comptes du modèle).
3. Conversions :

| Source | Transformation |
| :--- | :--- |
| `settings.json` | `UPDATE` de la ligne `user_settings` créée à l'étape 1 (champs invalides ignorés) ; `interpreter = 'none'` → `local` ; `max_items` > plafond de la prudence → `NULL` (E-3) ; `answers` → `memorized_answers`. |
| `history.jsonl` | `Item.source` → `kind` ; `id` `job:123` / `post:…` / `person:<slug>` → `external_id` sans préfixe ; `source_code = 'linkedin'` ; `scraped_at` → `collected_at` ; `person` : `text = NULL`, compétences extraites par `nlp` ; dernière version d'un `id` gagnante ; `keyword` de **toutes** les lignes d'un même `id` fusionnés dans `keywords` ; `purge_after` calculé (§ 6), `NULL` pour une offre annotée ou candidatée, lignes déjà expirées ignorées. |
| `labels.json` | jointure sur `external_id` ; libellé orphelin ignoré. |
| `applications.json` | `item_id` par `external_id` (offre absente de l'historique : création d'un `collected_items` minimal `job` avec titre/entreprise/url) ; `history` → `application_events` avec `statut`→`status`, `candidature`→`apply` ; `contacts` → `company_contacts` (`name`→`full_name`, `source` `linkedin`→`linkedin_equipe_recrutement`, `legal_basis = 'interet_legitime'`). |
| `session.json` | chiffré Fernet → `platform_sessions('linkedin')` ; ignoré si `logged_in_at` > 72 h. |
| `activity.jsonl`, `tasks.jsonl` | dates locales naïves → `timestamptz` avec le fuseau de la machine (`datetime.astimezone()`) ; tâches : 100 dernières, `id = uuid5(espace de noms fixe, ancien id)` (relance sans doublon). |
| `matching_model.json` | version 1 active ; `n_positive` / `n_negative` = nombre de 1 et de 0 de `labels.json` (à défaut, des `labels` déjà en base). |
| `reports/*.json` | slug = nom de fichier ; `interpreter_used = 'none'` → `local` sans texte (rapport historique conservé tel quel). |

4. Après `COMMIT` seulement : renommage de chaque fichier importé (ou reconnu) en `<nom>.migrated` (jamais
   supprimé). Les dossiers `reports/` (HTML) et `applications/` restent en place ; leurs chemins sont stockés.
   Aucun JSON de rapport n'est plus écrit sur disque.
5. Une erreur annule tout : aucune ligne insérée, aucun fichier renommé. Tests : `tests/test_import_json.py`
   (idempotence, erreur simulée, crash entre `COMMIT` et renommages, conversions, rapports identiques sous deux noms
   importés tous deux).

### 3.5 Index
Ceux de l'ERD § 2. Règle : toute colonne de `WHERE`/`JOIN` fréquente a un index dont `user_id` est la première
colonne ; les index partiels servent aux requêtes de purge et aux tâches actives. Pas d'index sur `jsonb` tant
qu'aucune requête ne filtre dedans.

## 4. Requêtes de Référence

```sql
-- Plafond quotidien de messages (jour local de l'utilisateur, fuseau Europe/Paris par exemple)
SELECT count(*) FROM outreach_messages
WHERE user_id = :uid AND status IN ('confirmé', 'envoyé')
  AND confirmed_at >= (date_trunc('day', now() AT TIME ZONE :tz) AT TIME ZONE :tz);

-- Blocage en cours (I-06) : dernier blocked non suivi d'un unblocked, de moins de 6 h
SELECT at, data FROM activity_events
WHERE user_id = :uid AND kind IN ('blocked', 'unblocked')
ORDER BY at DESC LIMIT 1;          -- en cours si kind = 'blocked' AND at > now() - interval '6 hours'

-- Fenêtre de 30 jours des tendances (db/repositories/items.py::window_items) : GIN sur keywords
SELECT * FROM collected_items
WHERE user_id = :uid AND keywords @> ARRAY[lower(trim(:kw))]
  AND coalesce(posted_at, collected_at::date) >= current_date - 30
ORDER BY collected_at, id;
```

Les messages `confirmé` comptent dans le plafond : un envoi réservé puis échoué libère sa place
(`status = 'échec'`), un envoi en cours ne la libère pas.

## 5. Transactions Critiques

### 5.1 Réservation d'un message (`EnvoyerMessage`) `[ACTUEL]` (vague 2, G-6/E-11)
Implémentée par `web/services.reserve_message()` (ORM SQLAlchemy, pas de SQL brut) et
`db/repositories/outreach.py` ; équivalent fonctionnel :
```sql
BEGIN;
SELECT id FROM applications WHERE id = :app AND user_id = :uid FOR UPDATE;          -- un envoi à la fois par candidature
SELECT message_daily_max FROM user_settings WHERE user_id = :uid FOR UPDATE;         -- sérialise le plafond (E-11)
-- [CIBLE] vague 4 : SELECT 1 FROM suppression_list WHERE user_id = :uid AND recipient_hash = :h ;
--                   trouvé ⇒ ROLLBACK + 409 (I-13) — absent à ce jour.
SELECT count(*) FROM outreach_messages WHERE …;                                      -- ≥ plafond ⇒ ROLLBACK + 429
INSERT INTO outreach_messages (…, status) VALUES (…, 'confirmé') RETURNING id;
COMMIT;
-- envoi SMTP / LinkedIn hors transaction, puis :
BEGIN;
UPDATE outreach_messages SET status = 'envoyé', sent_at = now(), smtp_message_id = :mid WHERE id = :id;
UPDATE applications SET status = 'contactée', updated_at = now()
  WHERE id = :app AND status IN ('préparée', 'externe', 'échec');                    -- advance_status (I-03)
UPDATE company_contacts SET last_contacted_at = now(), purge_after = now() + interval '3 years' WHERE id = :contact;
INSERT INTO application_events (application_id, kind, details) VALUES (:app, 'message', :details);
INSERT INTO activity_events (user_id, kind, data) VALUES (:uid, 'message_email', :data);
COMMIT;
```
`recipient_hash` = `sha256(lower(trim(email)))` ou `sha256(url de profil normalisée sans paramètres ni « / » final)`
(utilisé seulement à partir de la vague 4, pas encore en base).

Précisions livrées (`db/repositories/outreach.py::count_day`, `insert_confirmed`, `mark_sent`, `mark_failed`,
`note_uncertain`, `release`) :
- **Comptage du jour** = `max(réservations du jour dans outreach_messages hors status='échec', événements
  message_email/message_linkedin du jour dans activity_events)`. Le second terme couvre la **transition** : des
  messages envoyés avant l'existence de la réservation (vague 1 et avant) n'existent que comme événements
  d'activité, jamais l'inverse (chaque réservation produit aussi un événement à l'envoi) — un simple `count(*)`
  sur `outreach_messages` sous-compterait ces anciens envois le jour du passage à la vague 2.
- **Libération si la tâche n'a jamais démarré** : deux cas distincts.
  - La création de la tâche échoue de façon synchrone dans le routeur (ex. `409` « Un envoi est déjà en cours »,
    déclenché par `has_active()` *après* la réservation) : la ligne est **supprimée**
    (`outreach_repo.release`, `DELETE … WHERE status='confirmé' AND sent_at IS NULL`) — aucune trace.
  - La tâche est créée mais annulée avant que sa fonction démarre (attente du verrou navigateur interrompue,
    `TaskManager.submit(..., on_abort=...)`, `backend/05` § 1.2) : la réservation passe en `échec` avec
    `error = "Tâche annulée avant l'envoi."` (`outreach_repo.mark_failed` via `services.abort_message`) — la
    tentative reste visible, la place est rendue au plafond.
- **`outreach.UncertainDelivery`** (envoi LinkedIn dont la confirmation visuelle n'a pas pu être vérifiée après le
  clic sur « Envoyer », cf. `05` § 2.4) : `note_uncertain()` laisse `status='confirmé'` (pas `envoyé`, pas
  `échec`) et renseigne `error` — affiché « à vérifier » côté UI, compte toujours dans le plafond du jour,
  **jamais** renvoyé automatiquement (I-04, aucune relance d'un envoi, `backend/05` § 1.6).
- Le champ `contact_id` de la réservation est résolu à l'insertion (`outreach_repo.find_contact`, recherche par
  email ou URL de profil parmi les `company_contacts` de la candidature) ; `mark_sent()` l'utilise pour repousser
  `last_contacted_at` / `purge_after` du contact (I-14).

### 5.2 Autres transactions
| Cas | Contenu de la transaction |
| :--- | :--- |
| Changement de statut manuel | `FOR UPDATE` sur `applications` + `UPDATE` + `INSERT application_events(kind='status')`. |
| Candidature soumise `[ACTUEL]` (vague 2) | `FOR UPDATE` sur `user_settings` (`activity.check_quota_locked`) + contrôle `count(apply du jour)` au **démarrage** de la tâche (`services.apply_job`, avant tout accès au dossier) ; à la fin : statut + `apply_state` + `labels` (valeur 1) + `activity_events('apply')`. |
| Consentement `[ACTUEL]` (vague 3, `db/repositories/sources.py`) | `UPDATE source_consents SET revoked_at = now() WHERE … AND revoked_at IS NULL` + `INSERT`. |
| Activation d'un modèle `[ACTUEL]` | `FOR UPDATE` sur `user_settings` + `UPDATE matching_models SET is_active = false WHERE user_id = :uid AND is_active` + `INSERT … (version = max+1, is_active = true)`. |
| Collecte `[ACTUEL]` | insertion par lots de 100 (`ON CONFLICT (user_id, source_code, kind, external_id) DO UPDATE SET title, text, skills, keywords = anciens ∪ nouveaux, collected_at, purge_after` — ce dernier reste `NULL` s'il l'était) — une transaction par lot pour ne pas perdre une collecte interrompue. |

Niveau d'isolation : `READ COMMITTED` (défaut) ; les verrous explicites ci-dessus suffisent.

## 6. Purge de Rétention (I-10, I-14)

`RetentionService.purge()` s'exécute au démarrage puis toutes les 24 h (minuterie du `TaskManager`, tâche
`kind="purge"` non navigateur `[CIBLE]`, à ajouter au glossaire). `purge_after` est calculé à l'insertion :

| Table | `purge_after` | Remise à `NULL` |
| :--- | :--- | :--- |
| `collected_items` `post` / `person` | `collected_at + 30 days` | jamais |
| `collected_items` `job` | `collected_at + 90 days` | quand une candidature la référence ou qu'elle est annotée |
| `company_contacts` | `coalesce(last_contacted_at, collected_at) + 3 years` | recalculé à chaque contact |

```sql
BEGIN;
DELETE FROM collected_items ci
WHERE ci.user_id = :uid AND ci.purge_after < now()
  AND NOT EXISTS (SELECT 1 FROM applications a WHERE a.item_id = ci.id);
DELETE FROM company_contacts WHERE user_id = :uid AND purge_after < now();
DELETE FROM platform_sessions WHERE user_id = :uid AND expires_at < now();
DELETE FROM activity_events WHERE user_id = :uid AND at < now() - interval '12 months';
DELETE FROM tasks WHERE user_id = :uid AND created_at < now() - interval '12 months'
  AND status IN ('terminée', 'échec', 'annulée');
COMMIT;
```
`labels` suit `collected_items` (`ON DELETE CASCADE`). `suppression_list` n'est jamais purgée (empreinte seule).
Le nombre de lignes supprimées par table est journalisé (sans contenu).

## 7. Sauvegarde & Restauration

- **Sauvegarde** (documentée dans le README) :
  ```bash
  pg_dump -Fc -d linkedin_market -f sauvegarde-$(date +%F).dump
  # avec compose.yaml :
  docker compose exec -T db pg_dump -Fc -U linkedin_market linkedin_market > sauvegarde-$(date +%F).dump
  ```
  À sauvegarder **avec** : `linkedin_market_data/reports/`, `linkedin_market_data/applications/` et `.env`
  (contient `LM_SECRET_KEY` ; sans elle les cookies chiffrés sont illisibles — conséquence limitée : une nouvelle
  connexion LinkedIn).
- **Restauration** : `pg_restore --clean --if-exists -d linkedin_market sauvegarde.dump`, puis lancement normal
  (`alembic upgrade head` rattrape le schéma si la sauvegarde est plus ancienne).
- **Sauvegarde automatique** avant une migration destructive : `linkedin_market_data/backups/avant-<rev>.dump`,
  5 dernières conservées.
- Les sauvegardes contiennent des données personnelles de tiers : même niveau de protection que la base
  (disque de l'utilisateur, jamais dans le dépôt Git — `linkedin_market_data/` est dans `.gitignore`).

## 8. Tests

- `pytest` utilise une base PostgreSQL 18 jetable (`LM_TEST_DATABASE_URL`, service `postgres:18` en CI,
  lot 1 `04` § 6) ; le nom de la base **doit finir par `_test`** et l'URL **différer de `LM_DATABASE_URL`**,
  sinon la session pytest s'arrête (protection de la vraie base). `LM_SECRET_KEY` éphémère générée par la
  fixture ; l'écriture d'une clé dans `.env` est refusée sous pytest. `[ACTUEL]` (vague 2) : `pytest_configure`
  retire aussi du processus les identifiants réels du `.env` racine (`LINKEDIN_EMAIL`, `LINKEDIN_PASSWORD`,
  `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_SECURITY`, `ANTHROPIC_API_KEY`) : sans
  eux, aucune tâche lancée par un test ne peut se connecter à un vrai service ; un test qui en a besoin pose une
  valeur factice avec `monkeypatch.setenv`. Schéma : `DROP SCHEMA public CASCADE; CREATE SCHEMA public;` puis `alembic upgrade head`
  une fois par session (teste aussi `0001`). Isolation : avant chaque test marqué `db`, `TRUNCATE` de toutes les
  tables sauf `sources` et `alembic_version` (`RESTART IDENTITY CASCADE`) puis création de l'utilisateur de test.
  Pas de transaction annulée : les tâches tournent dans des threads du `TaskManager`, qui ouvrent leurs propres
  connexions (une connexion psycopg ne se partage pas entre threads).
- Marqueur `db` : sans `LM_TEST_DATABASE_URL`, les tests marqués sont **ignorés** avec le message « Tests
  PostgreSQL ignorés : définis LM_TEST_DATABASE_URL (base dont le nom finit par _test). » ; la porte qualité de
  l'architecte exige la variable et **0 test ignoré**.
- Tests obligatoires de l'import : `import_legacy_files` sur les fixtures JSON de `tests/fixtures/data/`, puis
  seconde exécution sans effet (idempotence), puis vérification des conversions (E-3, E-7, E-15).
- `[ACTUEL]` la fixture `conftest.data_dir` redirige tous les chemins de `config` vers `tmp_path` ; elle reste
  utile pour les fichiers (rapports HTML, dossiers de candidature).

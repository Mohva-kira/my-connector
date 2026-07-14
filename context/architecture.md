Markdown# Architecture Documentation: Intelligent Project Management & Email Assistant

Ce document décrit l'architecture technique, le modèle de données et les règles d'intégrité du système. L'application utilise une approche hybride cloud/edge pour maximiser la confidentialité des données tout en garantissant des performances de traitement et d'analyse élevées.

> **Note (Unit 24, 2026-07-12)** — Ce document reste globalement
> incohérent avec le code réel (Celery/RabbitMQ, base vectorielle,
> React Native mentionnés ci-dessous ne sont pas implémentés — la pile
> réelle est FastAPI + `arq`/Redis + PostgreSQL, un frontend React/Vite web
> seul ; voir `context/rfc-email-pipeline-v2.md` §0 et Unit 13 de
> `progress-tracker.md` pour le constat détaillé). Cette réconciliation
> complète reste différée par décision explicite. Seuls les ajouts réels de
> l'Unit 24 sont documentés ici, additivement :
> - `email_analyzer.db.models.ProjectSummary` gagne `structured_content`
>   (JSONB — décisions/risques/échéances/prochaines étapes structurés par
>   LLM, voir `llm.ProjectSummaryLLM`), `llm_risk_level`, `schema_version`.
> - Nouvelle table `assistant_messages` (`AssistantMessage`) : mémoire
>   persistée de l'assistant conversationnel portefeuille (tenant + user),
>   distincte de `/api/chat` (scoped à un projet, éphémère côté frontend).
> - `User` gagne `last_login_at`/`previous_login_at` — référence pour le
>   Brief ("depuis votre dernière visite", `GET /api/brief`).
> - Nouveaux endpoints additifs (contrats existants inchangés, cohérent
>   avec le principe de stabilité du RFC §12) : `GET /api/brief`,
>   `GET/PATCH /api/actions`, `GET /api/timeline`, `GET /api/assistant/messages`,
>   `POST /api/assistant/chat`.
> - Frontend : nouvelle IA en sidebar (Brief/Projets/Agenda/Actions/
>   Assistant/Paramètres) pour les utilisateurs SaaS, remplaçant l'accueil
>   "formulaire d'analyse" — le mode legacy (`.env`, sans DB) garde l'ancien
>   comportement inchangé. Palette et jetons UI documentés à jour dans
>   `context/ui-context.md`.

---

## 1. Stack Technique (Technology Stack)

| Couche (Layer)            | Technologie                       | Rôle dans l'Architecture                                                                                                 |
| :------------------------ | :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **Frontend Web**          | ReactJS                           | Interface utilisateur d'administration, tableaux de bord de validation des projets et relecture des brouillons.          |
| **Frontend Mobile**       | React Native (Expo)               | Application mobile pour le suivi des projets en déplacement et les notifications d'analyses.                             |
| **Local Cache & Index**   | SQLite (SQLCipher)                | Stockage local chiffré sur le device pour les métadonnées et le texte des 30 derniers jours ($J-30$).                    |
| **Local Vector DB**       | Orama / Voy (JavaScript)          | Base vectorielle embarquée côté client pour la recherche de similarité et le RAG local sur 30 jours glissants.           |
| **Backend API & Core**    | Python                            | Orchestration de l'ingestion, parsing des pièces jointes, génération des prompts et gestion de la sécurité.              |
| **Task Queue & Broker**   | Celery + Redis                    | File d'attente asynchrone pour le traitement lourd des emails et des fichiers (PDF/Excel) sans blocage.                  |
| **Base de Données Cloud** | PostgreSQL                        | Base centrale relationnelle stockant la persistance globale, les historiques complets et les données chiffrées au repos. |
| **Intelligence Engine**   | Cloud LLM APIs (OpenAI/Anthropic) | Moteur d'IA distant sollicité pour l'extraction d'entités, la classification de projets et la génération de brouillons.  |

---

## 2. Frontières du Système (System Boundaries)

L'organisation du dépôt de code (Repository structure) sépare strictement les responsabilités pour garantir l'étanchéité de la logique Edge et Cloud.

```text
├── apps/
│   ├── web/                     # Client ReactJS : Rendu UI et gestion de la base vectorielle locale WASM
│   └── mobile/                  # Client React Native : UI mobile et gestion SQLite/Vector chiffré local
├── backend/
│   ├── api/                     # Points d'accès Python (Endpoints API, Auth, validation des requêtes)
│   ├── core/                    # Logique métier : Modèles IA (Prompts), extraction d'entités, résolution d'alias
│   ├── workers/                 # Tâches Celery : Scripts de polling IMAP/Gmail et parsers de documents (PDF/Excel)
│   └── crypto/                  # Fonctions de chiffrement/déchiffrement des emails (AES-256) via clés du .env
└── packages/
    └── database/                # Schémas de migration PostgreSQL et définitions des entités de la base centrale
3. Modèle de Stockage (Storage Model)Le système segmente les données en trois états distincts : la persistance centralisée (Cloud), le cache opérationnel (Device), et la file d'attente volatile (Cache de transit).Base de Données Relationnelle Cloud (PostgreSQL)Contenu : Profils utilisateurs, configurations des comptes emails, jetons OAuth (Refresh Tokens) chiffrés, tables de correspondance des projets et de leurs alias.Sécurité : Stockage de l'historique complet des corps d'emails et du texte extrait des pièces jointes, intégralement chiffré au repos (AES-256) via une clé secrète système (ENCRYPTION_KEY) stockée dans les variables d'environnement.Cache et Indexation Côté Client (Device Storage)Contenu : Une base SQLite locale contenant les métadonnées des emails (Expéditeur, Sujet, Projet associé) des 30 derniers jours. Un index vectoriel (Orama/Voy) stockant les embeddings (vecteurs) de ces mêmes 30 jours.Cycle de vie : Au démarrage de l'application (Web ou Mobile), un script de nettoyage autonome supprime physiquement toutes les lignes et vecteurs dont le timestamp est plus ancien que $J-30$.Cache Temporel et File d'Attente (Redis)Contenu : États des tâches asynchrones Celery, files d'attente des identifiants d'emails à traiter, et mise en cache temporaire du timestamp de dernière synchronisation (last_synced_at) pendant l'exécution des scripts de polling.4. Modèle d'Authentification et d'Accès (Auth & Access Model)Authentification Applicative : Gérée via des jetons sécurisés (JWT) à courte durée de vie émis par le backend Python après vérification des identifiants ou de la session SSO.Accès aux Boîtes Mails Third-Party :Pour Gmail : Stockage sécurisé du Refresh Token obtenu via le flux Google OAuth 2.0.Pour l'IMAP Entreprise : Chiffrement symétrique en base de données des identifiants/mots de passe de connexion.Propriété et Isolation des Données (Multi-Tenancy) :Au niveau Cloud : Chaque table PostgreSQL critique contient une clé étrangère user_id. Toutes les requêtes SQL de l'API appliquent un filtrage strict sur cet identifiant.Au niveau Edge : L'isolation est physique. Chaque utilisateur possède son propre fichier SQLite et son index vectoriel local, isolés dans le stockage applicatif de son navigateur ou de son smartphone. Aucun partage de contexte vectoriel n'est possible entre deux sessions utilisateur.5. Modèles d'IA et de Tâches d'Arrière-Plan (AI & Background Tasks)Le système s'appuie sur une architecture orientée événements pour découpler la détection des emails du traitement analytique de l'IA.Pipeline d'Ingestion (Cron & Workers)Le Déclencheur (Cron Job) : Un script Python s'exécute toutes les heures. Il récupère en base la date last_synced_at de chaque utilisateur et interroge les serveurs Gmail/IMAP pour lister uniquement les emails reçus après cette date.La File d'Attente (Celery Queue) : Pour chaque email détecté, le Cron pousse un message léger contenant le message_id et les données d'accès dans la file d'attente Redis.Le Traitement (Celery Workers) : Les processus ouvriers (Workers) Python récupèrent les tâches, téléchargent l'email et ses pièces jointes (PDF/Excel), extraisent le contenu textuel brut, puis chiffrent et sauvegardent le tout dans PostgreSQL.Orchestration du RAG Hybride (Retrieval-Augmented Generation)Requêtes de routine (Recherche locale) : L'utilisateur interroge l'assistant depuis l'interface $\rightarrow$ Recherche de similarité exécutée localement sur l'index vectoriel du device ($J-30$) $\rightarrow$ Extraction des segments textuels pertinents $\rightarrow$ Envoi exclusif de ces segments et de la question à l'API du Cloud LLM.Analyses Approfondies (Recherche Cloud) : Si l'utilisateur demande une analyse historique ($> 30$ jours), la requête transite par l'API Python. Le serveur déchiffre temporairement en mémoire (RAM) les archives d'emails ciblées, exécute le prompt et le traitement lourd via le Cloud LLM, enregistre le résultat chiffré dans PostgreSQL et envoie la réponse finale au device pour affichage.6. Invariants du Codebase (Codebase Invariants)Les invariants sont des règles architecturales absolues. Le code ne doit jamais violer les quatre règles suivantes sous peine de refus immédiat lors de la relecture technique (Code Review) :Règle de Cardinalité Unique (Strict 1:1 Email-Project) : Un email traité ne peut être associé qu'à un et un seul projet principal en base de données. Le système doit utiliser le système d'alias ou le score de confiance du LLM pour trancher, mais l'entité Email ne doit jamais posséder une relation de type Many-to-Many avec l'entité Project.Règle de Chiffrement des Données Brutes (Zero Clear-Text Cloud Storage) : Aucun corps d'email brut (body_content) ni texte extrait de pièce jointe (attachment_parsed_text) ne doit être stocké en clair dans les tables de persistance PostgreSQL. Le déchiffrement n'est autorisé que de manière éphémère en mémoire vive (RAM) lors de l'exécution des tâches d'analyse ou de transmission sécurisée au LLM.Règle de Confinement Spatiale du Cache (Hard 30-Day Device Limit) : La base vectorielle et l'instance SQLite locales du device ne doivent contenir aucune donnée dont le timestamp de création dépasse 30 jours calendaires. Le script de purge local est obligatoire, s'exécute de manière synchrone au démarrage de l'application et ne dépend pas d'un ordre du serveur Cloud.Règle d'Isolation des Traitements Lourds (No Heavy I/O in API Thread) : Aucun appel réseau vers les protocoles IMAP/Gmail, aucun parsing de fichier joint (PDF/Excel) et aucun appel de synchronisation initiale de LLM ne doit être exécuté à l'intérieur du cycle de requête/réponse de l'API HTTP (Thread principal). Toutes ces opérations doivent obligatoirement être déléguées de manière asynchrone à la file d'attente Celery.
```

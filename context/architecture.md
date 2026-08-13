# Architecture Documentation: my-connector — Personal Executive AI Assistant

> **Pivot stratégique (2026-07-17)** — Ce document a été entièrement
> réécrit. Il remplace le texte précédent, qui décrivait une pile technique
> jamais implémentée (Celery/RabbitMQ, base vectorielle client Orama/Voy,
> React Native, SQLCipher) et les notes datées Unit 24 / Units 33-41 qui
> documentaient additivement l'écart avec le code réel. Le contenu ci-dessous
> décrit : (1) l'implémentation actuelle, (2) l'architecture court/moyen
> terme déjà planifiée dans `context/evolution-plan.md` et
> `context/rfc-email-pipeline-v2.md`, (3) la vision long terme "Executive
> Assistant" définie dans `context/project-overview.md`. Voir
> `context/progress-tracker.md` pour l'historique unité par unité.

---

## 1. Stack technique actuelle

| Couche                  | Technologie                          | Rôle                                                                 |
| :----------------------- | :------------------------------------ | :-------------------------------------------------------------------- |
| **Frontend web**          | React + Vite + Tailwind (web only)    | Interface utilisateur, chat-first (`ChatHome.tsx`), pas d'app mobile. |
| **Backend API**          | FastAPI (Python)                      | Endpoints REST, orchestration, auth JWT, multi-tenant.                |
| **Base de données**      | PostgreSQL + SQLAlchemy (ORM)         | Persistance relationnelle centrale, migrations Alembic.               |
| **File d'attente / jobs**| Redis + `arq`                         | Jobs asynchrones (analyse, sync planifiée, refresh agenda) hors du thread FastAPI. |
| **Moteur IA**            | OpenAI (gpt-4o-mini) + Google Gemini  | Extraction structurée, classification, résumés, chat.                 |
| **Paiement**             | CinetPay (XOF)                        | Facturation SaaS (non documenté avant ce pivot).                      |

Il n'y a **pas** de Celery/RabbitMQ, pas de base vectorielle côté client
(Orama/Voy), pas de cache SQLite/SQLCipher device, pas d'application React
Native. Ces éléments appartenaient à une vision antérieure jamais codée.

---

## 2. Frontières du système (réel)

> Mis à jour 2026-07-19 (Units 45-57) : le pivot exécutif a ajouté trois
> couches (`repositories/`, `services/`, `tools/`+`agent/`) et 4 routeurs
> depuis la version précédente de cette section — voir
> `context/progress-tracker.md` Units 45-57 pour le détail unité par unité.

```text
services/email_analyzer/
├── api/
│   ├── main.py                  # App FastAPI, /api/analyze, /api/chat, /api/draft, healthcheck
│   └── routers/
│       ├── auth.py              # /api/auth/register, /login, /switch, /me
│       ├── tenants.py           # /api/tenants, config IMAP
│       ├── billing.py           # /api/billing/plans, /checkout, /cinetpay/notify
│       ├── oauth.py             # Gmail/Outlook OAuth2 (authorize/callback/connections)
│       ├── projects.py          # CRUD projets, fusion, doublons
│       ├── actions.py           # /api/actions (SuggestedAction)
│       ├── agenda.py            # /api/agenda, rendez-vous
│       ├── brief.py             # /api/brief, /api/timeline (délègue à BriefService/TimelineService)
│       ├── assistant.py         # /api/assistant/messages, /api/assistant/chat (délègue à ConversationService)
│       ├── memory.py            # /api/memory, /api/memory/extract (Unit 46)
│       ├── knowledge.py         # /api/knowledge/contacts, /organizations, /projects/{id}/sync (Unit 47)
│       ├── objectives.py        # /api/objectives, jalons (Unit 54)
│       └── profile.py           # /api/profile (Contexte Personnel, 2026-07-30)
├── email_analyzer/
│   ├── db/models.py             # Modèles SQLAlchemy (voir §3)
│   ├── repositories/            # Accès SQL, un fichier par concept (Unit 45+) : conversation_,
│   │                             #   memory_, knowledge_, brief_, timeline_, objective_,
│   │                             #   profile_repository.py (2026-07-30)
│   ├── services/                # Logique métier, contrôleurs HTTP fins délèguent ici (Unit 45+) :
│   │                             #   conversation_, memory_, knowledge_, context_, brief_,
│   │                             #   timeline_, objective_, profile_service.py (2026-07-30),
│   │                             #   tool_executor.py (ce
│   │                             #   dernier remplace email_analyzer/agent/, voir §7)
│   ├── tools/                   # Enveloppes en lecture seule appelées par ToolExecutor (Unit 50+) :
│   │                             #   search_projects_, summarize_project_, search_emails_,
│   │                             #   search_memory_, find_contact_, list_meetings_,
│   │                             #   draft_reply_tool.py ; base.py définit `Tool` (typing.Protocol,
│   │                             #   pas d'héritage) — inchangés par l'unification du chat (§7)
│   ├── jobs.py                  # État des jobs dans Redis (create/set_status/get_job)
│   ├── analysis_tasks.py        # Tâches arq (WorkerSettings) : run_analysis_legacy/saas,
│   │                             #   run_fasttrack_refresh, run_domain_discovery,
│   │                             #   run_scheduled_sync, run_agenda_refresh
│   ├── llm.py                   # Intégration OpenAI/Gemini : résumés, extraction structurée, chat,
│   │                             #   extraction mémoire, formatage de contexte (Memory/Knowledge/Agent)
│   ├── ai_intelligent.py        # Scoring/analyse (modèle ML préchargé au démarrage)
│   ├── classification.py        # Score d'importance, tags, priorités
│   └── saas_logic.py            # Logique multi-tenant SaaS
└── frontend/src/
    ├── App.tsx                  # Routes : /login, /, /settings, /projects, /projects/:id, /agenda,
    │                             #   /actions, /memory, /objectives
    ├── pages/ChatHome.tsx       # Accueil chat-first unique (legacy + SaaS)
    ├── pages/MemoryPage.tsx     # Souvenirs + contacts (Unit 53)
    ├── pages/ObjectivesPage.tsx # Objectifs + jalons + timeline (Unit 56)
    ├── hooks/useBrief.ts        # Consomme GET /api/brief
    ├── hooks/useLegacyHomeChat.ts
    ├── hooks/useSaasHomeChat.ts
    ├── hooks/useMemoryAndKnowledge.ts
    ├── hooks/useObjectivesAndTimeline.ts
    ├── hooks/useProfile.ts      # Contexte Personnel (2026-07-30)
    ├── components/chat/         # ChatThread, AnalysisCard
    ├── components/ProfileSettingsForm.tsx  # Section "Profil & habitudes" de SettingsPage (2026-07-30)
    └── theme/ThemeProvider.tsx  # Thème clair/sombre réel
```

---

## 3. Modèle de données actuel

Tables SQLAlchemy réelles (`email_analyzer/db/models.py`) :

- `User`, `Tenant`, `Membership` — comptes et multi-tenancy.
- `Plan`, `Subscription`, `BillingEvent`, `UsageCounter` — facturation SaaS.
- `OAuthConnection` — connexions Gmail/Outlook.
- `Project` — projet détecté/créé, avec alias.
- `Email` — email ingéré, lié à un projet (relation 1:1 stricte).
- `ProjectSummary` — résumé LLM d'un projet : `structured_content` (JSONB —
  décisions, risques, échéances, prochaines étapes, rendez-vous probables),
  `llm_risk_level`, `schema_version`.
- `SuggestedAction` — action suggérée par l'IA, statut suivi.
- `Appointment` — rendez-vous détecté (agenda IA).
- `AssistantMessage` — historique brut de chat persisté (tenant + user)
  pour le chat portefeuille SaaS ; un seul fil continu par couple
  tenant/utilisateur, pas de notion de "conversation" séparée.
- `Memory` (Unit 46) — fait durable extrait de l'historique de chat
  (`category` ∈ preference/fact/context, `confidence`, dédupliqué par
  contenu exact). Distinct d'`AssistantMessage` : la mémoire est un résumé
  extrait, pas un remplacement de l'historique brut.
- `Organization`, `Contact`, `ProjectContactLink` (Unit 47) — entités
  normalisées depuis `ProjectSummary.structured_content["companies"/
  "participants"]` (déjà extrait par le LLM du Fast-Track), liées aux
  projets. Pas de graphe, tables relationnelles classiques.
- `Objective`, `Milestone` (Unit 54) — objectif optionnellement rattaché à
  un `Project`, jalons associés ; progression calculée à la lecture (pas
  de table dédiée), `blocking_reason` en simple colonne (pas de table
  "Blocking Issue" dédiée).
- `Profile` (2026-07-30) — Contexte Personnel : un enregistrement par
  (tenant, user), unique (contrairement à `Memory`, plusieurs lignes
  possibles) — profil (nom, langues, fuseau, localisation, préférences en
  JSONB, famille en JSONB) et habitudes en colonnes typées (heures de
  réveil/sommeil/travail, jours de sport, plage de concentration). Saisi
  directement par l'utilisateur (Paramètres), jamais extrait par LLM —
  voir `context/project-overview.md` "Feuille de route — 25 domaines de
  contexte", domaine #1.

Il n'existe **toujours pas** de tables `Fact` générique (fusionné dans le
concept `Memory` ci-dessus), `Entity` générique, `Document`, `Insight`,
`Decision`, `ToolExecution`, `KnowledgeGraphEdge` — ces concepts de la
vision long terme (§7) restent non implémentés ou couverts différemment
(voir §7 pour le détail à jour).

---

## 4. Modèle d'authentification et d'accès

- **Authentification applicative** : JWT émis par le backend après
  vérification des identifiants (bcrypt).
- **Gmail / Outlook** : OAuth2, jetons stockés via `OAuthConnection`.
- **IMAP entreprise (mode legacy)** : configuration `.env`, sans base de
  données persistante — ce mode ne bénéficie pas du Brief, des Actions, ni
  de l'assistant portefeuille (qui dépendent de données persistées).
- **Isolation multi-tenant** : filtrage par `tenant_id`/`user_id` au niveau
  des requêtes ORM. Pas d'isolation physique par device (il n'y a pas de
  stockage côté client).

Le chiffrement au repos des corps d'email (`body_encrypted`) évoqué dans
d'anciennes versions de ce document **n'est pas implémenté** — voir
`context/rfc-email-pipeline-v2.md` §0 et `context/evolution-plan.md` Phase 1
qui le planifient.

---

## 5. Jobs asynchrones (réel)

- Le thread FastAPI ne fait jamais d'I/O lourd (appels IMAP/Gmail/Outlook,
  parsing, appels LLM) directement : ce travail est délégué à `arq` via
  Redis (`email_analyzer/jobs.py` + `analysis_tasks.py`).
- Suivi d'état de job dans Redis (`create_job`/`set_status`/
  `report_progress`/`get_job`, TTL ~2h15), interrogé par le frontend via
  polling (`GET /api/analyze/{job_id}`, `GET /api/jobs/{id}`).
- Synchronisation planifiée (cron `arq`), refresh "Fast-Track" à la demande
  par projet, découverte de domaines, refresh agenda.
- Un throttle Redis limite l'auto-sync au login (`claim_daily_auto_sync`).

---

## 6. Architecture planifiée (court/moyen terme, non implémentée)

Basé sur `context/evolution-plan.md` et `context/rfc-email-pipeline-v2.md`
(RFC, statut "Proposition — non implémentée") :

- **Recherche sémantique centralisée via `pgvector`** dans PostgreSQL —
  pas de base vectorielle côté client, embeddings stockés dans une colonne
  `vector`, isolation par `tenant_id`.
- **Sync incrémental réel** : curseur par UID IMAP / `historyId` Gmail /
  `deltaLink` Outlook, au lieu du simple timestamp `last_synced_at` actuel.
- **Threading réel des conversations** (algorithme JWZ, `Message-ID`/
  `In-Reply-To`/`References`), au lieu du regroupement par sujet actuel.
- **Gestion des quotas/coûts LLM** : rate-limiting par tenant, cache de
  réponses LLM par hash de prompt.
- **Chiffrement au repos** des corps d'email et du texte extrait des pièces
  jointes.

Ces éléments sont un prérequis technique pour la mémoire persistante et le
knowledge graph décrits en §7 (la recherche sémantique `pgvector` est
notamment la base technique la plus proche d'un futur Memory Engine), mais
ne constituent pas encore ces fonctionnalités elles-mêmes.

---

## 7. Vision long terme — Executive Assistant (non implémentée)

Direction produit définie dans `context/project-overview.md`. Cible
architecturale :

```
                 Utilisateur
                      │
           Conversation naturelle
                      │
          Assistant IA Exécutif
                      │
         Moteur de récupération de contexte
                      │
      ┌────────────┼─────────────┐
      │            │             │
   Mémoire     Knowledge      Tools
                Graph
      │            │             │
      └────────────┼─────────────┘
                      │
        Sources d'information connectées
     (email, calendrier, documents, Slack,
      GitHub, CRM, fichiers locaux, ...)
```

- **Memory Engine** — fondation posée (Unit 46) : table `Memory` (§3),
  extraction structurée via `llm.extract_memory_candidates_openai/_gemini`,
  orchestrée par `MemoryService`. Déclenchement **explicite**
  (`POST /api/memory/extract`), pas automatique après chaque message. Pas
  de recherche vectorielle/sémantique (dépend de `pgvector`, §6, non fait).
- **Knowledge Graph** — fondation posée (Unit 47), sans graphe : tables
  relationnelles `Organization`/`Contact`/`ProjectContactLink` (§3),
  alimentées sans nouvel appel LLM par `KnowledgeService`
  (`sync_from_project_summary`, déclenchement explicite). Pas de
  raisonnement relationnel multi-sauts, pas de résolution contact↔
  organisation fiable (v1 lie tous les participants d'un projet à la
  première société citée).
- **Context Engine** — fondation posée (Unit 48) : `ContextService`
  assemble le prompt système du chat portefeuille à partir de
  Projects + Memory + Knowledge + **Profile** (2026-07-30, voir §3) ;
  `ConversationService` délègue entièrement la construction de son
  contexte à ce service. Emails/Documents/réunions ne sont pas des
  sources directes (les emails entrent indirectement via les résumés de
  projet). Voir `context/project-overview.md` "Feuille de route — 25
  domaines de contexte" (2026-07-30) pour le statut détaillé de chacun
  des domaines de la vision long terme "second cerveau" (Calendrier,
  Tâches, Documents, Communication, Finance, Santé, ...) — la plupart
  restent non commencés, bloqués sur un choix de fournisseur externe et
  des identifiants API que l'utilisateur n'a pas encore fournis.
- **Couche Agent / Tools** — **unifiée dans le chat portefeuille** (unité
  postérieure à Unit 57, 2026-07-27) : la décision Unit 57 ("coexistence
  durable, pas de fusion tant qu'aucun signal concret ne la justifie") est
  explicitement renversée à la demande de l'utilisateur, qui veut une seule
  surface de chat aussi capable et fluide que les assistants du marché
  (ChatGPT/Gemini). `POST /api/agent/query` et tout `email_analyzer/agent/`
  (`Planner`, `Executor`, `ResponseBuilder`, `Agent`) sont **supprimés** —
  `email_analyzer/tools/` (7 outils en lecture seule, `Tool` =
  `typing.Protocol`, pas d'héritage, inchangés) est désormais orchestré par
  `ToolExecutor` (`email_analyzer/services/tool_executor.py`), appelé
  directement par `ConversationService.stream_message`. La sélection
  d'outils n'est plus déterministe par mots-clés : elle passe par le
  tool-calling natif d'OpenAI/Gemini (`llm.stream_portfolio_assistant_chat_
  openai/_gemini`, paramètre `tools`) — le modèle décide lui-même, à chaque
  tour, s'il répond directement depuis le contexte déjà résumé
  (`ContextService`, inchangé) ou s'il doit appeler un ou plusieurs outils
  pour aller chercher des données plus fraîches/précises (emails bruts,
  rendez-vous, résumé de projet à jour, brouillon de réponse) — jusqu'à
  `_MAX_TOOL_ROUNDS = 4` allers-retours par message utilisateur. Les trois
  outils qui exigeaient un `project_id` que le texte libre ne permettait pas
  de résoudre (`summarize_project`, `search_emails`, `draft_reply`) sont
  désormais atteignables : `ToolExecutor` résout un `project_name` en
  langage naturel (fourni par le LLM, qui voit déjà les noms de projets dans
  le contexte) vers un `project_id` réel (correspondance exacte, puis
  sous-chaîne, puis floue via `difflib`), et renvoie un message d'erreur
  explicite au LLM en cas d'ambiguïté plutôt que de planter. La réponse du
  chat (`POST /api/assistant/conversations/{id}/chat`) est désormais un
  flux `text/event-stream` (SSE, voir `api/sse.py`) — premier streaming
  réel de ce repo, remplaçant le faux effet machine-à-écrire côté
  frontend.

---

## 8. Invariants du codebase (réels, applicables au code actuel)

1. **Cardinalité stricte 1:1 Email-Project** — un `Email` est rattaché à un
   seul `Project` en base, jamais une relation many-to-many.
2. **Aucun SQL brut** — toutes les requêtes passent par l'ORM SQLAlchemy.
3. **Isolation multi-tenant** — toute requête sur des tables liées à un
   tenant filtre explicitement sur `tenant_id`/`user_id`.
4. **Pas d'I/O lourd dans le thread FastAPI** — appels IMAP/Gmail/Outlook,
   parsing de fichiers, appels LLM synchrones : toujours délégués à un job
   `arq`, jamais exécutés dans le cycle requête/réponse HTTP. Exception de
   longue date, documentée et assumée : le chat interactif
   (`POST /api/assistant/conversations/{id}/chat`) appelle le LLM de façon
   synchrone dans le cycle requête/réponse — FastAPI exécute cette route
   `def` dans un threadpool, pas dans la boucle événementielle, donc
   l'invariant (ne pas bloquer l'event loop) reste respecté. Depuis
   l'unification Agent/Conversation (§7), ce même thread peut enchaîner
   plusieurs appels LLM séquentiels (boucle agentique, jusqu'à
   `_MAX_TOOL_ROUNDS`) et streamer la réponse via `StreamingResponse` (un
   générateur synchrone, itéré depuis ce même thread) — une extension de
   cette exception déjà acceptée, pas une nouvelle violation ; l'invariant
   continue de s'appliquer sans changement à l'analyse/synchronisation en
   masse, qui reste entièrement `arq`.
5. **Model → Repository → Service → API** (établi Units 45-57, appliqué à
   tout code touché depuis) — un routeur ne fait jamais de requête SQL
   directe pour une nouvelle fonctionnalité : la requête vit dans un
   `*_repository.py`, l'orchestration/validation dans un `*_service.py`,
   le routeur ne fait qu'authentifier/valider la requête HTTP et appeler
   le service. Les routeurs plus anciens non touchés depuis (`projects.py`,
   `agenda.py`) gardent encore leurs requêtes inline — à faire migrer
   seulement à l'occasion d'un changement réel de leur logique, pas par
   nettoyage préventif.

Les anciens invariants "Zero Clear-Text Cloud Storage" et "Hard 30-Day
Device Cache Limit" sont retirés de cette liste : ils décrivaient un
chiffrement au repos et un cache device qui n'ont jamais été implémentés
(voir §4 et §6 pour leur statut réel/planifié).

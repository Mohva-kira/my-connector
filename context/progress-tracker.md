# Progress Tracker

Update this file after every meaningful implementation change.

## Current Phase

- Assessment & Alignment

## Current Goal

- Align the existing codebase with the architecture and standards defined in the context files, then implement missing features incrementally.

## Completed

- Context files written (project-overview.md, architecture.md, ui-context.md, ai-workflow-rules.md)
- code-standards.md written (was a blank placeholder — now complete)
- Python FastAPI backend exists at `services/email_analyzer/` with:
  - Multi-tenant SaaS auth (JWT + bcrypt)
  - IMAP-based email fetching (existing)
  - OpenAI + Gemini LLM integration
  - CinetPay billing integration
  - PostgreSQL via SQLAlchemy + Alembic migrations (client **psycopg** v3, pas psycopg2 — évite UnicodeDecodeError sous Windows avec locales CP1252)
  - Fernet encryption for IMAP credentials
- React + Vite + TailwindCSS frontend exists at `services/email_analyzer/frontend/`
- **Unit 1 — Gmail OAuth2 Integration** (2026-05-04):
  - `OAuthConnection` model added to `db/models.py` with encrypted token storage
  - `Tenant` model updated with `oauth_connections` relationship
  - Alembic migration `002_gmail_oauth.py` creates `oauth_connections` table
  - `email_analyzer/gmail_oauth.py`: full OAuth flow (authorize URL, token exchange, token refresh, email fetching, normalization)
  - `api/routers/oauth.py`: 4 endpoints — GET `/api/oauth/gmail/authorize`, GET `/api/oauth/gmail/callback`, GET `/api/oauth/connections`, DELETE `/api/oauth/connections/{id}`
  - `api/main.py`: oauth router registered
  - `requirements.txt`: added `google-auth-oauthlib`, `google-auth-httplib2`, `google-api-python-client`
  - `.env.example`: added `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET` docs

- **Unit 6 — Analyse asynchrone (jobs) + mitigations timeout 504** (2026-07-05) :
  - Cause : `/api/analyze` faisait tout le travail (IMAP + ML + LLM) de façon
    synchrone dans la requête → dépassait le timeout de passerelle (Nginx + zrok) → 504.
  - `email_analyzer/ai_intelligent.py` : `get_shared_analyzer()` — analyseur ML
    singleton, chargé une seule fois (fin du cold-load par requête).
  - `email_analyzer/project_mail.py` : utilise `get_shared_analyzer()` au lieu de
    reconstruire `EmailIntelligentAnalyzer()` à chaque analyse.
  - `email_analyzer/config.py` + `email_analyzer/llm.py` : `LLM_TIMEOUT_SECONDS`
    (défaut 60 s) — timeout client LLM abaissé (< passerelle).
  - `email_analyzer/jobs.py` (nouveau) : store de jobs en mémoire + `ThreadPoolExecutor`
    (max_workers=2) + purge TTL. `create_job` / `submit` / `get_job`.
  - `api/main.py` : `POST /api/analyze` renvoie `202 {job_id, status:"pending"}` et
    lance l'analyse en tâche de fond ; nouveau `GET /api/analyze/{job_id}` (statut +
    isolation par tenant). Pré-chauffe l'analyseur ML au `lifespan`.
  - `frontend/src/pages/HomePage.tsx` : POST puis polling de `/api/analyze/{job_id}`
    (intervalle 2,5 s, plafond 5 min, timeout 30 s par requête).
  - `deploy/nginx-api.conf` (nouveau) : timeouts proxy documentés + note zrok +
    consigne uvicorn 1 worker.
  - `.env.example` : `LLM_TIMEOUT_SECONDS`, recommandation `EMAIL_ANALYZER_USE_LOCAL_ML=false` en prod.
  - Contrainte : lancer uvicorn avec **1 worker** (store en mémoire non partagé).
    Évolution multi-worker → table `analysis_jobs` en DB.

- **Unit 3 — Thème sombre (ui-context.md) + chargement séquentiel par lots +
  rendu progressif + gamification** (2026-07-05) :
  - Thème sombre appliqué à tout le frontend (tokens `ui-context.md` ajoutés
    dans `frontend/tailwind.config.js` — couleurs `bg-*`, `text-*`,
    `border-*`, `success/warning/danger/info`, `ai-*`, `priority-*`,
    `accent-*`, radius, police Inter via `index.html`) ; tous les composants
    (`AppShell`, `HomePage`, `AnalysisDashboard`, `ConversationalAssistant`,
    `AuthPanel`, `SaasPanels`, `LoginPage`, `SettingsPage`, `App`) recolorés,
    plus aucune classe `stone-*`/`slate-*`/`bg-white` résiduelle (vérifié par
    grep).
  - `email_analyzer/config.py` : `batch_threshold()` (env
    `EMAIL_ANALYZER_BATCH_THRESHOLD`, défaut 15) et `batch_chunk_size()` (env
    `EMAIL_ANALYZER_BATCH_SIZE`, défaut 10).
  - `email_analyzer/project_mail.py` : `search_project_emails(...,
    on_batch=None)` — au-delà du seuil, émet une progression + un partiel
    `{nb_emails, emails_critiques}` (via `identify_critical_emails`,
    rule-based, sans coût ML) après chaque lot de 10 ; sous le seuil,
    comportement strictement inchangé (`on_batch` jamais appelé). Callback
    protégé par `try/except` (ne casse jamais l'analyse).
  - `email_analyzer/analyzer.py` : `EmailProcessor.process_latest_emails`
    passe `on_batch` jusqu'à `search_project_emails`.
  - `email_analyzer/jobs.py` : champs `processed`/`total`/`partial` sur le
    job + nouvelle fonction `report_progress(job_id, processed, total,
    partial)`.
  - `api/main.py` : `_run_analysis_legacy`/`_run_analysis_saas` branchent
    `on_batch` sur `report_progress` ; `GET /api/analyze/{job_id}` expose
    additivement `progress: {processed, total}` et `partial` (rétrocompatible,
    `record_analysis_usage` toujours comptabilisé une seule fois en fin de job).
  - `frontend/src/pages/HomePage.tsx` : `pollAnalysis` accepte un callback
    `onTick` qui alimente `progress`/`partialBlock` à chaque poll ;
    `AnalysisDashboard` réutilisé tel quel pour le rendu progressif (aucun
    nouveau composant liste d'emails) ; barre de progression réelle + pastille
    `.animate-batch-pop` ("+10 emails analysés") + messages de statut rotatifs
    ; repli sur le skeleton existant tant qu'aucun lot n'est arrivé (petits
    mailboxes ≤15 emails : comportement inchangé, jamais de barre affichée).
  - `frontend/src/index.css` : keyframes `batch-pop` + `.animate-batch-pop`.
  - Vérifié en conditions réelles (compte IMAP réel du dépôt, `imap_configured:
    true`) : sur une recherche à 15749 candidats, progression par paliers
    exacts de 10 avec `partial.nb_emails`/`emails_critiques` qui grossissent ;
    sur une recherche ≤15, `progress.total` reste à 0 et le `result` final
    conserve la forme d'origine. Vérifié aussi : `tsc --noEmit` sans erreur,
    `vite build` transforme les 41 modules sans erreur (échec de build limité
    au nettoyage du dossier `dist/` pré-existant, appartenant à `root` —
    problème d'environnement local sans rapport avec le code).
  - Non vérifié dans cette session (pas d'outil navigateur disponible) : rendu
    visuel réel des animations/dark theme dans un navigateur, et le chemin
    SaaS (`DATABASE_URL` non configuré dans cet environnement de dev local,
    `saas_enabled: false`) — la non-duplication de `record_analysis_usage`
    a été vérifiée par lecture de code uniquement.

- **Unit 3.1 — Bornage IMAP de `/api/chat`** (2026-07-06) :
  - Cause : `fetch_last_n_emails_for_chat` (contexte du chat) appelait
    `search_project_emails` en scannant TOUS les emails candidats de la
    fenêtre IMAP avant de tronquer aux `n` derniers — même risque de lenteur
    que `/api/analyze` avant l'Unit 6, mais sur un endpoint resté synchrone
    (pas de job/polling). Décision utilisateur : garder `/api/chat`
    synchrone, corriger le fetch à la racine plutôt que d'ajouter un
    job/polling, et animer l'attente côté frontend.
  - `email_analyzer/project_mail.py` : extraction du corps par-email dans
    `_process_one_email_id` (partagé par les deux modes de balayage) ;
    nouveau paramètre `max_matches` sur `search_project_emails` — balaye les
    `email_ids` en ordre inverse (plus récents d'abord) et s'arrête dès que
    chaque filtre a atteint `max_matches` correspondances, puis réordonne en
    chronologique avant de retourner. `on_batch` ignoré dans ce mode.
  - `email_analyzer/analyzer.py` : `fetch_last_n_emails_for_chat` passe
    `max_matches=n` au lieu de scanner puis tronquer.
  - `frontend/src/ConversationalAssistant.tsx` : indicateur "en train
    d'écrire" animé (`TypingDots`, keyframes `typing-dot` dans `index.css`)
    + messages de statut rotatifs (`BOOTSTRAP_STATUS_MESSAGES` /
    `REPLY_STATUS_MESSAGES`) remplaçant les textes statiques "Connexion à
    l'assistant…" / "Réponse en cours…".
  - Vérifié : chemin `/api/analyze` (balayage avant + `on_batch`) inchangé
    après refactor (tests avec IMAP mocké, ticks identiques). Nouveau
    balayage arrière testé avec correspondances éparses : retrouve bien les
    N dernières correspondances en ordre chronologique, sans scanner tout le
    mailbox (vérifié par comptage des appels `fetch`). Test réel sur le
    compte IMAP du dépôt avec le même filtre large que précédemment (15749
    candidats) : `/api/chat` répond maintenant en ~42 s (dominé par la
    génération LLM), au lieu de risquer un scan complet.
  - `tsc --noEmit` : OK après les changements frontend.

- **Unit 7 — Gamified Loading Experience, Step 1: module init** (2026-07-06) :
  - Contexte : transformer l'attente de synchronisation/analyse (déjà gérée
    par le polling `jobs.py` + `HomePage.tsx`, Unit 6) en une visualisation
    animée du pipeline de traitement, découplée de React.
  - **Écart accepté vs. la demande initiale** : la demande visait Phaser 3 +
    `ion-phaser` + React 19. Le frontend existant est React **18.3.1**
    (`services/email_analyzer/frontend/`), et `@ion-phaser/react` déclare un
    peer dep `react: ^16.7.0` (dernier release ancien) → conflit garanti avec
    18 et 19. Décision utilisateur : abandon complet de Phaser/ion-phaser,
    réécriture avec React + Framer Motion, React reste en 18.3.1.
  - **Échelle** : la contrainte "5000 emails / 60fps / object pool" (pensée
    pour un moteur canvas) est réinterprétée pour du DOM : le convoyeur
    n'affiche jamais plus de `MAX_VISIBLE_ENVELOPES` (30) enveloppes montées
    à la fois ; le HUD reflète toujours les vrais totaux backend quel que
    soit le volume.
  - Dépendances ajoutées : `zustand@5`, `@tanstack/react-query@5`,
    `framer-motion@12` (aucun conflit de peer dependency à l'installation).
  - Fichiers créés (`frontend/src/`) :
    - `sync-experience/types.ts` — `SyncStage`, `EmailEnvelopeModel`,
      `SyncStats`, `SyncEventMap` (contrat partagé store/EventBus/UI).
    - `sync-experience/eventBus/EventBus.ts` — pub/sub typé singleton,
      réservé aux effets ponctuels (floating text, particules, célébration),
      jamais utilisé pour l'état de rendu continu (ça reste du re-render
      React normal via les sélecteurs Zustand).
    - `store/syncStore.ts` — état plat (stage, stats, envelopes cappées,
      currentTask) + setters uniquement ; n'émet jamais vers l'EventBus.
    - `hooks/useSyncGame.ts` — seul pont store → EventBus : `subscribe` au
      store, diff des compteurs agrégés, émission des événements
      (`STAGE_CHANGED`, `PROGRESS_UPDATED`, `EMAIL_IMPORTED`, `ACTION_FOUND`,
      `DEADLINE_FOUND`, `URGENT_FOUND`, `SYNC_COMPLETED`).
    - `sync-experience/components/SyncExperience.tsx` — placeholder
      minimal (stage + progress) prouvant le câblage store → hook →
      composant ; sera remplacé progressivement par PipelineTrack, Robot,
      EmailEnvelope, HUD, FloatingText.
    - `components/SyncGame.tsx` — point d'entrée public, monte
      `useSyncGame()` puis `<SyncExperience/>`.
  - Vérifié : `tsc --noEmit` sans erreur ; `vite build` transforme les 41
    modules sans erreur (échec ensuite limité au nettoyage de `dist/`
    pré-existant appartenant à `root`, problème d'environnement local déjà
    documenté ci-dessus, sans rapport avec ce code).
  - Pas encore fait : intégration au flux réel de `/api/analyze/{job_id}`
    (React Query), composants visuels (stations, robot, enveloppes, HUD,
    floating text), montage dans une page. Attente de validation utilisateur
    avant l'étape 2.

## Completed (suite)

- **Unit 8 — Réconciliation stack architecture.md (docs seulement)** (2026-07-06) :
  - Contexte : `context/evolution-plan.md` (Phase 1) a formalisé la décision
    de baser toute la roadmap sur l'état réel du code (Python FastAPI) plutôt
    que sur la cible non implémentée de `architecture.md` (Node.js/NestJS +
    BullMQ/Redis).
  - `context/architecture.md` : table Technology Stack (`Backend API` →
    FastAPI (Python), `Background Jobs` → Redis + arq), diagramme
    High-Level Architecture, diagramme Asynchronous Job Architecture,
    section Queue Processing, et route Fast-Track (`:id` → `{id}`, syntaxe
    FastAPI) — tous les mentions NestJS/Node.js/BullMQ remplacées par
    FastAPI/Redis+arq (vérifié par grep, aucune occurrence restante).
  - Aucun changement de code : le job store en mémoire (`jobs.py`,
    `ThreadPoolExecutor`, contrainte 1 worker uvicorn) reste inchangé — sa
    migration vers Redis/arq est une unité séparée (prochaine étape,
    Phase 1 du plan d'évolution).

- **Unit 9 — Migration du store de jobs : mémoire (ThreadPoolExecutor) → Redis + arq**
  (2026-07-06) :
  - Contexte : `context/evolution-plan.md` (Phase 1) identifiait la contrainte
    « 1 seul worker uvicorn » (état de job non partagé entre processus,
    `jobs.py` historique) comme le blocker technique le plus critique avant
    toute mise à l'échelle (Fast-Track, multi-tenant).
  - `email_analyzer/config.py` : `redis_url()` (env `REDIS_URL`, défaut
    `redis://localhost:6379/0`).
  - `email_analyzer/jobs.py` : réécrit intégralement. L'état d'un job
    (`status/result/error/tenant_id/processed/total/partial`) est stocké dans
    Redis (clé `myconnector:job:{id}`, JSON, `EX` = 30 min — remplace le TTL
    géré manuellement en mémoire). Nouvelle fonction `enqueue(job_id,
    task_name, *args)` qui met le job en file via `arq` au lieu d'exécuter un
    closure dans un `ThreadPoolExecutor` local ; nouvelle fonction
    `set_status(job_id, status, result=, error=)` (remplace la logique
    interne de l'ancien `submit`). `report_progress`/`get_job` gardent leur
    signature.
  - `email_analyzer/analysis_tasks.py` (nouveau) : héberge les tâches arq
    `run_analysis_legacy`/`run_analysis_saas` (déplacées depuis
    `api/main.py` — un worker arq tourne dans un process séparé, il ne peut
    pas recevoir de closure Python). Chaque tâche appelle la logique
    d'analyse existante via `asyncio.to_thread` (le pipeline IMAP/ML/LLM
    reste synchrone, seule l'orchestration devient asynchrone) puis appelle
    `jobs.set_status`. `WorkerSettings` (2 jobs concurrents par process,
    `on_startup` précharge l'analyseur ML partagé — même logique que le
    lifespan FastAPI côté API).
  - `api/main.py` : `_run_analysis_legacy`/`_run_analysis_saas` supprimées
    (déplacées) ; `/api/analyze` appelle désormais `jobs.enqueue(job_id,
    "run_analysis_legacy"|"run_analysis_saas", ...)` avec des arguments
    primitifs (plus l'objet `AnalyzeRequest` complet) ; `/api/analyze/{job_id}`
    inchangé (même forme de réponse).
  - `requirements.txt` (racine) : `arq>=0.26,<1` ajouté.
  - `.env.example` (racine) : `REDIS_URL` documenté.
  - `services/email_analyzer/deploy/nginx-api.conf` : suppression de la
    consigne « uvicorn --workers 1 » (obsolète, l'état vit dans Redis) ;
    ajout de la consigne pour lancer le worker arq séparément (`arq
    email_analyzer.analysis_tasks.WorkerSettings`), **obligatoire** — sans
    lui les jobs restent `pending` indéfiniment.
  - Vérifié end-to-end en conditions réelles (Redis local installé par
    l'utilisateur, `.env` avec identifiants IMAP réels, `assistant_provider:
    "none"` pour éviter une dépendance à une clé LLM) : (1) `jobs.create_job`
    + `jobs.enqueue` en Python direct → worker arq exécute réellement le job
    (process séparé, log arq confirmé) → `jobs.get_job` renvoie `status:
    "done"` avec un résultat IMAP réel ; (2) `uvicorn` démarré isolément (port
    de test) + `curl POST /api/analyze` puis polling `GET
    /api/analyze/{job_id}` → même contrat de réponse qu'avant la migration
    (`{status, result, error, progress, partial}`), aucune régression côté
    frontend. `python -m py_compile` OK sur les 4 fichiers modifiés/créés.
    Processus de test arrêtés après vérification (aucun process de test ne
    reste en arrière-plan).
  - Non vérifié : comportement sous charge concurrente réelle (plusieurs
    jobs simultanés, plusieurs process worker arq) — hors périmètre de cette
    unité, à couvrir si le volume de jobs augmente.

## In Progress

- Unit 7 — Gamified Loading Experience : étape 1/8 livrée, en attente de
  validation utilisateur avant l'étape 2 (câblage React Query ↔ Zustand ↔
  EventBus).

## Next Up

- **Unit 7, étape 2** : câblage `React Query → syncStore` (alimenter le
  store depuis le polling `/api/analyze/{job_id}` existant), puis
  vérification du flux `useSyncGame → EventBus`.
- **Unit 2 — Outlook (Microsoft Graph) OAuth2 Integration**: same pattern as Gmail, using Microsoft Identity Platform
- **Unit 4 — APScheduler / Celery batch processing** (2x/day email sync job, replacing BullMQ since stack is Python)
- **Unit 5 — Email importance scoring engine** (rules + AI hybrid using existing LLM integration)

## Open Questions

1. **Gmail OAuth analysis integration**: The `/api/analyze` endpoint still uses IMAP. Unit 2.5 should wire `gmail_oauth.fetch_emails()` into `EmailProcessor` so Gmail OAuth connections are used automatically when IMAP is not configured.
2. **Billing**: CinetPay (XOF) is already integrated. Architecture.md does not specify a payment provider. CinetPay retained as-is.

## Architecture Decisions

- **Stack decision resolved (2026-07-06)**: `architecture.md` updated to reference FastAPI (Python) + Redis/`arq` instead of NestJS (Node.js) + BullMQ, aligning the documented architecture with the actual implementation. `arq` chosen over Celery/RQ for the job-queue migration because it is asyncio-native and matches the existing async FastAPI codebase. See `context/evolution-plan.md` Phase 1. The in-memory `ThreadPoolExecutor` job store in `jobs.py` was migrated to Redis + `arq` the same day — see Unit 9 below.
- Database is PostgreSQL with SQLAlchemy ORM (aligns with architecture)
- AI layer uses OpenAI (gpt-4o-mini) + Gemini — architecture specifies GPT-4.1/GPT-4o-mini (aligns)
- Frontend is React + Vite (aligns with architecture)
- Email access is currently IMAP-based, not Gmail/Outlook OAuth2 (gap to close)
- Gamified Loading Experience (Unit 7) uses React + Framer Motion, not Phaser 3/ion-phaser as originally briefed — ion-phaser's peer dep (`react@^16.7.0`) conflicts with the installed React 18.3.1; user chose to drop the canvas engine entirely rather than force an incompatible dependency or do an unrelated React 19 upgrade

## Session Notes

- 2026-05-04: `requirements.txt` — `psycopg2-binary` remplacé par `psycopg[binary]` ; `normalize_database_url` force `postgresql+psycopg` pour corriger UnicodeDecodeError à la connexion sous Windows (messages serveur en CP1252).
- 2026-05-04: Doc de lancement API (`api/main.py`) — `cd services/email_analyzer` + uvicorn (évite PYTHONPATH) ; note PowerShell ($env:PYTHONPATH) car `set` (CMD) ne définit pas l’environnement sous PS.
- 2025-05-04: Full project audit completed. Existing code is a working SaaS email analyzer using IMAP.
  Target architecture requires Gmail/Outlook OAuth2 and Node.js backend. Stack decision needed from user before proceeding.
- code-standards.md was a blank prompt template — has been replaced with real standards.
- Next action: user confirms stack direction, then implement Unit 1 (Gmail OAuth2 integration).

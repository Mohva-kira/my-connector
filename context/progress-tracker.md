# Progress Tracker

Update this file after every meaningful implementation change.

## Current Phase

- Assistant-first frontend/backend refonte (Brief, sidebar, portfolio
  assistant) — see Unit 24. Legacy per-project analysis tool remains for
  non-SaaS (`.env`-only) users.

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
  - **Bug trouvé et corrigé (2026-07-06)** : le worker arq échouait sur toute
    analyse en mode SaaS avec `RuntimeError("DATABASE_URL non configuré")`
    même quand `.env` contenait bien `DATABASE_URL`. Cause : `api/main.py` et
    `alembic/env.py` appellent `load_dotenv()` explicitement, mais
    `analysis_tasks.py` (le module chargé par `arq
    email_analyzer.analysis_tasks.WorkerSettings`, process séparé d'uvicorn)
    ne le faisait pas — confirmé en inspectant `/proc/<pid>/environ` du
    worker en cours d'exécution : ni `DATABASE_URL` ni `REDIS_URL` n'y
    figuraient, alors qu'ils sont définis dans le `.env` racine. Corrigé en
    ajoutant les mêmes appels `load_dotenv(_REPO_ROOT / ".env")` /
    `load_dotenv(_SERVICE_ROOT / ".env")` en tête de `analysis_tasks.py`,
    avant l'import de `email_analyzer.config` (dont `redis_url()` est évalué
    au chargement du module, dans le corps de `WorkerSettings`). Vérifié :
    `init_db()` fixe désormais `SessionLocal` même avec `env -i` (environnement
    parent vidé). Worker relancé pour prendre en compte le correctif.

- **Unit 10 — Persistance Project/Email/ProjectSummary/SuggestedAction (préalable au Fast-Track)**
  (2026-07-06) :
  - Contexte : le vrai Fast-Track (architecture.md, Process 2) suppose un
    `last_processed_email_timestamp` stocké par projet pour calculer un
    delta ; ce préalable n'existait pas du tout (aucune table
    projects/emails/project_summaries/suggested_actions dans
    `db/models.py`, tout le pipeline `/api/analyze` est un scan IMAP live
    sans état persisté). Décision utilisateur : construire cette
    persistance d'abord, avant tout endpoint Fast-Track.
  - `email_analyzer/db/models.py` : ajout de `Project`, `Email`,
    `ProjectSummary`, `SuggestedAction` + enums `ProjectStatus`,
    `RecipientStatus`, `ProjectSentiment`, `SuggestedActionStatus` (même
    convention que les enums existants : `str, enum.Enum`, stockés en
    `String`, pas d'enum Postgres natif). Scoping par `tenant_id` (pas
    `user_id` comme dans le schéma cible d'architecture.md — le modèle de
    données réel n'a que des tenants, pas d'utilisateurs propriétaires de
    boîte mail individuels). `Email.external_id` + contrainte unique
    `(tenant_id, external_id)` pour la dédup exigée par
    `code-standards.md §8`. `ProjectSummary.project_id` unique (une seule
    ligne courante par projet, régénérée en place — pas un historique).
    `Tenant.projects` ajouté en back-ref. `Email.body_encrypted` est une
    colonne texte simple à ce stade — le chiffrement Fernet réel reste une
    unité séparée, pas encore câblée.
  - `alembic/versions/003_projects.py` (nouveau) : migration créant les 4
    tables + index, avec `downgrade()` symétrique.
  - Vérifié en conditions réelles (PostgreSQL local installé par
    l'utilisateur, base `connector`, `DATABASE_URL` renseigné dans `.env`) :
    `alembic upgrade head` applique proprement sur 002 existant ;
    `alembic downgrade -1` puis `upgrade head` round-trip sans erreur ;
    insert/lecture réels via l'ORM (Tenant → Project → Email/ProjectSummary/
    SuggestedAction, relations et back-refs fonctionnelles) ; la contrainte
    unique `(tenant_id, external_id)` rejette bien un doublon
    (`IntegrityError`). Données de test nettoyées après vérification.
    `python -m py_compile` OK.
  - Pas encore fait : aucun code applicatif n'écrit ou ne lit encore ces
    tables (le pipeline `/api/analyze` reste un scan IMAP live, inchangé).
    C'est le prérequis pour l'unité suivante (endpoint Fast-Track réel +
    écriture des résultats d'analyse dans `Email`/`ProjectSummary`).

- **Unit 7, étape 2 — Câblage du polling réel sur le `syncStore`** (2026-07-06) :
  - Contexte : l'étape 1 avait posé le store/EventBus/placeholder sans les
    alimenter avec de vraies données. Décision utilisateur : garder le
    mécanisme de polling existant (`pollAnalysis`, Unit 6, déjà validé en
    conditions réelles) plutôt que de le remplacer par `React Query`
    (dépendance ajoutée en étape 1 mais restée inutilisée — aucun composant
    ne la consomme encore, à réévaluer si un vrai besoin de cache/retry
    apparaît) ; mapper les `SyncStage` sur les seuls signaux réellement émis
    par le backend plutôt que d'en simuler par paliers de temps.
  - `sync-experience/bridgeAnalyzeJob.ts` (nouveau) : `applyAnalyzeTickToStore(tick,
    store)`, fonction pure sans dépendance React. Mapping **minimal honnête** :
    `status: "pending"` → `CONNECTING`, `"running"` → `IMPORTING`, `"done"` →
    `COMPLETED`, `"error"` → `store.reset()`. Les 6 stages intermédiaires
    (`NORMALIZING`…`GENERATING_BRIEFING`) restent définis dans le type mais ne
    sont jamais atteints — le backend n'émet aucun signal de pipeline
    granulaire aujourd'hui, et cette passerelle n'en invente pas. Stats
    dérivées uniquement de données réelles : `emailsProcessed`/`emailsTotal`
    depuis `progress.{processed,total}`, `urgentFound` = somme des longueurs
    de `partial[*].emails_critiques` (comptage réel d'emails critiques
    détectés par lots, rule-based, cf. Unit 3). `actionsFound`/`deadlinesFound`
    restent à 0 pendant la progression — aucune donnée réelle ne les
    alimente avant la fin de l'analyse LLM.
  - `hooks/useSyncGame.ts` : logique de souscription extraite dans
    `subscribeSyncStoreToEventBus()` (pure, sans React) ; `useSyncGame` ne
    fait plus que l'appeler dans un `useEffect`. Permet de piloter le flux
    store → EventBus dans un script `tsx` sans monter React, pour vérifier
    réellement le câblage plutôt que de se fier à la seule compilation.
  - `pages/HomePage.tsx` : `useSyncStore.getState().reset()` au lancement
    d'une analyse et dans le `catch` (échecs réseau/HTTP qui ne passent pas
    par un tick `"error"`) ; `applyAnalyzeTickToStore(...)` appelé dans le
    callback `onTick` déjà existant de `pollAnalysis`, en plus (pas en
    remplacement) de la mise à jour du state React local
    (`progress`/`partialBlock`) qui alimente l'UI actuelle — aucune requête
    HTTP dupliquée, aucune régression du flux Unit 6/3.
  - **Bug découvert et corrigé pendant la vérification** : `setStage(...)`
    puis `setStats(...)` sont deux appels `set()` Zustand distincts, chacun
    notifiant immédiatement les abonnés (`useSyncGame`). En appelant
    `setStage("COMPLETED")` avant `setStats(...)`, l'événement
    `SYNC_COMPLETED` se déclenchait avec les stats du tick **précédent**
    (stats pas encore mises à jour). Corrigé en réordonnant
    `bridgeAnalyzeJob.ts` pour appeler `setStats` avant `setStage` — les
    stats sont donc toujours à jour au moment où `STAGE_CHANGED`/
    `SYNC_COMPLETED` sont émis.
  - Vérifié : `tsc --noEmit` sans erreur ; `vite build` transforme les 46
    modules sans erreur (échec ensuite limité au nettoyage de `dist/`
    pré-existant appartenant à `root`, problème d'environnement déjà
    documenté, sans rapport avec le code). Flux store → EventBus vérifié
    avec un script `tsx` temporaire (créé puis supprimé) rejouant une
    séquence de ticks réalistes (`pending` → `running` ×2 avec
    `progress`/`partial` croissants → `done`, puis un tick `error` isolé) :
    tous les événements (`STAGE_CHANGED`, `EMAIL_IMPORTED`, `URGENT_FOUND`,
    `PROGRESS_UPDATED`, `SYNC_COMPLETED`) sont émis avec les payloads
    attendus, `SYNC_COMPLETED` porte les stats finales correctes après
    correction du bug, et le tick `error` réinitialise bien le store à
    l'état initial.
  - Non vérifié : rendu réel dans le navigateur avec un job d'analyse
    IMAP/Redis live (pas d'identifiants IMAP configurés dans cet
    environnement de session — `.env` absent) ; `<SyncGame />` n'est
    toujours pas monté dans `HomePage.tsx` (le composant reste un
    placeholder texte, la UI de résultat actuelle — barre de progression,
    `partialBlock` — n'a pas été touchée pour éviter toute régression
    visuelle avant que les vrais composants (PipelineTrack, Robot,
    EmailEnvelope, HUD, FloatingText) existent).

- **Unit 7, étape 3 — Premier composant visuel réel (HUD)** (2026-07-06) :
  - Contexte : décision utilisateur explicite de passer en mode autonome
    pour la suite de ce chantier (« continue l'implémentation sans demander
    mon avis, en tant qu'administrateur prends les décisions recommandées
    et valide les suggestions ») — les choix ci-dessous sont donc pris et
    documentés directement, sans question de clarification préalable.
  - `sync-experience/components/HUD.tsx` (nouveau) : composant de lecture
    seule branché sur `useSyncStore` via sélecteurs (`stage`, `stats`).
    Libellés FR par `SyncStage` (`STAGE_LABELS`, seuls `CONNECTING`,
    `IMPORTING`, `COMPLETED` sont réellement atteignables aujourd'hui, cf.
    étape 2) ; barre de progression + compteur `processed/total` affichés
    uniquement si `emailsTotal > 0` (sinon message d'attente neutre, jamais
    de fausse barre à 0%) ; badge `urgentFound` (token `danger`, cohérent
    avec `color-sentiment-critical` de `ui-context.md`) affiché seulement si
    `> 0` — conforme au principe *Calm Workspace* (pas de badge à zéro en
    permanence).
  - `sync-experience/components/SyncExperience.tsx` : remplace le
    placeholder texte (`{stage} / {progress}%`) par `<HUD />`. Toujours pas
    monté dans `HomePage.tsx` — l'UI de résultat actuelle (barre + skeleton
    + `partialBlock`, Unit 3) reste inchangée tant que `SyncExperience` n'a
    pas tous ses éléments visuels (PipelineTrack, Robot, EmailEnvelope,
    FloatingText restent à construire).
  - Vérifié : `tsc --noEmit` OK ; `vite build` transforme les modules sans
    erreur (échec ensuite limité au même `EACCES` sur `dist/`
    pré-existant, sans rapport). Tentative de vérification visuelle réelle
    dans un navigateur headless (Chromium système + `playwright-core`
    installé temporairement en `--no-save`, route `/dev/sync-preview`
    temporaire montant `<SyncGame/>` avec des stats seedées) : **Chromium
    plante immédiatement (SIGTRAP) au lancement dans cet environnement**,
    avec ou sans `--no-sandbox`/`--disable-gpu`/`--disable-dev-shm-usage`,
    y compris invoqué directement hors de Playwright — confirme que
    l'environnement de session ne permet pas de lancer un navigateur
    (cohérent avec toutes les unités précédentes : « pas d'outil navigateur
    disponible »). La route de test temporaire, le paquet `playwright-core`
    et le script de capture ont été entièrement retirés après l'essai
    (`git diff` sur `App.tsx`/`package.json` confirmé identique à avant
    l'essai).
  - Non vérifié : rendu visuel réel du HUD (dark theme, alignement,
    lisibilité) — nécessite soit un navigateur dans l'environnement de
    session, soit une vérification manuelle par l'utilisateur en lançant
    `npm run dev` localement.

- **Unit 11 — Fast-Track réel : découverte + CRUD Project manquant + bug
  critique du worker corrigé** (2026-07-06) :
  - **Découverte importante** : en lisant le code avant d'implémenter
    (comme l'exige `CLAUDE.md`), j'ai trouvé que `POST
    /api/projects/{id}/refresh` (`api/main.py`) et toute la logique de
    persistance Fast-Track (`_run_fasttrack_sync` dans
    `email_analyzer/analysis_tasks.py` : delta via `EmailProcessor.
    process_delta`, écriture `Email`/`ProjectSummary`/`SuggestedAction`,
    dédup par `external_id`, mapping sentiment depuis le niveau de risque)
    **existaient déjà dans l'arbre de travail, non committées, et non
    documentées ici** — `progress-tracker.md` affirmait à tort après Unit 10
    qu'« aucun code applicatif n'écrit encore ces tables ». Origine
    antérieure à cette session (confirmé via `git diff` contre le dernier
    commit `07e5926`). Je n'ai pas réimplémenté cette logique : je l'ai lue,
    vérifiée, et corrigée là où elle était cassée (voir bug ci-dessous).
    Cette entrée corrige rétroactivement le tracker.
  - **Gap réel identifié** : aucun moyen d'appeler cet endpoint n'existait
    car aucun `Project` ne pouvait jamais être créé (pas de `POST
    /api/projects`, pas de liste). `api/routers/projects.py` (nouveau,
    écrit cette session) : `POST /api/projects` (create), `GET
    /api/projects` (liste, scindé par tenant, inclut `summary_content`/
    `sentiment`/`last_processed_email_timestamp` depuis `ProjectSummary`
    si présent), `GET /api/projects/{id}` (détail + `suggested_actions`).
    Suit exactement le pattern des routers existants (`_require_saas()`
    dupliqué localement comme dans `tenants.py`/`auth.py`/`oauth.py`/
    `billing.py` — convention du projet, pas une nouvelle abstraction).
    Enregistré dans `api/main.py` (2 lignes : import + `include_router`).
  - **Bug critique trouvé et corrigé** : le worker `arq`
    (`analysis_tasks.py`) n'appelait jamais `init_db()` — seul le
    `lifespan` FastAPI (process `uvicorn`) le fait. Résultat : dans le
    process worker (séparé depuis Unit 9), `SessionLocal` restait `None`
    en permanence, et **toute tâche touchant la DB échouait avec
    « DATABASE_URL non configuré » même quand la variable d'environnement
    était correctement définie** — reproduit avec `/api/analyze` en mode
    SaaS (pas seulement avec mon nouveau Fast-Track). Ce bug affecte donc
    aussi une fonctionnalité déjà livrée (Unit 9). Corrigé en ajoutant
    `init_db()` dans `on_startup` (`analysis_tasks.py`), au même endroit où
    l'analyseur ML partagé est déjà pré-chargé par process worker.
  - Vérifié en conditions réelles, environnement jetable et démonté après
    coup (pas de `.env`/identifiants dans cette session) :
    - Cluster PostgreSQL local jetable (`initdb`/`pg_ctl`, port dédié,
      socket dans `/tmp`) + Redis local (déjà disponible) + `alembic
      upgrade head` (001→003) appliqué proprement.
    - `uvicorn` + un worker `arq` réel lancés comme deux process séparés
      (topologie exacte d'Unit 9) contre ce cluster.
    - Tenant/utilisateur/jeton créés directement en base (le flux HTTP
      `POST /api/auth/register` est bloqué par un bug **préexistant, sans
      rapport** : `passlib`+`bcrypt` incompatibles sous Python 3.13 dans cet
      environnement — `ValueError: password cannot be longer than 72
      bytes` sur un mot de passe pourtant valide ; non corrigé, hors
      périmètre de cette unité, à traiter séparément si l'utilisateur le
      confirme).
    - Flux HTTP réel : `POST /api/projects` → `GET /api/projects` → `GET
      /api/projects/{id}` (CRUD complet, scoping tenant vérifié) ; `POST
      /api/projects/{id}/refresh` → `202 {job_id}` → `GET
      /api/analyze/{job_id}` jusqu'à `error` **avant** le correctif (« DATABASE_URL
      non configuré ») puis jusqu'à l'erreur métier attendue **après** le
      correctif (« Identifiants IMAP manquants » — normal, tenant de test
      sans IMAP configuré) ; `/api/analyze` (SaaS) confirmé cassé par le
      même bug puis réparé par le même correctif.
    - Logique de persistance (`_run_fasttrack_sync`) testée en appel direct
      (contourne `arq`/HTTP) avec `EmailProcessor.process_delta` mocké
      (aucun IMAP/LLM réel disponible) : 2 emails factices insérés avec le
      bon `recipient_status` (`direct_to`/`cc` selon le champ `To:`) et
      dédupliqués par `external_id` ; `ProjectSummary.sentiment` correctement
      dérivé de `niveau_risque=CRITIQUE` → `under_tension` ; `SuggestedAction`
      créée uniquement pour risque non-FAIBLE ; second appel sans nouvel
      email vérifié idempotent (pas de doublon `Email`, pas de nouvelle
      `SuggestedAction`, résumé/sentiment inchangés).
    - `python -m py_compile` OK sur tous les fichiers touchés. Tout
      l'environnement jetable (Postgres, process `uvicorn`/`arq`, sockets,
      `pgdata`) démonté et supprimé après vérification ; `redis-cli flushdb`
      sur la DB de test (index 1, jamais celle de prod par défaut).
  - Non vérifié : le vrai chemin `process_delta` avec un IMAP réel (dépend
    d'identifiants que cette session n'a pas) ; le flux HTTP `register`
    (bug bcrypt préexistant bloquant) ; comportement sous charge concurrente.
  - **⚠️ Signalé à l'utilisateur** : `git status` montre un très grand nombre
    de fichiers modifiés non committés dans tout le dépôt (au-delà de cette
    session — état déjà présent au tout début de la conversation). Une
    partie est probablement du bruit de fin de ligne, mais le Fast-Track
    découvert ci-dessus prouve qu'il y a aussi du vrai code fonctionnel non
    committé. Recommandation : committer bientôt pour ne pas risquer de
    perdre ce travail.

- **Unit 12 — Frontend Project Hub** (2026-07-06) :
  - Contexte : Unit 11 a rendu le Fast-Track fonctionnel côté backend, mais
    aucune UI n'existait pour créer un projet, en voir la liste, ou
    déclencher un rafraîchissement — écart comblé ici, conformément à la
    spec `architecture.md` (`### Project Hub`, `### Fast-Track Refresh`) :
    grille responsive de cartes, chacune avec résumé, indicateur de santé,
    sentiment IA, actions en attente, horodatage de dernière mise à jour.
  - **Ajout backend minimal, directement motivé par la spec** :
    `api/routers/projects.py` — `pending_actions_count` ajouté à
    `ProjectOut`/`ProjectDetailOut` (compte les `SuggestedAction` au statut
    `pending` uniquement) : la spec exige "Outstanding actions" sur la
    carte, et l'endpoint liste ne remontait jusqu'ici aucune info sur les
    actions (seul le détail les exposait). Requête de comptage par projet
    (pas de sous-requête agrégée) — accepté vu le volume de projets par
    tenant attendu à ce stade, à revisiter seulement si ça devient un vrai
    goulot.
  - `frontend/src/ProjectHub.tsx` (nouveau) : `ProjectHub` (liste + création)
    et `ProjectCard` (une carte, refresh Fast-Track local). Mapping
    sentiment → couleur **sans nouveau token** : les 3 valeurs réelles de
    `ProjectSentiment` (`on_track`/`under_tension`/`awaiting_feedback`)
    correspondent déjà exactement à 3 des 5 couleurs de la "Project Health
    Scale" de `ui-context.md` (`success`/`danger`/`warning`, mêmes valeurs
    hex) — les 2 niveaux intermédiaires (Needs Attention/Delayed) n'ont pas
    de token dédié ajouté puisque le backend ne les distingue pas.
  - Fast-Track par carte : `POST /api/projects/{id}/refresh` puis polling de
    `GET /api/analyze/{job_id}` (réutilise tel quel l'endpoint générique de
    `jobs.py` — même pattern que `pollAnalysis` dans `HomePage.tsx`, dupliqué
    ici en `pollRefreshJob` plutôt que partagé : portées différentes —
    accepté, pas de nouvelle abstraction cross-fichier pour deux call sites).
    État `refreshing` **local à `ProjectCard`** : seule la carte concernée
    pulse (`.animate-fasttrack-pulse`, nouveau), jamais d'overlay global —
    respecte explicitement `architecture.md` *"Only the affected project
    card is updated"*. Au succès, le résultat du job (`sentiment`/`content`/
    `last_processed_email_timestamp`) patch directement la ligne du projet
    dans l'état local (pas de refetch complet de la liste).
  - `frontend/tailwind.config.js` : 2 tokens ajoutés (`fasttrack-active`
    `#60A5FA`, `fasttrack-bg` `rgba(59,130,246,.10)`), valeurs de
    `ui-context.md` "Fast-Track Tokens" — `fasttrack-idle` non ajouté, la
    valeur `#3B82F6` existe déjà sous le nom `info` et est réutilisée telle
    quelle (pas de doublon de token pour la même couleur).
  - `frontend/src/index.css` : `@keyframes fasttrack-pulse` +
    `.animate-fasttrack-pulse` (pulsation de `box-shadow`, même convention
    que `batch-pop`/`skeleton-pulse`/`typing-dot` déjà en place).
  - `frontend/src/pages/ProjectHubPage.tsx` (nouveau) : garde-fous
    identiques à `SettingsPage.tsx` (`saasEnabled` → sinon `Navigate` vers
    `/`, token absent → `/login`, `me` en cours de chargement → spinner) —
    le Project Hub est strictement SaaS (les projets sont scopés par
    tenant), pas de repli legacy.
  - `App.tsx` (route `/projects`) et `components/AppShell.tsx` (lien nav
    "Projets") mis à jour.
  - Vérifié : `tsc --noEmit` sans erreur ; `vite build` transforme les 48
    modules sans erreur (même échec `EACCES` sur `dist/` pré-existant,
    sans rapport, déjà documenté). Contrat backend réel re-vérifié avec un
    cluster Postgres jetable (même méthode qu'Unit 11, démonté après coup) :
    `POST/GET /api/projects` et `GET /api/projects/{id}` renvoient
    exactement la forme attendue par `ProjectListItem`/`ProjectDetailOut`
    côté frontend ; `pending_actions_count` vérifié à `1` avec 1 action
    `pending` + 1 `completed` seedées directement en base (compte bien
    seulement les `pending`, pas les autres statuts).
  - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
    d'environnement que pour le HUD d'Unit 7 — Chromium système crashe au
    lancement dans cette session, cf. entrée Unit 7 étape 3 ; pas retenté
    ici, la cause est déjà identifiée comme un problème d'environnement,
    pas de code) ; le flux de polling `pollRefreshJob` face à un vrai job
    Fast-Track déclenché depuis le navigateur (le contrat HTTP est vérifié,
    mais pas le rendu de la pulsation/l'état `refreshing` en conditions
    réelles) ; recommandé à l'utilisateur de lancer `npm run dev` +
    `uvicorn`/`arq` localement pour un premier passage visuel.

- **Fix — "Analyse complète" échouait prématurément avec un timeout côté
  client alors que le job backend tournait toujours** (2026-07-08) :
  - Cause racine (double, flux `HomePage.tsx` / `run_analysis_saas` /
    `run_analysis_legacy` uniquement — Fast-Track hors scope, confirmé avec
    l'utilisateur) :
    1. Frontend — `pollAnalysis` (`HomePage.tsx`) laissait l'erreur d'un
       **seul** tick (`AbortSignal.timeout(30s)`, hoquet réseau) se propager
       et tuer tout le run, alors que le budget global de polling est de
       5 min et que le job backend continuait souvent de tourner sainement.
    2. Backend — aucun timeout explicite sur les connexions IMAP
       (`imaplib.IMAP4_SSL`/`IMAP4`) ni sur l'appel Gemini
       (`generate_content`), combiné au `job_timeout` implicite de 300 s
       d'arq et au `max_tries` par défaut (5) : un job lent se faisait tuer
       puis **rejouer jusqu'à 5 fois**, refaisant intégralement le fetch
       IMAP + les résumés LLM à chaque tentative (aucun état intermédiaire
       persisté avant le `db.commit()` final).
  - Extraction des emails et corrélation job_id/tenant_id à travers
    create_job → enqueue → poll → Redis : **vérifiées correctes**, aucun bug
    trouvé — le mécanisme de réponse progressive (`on_batch` →
    `report_progress` → `processed`/`total`/`partial` → `onTick` →
    `setProgress`/`setPartialBlock`/`bridgeAnalyzeJob`) était déjà câblé et
    fonctionnel ; il était juste interrompu par le bug (1) ci-dessus.
  - `frontend/src/pages/HomePage.tsx` : `pollAnalysis` — le fetch d'un tick
    est maintenant dans un `try/catch` interne à la boucle ; une erreur
    transitoire fait `continue` au lieu de propager, la boucle ne s'arrête
    que sur `deadline` global dépassé ou `status === "error"` explicite du
    backend.
  - `email_analyzer/config.py` : nouveau `imap_timeout_seconds()` (env
    `IMAP_TIMEOUT_SECONDS`, défaut 30 s), même pattern que
    `llm_timeout_seconds()`.
  - `email_analyzer/project_mail.py` : `connect()` passe `timeout=` sur les
    3 sites de construction `imaplib.IMAP4_SSL`/`IMAP4`.
  - `email_analyzer/llm.py` : `generate_gemini_assistant_summary` passe
    `request_options={"timeout": llm_timeout_seconds()}` à `generate_content`
    (alignement sur le client OpenAI qui avait déjà `timeout=` depuis
    Unit 6 ; l'autre site Gemini du fichier, `chat.send_message` pour
    `/api/chat`, non touché — hors scope de ce flux).
  - `email_analyzer/analysis_tasks.py` : `WorkerSettings.functions` — `
    run_analysis_legacy`/`run_analysis_saas` enveloppés en `arq.func(...,
    max_tries=1)` (travail non idempotent, un retry ne fait que redoubler le
    coût réseau/LLM) ; `run_fasttrack_refresh` inchangé (hors scope).
  - Vérifié : `python -m py_compile` sur les 4 fichiers backend modifiés ;
    import direct de `WorkerSettings.functions` confirmant `max_tries=1` sur
    les 2 bonnes entrées ; `tsc --noEmit` sans erreur côté frontend.
  - Non vérifié : passage réel dans le navigateur avec un vrai worker arq +
    un compte IMAP lent (même contrainte d'environnement que les entrées
    précédentes — pas de Chromium fonctionnel dans cette session).

- **Unit 13 — Stabilisation du dépôt : commit du travail en attente + secrets non trackés**
  (2026-07-09) :
  - Contexte : reprise en **mode autonome intégral** demandée explicitement
    par l'utilisateur (« fait tout le traitement sans demander de validation
    [...] tu es l'administrateur ») — plan validé couvrant Units 13 à 19
    ci-dessous, exécuté sans pause de validation entre unités.
  - **Découverte avant tout traitement** : les 6 fichiers `context/*.md`
    (architecture, project-overview, ui-context, code-standards,
    ai-workflow-rules, evolution-plan) sont non committés et incohérents
    entre eux et avec HEAD (ex. `evolution-plan.md` non committé propose un
    pivot « V2 » pgvector, `architecture.md` non committé décrit encore un
    cache vectoriel device Orama/Voy, `code-standards.md` — committé comme
    non committé — décrit encore Node.js/NestJS/BullMQ). **Décision
    utilisateur** : ignorer cette incohérence pour cette session, ne pas
    toucher ces 6 fichiers, avancer sur le backlog concret en se basant sur
    ce fichier (`progress-tracker.md`, en avance sur HEAD et cohérent avec
    le code réel) plutôt que sur `architecture.md`/`code-standards.md`. À
    reprendre dans une session dédiée.
  - **Découverte critique de sécurité** : plusieurs fichiers déjà **committés**
    dans l'historique git (`email_analysis_cache.json` racine et
    `services/email_analyzer/`, `rapport.json`, `rapport_ia.json`, `resume`,
    `resumer`, `10_2025.json`,
    `.cache_tenants/tenant_c4c275d0-....json`) contiennent des **corps
    d'emails clients réels** — dont un email contenant en clair des
    identifiants d'administration (pgAdmin/RabbitMQ/APISIX) d'un client
    (BBCI/Ezikash). Un fichier équivalent non tracké
    (`tenant_566b33cb-....json`) contenait la même catégorie de données.
    **Remédiation appliquée** (réversible, ne réécrit pas l'historique
    git) : `.gitignore` étendu (ces motifs + `__pycache__/`/`*.pyc`) et
    `git rm --cached` sur les fichiers trackés (restent sur disque,
    seulement retirés du suivi futur). **Non fait, décision explicitement
    laissée à l'utilisateur** : purge de l'historique git existant (ces
    données restent lisibles dans les commits passés) et rotation des
    identifiants exposés (pgAdmin/RabbitMQ/APISIX du client BBCI/Ezikash)
    — actions à fort rayon d'effet, hors du périmètre d'une décision prise
    de façon autonome.
  - Commit du travail réel précédemment non committé (routeur Fast-Track
    `api/routers/projects.py`, Project Hub, scaffolding `sync-experience`,
    fix timeout du 2026-07-08) — voir message de commit `0fd191f` pour le
    détail. Exclu du commit : les 6 fichiers `context/*.md` (décision
    ci-dessus) et ~100 fichiers marqués modifiés uniquement par un
    changement de mode Unix (`100644→100755`, aucun contenu réel changé).
  - Vérifié : `git status` propre sur tous les fichiers concernés après
    coup ; `git diff --cached` relu intégralement avant commit (aucun
    secret, aucun fichier `context/*.md`) ; fichiers ignorés confirmés via
    `git check-ignore -v`.

- **Unit 14 — Correctif définitif du bug bcrypt/passlib (Python 3.13)** (2026-07-09) :
  - Contexte : `POST /api/auth/register`/`/login` cassés depuis Unit 11
    (`ValueError: password cannot be longer than 72 bytes` sur des mots de
    passe valides), non corrigé faute de confirmation sur la cause racine.
  - Cause racine confirmée cette session : `passlib==1.7.4` (dernière
    version, non maintenue depuis 2020) est incompatible avec **Python
    3.13**, qui a supprimé le module stdlib `crypt` (PEP 594) sur lequel la
    détection de backend de `passlib` s'appuie partiellement — le pin
    `bcrypt>=4,<4.1` déjà en place (contournement d'un problème différent,
    `bcrypt.__about__`) n'y change rien, confirmé en reproduisant le bug
    dans cet environnement (`bcrypt==4.0.1` + `passlib==1.7.4` +
    Python 3.13.5).
  - **Décision (moindre risque)** : abandon complet de `passlib` au profit
    d'appels `bcrypt` directs. Format de hash bcrypt (`$2b$...`) strictement
    identique entre les deux — **aucune migration de données, aucun hash
    existant invalidé**.
  - `email_analyzer/auth_jwt.py` : `hash_password`/`verify_password`
    réécrits avec `bcrypt.hashpw`/`bcrypt.checkpw` directement ; troncature
    explicite à 72 octets UTF-8 (`_truncate_for_bcrypt`, documente un
    comportement que bcrypt appliquait déjà silencieusement).
  - `requirements.txt` (racine) : `passlib[bcrypt]` retiré ; pin `bcrypt`
    élargi à `>=4,<5` (la contrainte `<4.1` ne protégeait plus qu'un usage
    passlib désormais supprimé).
  - Vérifié en conditions réelles, environnement jetable démonté après coup
    (cluster Postgres local dédié, `initdb`/`pg_ctl`, port et socket
    dédiés ; Redis local existant, DB 3 ; `alembic upgrade head` 001→003
    propre) : `uvicorn` réel démarré, flux HTTP complet
    `POST /api/auth/register` (mot de passe réaliste avec accents) → `200`
    avec JWT valide ; `POST /api/auth/login` même identifiants → `200` ;
    mauvais mot de passe → `401`. Test unitaire direct du round-trip
    hash/verify, y compris la frontière de troncature à 72 octets (un mot
    de passe de 100 `x` et sa variante tronquée à 71+`y` sont bien
    distingués). `python -m py_compile` OK. Processus/cluster de test
    arrêtés et supprimés après vérification.

- **Unit 15 — Fermeture de l'Open Question #1 : Gmail OAuth branché sur `/api/analyze`**
  (2026-07-09) :
  - Contexte : Unit 1 (2026-05-04) avait livré le flux OAuth Gmail
    (`gmail_oauth.py`, `api/routers/oauth.py`) mais `EmailProcessor`/
    `processor_from_tenant` restaient strictement IMAP — un tenant Gmail-only
    n'avait aucun moyen de déclencher une analyse.
  - Découverte exploitée : le format d'email normalisé est déjà identique
    entre IMAP (`project_mail.py`, `extract_email_content`) et Gmail
    (`gmail_oauth._normalize_message`) — mêmes clés `subject/from/to/date/
    body/normalized_text`.
  - `email_analyzer/project_mail.py` : nouvelle méthode
    `EmailProjectAnalyzer.search_project_emails_from_list(emails,
    project_filters)` — même matching que `search_project_emails`
    (réutilise `check_project_relevance`/`extract_participants`, aucune
    logique dupliquée), mais sur une liste déjà récupérée au lieu d'itérer
    une connexion IMAP.
  - `email_analyzer/analyzer.py` : `EmailProcessor.__init__` gagne
    `gmail_connection` (objet `OAuthConnection`, duck-typed) et
    `use_env_fallback` (bool, défaut `True`). `process_latest_emails`
    bascule sur `_fetch_gmail_project_data` (nouvelle méthode privée,
    `gmail_oauth.fetch_emails` + `search_project_emails_from_list`, plafond
    200 messages/une page — pas de pagination Gmail, limitation connue) quand
    aucun IMAP n'est configuré mais qu'une connexion Gmail l'est ; IMAP garde
    la priorité si les deux sont présents. Reste inchangé pour `process_delta`
    (Fast-Track) et `fetch_last_n_emails_for_chat` (`/api/chat`) —
    explicitement hors périmètre de cette unité, même pattern à répliquer
    plus tard si besoin confirmé.
  - **Bug latent trouvé et corrigé en cours de route** : `EmailProcessor.
    __init__` retombait sur `os.environ.get("IMAP_USER"/"IMAP_PASSWORD")`
    même pour un processeur construit par `processor_from_tenant` (mode
    SaaS, `load_env=False`) dès lors que ces variables étaient définies
    globalement dans le process (cas réel de cet environnement de dev via
    le `.env` racine, chargé par `api/main.py`). Un tenant SaaS sans IMAP
    configuré aurait donc silencieusement analysé la boîte IMAP globale du
    serveur au lieu de basculer sur Gmail ou de renvoyer une erreur claire —
    risque de fuite de données inter-tenant. Corrigé par le nouveau flag
    `use_env_fallback=False`, mis systématiquement par `processor_from_tenant`
    (les deux branches IMAP et Gmail) ; le mode legacy (`_run_legacy_sync`,
    pas de notion de tenant) garde `use_env_fallback=True` par défaut,
    comportement inchangé.
  - `email_analyzer/saas_logic.py` : `processor_from_tenant` restructuré —
    branche IMAP inchangée si `imap_password_encrypted` présent, sinon
    cherche une `OAuthConnection(provider="gmail")` via la relation déjà
    existante `tenant.oauth_connections` (aucune requête DB supplémentaire
    à écrire).
  - `email_analyzer/analysis_tasks.py` : `_run_saas_sync` persiste
    immédiatement (commit isolé, séparé du commit de fin de job) un
    éventuel `proc.last_gmail_token_refresh` via la nouvelle fonction
    `_persist_gmail_token_refresh` — un rafraîchissement de token réussi
    côté Google ne doit pas être perdu si l'analyse échoue plus loin.
  - Vérifié : `python -m py_compile` sur les 4 fichiers. Trois scénarios
    testés en direct Python avec `gmail_oauth.fetch_emails` mocké (pas de
    compte Gmail réel disponible dans cet environnement) : (1) tenant
    Gmail-only → email correspondant bien matché, `nb_emails == 1`,
    `token_refresh` capturé ; (2) tenant avec IMAP **et** Gmail configurés
    → IMAP prioritaire, `gmail_oauth.fetch_emails` jamais appelé ; (3)
    tenant sans aucune source → message d'erreur original inchangé. Puis
    vérification bout-en-bout en conditions réelles (cluster Postgres
    jetable dédié, `alembic upgrade head` 001→003, tenant + `OAuthConnection`
    Gmail réels insérés en base, `_run_saas_sync` appelé directement avec
    `fetch_emails` mocké) : résultat d'analyse correct **et** le nouveau
    token d'accès rafraîchi relu depuis Postgres après coup (déchiffré avec
    `ENCRYPTION_KEY`, confirmé identique à la valeur mockée). Environnement
    jetable démonté après vérification.
  - Non vérifié : échange réel avec un compte Gmail (aucun compte OAuth
    disponible dans cet environnement, comme pour Unit 1) ; pagination
    au-delà de 200 messages (non implémentée, limitation documentée
    ci-dessus).

- **Unit 16 — Frontend : UI de connexion des comptes (Gmail + Outlook)** (2026-07-09) :
  - Contexte : recherche sur `frontend/src/` avant implémentation (`grep -rl
    oauth`) — **aucune UI n'appelait `/api/oauth/*`**, ni pour Gmail (livré
    backend-only en Unit 1, 2026-05-04) ni a fortiori pour Outlook (pas
    encore implémenté à ce stade, voir Unit 17 ci-dessous). Les deux
    intégrations OAuth étaient inutilisables depuis l'app. Un seul
    composant provider-agnostic couvre les deux plutôt que de dupliquer la
    même UI à quelques unités d'écart.
  - `frontend/src/ConnectedAccounts.tsx` (nouveau) : section « Comptes
    connectés » — liste via `GET /api/oauth/connections` (déjà
    provider-agnostic côté API), boutons « Connecter Gmail »/« Connecter
    Outlook » (`GET /api/oauth/{provider}/authorize` puis redirection
    complète de la page vers l'URL retournée — nécessaire pour l'écran de
    consentement Google/Microsoft), lecture unique du retour de callback
    (`?oauth=success|error&provider=...`, déjà émis par
    `api/routers/oauth.py`) avec nettoyage de l'URL via
    `history.replaceState` pour ne pas le réafficher à un rafraîchissement,
    déconnexion via `DELETE /api/oauth/connections/{id}`. Réutilise
    `apiFetch`/le pattern `parseJson`/gestion d'erreur déjà en place dans
    `SaasPanels.tsx` (aucune nouvelle abstraction réseau).
  - `frontend/src/pages/SettingsPage.tsx` : `<ConnectedAccounts />` monté
    entre la section IMAP et la section Facturation.
  - Vérifié : `tsc --noEmit` sans erreur ; `vite build` transforme les 49
    modules sans erreur (échec ensuite limité au même `EACCES` sur `dist/`
    pré-existant appartenant à `root`, sans rapport, déjà documenté dans les
    unités précédentes).
  - Non vérifié : rendu visuel réel dans un navigateur (Chromium système
    crashe au lancement dans cette session, cf. Units 7/12) ; le bouton
    « Connecter Outlook » pointe vers un endpoint qui n'existe pas encore à
    ce stade de la session (`/api/oauth/outlook/authorize`, livré par
    Unit 17 juste après) — cohérent une fois les deux unités appliquées
    ensemble, comme prévu par le plan.

- **Unit 17 — Outlook OAuth2 (Microsoft Graph), miroir de Gmail** (2026-07-09) :
  - Contexte : Next Up historique « Unit 2 » — `OAuthConnection.provider`
    était déjà générique (pas de contrainte `gmail`-only en base), donc un
    nouveau provider plutôt qu'un nouveau concept.
  - **Décision (moindre risque)** : flux OAuth2 + Microsoft Graph en appels
    HTTP directs via `httpx` (déjà une dépendance du projet), sans ajouter
    `msal` — les endpoints Microsoft identity platform v2.0 et Graph sont de
    l'OAuth2/OIDC et du REST standard.
  - `email_analyzer/outlook_oauth.py` (nouveau) : même forme que
    `gmail_oauth.py` — `build_authorization_url`, `exchange_code_for_tokens`,
    `refresh_access_token` (gère la rotation du refresh_token, propre à
    Microsoft — Google ne la fait pas), `get_connected_email`,
    `build_outlook_filter` (équivalent de `build_gmail_query`, un `$filter`
    OData bornant par date), `fetch_emails`, `_normalize_message` (mêmes
    clés de sortie que Gmail : `id/subject/from/to/date/body/
    normalized_text` — réutilise `gmail_oauth._strip_html` pour le nettoyage
    HTML plutôt que de dupliquer le regex).
  - `api/routers/oauth.py` : `outlook_authorize`/`outlook_callback` ajoutés
    au même router (`/api/oauth/outlook/...`), réutilisant
    `_create_state_token`/`_decode_state_token` (déjà génériques).
    **Refactor de réutilisation** : la logique d'upsert `OAuthConnection`
    (dupliquée si copiée telle quelle) extraite en `_upsert_oauth_connection`
    partagée par `gmail_callback` et `outlook_callback`. `delete_connection`
    ne tentait déjà la révocation HTTP que si `provider == "gmail"` (code
    existant, aucun changement nécessaire — tolérant par construction pour
    un provider sans révocation implémentée).
  - `email_analyzer/analyzer.py` : `EmailProcessor` gagne
    `outlook_connection` (miroir de `gmail_connection`) et
    `last_outlook_token_refresh` ; `process_latest_emails` ajoute la branche
    Outlook (`_fetch_outlook_project_data`, même limitation connue que Gmail
    — 200 messages, une page, pas de pagination) ; priorité IMAP > Gmail >
    Outlook si plusieurs sources sont configurées (arbitraire côté
    Gmail/Outlook faute de signal pour départager, documenté en commentaire).
  - `email_analyzer/saas_logic.py` : `processor_from_tenant` cherche aussi
    une `OAuthConnection(provider="outlook")` en repli.
  - `email_analyzer/analysis_tasks.py` : `_persist_gmail_token_refresh`
    renommée `_persist_oauth_token_refresh` (généralisée aux deux
    providers) et gère en plus la rotation de refresh_token Outlook.
  - `.env.example` (racine) : section Outlook OAuth2 (`OUTLOOK_CLIENT_ID`,
    `OUTLOOK_CLIENT_SECRET`) avec instructions Azure AD.
  - `requirements.txt` : aucun ajout (httpx déjà présent).
  - Vérifié : `python -m py_compile` sur les 5 fichiers modifiés/créés.
    `build_authorization_url` produit une URL statiquement valide (`client_id`,
    `redirect_uri`, `state`, scopes Graph présents, encodage correct).
    `_normalize_message` testé directement : HTML correctement dépouillé
    (même comportement que `gmail_oauth._strip_html`), plusieurs
    destinataires bien joints. Deux scénarios `process_latest_emails` testés
    avec `outlook_oauth.fetch_emails` mocké (pas de compte Microsoft réel
    disponible) : (1) tenant Outlook-only → email correspondant bien matché,
    `token_refresh` capturé ; (2) tenant avec Gmail **et** Outlook connectés
    → Gmail prioritaire, `outlook_oauth.fetch_emails` jamais appelé. Puis
    vérification HTTP réelle bout-en-bout (cluster Postgres jetable dédié,
    `alembic upgrade head`, `uvicorn` réel, `OUTLOOK_CLIENT_ID`/`SECRET` de
    test) : `POST /api/auth/register` → `GET /api/oauth/outlook/authorize`
    (200, URL bien formée avec le bon state signé) → `GET
    /api/oauth/connections` (200, `[]`) → `GET /api/oauth/outlook/callback`
    avec `error=access_denied` et sans `code`/`state` → redirection `307`
    vers `/settings?oauth=error&provider=outlook` dans les deux cas.
    Environnement jetable démonté après vérification. `tsc --noEmit` toujours
    sans erreur côté frontend (aucun changement frontend dans cette unité,
    `ConnectedAccounts.tsx` d'Unit 16 pointe désormais vers un endpoint qui
    existe réellement).
  - Non vérifié : échange réel avec un compte Microsoft/Azure AD (aucun
    compte disponible dans cet environnement, comme pour Gmail) ; la
    rotation effective du refresh_token Outlook en conditions réelles.

- **Unit 18 — Sync planifiée 2x/jour via `arq` cron** (2026-07-09) :
  - Contexte : Next Up historique « Unit 4 » (« APScheduler / Celery batch
    processing ») — débloquée par Unit 11 (Fast-Track écrit réellement dans
    les tables Unit 10). Renommée/reformulée : la stack réelle est `arq`
    (Architecture Decisions), pas APScheduler ni Celery — planification via
    le support cron natif d'`arq` (`arq.cron.cron`), aucune nouvelle
    dépendance.
  - `email_analyzer/analysis_tasks.py` : nouvelle tâche
    `run_scheduled_sync(ctx)` — itère tous les `Project` au statut `active`
    (les `archived` sont exclus, inutile de les resynchroniser), appelle
    `_run_fasttrack_sync` pour chacun (réutilisation directe, même logique
    que le rafraîchissement manuel par carte — aucune duplication). Un échec
    par projet est loggé (`logger.exception`) sans jamais interrompre le
    traitement des projets suivants. Enregistrée dans
    `WorkerSettings.cron_jobs = [cron(run_scheduled_sync, hour={7, 19},
    minute=0, max_tries=1)]` — 7h/19h cohérent avec le « briefing matin/soir »
    de `project-overview.md`. `max_tries=1` : `run_scheduled_sync` gère déjà
    ses échecs par projet en interne, un retry global ne ferait que
    retraiter des projets déjà réussis.
  - **Limitation connue héritée d'Unit 15/17** : `_run_fasttrack_sync`
    appelle `EmailProcessor.process_delta`, resté strictement IMAP (le
    branchement Gmail/Outlook d'Unit 15/17 ne couvre que
    `process_latest_emails`, explicitement hors périmètre à l'époque). Les
    projets de tenants Gmail/Outlook-only échouent donc proprement (loggés,
    sans casser le batch) plutôt que d'être réellement synchronisés — pas
    un nouveau bug, la limitation était déjà documentée, cette unité la
    rend juste visible à l'échelle du cron plutôt que d'un seul flux manuel.
    À lever si confirmé comme un besoin réel (répliquer le même pattern que
    Unit 15 dans `process_delta`).
  - `services/email_analyzer/deploy/nginx-api.conf` : note ajoutée — le
    process worker `arq` déjà obligatoire (Unit 9) exécute aussi
    automatiquement cette sync planifiée, aucune commande de déploiement
    séparée (déduplication cron entre process worker gérée nativement par
    `arq` via Redis, `unique=True` par défaut).
  - Vérifié : `python -m py_compile` ; `WorkerSettings.cron_jobs` inspecté
    directement (hour `{7, 19}`, `minute=0`, `max_tries=1` bien appliqués).
    Vérification bout-en-bout en conditions réelles (cluster Postgres
    jetable dédié, `alembic upgrade head`) avec 3 tenants/projets seedés :
    (1) tenant avec IMAP configuré (`process_delta` mocké, pas d'IMAP réel
    disponible) → traité avec succès, `ProjectSummary.
    last_processed_email_timestamp` avancé ; (2) tenant sans aucune source
    → échec propre (`RuntimeError` "Identifiants IMAP manquants" loggé),
    aucune `ProjectSummary` créée, **le batch continue** ; (3) projet
    `archived` → jamais tenté (absent du log d'appels). Confirme les 3
    critères du plan : projets valides rafraîchis, échec d'un projet
    n'empêche pas les suivants, timestamp avancé correctement.
    Environnement jetable démonté après vérification.
  - Non vérifié : déclenchement réel à l'horaire cron (7h/19h) — vérifié
    uniquement par appel direct de `run_scheduled_sync()`, pas en laissant
    tourner un worker arq jusqu'à l'horaire ; comportement sous plusieurs
    process worker arq simultanés (dédup `unique=True`, documentée mais pas
    testée avec 2 process réels).

- **Unit 19 — Moteur de scoring d'importance par email (rules + AI hybride)** (2026-07-09) :
  - Contexte : Next Up historique « Unit 5 ». Dernière unité du plan
    d'implémentation autonome validé cette session (Units 13-19).
  - Découverte de réutilisation : `ai_intelligent.py` avait déjà un
    dictionnaire de mots-clés pondérés (`risk_keywords`) et une méthode
    `identify_critical_emails` flaggant des emails individuels par
    mots-clés urgents — mais rien n'était persisté au niveau d'un email
    individuel (`Email`, Unit 10, n'avait pas de colonne de score).
  - **Décision (moindre risque)** : pas de nouvel appel LLM par email (coût/
    latence non maîtrisés à l'échelle d'une sync planifiée batch, cf. Unit
    18) — score hybride calculé au moment de la persistance à partir de
    signaux déjà disponibles sans coût réseau supplémentaire :
    - **Règles** : mêmes `risk_keywords` que `identify_critical_emails`,
      somme pondérée plafonnée puis mise à l'échelle (60 pts max).
    - **Adressage** : +15 si `recipient_status == "direct_to"`, 0 si `"cc"`
      (signal déjà calculé par `_run_fasttrack_sync`, aucun calcul
      supplémentaire).
    - **IA** : plancher (`_IMPORTANCE_RISK_LEVEL_FLOOR` : CRITIQUE=40,
      MODÉRÉ=20, FAIBLE=0) dérivé du `niveau_risque` déjà produit par
      l'appel LLM du même cycle de résumé (extrait de `delta["summary"]`
      avant la boucle de persistance des emails, un seul niveau de risque
      pour tout le lot).
    - Résultat borné `[0, 100]`.
  - `alembic/versions/004_email_importance_score.py` (nouveau) : ajoute
    `emails.importance_score` (`Integer`, nullable — les lignes existantes
    ne sont pas rétro-calculées), `downgrade()` symétrique. Suit le pattern
    de `003_projects.py`.
  - `email_analyzer/db/models.py` : colonne `Email.importance_score`.
  - `email_analyzer/ai_intelligent.py` : nouvelle méthode
    `EmailIntelligentAnalyzer.score_email_importance(email_data,
    recipient_status, niveau_risque)` + constante module
    `_IMPORTANCE_RISK_LEVEL_FLOOR`.
  - `email_analyzer/analysis_tasks.py` : `_run_fasttrack_sync` extrait
    `risk_level_for_scoring` de `delta["summary"]` avant la boucle, appelle
    `get_shared_analyzer().score_email_importance(...)` pour chaque email
    persisté et passe le résultat à `Email(..., importance_score=...)`.
    Couvre à la fois le Fast-Track manuel (carte projet) et la sync
    planifiée (Unit 18, qui appelle la même fonction) — un seul point
    d'écriture, pas de logique dupliquée.
  - `api/routers/projects.py` : nouveau schéma `EmailOut`
    (`id/subject/received_at/recipient_status/importance_score`) ;
    `ProjectDetailOut.top_emails` — les 10 emails du projet triés par
    `importance_score` décroissant (`NULLS LAST` pour les emails persistés
    avant cette unité) puis `received_at` décroissant. Pas de nouvel
    endpoint de liste d'emails paginé (hors périmètre, `top_emails` suffit
    à « permettre un tri côté frontend » comme demandé) ; aucun changement
    frontend dans cette unité (le tri/affichage réel de `top_emails` reste
    à construire si un besoin UI se confirme).
  - Vérifié : `python -m py_compile` sur les 4 fichiers Python
    modifiés/créés. `score_email_importance` testé directement sur des cas
    connus : email urgent + `direct_to` + `CRITIQUE` → score 88 (`≥ 80`
    attendu) ; email neutre + `cc` + `FAIBLE` → score 0 ; cas extrême
    (tous les mots-clés de risque présents) → plafonné à 100, jamais
    au-delà. Vérification bout-en-bout en conditions réelles (cluster
    Postgres jetable dédié) : `alembic upgrade head` (001→004) propre,
    `downgrade -1` puis `upgrade head` round-trip confirmant la colonne
    disparaît puis réapparaît ; `_run_fasttrack_sync` appelé avec
    `process_delta` mocké (2 emails factices, un urgent+direct_to, un
    neutre+cc) → les deux persistés avec le bon score, le score le plus
    élevé correspondant bien à l'email urgent/direct_to attendu. Puis test
    HTTP réel (`uvicorn`) : `POST /api/auth/register` → `POST
    /api/projects` → 3 `Email` seedées directement en base (scores 90, 10,
    `NULL`) → `GET /api/projects/{id}` renvoie `top_emails` correctement
    trié (90, puis 10, puis `NULL` en dernier). Environnement jetable
    démonté après vérification. `tsc --noEmit` toujours sans erreur côté
    frontend (aucun changement frontend dans cette unité).
  - Non vérifié : pertinence du score sur un vrai corpus d'emails clients
    (les poids/seuils sont un premier jet raisonnable, pas calibrés sur des
    données réelles) ; rendu/tri réel dans l'UI (pas encore construit).

- **Unit 20 — Classification multi-critères + taxonomie de tags par email** (2026-07-09) :
  - Contexte : à partir d'une vision produit "Coach IA" très large (classification,
    tags, décisions, actions, calendrier, base documentaire, recherche hybride,
    chatbot source-cité, dashboard) transmise par l'utilisateur, plan discuté et
    scindé — l'utilisateur a choisi de démarrer par « classification + tagging
    plus intelligents » et de rester 100% cloud pour l'IA (pas de tier LLM local,
    l'invariant `architecture.md` reste inchangé).
  - **Découverte confirmant exactement le gap de la vision** : la classification
    était `EmailProjectAnalyzer.check_project_relevance` (`project_mail.py`) —
    une simple sous-chaîne du **nom du projet** dans le texte de l'email.
    `Project.rules_matrix` (JSONB, `db/models.py`) existait déjà avec un
    commentaire décrivant mots-clés/adresses/domaines mais n'était **lu nulle
    part** — colonne morte. Aucune taxonomie de tags n'existait sur
    `Email`/`Project`/`SuggestedAction`.
  - `email_analyzer/classification.py` (nouveau, module pur sans I/O) :
    - `ProjectRules.from_dict` formalise enfin le contenu de `rules_matrix`
      (keywords/sender_domains/sender_emails/client_names/company_names/
      reference_numbers) ; dégrade silencieusement tout JSON malformé vers des
      listes vides (jamais d'exception).
    - `score_project_relevance` : score de confiance 0-100 combinant nom de
      projet (45 pts — à lui seul égal à `MATCH_THRESHOLD`, donc **bit-à-bit
      identique à l'ancien comportement en sous-chaîne** quand `rules_matrix`
      et participants connus sont absents), mots-clés, adresse/domaine
      expéditeur, participants déjà connus du projet **dans le même balayage**
      (pas de colonne `sender` persistée sur `Email` → pas de roster durable
      inter-exécutions, limitation documentée), noms client/société, références
      internes (regex `[A-Z]{2,}-\d{2,}`).
    - `derive_tags` : catégorise le dictionnaire plat `risk_keywords`
      (`ai_intelligent.py`) en tags métier (Urgent, Bloquant, Bug, Finance,
      Facturation, Livraison, Technique, Client, Sécurité, Juridique, RH,
      Commercial, Validation, Production, Support) + un tag de priorité
      (Critique/Haute/Moyenne/Faible) dérivé de `importance_score` déjà calculé
      (mêmes bornes 80/60/35 que Unit 19). Aucun nouvel appel LLM.
  - `email_analyzer/project_mail.py` : `check_project_relevance`,
    `_process_one_email_id`, `search_project_emails`,
    `search_project_emails_from_list` gagnent un paramètre optionnel
    `rules_map` (+ `participants_map` construit via `.items()`, jamais un accès
    par crochet, pour ne pas déclencher la factory du `defaultdict` sur un
    projet pas encore matché). Sans `rules_map`, comportement inchangé.
  - `email_analyzer/analyzer.py` : `process_delta` gagne `rules_matrix:
    Optional[dict] = None`, construit le `rules_map` et le passe à
    `search_project_emails`. Les autres appelants (`process_latest_emails`,
    `fetch_last_n_emails_for_chat`, `legacy_cli.py`) restent inchangés.
  - `email_analyzer/analysis_tasks.py` (`_run_fasttrack_sync`, couvre aussi
    `run_scheduled_sync` qui appelle la même fonction) : passe
    `rules_matrix=project.rules_matrix` à `process_delta` ; à la persistance de
    chaque email, recalcule (pur, sans I/O supplémentaire)
    `score_project_relevance`/`derive_tags` et les stocke sur
    `Email.classification_score`/`Email.tags`.
  - `alembic/versions/005_email_tags_classification.py` (nouveau) : ajoute
    `emails.tags` (JSONB) et `emails.classification_score` (Integer), nullable,
    pas de rétro-calcul (même discipline que 004). `db/models.py` mis à jour.
  - `api/routers/projects.py` : `EmailOut` expose `tags`/`classification_score` ;
    `ProjectOut`/`_project_out` expose désormais `rules_matrix` en lecture
    (auparavant write-only via `POST`) ; nouveau `PATCH /api/projects/{id}`
    (name/rules_matrix), même schéma que `PATCH /api/tenants/{id}/imap`.
  - `frontend/src/ProjectHub.tsx` : `RulesMatrixEditor` (par carte projet, replié
    par défaut, PATCH au clic) + section optionnelle repliée dans
    `CreateProjectForm` pour saisir les 6 signaux en listes séparées par
    virgules. `top_emails`/`GET /api/projects/{id}` restent non appelés ailleurs
    dans le frontend (même situation que Unit 19) — pas d'UI de liste d'emails
    construite, l'API suffit pour rendre `tags`/`classification_score`
    exploitables plus tard.
  - Vérifié : `python -m py_compile` sur tous les fichiers backend
    modifiés/créés. Appels directs sur fixtures construites (sans DB/réseau) :
    nom de projet seul → `matched=True`, `score==45`, identique au booléen de
    l'ancienne sous-chaîne sur plusieurs cas construits à la main ; email sans
    nom de projet mais `sender_email` + participant connu combinés → `matched=
    True` (nouveau vrai positif) ; email non pertinent + règles vides →
    `matched=False`, `score==0` ; `derive_tags` sur un email
    urgent/bloqué/facture avec `importance_score=88` vs `0` → bons tags +
    bonne bascule de priorité ; `ProjectRules.from_dict` sur entrées
    malformées (`None`, dict vide, listes mal typées, valeur non-dict comme
    `"garbage"`/`42`) → ne lève jamais. Cluster PostgreSQL local jetable
    (`initdb`/`pg_ctl`, port et socket dédiés, démonté après coup) :
    `alembic upgrade head` 001→005 propre ; `downgrade -1` puis `upgrade head`
    round-trip confirmant que `tags`/`classification_score` disparaissent puis
    réapparaissent ; `_run_fasttrack_sync` appelé avec `processor_from_tenant`
    et `process_delta` mockés (fixtures construites) — confirmé que
    `rules_matrix` du projet est bien transmis à `process_delta`, que les
    emails sont persistés avec les bons `tags`/`classification_score`, et
    qu'un projet sans `rules_matrix` matché uniquement par son nom obtient
    `classification_score == 45` (non-régression bit-à-bit confirmée en
    conditions quasi réelles, pas seulement en unitaire pur). `uvicorn` réel
    lancé contre ce cluster : `POST /api/auth/register` → `POST /api/projects`
    (avec `rules_matrix`) → `GET /api/projects` (rules_matrix bien exposé) →
    `PATCH /api/projects/{id}` (mise à jour bien persistée et renvoyée) →
    `GET /api/projects/{id}` (round-trip confirmé) → `PATCH` sur un id
    inexistant → `404`. Environnement jetable entièrement démonté après
    vérification (process uvicorn arrêté, cluster Postgres stoppé et supprimé).
    `tsc --noEmit` sans erreur ; `vite build` transforme les 49 modules sans
    erreur (échec ensuite limité au même `EACCES` pré-existant sur `dist/`,
    sans rapport, déjà documenté dans les unités précédentes).
  - **Limitation notée en cours de vérification** (pas un bug introduit par
    cette unité, comportement déjà présent dans `risk_keywords`/
    `identify_critical_emails`) : le matching par mots-clés est une simple
    sous-chaîne sans frontière de mot — ex. "rapidement" contient
    littéralement "api", ce qui peut déclencher le tag Technique par faux
    positif. Cohérent avec le style existant du reste du fichier
    `ai_intelligent.py`, non corrigé ici pour rester dans le périmètre défini.
  - Non fait / hors périmètre explicite (voir le plan) : signaux basés sur les
    pièces jointes (aucun parsing PDF/Word n'existe) ; classification
    hiérarchique multi-niveaux et détection de type de document ; auto-
    découverte cross-mailbox du projet d'un email ("Zero-Touch Project
    Creation") ; tags au niveau Projet/Action (seul `Email` est tagué) ;
    roster durable de participants inter-exécutions (nécessiterait une colonne
    `sender` sur `Email`) ; calibration des poids/seuils sur un vrai corpus.

- **Unit 21 — Auto-sync au login, cron élargi, fenêtres 90/120/240j, tags de
  référence + fils de discussion, résumé "assistant de direction"** (2026-07-11) :
  - Contexte : demande utilisateur en 2 temps — (1) remplacer la saisie
    manuelle du filtre "Projet" par un déclenchement automatique à la
    connexion, corriger le symptôme "le cron tourne souvent sans rien
    trouver" (fenêtre de repli Fast-Track figée à 30j), et proposer d'élargir
    la recherche après un résultat pauvre ; (2) ajouter des préréglages
    90/120/240 jours, faire remonter des tags de référence et un
    regroupement des emails par sujet pour affiner le résumé, et élever la
    qualité du résumé au niveau "assistant de direction".
  - **Fenêtre de repli Fast-Track paramétrable** (`email_analyzer/analyzer.py::
    EmailProcessor.process_delta`, nouveau `fallback_days: int = 30`, ne
    change rien par défaut) :
    - `analysis_tasks.py::_run_fasttrack_sync`/`run_fasttrack_refresh` :
      nouveaux `fallback_days`/`force_days` (ce dernier ignore
      `last_processed_email_timestamp` et force un rescan complet — utilisé
      par les boutons "chercher sur plus de jours").
    - `run_scheduled_sync` (cron 7h/19h) : passe désormais `fallback_days=60`
      (repli plus généreux pour les projets jamais synchronisés — seule
      fenêtre concernée, les deltas incrémentaux normaux ne sont pas
      touchés).
    - `api/main.py::refresh_project` (`POST /api/projects/{id}/refresh`) :
      nouveau paramètre de requête optionnel `force_days`.
  - **Auto-sync au login, throttlé 1x/24h/tenant** : `jobs.py::
    claim_daily_auto_sync` (Redis `SET NX EX 86400`) ; `saas_logic.py::
    trigger_login_auto_sync` (Fast-Track sur tous les `Project` actifs du
    tenant, `fallback_days=20`) ; `api/routers/auth.py::login` l'appelle et
    renvoie `TokenResponse.auto_sync_jobs` (additif, vide pour
    register/switch). Frontend : `apiClient.ts` (`setPendingAutoSync`/
    `takePendingAutoSync`, sessionStorage — survit à la redirection
    `/login` → `/`), `AuthPanel.tsx` (capture au login), `ProjectHub.tsx`
    (bannière "Mise à jour automatique de N projet(s) en cours…", chaque
    `ProjectCard` concernée suit son job via `pollRefreshJob` sans nouveau
    POST, même état visuel `refreshing` que le clic manuel).
  - **Préréglages 90/120/240 jours** : `config.py::VALID_PERIODS` étendu (la
    logique jours de `period.py` généralisait déjà `int(period) - 1`, aucun
    changement de logique nécessaire) ; options ajoutées au `<select>`
    "Préréglage" de `HomePage.tsx`.
  - **Regroupement par sujet (fils de discussion)** : `project_mail.py::
    normalize_subject`/`group_emails_by_subject` (nouveau, aucune base
    existante — pas de threading par en-têtes In-Reply-To/References,
    volontairement limité au signal `subject` déjà extrait). Intégré dans
    `generate_intelligent_summary` : nouvelles clés `threads` (top 10,
    sans le champ interne `latest_email`) et `tags_reference` (agrégation de
    `classification.derive_tags` par email, top 10) sur le dict JSON déjà
    renvoyé par `/api/analyze` — additif, aucune migration DB (résultat
    transitoire, pas d'historisation). Relayé aussi dans le retour de
    `_run_fasttrack_sync` pour le Project Hub.
  - **Résumé "assistant de direction"** : `llm.py::build_llm_thread_corpus`
    (corpus par fil plutôt que par email brut — un seul représentant par
    fil, réduit le bruit des relances) + `format_tags_reference_context`
    (injecte les tags dominants dans le prompt) + nouveau
    `_EXECUTIVE_SYSTEM_PROMPT` partagé OpenAI/Gemini (structure imposée en 5
    sections : synthèse en une phrase, points clés, décisions/risques,
    échéances, recommandation priorisée). `generate_openai_assistant_summary`/
    `generate_gemini_assistant_summary` prennent désormais `threads` (au lieu
    d'`emails`) + `tags_reference` optionnel ; seul appelant
    (`project_mail.py::generate_intelligent_summary`) mis à jour en
    conséquence. Contrat de sortie inchangé (`{"texte", "modèle", "erreur",
    ...}`), pas de passage à un JSON structuré multi-champs.
  - **Rendu frontend (comblait un vide réel)** : `résumé_assistant.texte`
    n'était jusqu'ici jamais affiché nulle part dans le frontend (confirmé
    par recherche avant implémentation). `AnalysisDashboard.tsx` : nouvelle
    carte "Synthèse (assistant de direction)" (premier rendu de ce champ),
    section "Tags dominants" (chips) et "Fils de discussion" (liste). Mêmes
    tokens Tailwind que l'existant (`bg-surface`, `border-border-default`,
    `rounded-full`), aucun nouveau composant lourd.
  - **Boutons "chercher sur plus de jours" (60/90/120/240)** : `HomePage.tsx`
    — `runAnalyze` scindé en `launchAnalysis(overrideDays?)` (réutilisable) +
    wrapper `runAnalyze` pour le submit du formulaire ; boutons affichés dans
    le bloc "Aucun email trouvé" existant, uniquement pour les valeurs
    strictement supérieures à la fenêtre courante. `ProjectHub.tsx` —
    mêmes boutons par carte, affichés seulement si le refresh était un
    **premier sync** revenu à 0 nouvel email (`new_emails === 0` et
    `last_processed_email_timestamp` absent avant l'appel) : un delta à 0
    sur un projet déjà synchronisé reste un état normal, pas un signal
    d'échec — condition volontairement restrictive pour rester honnête.
  - Vérifié : `python -m py_compile` sur tous les fichiers backend modifiés
    (`analyzer.py`, `analysis_tasks.py`, `jobs.py`, `saas_logic.py`,
    `api/routers/auth.py`, `api/main.py`, `config.py`, `project_mail.py`,
    `llm.py`, `cli.py`, `legacy_cli.py`) ; `tsc --noEmit` frontend sans
    erreur ; `vite build` transforme les 49 modules sans erreur (échec
    ensuite limité au même `EACCES` pré-existant sur `dist/` appartenant à
    `root`, environnement local, déjà documenté dans les unités
    précédentes — sans rapport avec ce code).
  - Non vérifié (pas d'environnement IMAP/Postgres/Redis/navigateur
    disponible dans cette session) : comportement end-to-end réel (login →
    jobs auto-sync effectivement enqueués et visibles côté Project Hub ;
    throttle Redis 24h ; contenu réel du résumé LLM avec le nouveau prompt ;
    rendu visuel des nouvelles sections). Recommandé à l'utilisateur de
    vérifier avec `npm run dev` + `uvicorn`/`arq` locaux avant mise en
    production, en particulier le rendu du résumé exécutif et le
    déclenchement de l'auto-sync sur un compte réel avec plusieurs projets.

- **Unit 23 — Analyse sans filtre sur la page d'accueil, sujets détectés
  cliquables (mode autonome)** (2026-07-11) :
  - Contexte : demande utilisateur « unifier projets et accueils [...]
    supprimer le filtre projet sur la page d'accueil pour lancer une analyse
    sans filtre, une fois les emails récupérés proposer les sujets à
    analyser ». Décision prise en mode autonome (voir
    `feedback-autonomous-implementation-mode` en mémoire) sur deux points
    d'ambiguïté réelle plutôt que de bloquer sur une question :
    1. **Ce que "unifier" signifie concrètement** : pas de fusion des routes
       `/` et `/projects` (Project Hub reste un CRUD séparé, scope SaaS) —
       l'unification se fait par le comportement décrit juste après : la page
       d'accueil ne demande plus un nom de projet à l'avance, elle en propose
       après coup, à partir du contenu réel de la boîte mail. Option retenue
       car surface de changement minimale (pas de refonte de navigation) et
       cohérente avec la préférence utilisateur déjà établie (voir
       `feedback-ask-before-ambiguous-scope`, "moins de surface touchée").
    2. **Comment "proposer les sujets" sans fabriquer une fonctionnalité de
       clustering non demandée** : `evolution-plan.md` classe le "Historical
       Clustering Worker" (découverte automatique de projets) en Phase 2, non
       construit (confirmé par exploration du code avant d'implémenter — zéro
       logique de clustering/proposition nulle part dans le backend). Plutôt
       que de construire ce sous-système, réutilisation du regroupement par
       fil de discussion déjà livré aujourd'hui (Unit 21,
       `project_mail.py::group_emails_by_subject`, champ `threads` du rapport
       JSON) : une analyse sans filtre récupère tous les emails de la
       fenêtre dans un seul bucket, le résumé IA généré dessus expose déjà
       `threads` (sujets groupés) — il ne restait qu'à les rendre cliquables
       côté frontend pour relancer une analyse ciblée. Aucune nouvelle
       classification IA, aucun nouveau modèle de données.
  - `email_analyzer/project_mail.py` : `_process_one_email_id`,
    `search_project_emails`, `search_project_emails_from_list` gagnent un
    paramètre optionnel `catch_all_key` — un email qui ne correspond à aucun
    filtre est tout de même conservé sous cette clé unique au lieu d'être
    ignoré. Avec `project_filters` vide, `_uid_search_narrow_ids` renvoyait
    déjà `None` (aucune clause à construire) donc le fallback SINCE large
    existant suffit à récupérer tous les emails de la fenêtre — aucun
    changement nécessaire côté recherche IMAP elle-même.
  - `email_analyzer/analyzer.py` : `EmailProcessor.process_latest_emails`
    accepte désormais `project_filter: Optional[...]` — `None`/liste vide
    déclenche le mode sans filtre, tous les emails regroupés sous la nouvelle
    constante `EmailProcessor.NO_FILTER_RESULT_KEY = "Tous les emails"`
    (propagée aux chemins IMAP, Gmail et Outlook). Message `_empty` distinct
    en mode sans filtre (ne parle plus de "filtre" puisqu'il n'y en a pas).
  - `email_analyzer/analysis_tasks.py` : `_run_legacy_sync`/`_run_saas_sync`
    (+ les tâches arq `run_analysis_legacy`/`run_analysis_saas`) acceptent
    `project: Optional[str]` ; chaîne vide/`None` transmise telle quelle à
    `process_latest_emails`.
  - `api/main.py` : `AnalyzeRequest.project` devient `Optional[str] = None`
    (au lieu de `Field(..., min_length=1)`) ; normalisation
    `(body.project or "").strip() or None` avant `enqueue(...)`, sur les deux
    branches (legacy et SaaS).
  - `frontend/src/pages/HomePage.tsx` : champ "Projet (filtre)" devient
    "Projet (filtre, optionnel)" avec aide contextuelle, `required` retiré,
    bouton "Lancer l'analyse" n'est plus désactivé par un champ vide ;
    `launchAnalysis` gagne un second paramètre optionnel `overrideProject`
    (même pattern qu'`overrideDays` déjà existant) pour relancer une analyse
    ciblée depuis un sujet cliqué ; nouvelle constante
    `NO_FILTER_RESULT_KEY = "Tous les emails"` (doit rester synchronisée avec
    `EmailProcessor.NO_FILTER_RESULT_KEY` côté backend — pas de source
    partagée entre Python et TypeScript dans ce dépôt) utilisée pour savoir
    quand proposer les sujets cliquables et adapter le message de succès
    ("Tous les emails de la période ont été analysés." au lieu de "N
    projet(s) dans le rapport.").
  - `frontend/src/AnalysisDashboard.tsx` : nouvelle prop optionnelle
    `onSelectThread?: (subject: string) => void` — quand fournie, chaque
    entrée de "Fils de discussion" (déjà affichée depuis Unit 21, jusqu'ici
    statique) devient un bouton cliquable et le titre de section devient
    "Sujets détectés — cliquez pour approfondir" ; sans la prop (tous les
    autres appelants), comportement strictement inchangé.
  - Vérifié : `python -m py_compile` sur les 4 fichiers backend modifiés ;
    `tsc --noEmit` sans erreur ; `vite build` transforme les 49 modules sans
    erreur (échec ensuite limité au même `EACCES` pré-existant sur `dist/`
    appartenant à `root`, déjà documenté dans toutes les unités précédentes,
    sans rapport avec ce code). Trois scénarios rejoués en appel direct avec
    `EmailProjectAnalyzer`/IMAP mockés (aucun environnement IMAP réel
    disponible dans cette session) : (1) mode sans filtre —
    `search_project_emails` bien appelé avec `project_filters=[]` et
    `catch_all_key="Tous les emails"`, résultat du bucket unique bien
    retourné tel quel ; (2) mode sans filtre + boîte vide sur la fenêtre —
    `_empty` avec le nouveau message générique (ne mentionne plus de
    "filtre") ; (3) régression — un filtre simple (`"BBCI"`) produit toujours
    exactement le même appel/résultat qu'avant cette unité
    (`catch_all_key=None`, comportement bit-à-bit inchangé). Logique de
    bascule catch-all de `_process_one_email_id` testée isolément (email sans
    correspondance + `catch_all_key` fourni → bucketé sous cette clé ; sans
    `catch_all_key` → toujours ignoré comme avant).
  - Non vérifié (pas d'environnement IMAP/navigateur disponible dans cette
    session) : rendu visuel réel (nouveau libellé du champ, sujets cliquables
    dans `AnalysisDashboard`, message de succès) ; comportement end-to-end
    avec une vraie boîte IMAP volumineuse (temps de scan sans narrowing IMAP
    côté serveur — le fallback SINCE large peut ramener plus de candidats
    qu'un scan filtré, comportement attendu et documenté mais pas mesuré en
    conditions réelles). Recommandé à l'utilisateur de lancer `npm run dev` +
    `uvicorn`/`arq` locaux, cliquer "Lancer l'analyse" sans rien saisir dans
    le champ Projet, et confirmer que les sujets proposés sont pertinents et
    cliquables.
  - Non fait / explicitement hors périmètre (voir décision 2 ci-dessus) :
    classification des emails "sans filtre" contre les `Project` déjà
    enregistrés du tenant (aurait permis de router automatiquement vers un
    projet existant plutôt qu'un bucket générique) ; conversion d'un sujet
    détecté en `Project` persisté (bouton "créer un projet à partir de ce
    sujet") ; fusion réelle des routes `/` et `/projects`. Ces trois pistes
    sont des extensions naturelles à prioriser séparément avec l'utilisateur
    si le comportement actuel s'avère insuffisant à l'usage.

- **Unit 22 — RFC pipeline v2 (document seulement, aucun code)** (2026-07-11) :
  - Contexte : demande utilisateur d'un document d'architecture complet (RFC/ADR)
    proposant une refonte du pipeline d'analyse d'emails pour tenir à l'échelle de
    plusieurs milliers/millions d'emails, plusieurs providers, plusieurs tenants.
  - `context/rfc-email-pipeline-v2.md` (nouveau) : analyse critique de l'existant
    (ancrée dans le code réel — `analyzer.py`, `project_mail.py`, `classification.py`,
    `llm.py`, `db/models.py`, etc., pas une réécriture abstraite), architecture cible
    en 17 étapes, découpage modulaire, threading JWZ, détection automatique de
    projets (clustering par thread + pgvector), mémoire projet (`project_aliases`),
    matching hybride, sync incrémentale par provider (`sync_checkpoints`, UID/
    historyId/deltaLink), résumé hiérarchique à 3 paliers avec cache par hash,
    contrat JSON structuré LLM (OpenAI Structured Outputs / Gemini `response_schema`),
    nouveaux modèles de données, pipeline `arq` à 2 files, gestion d'erreurs,
    diagrammes Mermaid, roadmap en 9 phases indépendantes (0-8) + tableau Quick Wins
    vs Long Terme.
  - Explicitement **cohérent avec** `context/evolution-plan.md` (pgvector déjà
    décidé, 2 files arq déjà nommées, Historical Clustering Worker/Validation Board
    déjà nommés) plutôt que de re-débattre ces choix ; **ne modifie pas**
    `architecture.md`/`evolution-plan.md`/`project-overview.md` (réconciliation des 6
    fichiers `context/*.md` toujours différée par décision Unit 13, non traitée ici).
  - Rien n'est implémenté : aucune migration, aucun code applicatif touché. Les 9
    phases de la roadmap restent toutes à faire — à démarrer par la Phase 0
    (fondations, migrations additives seulement) si l'utilisateur valide la direction.
  - Pas de vérification technique applicable (document seul) — relecture de cohérence
    interne effectuée (chaque section cite le fichier/fonction réel vérifié par 2
    agents Explore + lecture directe avant rédaction).

- **Unit 24 — Refonte "assistante proactive" : Brief quotidien, sidebar,
  assistant permanent portefeuille (2026-07-12)** :
  - Contexte : proposition produit détaillée de l'utilisateur (voir le plan
    approuvé, `~/.claude/plans/frontend-transform-ai-une-sharded-creek.md`)
    demandant de transformer l'outil d'analyse à la demande en assistante
    proactive — page d'accueil "Brief" (« Bonjour Mohamed, voici ce qui a
    changé… »), sidebar (Brief/Projets/Agenda/Actions/Assistant/Paramètres),
    chat permanent comme point d'entrée principal, palette claire "bureau".
    5 décisions clarifiées via questions ciblées : refonte complète (mais
    livrée en petites unités, `ai-workflow-rules.md` §2), backend structuré
    inclus, assistant = point d'entrée principal, Brief livré en premier,
    formulaire classique retiré pour les utilisateurs SaaS. Découverte clé en
    amont : `context/rfc-email-pipeline-v2.md` (Unit 22) propose déjà en
    Phase 5 exactement le schéma structuré nécessaire (`structured_content`
    JSONB additif sur `ProjectSummary`) — ce chantier en dépend explicitement
    plutôt que d'inventer un schéma concurrent ; les autres phases du RFC
    (0-4, 6-8 : checkpoints de sync, threading, mémoire projet, clustering)
    restent hors périmètre, non traitées ici.
  - **Backend — B1 (schéma LLM structuré)** : `email_analyzer/llm.py` gagne
    `ProjectSummaryLLM`/`DecisionItem`/`RiskItem`/`NextStepItem`/`DeadlineItem`
    (Pydantic, aucun champ avec valeur par défaut — voir bug ci-dessous) et
    `extract_structured_project_summary_openai`/`_gemini` (OpenAI Structured
    Outputs via `chat.completions.parse`, Gemini via
    `response_schema=ProjectSummaryLLM`). `db/models.py::ProjectSummary`
    gagne `structured_content` (JSONB), `llm_risk_level`, `schema_version` —
    additifs, jamais fusionnés avec `sentiment`/`ai_intelligent.
    calculate_risk_score` (garde-fou explicite du RFC §11). Migration Alembic
    `006_structured_content` (le dépôt a un vrai setup Alembic —
    `alembic/versions/001-005` déjà présents ; la DB locale était en retard
    de 2 migrations, `alembic upgrade head` appliqué avant d'ajouter la
    nouvelle). `project_mail.py::generate_intelligent_summary` gagne un
    paramètre `include_structured: bool = False` (appel LLM additionnel coûte
    cher — n'active l'extraction structurée que pour le chemin persisté,
    jamais pour `/api/analyze` éphémère) ; `analyzer.py::process_delta` le
    propage. **Bug réel trouvé en vérification live** : Gemini
    (`google-generativeai` 0.8) rejette toute clé `"default"` dans le schéma
    JSON envoyé (`Unknown field for Schema: default`) — corrigé en retirant
    toute valeur par défaut Python des modèles Pydantic (y compris
    `Optional[...] = None`), les rendant "requis mais nullable" plutôt
    qu'optionnels, ce qui satisfait aussi le mode strict d'OpenAI.
  - **Backend — B2 (actions multiples)** : `analysis_tasks.py::
    _run_fasttrack_sync` remplace l'unique `SuggestedAction` dérivée de
    `recommandation` par une ligne par item de `structured_content.next_steps`
    + `.deadlines` (nouveau helper `_parse_iso_date`) ; repli sur l'ancien
    comportement (une action depuis `recommandation`) si l'extraction
    structurée est absente/en échec.
  - **Backend — B4 (depuis votre dernière visite)** : `User.last_login_at` +
    `User.previous_login_at` (migration `007_user_login_timestamps`) —
    `previous_login_at` est décalé (`= last_login_at` puis `last_login_at =
    now()`) uniquement dans `api/routers/auth.py::login`, jamais `switch` ;
    exposé via `MeResponse.previous_login_at`. C'est `previous_login_at` (pas
    `last_login_at`) qui sert de référence au Brief — reste stable pendant
    toute la session en cours au lieu de se réinitialiser à chaque
    rechargement de page.
  - **Backend — B3 (agrégation portefeuille)** : nouveaux routers
    `api/routers/actions.py` (`GET/PATCH /api/actions`, tenant-wide,
    n'existait nulle part avant — `SuggestedAction` n'était visible que
    nichée dans `GET /api/projects/{id}`) et `api/routers/brief.py`
    (`GET /api/brief` — compteurs "depuis votre dernière visite" + top 5
    actions recommandées ; `GET /api/timeline` — fusion `Email.received_at` +
    `ProjectSummary.updated_at` de tous les projets, aucune nouvelle table).
  - **Backend — B5 (assistant permanent)** : nouvelle table
    `assistant_messages` (migration `008_assistant_messages`,
    `AssistantMessage` model) — persiste la conversation par (tenant, user),
    contrairement à `/api/chat` existant (scoped à un projet, éphémère côté
    frontend). `llm.py` gagne `build_portfolio_context`/
    `build_portfolio_chat_system_prompt` (contexte = tous les projets du
    tenant, pas le corpus IMAP brut d'un seul) et
    `portfolio_assistant_chat_openai`/`_gemini` (mêmes mécaniques que
    `project_assistant_chat_*`, system prompt différent). Nouveau
    `api/routers/assistant.py` (`GET /api/assistant/messages`,
    `POST /api/assistant/chat`). **Bug réel trouvé en vérification live** :
    le chat renvoyait "OPENAI_API_KEY non définie" car le frontend imposait
    `assistant_provider: "openai"` alors que seul `GEMINI_API_KEY` est
    configuré dans cet environnement — corrigé en rendant
    `ChatBody.assistant_provider` optionnel et en ajoutant
    `_select_provider()` côté serveur (choisit le premier fournisseur dont la
    clé est réellement configurée) ; le frontend n'envoie plus de préférence.
  - **Frontend — F1 (thème)** : `tailwind.config.js` — palette sombre
    indigo remplacée par une palette claire "bureau" (fond blanc cassé
    `#FAFAF8`, bleu profond `#1D4ED8` en accent, vert/orange/rouge réservés
    aux statuts). Mêmes noms de jetons qu'avant → aucun composant à
    retoucher pour ce point seul (confirmé : `ProjectHub`/`AnalysisDashboard`
    héritent du nouveau thème sans modification).
  - **Frontend — F2 (sidebar)** : `components/AppShell.tsx` — nav plate à 3
    liens remplacée par une sidebar (Brief/Projets/Agenda/Actions/Assistant/
    Paramètres + Documents/Clients grisés "Bientôt", aucune page derrière
    faute de données réelles — parsing de pièces jointes et entité
    client/contact non implémentés). Uniquement pour `saasEnabled && me` ; le
    mode legacy garde son layout inchangé.
  - **Frontend — F3/F6 (page Brief + assistant permanent)** : nouveau
    `pages/BriefPage.tsx` (compteurs depuis `GET /api/brief`, liste
    "recommandations du jour", accueil SaaS — `HomePage.tsx` délègue à
    `BriefPage` pour `saasEnabled && me`, l'ancien formulaire "Lancer
    l'analyse" ne reste que sous la branche `!saasEnabled`) et nouveau
    `PortfolioAssistant.tsx` (charge/persiste via `/api/assistant/*`, affiché
    en continu sur le Brief plutôt que caché derrière un bouton flottant).
  - **Frontend — F4 (Agenda/Actions)** : nouveaux `pages/AgendaPage.tsx`
    (échéances triées, tous projets) et `pages/ActionsPage.tsx` (liste
    groupée Aujourd'hui/Cette semaine/Plus tard, cocher/écarter via
    `PATCH /api/actions/{id}`) ; `actionsApi.ts` factorise le type `ActionOut`
    et les deux appels réseau partagés avec `BriefPage`. Décisions/risques
    n'ont pas eu de pages dédiées séparées (contrairement à la proposition
    initiale à 3 pages quasi identiques) : ils vivent dans la carte projet
    (F5) plutôt que dupliquer la liste todo trois fois.
  - **Frontend — F5 (Project Hub enrichi)** : `api/routers/projects.py::
    ProjectOut` gagne `structured_content` (additif) ; `ProjectHub.tsx`
    gagne une section repliable "Décisions & risques" par carte (badges
    colorés par niveau : critique=rouge, modéré=orange, faible=vert) ;
    `_run_fasttrack_sync` renvoie désormais `structured_content` dans son
    résultat pour que la carte se mette à jour sans reload après un
    rafraîchissement manuel.
  - **Frontend — F7 (ton)** : `HomePage.tsx` (mode legacy uniquement — le
    Brief avait déjà un ton naturel dès l'écriture) : `LOADING_STATUS_
    MESSAGES` et le bandeau de succès reformulés à la première personne
    ("J'ai terminé l'analyse de vos échanges" au lieu de "Analyse terminée").
    `ConversationalAssistant.tsx` avait déjà le bon ton, aucun changement
    nécessaire.
  - **Frontend — F8 (recherche)** : `components/AppShell.tsx` gagne un champ
    de recherche dans la sidebar (accessible depuis toutes les pages, pas
    seulement le Brief) qui navigue vers `/assistant?q=…` ; `AssistantPage.tsx`
    lit le paramètre et `PortfolioAssistant` l'envoie automatiquement au
    montage (prop `initialQuestion`) — pas de moteur de recherche séparé,
    réutilise le même LLM/contexte que l'assistant (scope volontairement
    limité aux données structurées déjà en base, pas de recherche sémantique
    plein texte sur les emails — nécessiterait pgvector, non demandé).
  - **Vérifié** (session avec accès réel à Postgres local — `DATABASE_URL`
    dans `.env`, juste précédé d'un espace qui avait fait manquer sa détection
    initiale — et à un vrai `GEMINI_API_KEY`) :
    - `alembic upgrade head` exécuté réellement (003→008) contre la DB locale ;
      colonnes/table vérifiées via `psql \d`.
    - Appel Gemini réel (`extract_structured_project_summary_gemini`) avec un
      thread email fabriqué → JSON structuré valide, décisions/risques/
      échéances cohérents avec le texte source.
    - Persistance round-trip `ProjectSummary.structured_content` via
      SQLAlchemy contre la DB réelle.
    - `_parse_iso_date`/insertion multiple de `SuggestedAction` (B2) rejouée
      avec des données synthétiques → 2 lignes correctes, description vide
      bien filtrée.
    - Endpoints `GET/PATCH /api/actions`, `GET /api/brief`, `GET /api/timeline`
      exercés via `FastAPI TestClient` + données seedées réelles (compteurs
      `new_projects`/`pending_actions`/`upcoming_deadlines`/
      `important_emails`/`at_risk_projects` tous corrects).
    - Flux login réel (register → login ×2) confirmant le décalage correct de
      `previous_login_at`.
    - Assistant permanent : conversation réelle multi-tours via
      `POST /api/assistant/chat`, réponses correctement ancrées dans le
      contexte réel (ex. "Le projet VitBank est actuellement à risque…").
    - **Parcours navigateur complet** (Playwright + Chromium installés dans
      la session, `chromium-cli` indisponible) : `uvicorn` + `vite dev`
      lancés réellement, connexion via le vrai formulaire de login, capture
      d'écran de Brief/Projets/Agenda/Actions/Assistant — zéro erreur
      console. Deux bugs réels trouvés et corrigés grâce à cette
      vérification visuelle (Gemini `default`, sélection de fournisseur —
      voir ci-dessus) plus un troisième mineur : `greetingName` (BriefPage)
      renvoyait l'email complet au lieu du prénom quand l'adresse commence
      par un séparateur (`_`/`.`/`-`) — corrigé (`.find(Boolean)` au lieu de
      `[0]`).
    - `tsc --noEmit` et `vite build` (vers un `outDir` alternatif — `dist/`
      appartient à `root` suite à un build précédent, `EACCES` pré-existant
      déjà documenté dans les unités antérieures) propres sur l'ensemble des
      fichiers touchés.
    - Toutes les données de test (utilisateurs/tenants/projets `__visual_verify*`)
      nettoyées après vérification.
  - **Non fait / explicitement hors périmètre** : réconciliation complète de
    `architecture.md`/`project-overview.md` avec le code réel (différée par
    décision Unit 13, confirmée à nouveau ici — seule une note ciblée sur les
    ajouts de cette unité a été ajoutée à chacun, pas une réécriture ;
    `ui-context.md` a en revanche été entièrement corrigé, palette et jetons
    n'ayant plus aucun lien avec le thème sombre déjà obsolète avant cette
    unité) ; `HomePage.tsx` reste un fichier de ~750 lignes dont l'essentiel
    (formulaire, hooks d'analyse) ne s'exécute plus que pour les utilisateurs
    legacy — une extraction pour l'alléger serait un nettoyage naturel futur,
    pas fait ici pour ne pas élargir le scope ; pas de rate-limiting sur
    `/api/assistant/chat` (un utilisateur pourrait déclencher des appels LLM
    en boucle) ; les phases 0-4/6-8 du RFC (sync incrémentale réelle,
    threading, mémoire projet, résumé hiérarchique, clustering) restent à
    faire séparément.

- **Bug fix (2026-07-12)** — `UniqueViolation` sur `uq_email_tenant_external_id`
  en production lors d'un sync Fast-Track (155 emails d'un coup) : cause
  racine identifiée dans `_run_fasttrack_sync` (`analysis_tasks.py`), pattern
  check-then-insert (`SELECT` d'existence puis `db.add(Email(...))`, un seul
  `db.commit()` en fin de boucle) — deux syncs concurrents du même projet
  (cron 2x/jour `run_scheduled_sync` vs. rafraîchissement manuel/auto-login,
  aucun verrou entre les deux) peuvent tous les deux voir un email comme
  absent avant que l'un des deux ne commit, et celui qui commit en second
  fait échouer toute sa transaction. **Décision autonome** : plutôt qu'un
  verrou (aurait bloqué/sérialisé les jobs, plus complexe), l'insertion est
  passée d'un `db.add()` ORM par email à un batch `INSERT ... ON CONFLICT
  (tenant_id, external_id) DO NOTHING` (`sqlalchemy.dialects.postgresql.
  insert`, `.returning(id)` pour compter `persisted` avec précision) — le
  `SELECT` d'existence préalable est conservé comme filtre rapide (évite le
  scoring LLM/règles inutile pour les emails déjà connus) mais n'est plus le
  seul rempart ; un set `seen_external_ids` dédoublonne aussi les éventuels
  doublons internes à un même delta `process_delta`. Non vérifié contre une
  vraie DB dans cette session (pas d'accès interactif à Postgres accordé) —
  seule une vérification syntaxique (`ast.parse`) a été faite ; à confirmer
  au prochain sync réel ou lors d'une session avec accès DB.

- **Unit 21 — Bouton "Analyser" par projet (modal de plage temporelle)** (2026-07-12) :
  - Contexte : la carte projet (`ProjectHub.tsx`) n'avait qu'un bouton
    "Actualiser (Fast-Track)" (delta depuis `last_processed_email_timestamp`,
    sans contrôle utilisateur sur la fenêtre). Demande : un bouton "Analyser"
    ouvrant un modal pour choisir la période (1 mois/3 mois/6 mois/1 an/autre).
  - **Aucun changement backend** : `POST /api/projects/{id}/refresh` acceptait
    déjà `force_days` (1-365, `api/main.py::refresh_project`) — ignore
    `last_processed_email_timestamp` et force un rescan complet sur N jours.
    Mécanisme déjà exercé côté frontend par la fenêtre "chercher plus loin ?"
    (`WIDEN_WINDOW_DAYS`). Le nouveau bouton ne fait que réutiliser ce même
    paramètre, avec des présets exprimés en mois côté UI uniquement (30/90/
    180/365 jours — pas d'abstraction "mois" ajoutée côté backend).
  - `frontend/src/ProjectHub.tsx` : nouveau composant `AnalyzeRangeModal`
    (présets 1/3/6/12 mois + "Autre" avec `<input type="number" min={1}
    max={365}>`, validation avant tout appel réseau) ; markup modal identique
    au pattern déjà utilisé par `BillingModal`/`ImapSettingsModal`
    (`SaasPanels.tsx` : `fixed inset-0 z-50 ... bg-black/40`, conteneur
    `rounded-2xl bg-surface p-6 shadow-xl`) — pas de nouvelle convention de
    modal. `ProjectCard` gagne un état `showAnalyzeModal` et un bouton
    "Analyser" à côté d'"Actualiser (Fast-Track)". `handleRefresh` renvoie
    désormais un booléen de succès (au lieu de `void`) pour permettre au modal
    de se fermer automatiquement seulement en cas de succès (reste ouvert sur
    erreur, l'erreur s'affiche via la bannière `refreshError` déjà existante
    sur la carte) ; comportement des autres appelants (`onClick`, effet
    auto-sync) inchangé, la valeur de retour y est simplement ignorée.
  - Vérifié : `tsc --noEmit` sans erreur ; `vite build` (vers un `outDir`
    alternatif, `dist/` du projet toujours indisponible en écriture — même
    problème d'environnement documenté dans les unités précédentes) transforme
    les 55 modules sans erreur.
  - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
    d'environnement que les unités précédentes — pas de Chromium fonctionnel
    dans cette session) ; recommandé à l'utilisateur de lancer `npm run dev` +
    `uvicorn`/`arq` localement pour confirmer le modal et l'appel réseau
    `?force_days=N`.

- **Fix — `_run_fasttrack_sync` : résumé ML/basique stocké comme dict au lieu
  de texte, faisait échouer tout le commit (emails + résumé) sur les
  projets sans texte LLM** (2026-07-12) :
  - Contexte : deux tracebacks psycopg remontés par l'utilisateur depuis les
    logs de `run_scheduled_sync` (cron 2x/jour, `logger.exception(...)` sur
    échec par projet — d'où la trace SQL+params complète visible), à
    quelques minutes d'intervalle, pour le même projet.
  - **Premier traceback (`UniqueViolation` sur `uq_email_tenant_external_id`
    pendant l'insert batch)** : investigation montre que le code source
    contient déjà le correctif pour cette course (cron vs. rafraîchissement
    manuel concurrent, cf. Unit 10/commentaire `analysis_tasks.py` —
    `pg_insert(...).on_conflict_do_nothing(index_elements=["tenant_id",
    "external_id"])`) ; la forme du SQL de l'erreur (aucune clause `ON
    CONFLICT` visible) est cohérente avec un **worker `arq` encore sur
    l'ancien bytecode** (fichier modifié non committé — un process Python
    ne recharge pas le source à chaud). **Aucun changement de code** pour
    celui-ci : nécessite juste un redémarrage du worker `arq` pour charger
    le fichier déjà corrigé sur disque.
  - **Second traceback (`ProgrammingError: cannot adapt type 'dict'` sur
    `UPDATE project_summaries SET content=...`, bug réel)** : cause racine —
    `summary_block.get("résumé_automatique")` n'est pas le texte du résumé
    mais un dict imbriqué (`{"résumé_automatique": <texte>,
    "emails_analysés": ..., "méthode": ...}`, voir
    `ai_intelligent.generate_auto_summary`/`generate_basic_summary`), donc
    quand le résumé LLM (`résumé_assistant.texte`) est absent (repli
    ML/extraction basique, `assistant_provider="none"` ou échec LLM), le
    code assignait le dict entier à `ProjectSummary.content` (colonne
    `Text`) au lieu de son champ texte interne. Comme l'échec survient sur
    `db.commit()`, **toute la transaction était annulée** — y compris le
    batch d'emails déjà exécuté juste avant dans la même session : un
    projet retombant sur le résumé ML/basique perdait silencieusement la
    persistance de ses emails à chaque sync, cron comme manuel.
  - `email_analyzer/analysis_tasks.py` (`_run_fasttrack_sync`) : double
    déballage de `summary_block["résumé_automatique"]` avant utilisation,
    même convention que `project_mail.py:378-381`/`templates.py:49-51`
    (`isinstance(..., dict)` puis accès à la clé interne) — pattern réutilisé
    tel quel, pas de nouvelle abstraction.
  - Vérifié : `python -m py_compile` OK. Script jetable (créé puis supprimé)
    testant la fonction d'extraction isolée sur 4 cas — (1) forme ML
    reproduisant exactement le dict de l'erreur → texte plat extrait
    correctement ; (2) résumé LLM présent → prioritaire sur le repli ML ;
    (3) aucun des deux → repli sur le contenu précédent inchangé ; (4) forme
    non-dict défensive → pas de crash, repli silencieux (même garde
    `isinstance` que le code réutilisé) — les 4 cas passent.
  - Non vérifié : passage réel bout-en-bout avec Postgres/Redis/worker
    (pas d'environnement jetable monté cette session, correctif validé par
    test unitaire isolé de la logique d'extraction uniquement).
  - **⚠️ Signalé à l'utilisateur** : redémarrer le worker `arq`
    (`arq email_analyzer.analysis_tasks.WorkerSettings`) pour que le
    correctif `on_conflict_do_nothing` déjà présent sur disque (premier
    traceback) prenne effet — sans redémarrage, cette erreur peut se
    reproduire même après ce fix.

- **Unit 25 — Bouton "Analyser mes emails" sur le Brief : découverte de
  projets par domaine expéditeur** (2026-07-13) :
  - Contexte : demande utilisateur d'un bouton sur la page Brief qui scanne
    tous les emails des 3 derniers mois, affiche un modal de chargement
    animé, puis propose les expéditeurs regroupés par domaine (hors
    `mediasoftci.net`, domaine interne de l'entreprise) sous forme de cartes
    sélectionnables pour créer des `Project`. Recherche préalable (2 agents
    Explore) : fonctionnalité absente du code, aucune table/colonne domaine
    sur `Project`/`Email` — conçue comme une proposition éphémère (rien
    persisté avant que l'utilisateur choisisse un domaine), la persistance
    passant par `POST /api/projects` existant, inchangé.
  - `email_analyzer/analyzer.py` : le bloc de sélection de provider
    (IMAP/Gmail/Outlook) au début de `process_latest_emails` est extrait dans
    `_fetch_project_data` (partagé), sans changement de comportement.
    Nouvelle méthode `EmailProcessor.discover_sender_domains(days_back=90,
    exclude_domains=None, on_batch=None)` : récupère les emails sans filtre
    projet (réutilise `_fetch_project_data`/`NO_FILTER_RESULT_KEY`), extrait
    l'adresse via `email.utils.parseaddr`, regroupe par domaine
    (`email_count`, `sender_count`, `sample_senders`, `sample_subjects`,
    `latest_received_at`), exclut les domaines de `exclude_domains` **et
    leurs sous-domaines**. Aucun appel ML/LLM.
  - `email_analyzer/analysis_tasks.py` : nouvelle tâche arq
    `run_domain_discovery(ctx, job_id, tenant_id, days)` /
    `_run_domain_discovery_sync` — même schéma que `_run_saas_sync`
    (session DB dédiée, persistance du refresh token OAuth si rafraîchi),
    exclusion = `{"mediasoftci.net"} ∪ {domaine de tenant.imap_user}`, pas de
    comptage de quota (aucun coût LLM, même choix que le Fast-Track).
    Enregistrée dans `WorkerSettings.functions` (`max_tries=1`).
  - `api/routers/brief.py` : nouvel endpoint `POST /api/brief/discover-projects`
    (body `{days: int = 90}`, borné 1-365), même contrat `{job_id, status}`
    que `/api/analyze` ; polling réutilisé tel quel via `GET
    /api/analyze/{job_id}` (générique par `job_id`, aucune nouvelle route de
    polling).
  - Frontend : `discoverApi.ts` (nouveau, miroir de `actionsApi.ts`) —
    `startDomainDiscovery`/`pollDomainDiscoveryJob` (avec callback de
    progression `onTick`)/`createProjectFromDomain`. `DiscoverProjectsModal.tsx`
    (nouveau) — modal en 3 états (loading/results/message), même
    overlay/panel que `SaasPanels.tsx`/`AnalyzeRangeModal` ; grille de cartes
    sobres sélectionnables (bordure + coche, jetons `ui-context.md`
    existants) ; création séquentielle des projets sélectionnés
    (`rules_matrix.sender_domains` seedé automatiquement — alimente le
    signal de classification déjà pondéré, `classification.py:127`) ; erreurs
    partielles gérées par carte sans annulation globale. `BriefPage.tsx` :
    bouton "Analyser mes emails" dans l'en-tête, bannière de succès + refetch
    de `/api/brief` après création (compteur `new_projects` à jour sans
    reload).
  - **2 bugs réels trouvés et corrigés pendant la vérification en conditions
    réelles** (IMAP réel du dépôt, ~5900 candidats sur 90 jours, 499 sur 7
    jours) :
    1. `TypeError: can't compare offset-naive and offset-aware datetimes` sur
       `latest_received_at` — certains en-têtes `Date` d'emails réels
       reviennent naïfs (`parse_email_datetime`) alors que la majorité sont
       aware ; corrigé en normalisant chaque date via
       `period.normalize_datetime_naive_local` (déjà utilisé ailleurs dans le
       code, pas de nouvelle logique) avant comparaison.
    2. Un sous-domaine technique du domaine interne (`pmg.mediasoftci.net`,
       retours de bounce) passait le filtre d'exclusion car celui-ci ne
       testait que l'égalité stricte (`domain in exclude`) — corrigé en
       excluant aussi tout domaine se terminant par `.` + un domaine exclu.
    3. (Perf, pas un bug mais une découverte importante) : `on_batch` n'était
       pas câblé initialement sur `discover_sender_domains` → un premier test
       réel (90 jours, ~5863 candidats) est resté "running" sans aucun signal
       pendant plus de 15 minutes avant d'être interrompu pour investiguer.
       Corrigé en propageant `on_batch` (`_fetch_project_data` le supportait
       déjà) jusqu'à `report_progress` côté tâche arq, et en affichant une
       vraie barre de progression côté modal (`processed`/`total`, repli sur
       le texte rotatif tant que `total` vaut 0 — même principe "jamais de
       barre à 0%" que `HUD.tsx`). Budget de polling frontend relevé de 5 à
       32 min (`DISCOVER_POLL_TIMEOUT_MS`, aligné sur `job_timeout=1800` côté
       worker arq) ; tick réseau isolé qui échoue n'interrompt plus tout le
       polling (`try/catch` + `continue`, même correctif que `pollAnalysis`,
       `HomePage.tsx`, appliqué ici pour la même raison).
  - Vérifié en conditions réelles (cluster Postgres 17 jetable démonté après
    coup — `initdb`/`pg_ctl` sur un port/socket dédiés, aucun impact sur le
    cluster système `postgresql@17-main` ni sur l'environnement de dev
    existant de l'utilisateur repéré en cours de route sur le port 8000 —,
    Redis DB dédiée `5` vidée après usage, clé de chiffrement jetable générée
    pour la session au lieu de réutiliser celle de production) :
    - `alembic upgrade head` (001→008) propre sur la base jetable.
    - Flux HTTP complet réel : `register` → `auth/me` → `PATCH
      /api/tenants/{id}/imap` (identifiants IMAP réels du dépôt) → `POST
      /api/brief/discover-projects` → polling `GET /api/analyze/{job_id}`
      jusqu'à `done`, avec progression réelle observée (`processed`/`total`
      croissants, ex. `499/499`) → `POST /api/projects` avec un domaine
      découvert (`bci-banque.com`) → confirmé dans `GET /api/projects` avec
      le bon `rules_matrix` → `GET /api/brief` confirme `new_projects: 1`.
    - Domaines réels obtenus sur la boîte du dépôt (7 jours, 499 emails
      candidats) : `gmail.com`, `bni.ci`, `icloud.com`, `bhci.ci`,
      `bci-banque.com`, etc. — `mediasoftci.net` et son sous-domaine
      `pmg.mediasoftci.net` absents du résultat après les 2 correctifs.
    - `WorkerSettings.functions` réellement importé : confirme
      `run_domain_discovery` enregistré (5 fonctions + le cron).
    - `python -m py_compile` sur les 3 fichiers backend modifiés ; `tsc
      --noEmit` et `vite build` (57 modules) propres côté frontend.
  - **⚠️ Signalé à l'utilisateur** : le worker `arq` de l'environnement de
    dev existant (`.venv/bin/arq …WorkerSettings`, déjà en cours d'exécution
    avant cette session sur un autre projet/port) tourne avec l'ancien code
    chargé en mémoire — redémarrer ce worker pour que `run_domain_discovery`
    et les 2 correctifs ci-dessus soient pris en compte.
  - Non vérifié : rendu visuel réel du modal/des cartes dans un navigateur
    (pas d'outil navigateur disponible dans cette session) ; comportement
    exact si un job dépasse le `job_timeout` arq (1800 s) côté navigateur
    réel — le budget de polling frontend (32 min) a été aligné dessus mais le
    scénario "job tué par arq en plein scan" n'a pas été reproduit
    volontairement (risque de job bloqué en "running" jusqu'à expiration du
    TTL Redis, comportement préexistant partagé avec `run_analysis_saas`,
    hors périmètre de cette unité).

- **Unit 26 — Découverte de domaines : timeout sur grosses boîtes (5000+
  emails) résolu par fetch headers-only** (2026-07-13) :
  - Contexte : signalé par l'utilisateur — sur une boîte de 5000+ emails, le
    scan progresse jusqu'à ~1000 puis `TimeoutError` (`arq` tue le job au
    `job_timeout` global de 1800 s, `max_tries=1` donc tout le travail est
    perdu). Diagnostic initial demandé ("checkpoints tous les 150 emails +
    continuation en arrière-plan") ne suffisait pas seul : la progression
    (`on_batch`) existait déjà de bout en bout (Unit 25) ; le vrai goulot
    était le débit — `_process_one_email_id` récupérait le message complet
    (`RFC822` : corps + pièces jointes) un par un via IMAP, alors que
    `discover_sender_domains` n'utilise que `from`/`subject`/`date`. Décision
    validée avec l'utilisateur (AskUserQuestion) : corriger le débit réel en
    plus des checkpoints, pas seulement les rendre plus fréquents.
  - `email_analyzer/project_mail.py` : `_process_one_email_id` et
    `search_project_emails` gagnent un paramètre `headers_only: bool = False`.
    En mode `headers_only`, le FETCH IMAP devient
    `(BODY.PEEK[HEADER.FIELDS (FROM TO CC SUBJECT DATE MESSAGE-ID)])` au lieu
    de `(RFC822)` ; le cache email (lecture ET écriture) est bypassé dans ce
    mode car `_cached_content_usable` traite un `subject` non vide comme
    suffisant — écrire un contenu sans corps dans le cache partagé aurait pu
    faire croire à tort à une analyse filtrée ultérieure (même tenant,
    `process_latest_emails`) qu'elle a déjà le corps complet. Seul le
    balayage avant est concerné ; le balayage arrière borné (`max_matches`,
    contexte `/api/chat`) continue à récupérer le corps complet (nécessaire
    pour évaluer la pertinence projet). `BODY.PEEK` a aussi pour effet
    positif de ne plus marquer les emails scannés comme lus (contrairement à
    l'ancien fetch `RFC822`).
  - `email_analyzer/analyzer.py` : `_fetch_project_data` et
    `discover_sender_domains` propagent `headers_only=True` uniquement pour
    le chemin IMAP direct (Gmail/Outlook inchangés — un seul appel API,
    aucune distinction headers/corps possible côté fetch).
  - `email_analyzer/config.py` : nouveau seuil `large_scan_threshold()`
    (1000, env `EMAIL_ANALYZER_LARGE_SCAN_THRESHOLD`) et
    `large_scan_chunk_size()` (150, env
    `EMAIL_ANALYZER_LARGE_SCAN_CHUNK_SIZE`), sur le même modèle que
    `batch_threshold()`/`batch_chunk_size()`. `search_project_emails` bascule
    la taille de lot de progression sur 150 dès que le nombre de candidats
    dépasse 1000 (au lieu de 10) — répond littéralement à la demande
    utilisateur, et évite des allers-retours Redis inutiles sur un scan de
    plusieurs milliers d'emails.
  - `email_analyzer/analysis_tasks.py` : `run_domain_discovery` reçoit un
    `timeout` `arq` dédié de 7200 s (`func(..., timeout=7200)`) au lieu
    d'hériter des 1800 s globaux — filet de sécurité, pas le temps attendu
    (le fetch headers-only devrait rester très en dessous). Les autres jobs
    (`run_analysis_*`) restent sur les 1800 s globaux, inchangés.
  - `email_analyzer/jobs.py` : `_JOB_TTL_SECONDS` relevé de 30 min à 2h05 pour
    rester au-dessus du nouveau timeout `arq` de la découverte (le TTL Redis
    est déjà rafraîchi à chaque `report_progress`, ce changement ne sert que
    de filet de sécurité sur un job anormalement lent entre deux checkpoints).
  - Frontend `discoverApi.ts` : `DISCOVER_POLL_TIMEOUT_MS` aligné sur 126 min
    (marge sur les 120 min du nouveau timeout backend). Le `job_id` en cours
    est désormais persisté dans `localStorage`
    (`myconnector:discover-job`, avec horodatage de démarrage) : `startDomainDiscovery`
    l'écrit, `pollDomainDiscoveryJob` recalcule son budget depuis
    l'horodatage stocké (plutôt qu'un budget plein à chaque appel) et purge
    l'entrée à l'arrivée d'un état terminal (`done`/`error`) ou à l'expiration
    du budget. `getResumableDiscoveryJobId()` (nouveau, exporté) permet de se
    rattacher à un scan déjà en cours plutôt que d'en relancer un nouveau.
    `DiscoverProjectsModal.tsx` : appelle `getResumableDiscoveryJobId()`
    avant `startDomainDiscovery()` ; ajoute un texte d'info sous la barre de
    progression quand `total > 1000` ("boîte volumineuse… continue en
    arrière-plan… rouvrez pour voir la progression"). C'est ce mécanisme de
    reprise qui rend la continuation en arrière-plan réellement observable
    (fermeture/réouverture du modal, ou refresh de page, se rattachent au
    même job au lieu de le relancer et de jeter le travail IMAP déjà fait).
  - Vérifié : `python -m py_compile`/import réel des 5 fichiers backend
    modifiés ; test unitaire jetable avec IMAP mocké confirmant que
    `headers_only=True` (a) émet bien le FETCH `BODY.PEEK[HEADER.FIELDS...]`
    et non `RFC822`, (b) laisse `self.email_cache` vide (aucune écriture),
    (c) produit un `email_content` exploitable par `discover_sender_domains`
    (from/subject/date présents, corps vide) ; introspection du `Function`
    `arq` confirmant `run_domain_discovery.timeout_s == 7200` alors que les
    autres jobs restent à `None` (default global) ; `large_scan_threshold()`/
    `large_scan_chunk_size()` retournent bien 1000/150 par défaut. Frontend :
    `npx tsc --noEmit` propre sur les fichiers modifiés.
  - Non vérifié : `vite build` complet — `dist/assets/` de ce dépôt appartient
    à `root` (résidu d'un build antérieur lancé avec des privilèges élevés,
    sans lien avec cette session), `vite` ne peut pas le nettoyer
    (`EACCES`). Non traité (nécessiterait `sudo rm`, non exécuté sans
    confirmation explicite) — signalé à l'utilisateur. Scénario réel en
    conditions IMAP (boîte à 5000+ emails, mesure du débit avant/après) non
    reproduit dans cette session — pas d'accès à une telle boîte de test ;
    seule la logique (fetch spec, cache, seuils, timeout) a été vérifiée
    unitairement.

- **Unit 27 — Fix jobs figés à "running" pour toujours (`CancelledError` non
  capturée)** (2026-07-13) :
  - Contexte : utilisateur signale la découverte de projets ("Analyser" du
    Brief) bloquée à "40 / 5806 emails analysés" — jamais de progression, le
    modal reste ouvert indéfiniment. Boîte réelle de 5806 candidats (le
    scénario que Unit 26 n'avait pas pu tester faute d'accès à une boîte de
    cette taille).
  - Diagnostic : 4 jobs du tenant trouvés figés à `status: running` dans
    Redis (`myconnector:job:*`), certains depuis plus d'une heure, alors que
    le process worker `arq` actif n'avait que 20 min d'existence — jobs
    orphelins d'un worker précédent mort/redémarré en plein job. Cause
    racine dans `email_analyzer/analysis_tasks.py` : les 4 wrappers
    (`run_analysis_legacy/saas`, `run_fasttrack_refresh`,
    `run_domain_discovery`) ne capturaient que `except Exception`. Or arq
    tue un job qui dépasse son `job_timeout`/`timeout` via
    `asyncio.wait_for(task, timeout_s)` (`arq/worker.py:599`), qui lève
    `asyncio.CancelledError` **dans** la coroutine — et `CancelledError`
    n'est plus une sous-classe d'`Exception` depuis Python 3.8. Résultat :
    `set_status(job_id, STATUS_ERROR, ...)` n'était jamais appelé, le job
    restait figé à "running" avec la dernière progression connue jusqu'à
    l'expiration du TTL Redis (2h05, `jobs._JOB_TTL_SECONDS`) — le frontend
    n'a aucun moyen de détecter un job mort et continue de l'afficher comme
    "en cours".
  - Fix : les 4 wrappers capturent désormais
    `except (Exception, asyncio.CancelledError) as exc`, appellent
    `set_status(job_id, STATUS_ERROR, error=...)` (message de repli explicite
    si `str(exc)` est vide, ce qui est le cas pour `CancelledError`) puis
    `raise` pour ne pas casser la sémantique d'annulation côté arq (retry
    logic interne, `aborting_tasks`).
  - Nettoyage immédiat : les 4 jobs orphelins déjà figés ont été marqués
    `status: error` directement dans Redis (mutation manuelle, pas de
    nouveau code impliqué) pour débloquer l'utilisateur sans attendre
    l'expiration du TTL ; worker `arq` redondé pour charger le fix (aucun
    job en cours au moment du redémarrage, vérifié via l'absence de clé
    `arq:in-progress:*`).
  - Non résolu par ce fix : la cause du premier redémarrage/mort du worker
    précédent reste inconnue (pas de log persistant du process arq dans cet
    environnement — tourne dans un terminal sans redirection vers fichier
    avant cette session) ; si le pattern se reproduit, envisager de rediriger
    systématiquement stdout/stderr du worker vers un fichier de log.
  - Non traité (hors scope de ce fix ponctuel) : le thread lancé par
    `asyncio.to_thread` dans `_run_*_sync` n'est pas interrompu par
    l'annulation de la tâche asyncio parente — un job tué par `job_timeout`
    continue de tourner en arrière-plan dans son thread (fetch IMAP/appel
    LLM en cours) jusqu'à sa fin naturelle, même après que `set_status`
    marque le job "error". Pas d'impact utilisateur direct (le statut
    Redis est correct), mais gaspille des ressources et pourrait, dans de
    rares cas, écrire un résultat tardif après coup si `_run_*_sync` finit
    par retourner normalement (peu probable ici : la fonction ne rappelle
    plus `set_status` après un retour réussi de `asyncio.to_thread` puisque
    ce retour n'arrivera jamais côté coroutine annulée).

- **Unit 28 — Fiche détail d'une action (page Actions)** (2026-07-13) :
  - Contexte : demande utilisateur — cliquer sur une action dans la page
    Actions doit ouvrir un détail "soft, intuitif, facile à lire" avec le
    pourquoi, les personnes concernées, et un conseil pour éviter que la
    situation se reproduise. Aucune de ces 3 données n'existait — `SuggestedAction`
    (`db/models.py`) n'avait que `description`/`deadline`/`status` — donc pas
    question de les fabriquer côté frontend : remontée dans tout le pipeline
    jusqu'à l'extraction LLM.
  - `email_analyzer/llm.py` : `NextStepItem`/`DeadlineItem` (schéma
    `ProjectSummaryLLM`) gagnent `raison` et `conseil_prevention` (+
    `responsable` ajouté à `DeadlineItem`, symétrique à `NextStepItem` qui
    l'avait déjà mais ne le persistait nulle part avant cette unité) ;
    `_STRUCTURED_EXECUTIVE_SYSTEM_PROMPT` étendu pour les définir, avec
    consigne explicite de laisser `null` plutôt que d'inventer un conseil
    générique si le contexte email est insuffisant.
  - `email_analyzer/db/models.py` : `SuggestedAction` gagne `rationale`
    (Text), `stakeholder` (String 200), `advice` (Text), tous nullable.
  - `alembic/versions/009_suggested_action_detail.py` (nouveau) : 3 colonnes
    additives nullable, `downgrade()` symétrique.
  - `email_analyzer/analysis_tasks.py` : les 2 sites de création de
    `SuggestedAction` (boucle `next_steps`, boucle `deadlines`) persistent
    désormais `rationale`/`stakeholder`/`advice` depuis
    `step.get("raison")`/`step.get("responsable")`/`step.get("conseil_prevention")`.
    Le 3ᵉ site (repli sur `risk.get("recommandation")`, sans extraction
    structurée) reste sans ces champs — honnête : aucune donnée de ce type
    n'existe dans ce chemin de repli.
  - `api/routers/actions.py` (`ActionOut`) et `api/routers/projects.py`
    (`SuggestedActionOut`) : 3 champs additifs exposés, mêmes noms.
  - `frontend/src/actionsApi.ts` : `ActionOut` (type miroir) étendu à
    l'identique.
  - `frontend/src/pages/ActionsPage.tsx` : `ActionRow` devient cliquable
    (description/projet dans un `<button>` dédié, checkbox "fait" et
    "Ignorer" gardent `stopPropagation` pour ne pas ouvrir la fiche) ; nouveau
    `ActionDetailModal` — même pattern de modale que `DiscoverProjectsModal.tsx`
    (`fixed inset-0 bg-black/40` + carte `rounded-2xl bg-surface`), avec une
    animation d'entrée douce dédiée (`animate-modal-in`, nouveau keyframe
    fade+scale léger dans `index.css`, même convention que
    `animate-result-panel`). 3 sections "Pourquoi cette action" / "Personnes
    concernées" / "Pour éviter que ça se reproduise", chacune avec un repli
    textuel explicite (ex. *"Aucun contexte détaillé n'a pu être extrait pour
    cette action."*) plutôt qu'une section vide ou masquée — couvre les
    actions créées avant cette migration ou via le chemin de repli. Actions
    "Marquer comme fait"/"Ignorer" dupliquées dans la fiche (ferment la
    modale après action).
  - Vérifié en conditions réelles : `tsc --noEmit` OK ; `vite build`
    transforme les 57 modules sans erreur (échec ensuite limité au même
    `EACCES` sur `dist/` pré-existant appartenant à `root`, sans rapport,
    déjà documenté dans les unités précédentes) ; `python -m py_compile` OK
    sur les 6 fichiers backend touchés. Migration testée sur un cluster
    Postgres réel via une base jetable dédiée (`createdb
    my_connector_verify_tmp` dans le cluster local existant, jamais la base
    `connector` de dev) : `alembic upgrade head` (001→009) applique
    proprement, `downgrade -1` puis `upgrade head` round-trip sans erreur,
    schéma de `suggested_actions` vérifié colonne par colonne (`\d`) contre
    le modèle ORM. Insert/lecture ORM réels : une action avec les 3 champs
    remplis relue à l'identique, et une action créée sans ces champs (chemin
    de repli) confirmée `None` sans erreur ; `action_out(...)` (le mapper
    API) sérialisé en JSON et vérifié champ par champ sur l'action remplie.
    Base jetable supprimée (`dropdb`) après vérification.
  - Non vérifié : rendu visuel réel de la modale dans un navigateur (même
    contrainte d'environnement que les unités précédentes — pas de Chromium
    fonctionnel dans cette session) ; le chemin LLM réel produisant
    effectivement `raison`/`conseil_prevention`/`responsable` (dépend d'un
    vrai appel OpenAI/Gemini sur un corpus d'emails, non testé ici — seul le
    contrat Pydantic → dict → DB → API a été vérifié). Recommandé à
    l'utilisateur de lancer `npm run dev` localement pour un premier passage
    visuel, et de rafraîchir un vrai projet (Fast-Track) pour voir le LLM
    remplir ces champs en conditions réelles.

- **Unit 29 — Page de détails d'un projet** (2026-07-13) :
  - Contexte : demande utilisateur — cliquer sur un projet dans le Project
    Hub doit ouvrir une page dédiée avec ses détails. Découverte avant
    implémentation (comme l'exige `CLAUDE.md`) : `GET /api/projects/{id}`
    (`api/routers/projects.py:160-217`, `ProjectDetailOut`) existait déjà et
    exposait déjà tout ce qu'il fallait (`suggested_actions` avec
    `rationale`/`stakeholder`/`advice` depuis l'Unit 28, `top_emails` avec
    `tags`/`importance_score`) mais n'était appelé par **aucun** composant
    frontend. Aucun changement backend nécessaire pour cette unité.
  - `frontend/src/ProjectHub.tsx` : `Sentiment`, `SENTIMENT_LABEL`,
    `SENTIMENT_CLASS`, `StructuredContent`, `RISK_LEVEL_CLASS`,
    `formatTimestamp`, `DecisionsAndRisks` passés en `export` (étaient
    privés au fichier) pour réutilisation sur la nouvelle page, plutôt que
    de les dupliquer. `DecisionsAndRisks` gagne une prop optionnelle
    `defaultOpen` (défaut `false`, comportement de la carte inchangé) pour
    l'afficher déplié par défaut sur la page détail. Titre de
    `ProjectCard` enveloppé dans un `<Link to={`/projects/${id}`}>` — ciblé
    sur le titre plutôt que toute la carte pour ne pas avoir à ajouter
    `stopPropagation` sur les boutons Refresh/Analyser, l'éditeur de règles
    et l'accordéon Décisions & risques déjà présents dans la carte.
  - `frontend/src/pages/ActionsPage.tsx` : `DetailBlock` passé en `export`
    (composant du triptyque "Pourquoi / Personnes concernées / Conseil"
    créé pour la fiche action de l'Unit 28) — réutilisé tel quel sur la page
    détail projet plutôt que dupliqué.
  - `frontend/src/pages/ProjectDetailPage.tsx` (nouveau) : même garde-fous
    que `ProjectHubPage.tsx` (`saasEnabled`/token/`me`), `useParams` pour
    l'id de projet, `fetchProjectDetail` (appel direct `GET
    /api/projects/{id}`, même convention `apiFetch`/`parseResponseJson` que
    `actionsApi.ts` — pas de module API séparé pour un seul call site).
    Affiche dans l'ordre : lien retour, en-tête (nom/sentiment/dernière
    analyse), résumé, `DecisionsAndRisks` (toujours déplié), actions
    suggérées (toutes, pas seulement `pending` — historique complet
    pertinent sur la page d'un seul projet ; accordéon par action avec les 3
    `DetailBlock` + boutons "Fait"/"Ignorer" réutilisant
    `updateActionStatus` d'`actionsApi.ts`, mise à jour optimiste locale),
    emails les plus importants (sujet, date, `recipient_status`, tags,
    score d'importance). **Hors scope volontaire** : pas de boutons
    Fast-Track "Actualiser"/"Analyser" ni d'édition de `rules_matrix` sur
    cette page — ces actions restent sur la carte du Hub, la page détail
    reste une vue de lecture + gestion des actions suggérées.
  - `App.tsx` : route `/projects/:projectId` → `ProjectDetailPage`.
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` transforme les
    58 modules sans erreur (échec ensuite limité au même `EACCES` sur
    `dist/` pré-existant appartenant à `root`, sans rapport, déjà documenté
    dans les unités précédentes). Contrat backend re-vérifié en conditions
    réelles avec un cluster Postgres local jetable (base
    `my_connector_verify_tmp2`, `alembic upgrade head` 001→009, démontée
    après coup) : tenant/user/membership/projet/résumé/action/email de test
    créés via l'ORM, `GET /api/projects/{id}` appelé via
    `TestClient(app)` avec un vrai JWT — réponse `200` dont la forme exacte
    (clés de premier niveau + `suggested_actions[0]` + `top_emails[0]`)
    correspond precisément aux types `ProjectDetail`/`SuggestedActionDetail`/
    `ProjectEmailSummary` du nouveau fichier frontend (assertions
    programmatiques sur les clés, pas seulement une lecture visuelle du
    JSON).
  - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
    d'environnement que les unités précédentes — pas de Chromium
    fonctionnel dans cette session, cf. Unit 7 étape 3) ; recommandé à
    l'utilisateur de lancer `npm run dev` + `uvicorn` localement, ouvrir
    `/projects`, cliquer un projet, et confirmer l'affichage/la navigation
    retour.

- **Unit 30 — Filtres sur la page Projets (Statut, Santé, Recherche)**
  (2026-07-13) : `ProjectHub.tsx` — nouveau composant `ProjectFilterPanel`
  (panneau latéral repliable, fermé par défaut, badge du nombre de filtres
  actifs) avec recherche par nom, statut (Actifs par défaut / Archivés /
  Tous) et santé/sentiment (multi-sélection incluant un token frontend
  `"none"` pour "pas encore analysé", `sentiment === null`). Filtrage
  entièrement côté client via `filterProjects()` (fonction pure, `useMemo`)
  — **aucun changement backend** : `GET /api/projects` reste sans query
  params, comme décidé avec l'utilisateur (pas de filtre "client" basé sur
  `rules_matrix.client_names`, jugé peu fiable ; pas de filtre "actions en
  attente" pour cette itération). État des filtres piloté par
  `useSearchParams` (source de vérité unique, pas de state dupliqué) :
  persiste au refresh et est partageable via l'URL (`?status=`,
  `?sentiment=`, `?q=`), clés absentes de l'URL quand la valeur est celle
  par défaut. Nouvel état vide dédié ("Aucun projet ne correspond aux
  filtres" + bouton réinitialiser), distinct de l'état "aucun projet du
  tout" existant.
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` transforme les
    58 modules sans erreur (échec ensuite au même `EACCES` pré-existant sur
    `dist/`, sans rapport). Logique de `filterProjects()` vérifiée en
    isolation via un script `tsx` jetable (supprimé après coup) avec des
    payloads synthétiques mais réalistes (5 projets couvrant actif/archivé,
    les 3 sentiments + `null`, noms avec accents) : 7 cas passés (défaut
    actifs-seulement masque les archivés, filtre archivés, filtre "tous",
    token `"none"` matche `sentiment: null`, multi-sélection sentiment,
    recherche insensible à la casse/accents, combinaison sans résultat).
  - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
    d'environnement que les unités précédentes, pas de Chromium fonctionnel
    dans cette session) ; recommandé à l'utilisateur de lancer `npm run dev`
    + `uvicorn` localement, ouvrir `/projects`, et confirmer visuellement
    l'ouverture/fermeture du panneau, le style des pills actives, et le
    comportement au F5 avec des filtres dans l'URL.

- **Unit 31 — Agenda IA : rendez-vous, prochains retours probables, cron dédié**
  (2026-07-14) : voir le plan approuvé
  `~/.claude/plans/implementer-l-agenda-les-rendez-sequential-spring.md` pour
  le détail complet du cadrage. Résumé :
  - `alembic/versions/010_agenda.py` (nouveau) : table `appointments`
    (tenant_id/project_id/description/scheduled_at/participants JSONB/status/
    created_at/updated_at) + 4 colonnes nullables sur `project_summaries`
    (`probable_next_contact_date/_reason/_confidence`, `agenda_updated_at`).
  - `email_analyzer/db/models.py` : `AppointmentStatus`
    (`tentative`/`confirmed`/`cancelled`), modèle `Appointment`,
    `Project.appointments`, 4 nouvelles colonnes `ProjectSummary`.
  - `email_analyzer/llm.py` : `ProjectSummaryLLM` gagne `appointments:
    List[AppointmentItem]` et `probable_next_contact:
    Optional[ProbableContactItem]` ; `_STRUCTURED_EXECUTIVE_SYSTEM_PROMPT`
    étendu — instruction explicite de ne remplir `probable_next_contact` que
    sur signal réel dans les emails (jamais une date devinée sans ancrage,
    même discipline que `missing_information`).
  - `email_analyzer/analysis_tasks.py::_run_fasttrack_sync` : wipe/rebuild
    idempotent des `Appointment` du projet (même discipline que
    `SuggestedAction`) + écriture de
    `probable_next_contact_date/_reason/_confidence`/`agenda_updated_at` à
    chaque resynchronisation avec nouveaux emails (remis à `None` si le LLM
    ne renvoie plus de signal, jamais de valeur périmée conservée).
  - Nouveau cron `run_agenda_refresh` (`arq.cron`, portée réduite aux projets
    actifs `awaiting_feedback`/`under_tension`/`llm_risk_level CRITIQUE` via
    `_agenda_relevant_project_rows`, appelle `_run_fasttrack_sync` — aucune
    pipeline LLM dupliquée), cadence `config.agenda_refresh_cron_hours()`
    (nouvel accessor, env `AGENDA_REFRESH_CRON_HOURS`, défaut heures ouvrées
    toutes les 2h : `{7,9,11,13,15,17,19,21}`), enregistré dans
    `WorkerSettings.cron_jobs` à côté de `run_scheduled_sync`. Rafraîchissement
    à la demande : `run_agenda_refresh_for_tenant` (tâche arq, même sélection
    de projets scopée à un tenant), déclenché par `POST /api/agenda/refresh`.
  - `api/routers/agenda.py` (nouveau, enregistré dans `api/main.py`) :
    `GET /api/agenda` (appointments à venir + `awaiting_projects`/
    `at_risk_projects` avec prédicat identique à
    `brief.py::at_risk_projects`, + `agenda_updated_at` = plus ancienne
    préparation IA parmi les projets retournés), `PATCH
    /api/agenda/appointments/{id}` (confirmer/annuler, validation contre
    `AppointmentStatus`), `POST /api/agenda/refresh` (enqueue +
    `{job_id}`, même contrat générique `GET /api/analyze/{job_id}` que le
    reste du Fast-Track).
  - `frontend/src/agendaApi.ts` (nouveau) : types miroirs +
    `fetchAgenda`/`updateAppointmentStatus`/`requestAgendaRefresh`/
    `pollAgendaRefreshJob` (même convention `apiFetch`/`parseResponseJson` que
    `actionsApi.ts`, même forme de polling que `ProjectHub.tsx::
    pollRefreshJob`).
  - `frontend/src/pages/AgendaPage.tsx` : réécrite (l'ancienne version,
    untracked, n'affichait qu'une liste plate des `SuggestedAction` en attente
    avec deadline — conservée telle quelle comme 4ᵉ section "Échéances
    (actions)"). Nouvelles sections "Rendez-vous à venir" (confirmer/annuler
    par ligne), "Projets en attente — prochain retour probable", "Projets en
    rouge — prochain retour attendu". En-tête : "Dernière préparation IA : il
    y a Xh" (calculé depuis `agenda_updated_at`) + bouton "Rafraîchir" (pulse
    visuel `.animate-fasttrack-pulse`/tokens `fasttrack-*` pendant le job,
    même pattern que `ProjectCard.handleRefresh`). Aucun changement requis
    dans `AppShell.tsx`/`App.tsx` (route `/agenda` et lien sidebar déjà en
    place, untracked également).
  - Décisions de cadrage (mode autonome) : rendez-vous exclusivement extraits
    par l'IA (pas de formulaire de création manuelle — hors scope, ajout
    spéculatif sinon) ; utilisateur peut seulement confirmer/annuler.
  - Vérifié en conditions réelles : `python -m py_compile` sur tous les
    fichiers backend touchés ; cluster Postgres existant (`my_connector_verify_agenda`,
    base jetable dédiée, jamais `connector`) — `alembic upgrade head` (001→010)
    propre, `downgrade -1`/`upgrade head` round-trip sans erreur, schéma
    vérifié colonne par colonne (`\d appointments`, `\d project_summaries`)
    contre l'ORM. Appel direct de `_run_fasttrack_sync` avec un résultat LLM
    mocké (`unittest.mock.patch` sur `processor_from_tenant`) : rendez-vous
    avec date persisté, rendez-vous sans date exploitable correctement rejeté,
    `probable_next_contact_*`/`agenda_updated_at` correctement écrits ;
    deuxième appel sans nouvel email confirmé idempotent (aucun doublon
    `Appointment`). `GET /api/agenda` via `TestClient` avec tenant/projets/
    utilisateur/JWT réels (3 projets couvrant `awaiting_feedback`/
    `under_tension`/`on_track`, 2 rendez-vous) : partitionnement et tri
    corrects, `PATCH /api/agenda/appointments/{id}` confirmé (200 + statut
    changé) et rejet d'un statut invalide (422). Cron vérifié par lecture
    directe de `WorkerSettings.cron_jobs`/`.functions` :
    `run_agenda_refresh` bien enregistré avec la cadence attendue,
    `run_agenda_refresh_for_tenant` bien exposé comme fonction arq. Base
    jetable et scripts de vérification supprimés après coup. `npx tsc
    --noEmit` sans erreur ; `vite build` transforme les 59 modules sans
    erreur (échec ensuite limité au même `EACCES` pré-existant sur `dist/`
    appartenant à `root`, sans rapport, déjà documenté dans les unités
    précédentes).
  - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
    d'environnement que toutes les unités précédentes — pas de Chromium
    fonctionnel dans cette session) ; le chemin LLM réel produisant
    effectivement `appointments`/`probable_next_contact` (dépend d'un vrai
    appel OpenAI/Gemini sur un corpus d'emails avec de vrais signaux de
    rendez-vous/relance, non testé ici — seul le contrat Pydantic → DB → API
    a été vérifié) ; le cron en conditions réelles sur la durée (l'enregistrement
    dans `WorkerSettings` est vérifié, pas une exécution effective par un
    worker arq tournant plusieurs heures). Recommandé à l'utilisateur de
    lancer `npm run dev` + `uvicorn`/`arq` localement, ouvrir `/agenda`, et
    rafraîchir un vrai projet en attente/en rouge pour voir le LLM remplir ces
    champs en conditions réelles.

- **Unit 32 — Dédoublonnage des projets (découverte intelligente + fusion des
  doublons)** (2026-07-14) :
  - Contexte : demande utilisateur explicite — éviter les doublons dans la
    liste des projets ; le flux "Analyser" du Brief doit mettre à jour les
    projets déjà existants plutôt que les redupliquer, et ne créer que pour
    les domaines réellement nouveaux ; pouvoir fusionner les doublons déjà
    présents. Investigation préalable (agent Explore + lecture directe,
    comme l'exige `CLAUDE.md`) : **aucune déduplication n'existait nulle
    part** — `POST /api/projects` (`api/routers/projects.py`) insère sans
    jamais vérifier l'existant, et le flux de découverte
    (`discover_sender_domains` → `DiscoverProjectsModal.tsx` →
    `createProjectFromDomain`) proposait tous les domaines trouvés, y compris
    ceux déjà couverts par un projet (`rules_matrix.sender_domains`).
  - **Volet A — découverte** : `email_analyzer/analysis_tasks.py` —
    `_existing_domain_project_map(db, tenant_id)` (nouveau) construit
    domaine minuscule → `{id, name}` à partir des `rules_matrix.sender_domains`
    des projets actifs du tenant ; `_run_domain_discovery_sync` annote
    chaque domaine du résultat avec `existing_project_id`/
    `existing_project_name` après l'appel à `discover_sender_domains`
    (`analyzer.py` reste une fonction pure, sans accès DB — l'annotation se
    fait côté tâche arq qui a déjà la session ouverte).
    `frontend/src/discoverApi.ts` : `DiscoveredDomain` gagne ces 2 champs ;
    nouvelle fonction `updateProjectFromDomain(projectId)` (POST
    `/api/projects/{id}/refresh`, fire-and-forget — même endpoint Fast-Track
    que le bouton "Rafraîchir" du Project Hub). `DiscoverProjectsModal.tsx` :
    badge "Déjà suivi : {nom}" sur les domaines déjà couverts ;
    `handleConfirm` scinde la sélection (création vs. mise à jour) ; bouton
    de confirmation affiche les deux compteurs séparément
    (`confirmLabel`) ; prop `onProjectsCreated` renommée
    `onProjectsProcessed(created, updated)`. `BriefPage.tsx` : callback et
    message toast alignés sur les deux compteurs.
  - **Volet B — fusion des doublons existants** : `api/routers/projects.py` —
    `GET /api/projects/duplicates` (nouveau, déclaré avant `/{project_id}`
    pour ne pas être intercepté) regroupe les projets actifs du tenant dont
    `rules_matrix.sender_domains` se chevauche (seul signal retenu — `name`/
    `client_names`/`company_names` écartés, même précédent que l'Unit 30
    "jugé peu fiable") ; union-find pour qu'un projet à cheval sur plusieurs
    domaines n'apparaisse que dans un seul groupe. `POST
    /api/projects/{target_id}/merge` (nouveau, body `{source_ids}`) :
    fusionne `rules_matrix` (union dédupliquée insensible à la casse via
    `_merge_rules_matrix`), réassigne `Email`/`SuggestedAction`/`Appointment`
    des sources vers la cible (bulk `UPDATE`), supprime les `ProjectSummary`
    des sources (contrainte unique sur `project_id`, ne peut être
    réassignée) puis les `Project` sources, retourne le détail de la cible
    (réutilise `get_project`). Ne déclenche **aucun** rafraîchissement
    automatique (pas d'appel LLM caché) — documenté que l'utilisateur doit
    relancer un Fast-Track (idéalement avec `force_days`) après une fusion
    pour que le résumé reflète les emails plus anciens réassignés (le
    curseur `last_processed_email_timestamp` de la cible peut leur être
    postérieur, un refresh delta simple ne les verrait pas).
    `frontend/src/ProjectHub.tsx` : nouveau `DuplicatesPanel` (bandeau
    au-dessus de la grille, un groupe par domaine partagé, sélection radio de
    la cible préremplie sur le projet avec le plus d'emails, bouton
    "Fusionner" → retire les sources de la liste locale et patche la cible
    avec la réponse, sans refetch complet).
  - Vérifié en conditions réelles (cluster Postgres local existant, base
    jetable dédiée `my_connector_verify_dedup`, démontée après coup ;
    `alembic upgrade head` 001→010 propre) : (1) 2 projets de test partageant
    `clienta.com`/`ClientA.com` (casse différente) + 1 projet non lié → `GET
    /api/projects/duplicates` renvoie exactement 1 groupe avec les 2 bons
    projets et `shared_domains: ["clienta.com"]` ; (2) `Email`/
    `SuggestedAction`/`Appointment` seedés sur le projet source, appel réel
    `POST /api/projects/{target}/merge` via `TestClient(app)` avec un JWT
    réel → tous réassignés à la cible en base, `ProjectSummary`/`Project` du
    source disparus, `rules_matrix.sender_domains` de la cible = union
    dédupliquée (`clienta.com`, `extra.clienta.com`) ; `GET
    /api/projects/duplicates` revient vide après fusion ; auto-fusion et
    `source_id` inconnu rejetés (400/404) ; (3) `_run_domain_discovery_sync`
    appelé directement avec un `discover_sender_domains` mocké (2 domaines,
    dont un couvert par un projet existant en base) → le domaine couvert
    ressort bien annoté `existing_project_id`/`existing_project_name`
    corrects, l'autre `null`/`null`. `python -m py_compile` OK sur les 2
    fichiers backend touchés ; `npx tsc --noEmit` sans erreur ; `vite build`
    transforme les 59 modules sans erreur (échec ensuite limité au même
    `EACCES` pré-existant sur `dist/` appartenant à `root`, sans rapport,
    déjà documenté dans les unités précédentes).
  - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
    d'environnement que toutes les unités précédentes — pas de Chromium
    fonctionnel dans cette session) — recommandé à l'utilisateur de lancer
    `npm run dev` + `uvicorn`/`arq` localement, ouvrir `/projects` pour voir
    le bandeau de doublons et tester une fusion réelle, et le modal
    "Analyser" du Brief pour voir les badges "Déjà suivi" et le bouton de
    confirmation à deux compteurs.

## In Progress

- Unit 7 — Gamified Loading Experience : étape 3/8 livrée (HUD réel branché
  sur le store, non monté dans l'UI) puis **mise en pause décidée en mode
  autonome** (2026-07-06) : les étapes 4-8 (`PipelineTrack`, `Robot`,
  `EmailEnvelope`, `FloatingText`) supposent toutes une granularité que le
  backend n'expose pas encore (stages de pipeline détaillés, items
  individuels par email pendant le scan). Les continuer maintenant
  signifierait soit fabriquer des données (déjà écarté à l'étape 2), soit
  construire des composants qui ne s'activeront jamais tant que le signal
  réel n'existe pas — les deux vont à l'encontre de `ai-workflow-rules.md`
  (pas de logique spéculative/cachée) et de la préférence utilisateur
  établie pour le mapping honnête. Reprise naturelle une fois qu'une source
  de données plus granulaire existe (par ex. si Unit 11 ci-dessous finit
  par exposer une progression par email plutôt que par lot de 10).

## Next Up

- **Vérification visuelle de l'Agenda IA (Unit 31) par l'utilisateur** :
  `npm run dev` + `uvicorn`/`arq` locaux, ouvrir `/agenda`, cliquer
  "Rafraîchir" sur un tenant ayant des projets en attente/en rouge, confirmer
  l'affichage des 4 sections (rendez-vous, prochain retour probable en
  attente/en rouge, échéances) et que le LLM remplit bien `appointments`/
  `probable_next_contact` sur un vrai corpus d'emails — voir l'entrée Unit 31
  ci-dessus pour ce qui a et n'a pas pu être vérifié dans cette session (pas
  de navigateur disponible, pas d'appel LLM réel testé).
- **Vérification visuelle du Project Hub par l'utilisateur** : `npm run dev`
  + `uvicorn`/`arq` locaux, ouvrir `/projects`, créer un projet, déclencher
  un rafraîchissement Fast-Track et confirmer que seule la carte concernée
  pulse — voir l'entrée Unit 12 ci-dessus pour ce qui a et n'a pas pu être
  vérifié dans cette session (pas de navigateur disponible).
- **Vérification visuelle de la page détail projet (Unit 29) par
  l'utilisateur** : cliquer le nom d'un projet depuis `/projects`, confirmer
  l'affichage de `/projects/:id` (résumé, décisions & risques, actions
  suggérées dépliables, emails importants) et le lien retour — même
  contrainte d'environnement (pas de navigateur disponible dans cette
  session), voir l'entrée Unit 29 ci-dessus.
- ~~Outlook (Microsoft Graph) OAuth2 Integration~~ — **Fermée par Unit 17**
  (2026-07-09).
- ~~Sync planifiée 2x/jour via `arq` cron~~ — **Fermée par Unit 18** (2026-07-09).
- ~~Moteur de scoring d'importance par email~~ — **Fermée par Unit 19**
  (2026-07-09). Reste ouvert : calibration des poids sur un vrai corpus,
  affichage/tri de `top_emails` côté frontend.
- ~~Classification multi-critères + taxonomie de tags par email (slice 1 de la
  vision "Coach IA")~~ — **Fermée par Unit 20** (2026-07-09). Reste ouvert :
  calibration des poids/seuils sur un vrai corpus ; les 9 autres sous-systèmes
  de la vision Coach IA (décisions, actions, calendrier, base documentaire,
  recherche hybride, chatbot source-cité, dashboard, tags Projet/Action,
  roster de participants durable) restent à prioriser séparément avec
  l'utilisateur.
- **Doc d'implémentation en attente d'une session dédiée** : les 6 fichiers
  `context/*.md` restent incohérents entre eux et avec le code (voir entrée
  Unit 13 ci-dessus) — non traité par décision utilisateur explicite cette
  session, à reprendre séparément.

## Open Questions

1. ~~**Gmail OAuth analysis integration**~~ — **Fermée par Unit 15** (2026-07-09) :
   `processor_from_tenant` bascule automatiquement sur Gmail OAuth quand
   IMAP n'est pas configuré. `process_delta` (Fast-Track) et
   `fetch_last_n_emails_for_chat` restent IMAP-only (hors périmètre d'Unit 15).
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

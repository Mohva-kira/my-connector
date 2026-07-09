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

- **Vérification visuelle du Project Hub par l'utilisateur** : `npm run dev`
  + `uvicorn`/`arq` locaux, ouvrir `/projects`, créer un projet, déclencher
  un rafraîchissement Fast-Track et confirmer que seule la carte concernée
  pulse — voir l'entrée Unit 12 ci-dessus pour ce qui a et n'a pas pu être
  vérifié dans cette session (pas de navigateur disponible).
- **Unit 4 — APScheduler / Celery batch processing** (2x/day email sync job,
  replacing BullMQ since stack is Python) — débloqué maintenant qu'Unit 11
  écrit réellement dans les tables Unit 10.
- **Unit 2 — Outlook (Microsoft Graph) OAuth2 Integration**: same pattern as Gmail, using Microsoft Identity Platform
- **Unit 5 — Email importance scoring engine** (rules + AI hybrid using existing LLM integration)

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

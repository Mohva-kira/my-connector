# Progress Tracker

Update this file after every meaningful implementation change.

## Current Phase

- **Classification par sujet (tickets bancaires) + enrichissement projet
  (Unit 63, 2026-07-28)** — demande utilisateur : les mailboxes gérées sont
  des boîtes de support pour des clients banques (BDM Togo, BMS Mali, Versus
  Bank...) dont beaucoup de sujets portent un motif de ticket entre crochets
  (`[Banque_Plateforme_Numéro]`, ex. `[BDM TOGO_BDM MOBILE_12002]`,
  séparateurs `_`/`-` mélangés, aucun format fermé) signalant une réclamation
  client, et d'autres sujets thématiques sans crochets nécessitant une
  correspondance plus tolérante que la sous-chaîne exacte. Cinq décisions
  actées avec l'utilisateur avant implémentation : (1) une clé-crochet
  détectée doit d'abord tenter un match automatique sur un projet existant,
  sinon proposer à l'utilisateur de créer un nouveau projet OU de rattacher
  manuellement à un projet existant ; (2) nouveau tag "Réclamation" déclenché
  uniquement par la détection du motif crochet (indépendant des mots-clés du
  corps/tag "Client" existant) ; (3) format générique, pas de liste fermée de
  banques/plateformes ; (4) détection "avancée" (floue) pour les sujets
  thématiques sans crochets ; (5) enrichir le contenu projet pour que le chat
  portefeuille raisonne sur les tickets (volumes, thèmes récurrents) comme un
  assistant expérimenté. Contrainte de coût conservée : aucun nouvel appel
  LLM par email, tout le déterministe est en regex/SQL.
  - `email_analyzer/subject_tickets.py` (nouveau) — `parse_subject_ticket_key`,
    regex générique `[libellé][_-][ticket]` (≥3 chiffres), tolère séparateurs
    mixtes et préfixes Re:/Fwd:/Tr: répétés. Module pur, testé à la main sur
    les 3 exemples fournis + cas limites (`[v2]`, `[1]` correctement ignorés).
  - `email_analyzer/classification.py` — `ProjectRules.subject_keys` (7e clé
    `rules_matrix`) ; nouveau signal `subject_ticket_key` (poids 45, palier
    `project_name`) ; tag `"Réclamation"` ajouté dans `derive_tags`
    indépendamment de `TAG_KEYWORDS` ; `fuzzy_similarity` (Jaccard de tokens +
    containment + `difflib.SequenceMatcher`, stdlib pur) et signal additif
    `fuzzy_theme` (poids 15, ne se déclenche que si aucun `keyword` exact n'a
    matché) pour les sujets thématiques sans motif crochet.
  - Migration `alembic/versions/015_subject_tickets.py` — `emails.ticket_label`/
    `ticket_number` (colonnes dédiées, pas JSONB : l'énumération/comptage par
    projet a besoin d'un `GROUP BY` SQL bon marché) + index
    `(project_id, ticket_label)`. `db/models.py::Email` mis à jour.
  - `analysis_tasks.py::_run_fasttrack_sync` persiste `ticket_label`/
    `ticket_number` par email. `project_mail.py::_uid_search_narrow_ids`
    accepte désormais `rules_map` et inclut les `subject_keys` connus dans le
    pré-filtre IMAP côté serveur (sans ça, un email dont seule la clé de sujet
    matche aurait été exclu avant même `score_project_relevance`).
  - Découverte de tickets (mirroring exact de la découverte de domaines,
    `_run_domain_discovery_sync`) : `analyzer.py::discover_subject_ticket_keys`
    (scan headers-only sans filtre) → `analysis_tasks.py::
    _existing_subject_key_project_map`/`_run_subject_key_discovery_sync`
    (annotation `existing_project_id` si match exact/fuzzy ≥0.8,
    `suggested_project_id` si fuzzy ≥0.45, sinon libre) →
    `run_subject_key_discovery` (arq, timeout 7200s) →
    `POST /api/brief/discover-tickets` (`api/routers/brief.py`) → même
    polling générique `GET /api/analyze/{job_id}`.
  - `api/routers/projects.py` — `subject_keys` ajouté à
    `_RULES_MATRIX_LIST_KEYS` (le merge union existant le gère gratuitement,
    y compris via `/merge`) ; nouvel endpoint
    `POST /api/projects/{id}/attach-subject-key` (lecture-fusion-écriture,
    rattachement manuel choisi par l'utilisateur) ; `EmailOut` expose
    `ticket_label`/`ticket_number`.
  - Frontend : `discoverTicketsApi.ts` + `DiscoverTicketsModal.tsx` (sibling
    de `DiscoverProjectsModal.tsx`, pas un flag — ajoute un sélecteur
    "Rattacher à un projet existant" que l'original n'a pas), bouton
    "Détecter les tickets" dans `ChatHome.tsx` à côté d'"Analyser mes emails".
  - Enrichissement chat : `llm.py::ProjectSummaryLLM.recurring_issues`
    (nouveau, `TicketPatternItem` — thèmes récurrents qualitatifs, ≥2
    occurrences, complémentaire à l'énumération déterministe) + instruction
    ajoutée à `_STRUCTURED_EXECUTIVE_SYSTEM_PROMPT`. Nouvel outil à la demande
    `tools/list_tickets_tool.py` (`GROUP BY project_id, ticket_label`, câblé
    dans `tool_executor.py` : `TOOL_SPECS`, `_tool_classes`,
    `_PROJECT_NAME_TOOLS`, `_run_tool`) ; surfacé aussi dans
    `summarize_project_tool.py`. Signal toujours actif (sans appel outil) :
    `context_service.py::_open_ticket_counts` (une requête groupée, pas de
    N+1) → `open_ticket_count` par projet → rendu dans
    `llm.py::build_portfolio_context` ("Tickets/réclamations détectés : N").
  - Vérifié end-to-end contre Postgres réel (tenant/projet/emails jetables,
    nettoyés après coup) : `list_tickets` regroupe correctement 3 emails en 2
    tickets distincts, `score_project_relevance` atteint 100 sur le motif
    crochet + `subject_keys` appris, tag `Réclamation` présent,
    `open_ticket_count` correct. `tsc --noEmit` et la phase de transformation
    Vite (480 modules) passent sans erreur (le `vite build` complet échoue
    sur un `dist/` préexistant appartenant à `root`, sans rapport avec ce
    changement — non corrigé, permissions hors périmètre).
- **Unification Agent/Conversation : tool-calling natif LLM + streaming SSE
  (Unit 62, 2026-07-27)** — demande utilisateur explicite : un seul chat,
  aussi fluide qu'un ChatGPT/Gemini, basé "sur les données persistées
  d'abord, puis sur la base de données". Renverse explicitement la
  décision Unit 57 ("coexistence durable, pas de fusion tant qu'aucun
  signal concret ne la justifie") — le signal, ici, est la demande produit
  directe. Quatre décisions actées avec l'utilisateur avant implémentation :
  (1) streaming réel token par token (SSE), pas l'ancien effet
  machine-à-écrire simulé ; (2) fallback séquentiel piloté par le LLM
  (contexte résumé `ContextService` d'abord, tools sur données brutes
  seulement si insuffisant), pas une double passe systématique ; (3)
  tool-calling natif OpenAI/Gemini remplace le `Planner` déterministe par
  mots-clés — le modèle choisit lui-même ses outils ; (4) `/api/agent/query`
  et tout `email_analyzer/agent/` retirés une fois absorbés, legacy
  `/api/chat` (IMAP, sans DB) non touché.
  - `email_analyzer/llm.py` : nouvelles `stream_portfolio_assistant_chat_
    openai/_gemini` (générateurs, jamais d'exception — événements
    `token`/`tool_calls`/`done`/`error`), `build_openai_tool_specs`,
    `_gemini_tools_param`/`_gemini_history_with_tools`/
    `_gemini_last_turn_payload`. Suppression des fonctions devenues mortes
    (confirmé par grep, aucun autre appelant) : `portfolio_assistant_chat_
    openai/_gemini` (non-streaming), `build_agent_context`,
    `build_agent_chat_system_prompt`. `project_assistant_chat_*` (legacy
    `/api/chat`) et `_normalize_chat_messages`/`_gemini_history_from_
    messages`/`_gemini_history_with_embedded_system` (encore utilisés par
    le legacy) non touchés. Paragraphe ajouté à
    `build_portfolio_chat_system_prompt` explicitant au modèle la politique
    de fallback séquentiel (décision 2).
  - `email_analyzer/services/tool_executor.py` (nouveau) — successeur de
    l'ancien `agent/executor.py` : même Session SQLAlchemy fraîche par
    appel (thread-safety), même exécution parallèle
    (`ThreadPoolExecutor`), même isolation d'erreur par outil. Nouveau :
    `TOOL_SPECS` (schémas JSON exposés au LLM) et
    `_resolve_project_id` (correspondance exacte → sous-chaîne → floue via
    `difflib`, stdlib, pas de nouvelle dépendance) qui débloque enfin
    `summarize_project`/`search_emails`/`draft_reply` (exigent un
    `project_id` que l'ancien `Planner` ne pouvait jamais résoudre depuis du
    texte libre) via un `project_name` fourni par le LLM. Corrige au passage
    un bug constaté : `list_meetings` n'était jamais réellement scopé à un
    projet dans l'ancien `Executor` (catch-all sans `project_id`) malgré un
    paramètre optionnel déjà supporté par le tool.
  - `email_analyzer/services/conversation_service.py` : `send_message`
    remplacé par `stream_message` (générateur) — boucle agentique bornée
    (`_MAX_TOOL_ROUNDS = 4`) : appelle le LLM avec `tools=TOOL_SPECS`,
    réémet les tokens en flux, exécute les `tool_calls` demandés via
    `ToolExecutor`, reboucle jusqu'à une réponse finale. Fallback timeout
    inchangé (`ContextService.quick_fallback_summary`) ; ajout d'un cas non
    prévu par l'ancien chat tout-ou-rien : si une erreur non-timeout
    survient après que des tokens ont déjà été streamés (visibles par
    l'utilisateur), ce texte partiel est conservé et persisté (avec mention
    d'interruption) plutôt que perdu silencieusement. Persistance
    (`ConversationRepository.append_turn`) inchangée : toujours un seul
    tour utilisateur + un tour assistant final, aucune nouvelle table (pas
    de `ToolExecution`/audit — la transparence des appels d'outils reste un
    signal SSE éphémère).
  - `api/routers/assistant.py` + `api/sse.py` (nouveau, `format_sse_event`)
    — `POST /conversations/{id}/chat` retourne un `StreamingResponse`
    (`text/event-stream`) au lieu d'un JSON unique. Point d'attention géré
    explicitement : le générateur est amorcé (`next()`) dans le corps
    synchrone de la route *avant* d'être passé à `StreamingResponse`, pour
    que la validation immédiate (message vide, conversation introuvable,
    fournisseur non configuré) lève encore une vraie `HTTPException` avec
    le bon status code — sinon l'erreur surviendrait après l'envoi des
    entêtes HTTP 200.
  - Suppression : `api/routers/agent.py`, tout `email_analyzer/agent/`
    (`agent.py`, `planner.py`, `executor.py`, `response_builder.py`) ;
    `api/main.py` : retrait de l'import et de
    `app.include_router(agent_router.router)`.
  - Frontend : `assistantApi.ts::sendChatMessage` remplacé par
    `streamChatMessage` (fetch + `ReadableStream` manuel, pas d'EventSource
    — ne supporte ni POST ni Authorization). `useSaasHomeChat.ts::send` :
    placeholder assistant créé au premier token (pas immédiatement),
    rempli de façon incrémentale ; indicateur "Recherche en cours…"
    réutilisant le variant `kind:"status"` déjà existant (compteur
    d'appels d'outils en cours, pas un nouveau variant). `ChatThread.tsx` :
    suppression de `TypingText` (effet machine-à-écrire simulé par
    `setInterval` sur un texte déjà complet) — rendu direct du contenu
    puisque le streaming est désormais réel.
  - Docs mises à jour dans la même unité (`ai-workflow-rules.md` §5) :
    `architecture.md` §2 (arbre de fichiers), §7 (Agent/Tools réécrit), §8
    invariant 4 (nuance streaming) ; `project-overview.md` piliers 1/4/5,
    roadmap Phase 3, sections Scope/Hors-scope ; `code-standards.md` §3
    (nouvelle convention SSE).
  - Vérifié : chaque fichier Python compile (`py_compile`) et s'importe
    réellement (`import api.main` charge les 27 routes sans erreur, y
    compris toute la chaîne `llm.py`→`conversation_service.py`→
    `tool_executor.py`) ; grep de contrôle post-suppression sans résidu
    (`email_analyzer.agent`, anciennes fonctions `llm.py`) ; `npx tsc
    --noEmit` sans erreur ; `vite build` transforme 75 modules (échec
    ensuite limité au nettoyage `dist/` préexistant appartenant à `root`,
    problème d'environnement déjà documenté, sans rapport avec le code).
  - Non vérifié (pas d'environnement Postgres/clé LLM/navigateur disponible
    dans cette session) : appel réel au tool-calling OpenAI/Gemini de bout
    en bout (schéma exact des deltas `tool_calls` en streaming, format
    `function_call`/`function_response` Gemini), résolution `project_name`
    contre de vraies données, rendu SSE réel dans le navigateur (tokens
    progressifs, indicateur d'outil, historique après rechargement) —
    recommandé à l'utilisateur avant mise en production, avec au moins un
    provider (OpenAI ou Gemini) configuré et une vraie base Postgres.

- **Multi-conversation assistant : bouton "nouvelle conversation" + panneau
  d'historique (Unit 61, 2026-07-19)** — l'assistant portefeuille n'avait
  jusqu'ici qu'un seul historique continu par tenant+utilisateur (aucune
  notion de conversation séparée, gap documenté dans project-overview.md).
  Ajout d'une table `conversations` (migration `014_conversations.py`,
  backfill : chaque couple tenant/utilisateur existant récupère une
  "Conversation 1" couvrant tout son historique — vérifié sur la base
  réelle : 34 messages d'un seul couple correctement rattachés). API :
  `GET/POST /api/assistant/conversations`,
  `GET /api/assistant/conversations/{id}/messages`,
  `POST /api/assistant/conversations/{id}/chat` remplacent (coupure nette,
  pas de double API) `GET /api/assistant/messages`/`POST /api/assistant/chat`.
  Titre auto (premier message utilisateur tronqué, pas de renommage manuel
  ni de LLM d'auto-titrage — décision minimale assumée). Frontend :
  `useSaasHomeChat.ts` gère `conversations`/`activeConversationId`/
  `newConversation()`/`switchConversation()` ; nouveau panneau
  `ConversationHistoryPanel.tsx` (slide-over, même convention que
  `DiscoverProjectsModal.tsx`) + boutons dans `ChatHome.tsx` (mode SaaS
  uniquement, mode legacy inchangé — pas d'historique DB en legacy).
  `api/routers/memory.py::extract_memories` adapté (nouvelle méthode
  `ConversationService.list_history_for_active_conversation`, sur la
  conversation la plus récemment active, pour préserver le comportement
  d'extraction existant). Vérifié bout-en-bout sur la base réelle : migration
  + backfill, `list_conversations`/`create_conversation`/garde 404
  cross-tenant via appels directs au service, puis les quatre endpoints
  HTTP réels (create → chat avec vrai appel Gemini → titre auto-défini →
  liste réordonnée par `updated_at` → 404 sur id étranger). `npx tsc
  --noEmit` sans erreur. Non vérifié : passage réel dans le navigateur
  (même contrainte d'environnement que les entrées précédentes).
- **Chat rapide + analyse auto en arrière-plan avec indicateur (Unit 60,
  2026-07-19)** — trois demandes produit : (1) le chat ne doit plus
  déclencher d'opération lourde de type analyse, il répond selon les
  données déjà disponibles ; (2) l'analyse en arrière-plan (déjà
  automatique — cron + auto-sync login) doit avoir un indicateur visible ;
  (3) le chat doit répondre en moins de 3s. Voir l'entrée Unit 60
  ci-dessous, notamment la limite assumée sur le point (3) : le budget de
  3s est garanti côté temps de réponse utilisateur (timeout serré +
  repli sur les données disponibles), pas comme une promesse sur la
  latence réelle des fournisseurs LLM tiers.
- **Pivot exécution — DraftReplyTool (Unit 59, 2026-07-19)** — dernier
  tool "core" listé par le plan de pivot, complété. Réutilise le moteur de
  templates legacy déterministe (`templates.py::generate_response_draft`,
  aucune réécriture) via un adaptateur `ProjectSummary`→dict legacy ;
  exposé via `POST /api/projects/{id}/draft-reply` (le legacy
  `POST /api/draft` reste inchangé, toujours utilisé en mode IMAP sans
  tenant). Vérifié bout-en-bout (déterministe, pas de LLM) : sélection
  correcte du template alerte/routine selon le niveau de risque persisté.
  Voir l'entrée Unit 59 ci-dessous.
- **Rattrapage documentaire : architecture.md + code-standards.md (Unit
  58, 2026-07-19)** — `architecture.md` §2/§3/§7 étaient devenus factuellement
  faux après Units 45-57 (affirmaient encore que `Memory`/`Contact`/
  `Organization`/`Objective` n'existaient pas, alors qu'ils avaient été
  créés) : corrigé. `code-standards.md` intégralement réécrit (décrivait
  Celery/React Native/mypy 100%/couverture 80%/Bandit, aucun réel dans ce
  dépôt) pour décrire les conventions réellement suivies depuis le pivot.
  Voir l'entrée Unit 58 ci-dessous.
- **Pivot exécution — Décision Agent/chat + ListMeetingsTool (Unit 57,
  2026-07-19)** — question Agent vs. chat portefeuille (ouverte depuis Unit
  51) tranchée durablement : coexistence, pas de fusion (voir raisonnement
  détaillé ci-dessous). `ListMeetingsTool` (dernier tool "core" listé par
  le plan) créé et branché dans le Planner (déclenchement conditionnel sur
  motifs "réunion"/"rendez-vous"/"agenda", même mécanique que
  `find_contact`). Vérifié avec un vrai appel Gemini. Voir l'entrée Unit 57
  ci-dessous.
- **Pivot exécution — UI Objectifs & Timeline (Unit 56, 2026-07-19)** —
  première surface frontend pour Objectives (Phase 11) et Timeline (Phase
  12), toutes deux fonctionnelles côté backend depuis Units 54-55 mais
  jamais appelées depuis `frontend/src` avant cette unité (même situation
  que Memory/Knowledge avant Unit 53). Page `/objectives` : création
  d'objectif, jalons (ajout/complétion), bascule bloqué/débloqué, flux
  timeline en lecture seule. Voir l'entrée Unit 56 ci-dessous.
- **Pivot exécution — Phase 12 (Timeline unifiée) (Unit 55, 2026-07-17)**
  — `GET /api/timeline` fusionnait déjà emails + résumés ; étendu pour
  inclure aussi les actions suggérées et les rendez-vous d'agenda
  (`Appointment`), organisé en `TimelineService`/`TimelineRepository`
  (même geste qu'Unit 52 pour le Brief). Toutes les phases du plan de
  pivot (1 à 12) ont maintenant au moins une fondation implémentée. Voir
  l'entrée Unit 55 ci-dessous.
- **Pivot exécution — Phase 11 (Objectives, fondation) (Unit 54,
  2026-07-17)** — nouvelles tables `objectives`/`milestones`
  (`ObjectiveService`/`ObjectiveRepository`), optionnellement rattachées à
  un `Project` existant, progression calculée depuis les jalons (pas de
  table "Progress" séparée), `blocking_reason` en simple colonne (pas de
  table "Blocking Issue" séparée). CRUD complet
  (`POST/GET/PATCH /api/objectives`, jalons). Vérifié bout-en-bout sur la
  vraie base Postgres. Pas encore d'UI frontend. Voir l'entrée Unit 54
  ci-dessous.
- **Pivot exécution — Planner v2 (query-aware) + première UI Memory/Knowledge
  (Unit 53, 2026-07-17)** — `Planner.plan(query)` (Phase 9) ne retourne plus
  un plan fixe identique pour toute question : `search_projects`/
  `search_memory` restent toujours inclus, `find_contact` devient
  conditionnel (motifs "contact"/"qui est"/"coordonnées"/etc.) ;
  `Executor.execute` accepte désormais `query` et en dérive un mot-clé
  déterministe (premier mot capitalisé après le premier mot de la phrase —
  pas de NLP) transmis à `search_projects`/`search_memory`/`find_contact`
  pour filtrer réellement leurs résultats plutôt que de toujours lister
  tout le tenant. Décision utilisateur explicite en amont (question posée
  en mode plan) : continuer la suite du pivot exécutif plutôt que
  `evolution-plan.md` (V2, pgvector/quotas) ou la numérotation de
  `project-overview.md`. Décision explicite de ne **pas** brancher `POST
  /api/agent/query` dans le frontend cette fois (le chat portefeuille
  injecte déjà Projects+Memory+Knowledge via `ContextService` depuis Unit
  48 — donner une UI séparée à l'Agent aurait dupliqué une capacité sans
  l'améliorer tant que son plan était fixe). À la place : première UI pour
  Memory (Phase 5) et Knowledge (Phase 6), toutes deux vérifiées
  fonctionnelles côté backend depuis leur unité de fondation mais jamais
  appelées depuis `frontend/src` avant cette unité (confirmé par
  exploration : aucune occurrence de `/api/memory`/`/api/knowledge` dans le
  frontend). Voir l'entrée Unit 53 ci-dessous.
- **Pivot exécution — Phase 10 (Daily Brief, extraction) (Unit 52,
  2026-07-17)** — `GET /api/brief` existait déjà (fonctionnel) mais avec
  toute sa logique inline dans le routeur ; extraite vers
  `BriefService`/`BriefRepository`, comportement inchangé (vérifié
  bout-en-bout, compteurs identiques à ceux seedés). Voir l'entrée Unit 52
  ci-dessous.
- **Pivot exécution — Phase 9 (AI Agent, fondation) (Unit 51, 2026-07-17)**
  — `email_analyzer/agent/` (Planner/Executor/ResponseBuilder, plan fixe et
  déterministe, pas de sélection d'outils par le LLM), nouvel endpoint
  `POST /api/agent/query`, séparé de `/api/assistant/chat` (ne le remplace
  pas). Vérifié bout-en-bout avec un vrai appel Gemini. Voir l'entrée Unit
  51 ci-dessous.
- **Pivot exécution — Phase 8 (Tool Architecture, fondation) (Unit 50,
  2026-07-17)** — package `email_analyzer/tools/` : 5 tools en lecture
  seule (`SearchProjectsTool`, `SummarizeProjectTool`, `SearchEmailsTool`,
  `SearchMemoryTool`, `FindContactTool`), chacun une enveloppe fine autour
  d'un service/requête déjà existant. Pas encore appelés par un agent (Phase
  9) ni exposés via l'API. Voir l'entrée Unit 50 ci-dessous.
- **Vérification bout-en-bout réelle des Units 45-48 (Unit 49, 2026-07-17)**
  — migrations 011/012 appliquées sur la vraie base Postgres locale
  (`connector`), pipeline complet exercé avec de vrais appels Gemini
  (extraction mémoire + chat portefeuille) via l'app FastAPI réelle
  (`TestClient`). Tout fonctionne, y compris l'effet observable de la
  mémoire sur une vraie réponse LLM. Voir l'entrée Unit 49 ci-dessous.
- **Pivot exécution — Phase 7 (Context Engine, fondation) (Unit 48,
  2026-07-17)** — nouveau `ContextService` qui assemble le prompt système du
  chat portefeuille à partir de Projects + Memory (Unit 46) + Knowledge
  (Unit 47) ; `ConversationService` ne construit plus son contexte lui-même.
  Première fois que Memory/Knowledge influencent réellement une réponse de
  l'assistant. Voir l'entrée Unit 48 ci-dessous.
- **Pivot exécution — Phase 6 (Knowledge Layer, fondation) (Unit 47,
  2026-07-17)** — nouvelles tables `organizations`/`contacts`/
  `project_contact_links` + `KnowledgeService`/`KnowledgeRepository`,
  alimentées sans nouvel appel LLM (relit `ProjectSummary.structured_content`
  déjà produit par le Fast-Track), endpoints `GET /api/knowledge/contacts`,
  `GET /api/knowledge/organizations`, `POST /api/knowledge/projects/{id}/sync`.
  Voir l'entrée Unit 47 ci-dessous.
- **Pivot exécution — Phase 5 (Memory Engine, fondation) (Unit 46,
  2026-07-17)** — nouvelle table `memories` + `MemoryService`/
  `MemoryRepository`/extraction LLM structurée, endpoints
  `GET/DELETE /api/memory`, `POST /api/memory/extract`. Pas encore branché
  automatiquement dans le flux de chat. Voir l'entrée Unit 46 ci-dessous.
- **Pivot exécution — Phase 4 (Conversation layer, fondation) (Unit 45,
  2026-07-17)** — première brique de code du pivot stratégique (Unit 44) :
  extraction de la logique du chat portefeuille (`api/routers/assistant.py`)
  vers `ConversationService`/`ConversationRepository`, sans nouvelle table
  ni changement de comportement. Voir l'entrée Unit 45 ci-dessous.
- **Pivot stratégique documentaire : Email Assistant → Personal Executive
  AI Assistant (Unit 44, 2026-07-17)** — `context/project-overview.md` et
  `context/architecture.md` réécrits pour refléter la nouvelle vision long
  terme (Memory Engine, Knowledge Graph, Context Engine, Agent/Tools),
  clairement distinguée de l'implémentation actuelle. Voir l'entrée Unit 44
  ci-dessous.
- Refonte "chat-first" de l'accueil, unifiée legacy + SaaS (Units 33-42,
  2026-07-16) — voir Unit 24 pour la phase précédente ("assistante
  proactive", Brief séparé) qu'elle remplace. `pages/ChatHome.tsx` est
  désormais l'unique accueil (`/`) dans les deux modes ; thème clair/sombre
  réel ajouté.

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

- **Unit 33 — Jetons de couleur convertis en variables CSS (préalable au
  thème sombre)** (2026-07-16) :
  - Contexte : refonte de l'accueil en interface "chat-first" demandée par
    l'utilisateur (voir plan `refait-le-design-de-lucky-comet`), avec un
    vrai toggle clair/sombre global. `tailwind.config.js` définissait les
    ~20 jetons de couleur en hex/rgba statiques — aucune variable CSS,
    aucune variante `dark:`, confirmé par exploration (`context/ui-context.md`
    affirmait explicitement "pas de mode sombre").
  - `frontend/tailwind.config.js` : chaque jeton référence désormais une
    variable CSS `--color-<slug>-rgb` (triplet "R, G, B") via un helper
    `withOpacity(variableName, defaultOpacity)` qui reproduit le mécanisme
    natif de Tailwind pour les modificateurs d'opacité (`bg-danger/10`,
    etc.) — impossible avec un simple `var(--x)` puisque Tailwind ne peut
    pas réécrire l'alpha d'une couleur déjà résolue par le navigateur.
    Mêmes noms de jetons qu'avant → zéro changement dans les composants
    pour ce refactor seul.
  - Piège rencontré et corrigé : pour `ai-glow`/`accent-soft`/`fasttrack-bg`
    (halos toujours semi-transparents par défaut, ex. `bg-ai-glow` seul
    doit valoir 0.08 d'opacité), Tailwind injecte automatiquement sa
    propre variable d'opacité (`--tw-bg-opacity`, repart de `1`) dès
    qu'une couleur est une fonction — `opacityValue` n'est donc jamais
    `undefined`, le `defaultOpacity` de `withOpacity` ne se déclenchait
    jamais (vérifié en inspectant le CSS généré : `bg-ai-glow` sortait à
    opacité 1 au lieu de 0.08). Corrigé en sortant ces 3 jetons du helper
    et en les définissant comme chaînes statiques
    `rgba(var(--color-x-rgb), 0.08)` — un seul usage avec modificateur
    existait dans tout le repo (`bg-ai-glow/40` dans
    `ConversationalAssistant.tsx`, bandeau "Réponse rapide"), changé en
    valeur arbitraire `bg-[rgba(var(--color-ai-glow-rgb),0.4)]` pour
    préserver l'opacité voulue.
  - `frontend/src/index.css` : blocs `:root` (palette claire actuelle,
    portage à l'identique) et `[data-theme="dark"]` (nouvelle palette
    sombre navy/indigo inspirée de l'image de référence fournie par
    l'utilisateur — fond `#0B0F1A`/`#131A2C`, texte clair, accent/ai
    éclaircis vers un bleu/indigo plus lumineux) définissant les 24
    variables `--color-*-rgb`. `danger`/`priority-*` restent dans la même
    famille de teinte (rouge) dans les deux thèmes — règle
    `ui-context.md` (réservé à l'urgence réelle) préservée.
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` (vers un
    `--outDir` hors du dépôt pour contourner l'`EACCES` pré-existant sur
    `dist/`, déjà documenté dans les unités précédentes) transforme les 59
    modules sans erreur ; inspection du CSS généré confirme
    `.bg-danger\/10{background-color:rgba(var(--color-danger-rgb),.1)}`
    (modificateurs toujours fonctionnels), `.bg-ai-glow{background-color:
    rgba(var(--color-ai-glow-rgb),.08)}` (défaut correct après correctif),
    et présence des deux blocs `:root`/`[data-theme=dark]` avec les bonnes
    valeurs.
  - Non vérifié : rendu visuel réel en navigateur (pas de Chromium
    fonctionnel dans cette session, comme les unités précédentes) — le
    theme toggle lui-même arrive à l'Unit 34, `[data-theme="dark"]` n'est
    encore posé sur aucun élément.

- **Unit 34 — `ThemeProvider`/`ThemeToggle` + montage global** (2026-07-16) :
  - Suite de l'Unit 33 (tokens couleur en variables CSS). Nouveau
    `frontend/src/theme/ThemeProvider.tsx` : contexte React
    (`useTheme() -> {theme, setTheme, toggleTheme}`), thème initial lu
    dans `localStorage["email_analyzer_theme"]`, sinon
    `window.matchMedia("(prefers-color-scheme: dark)")` ; pose
    `document.documentElement.dataset.theme` (attribut `data-theme` lu
    par les blocs CSS de l'Unit 33) et persiste à chaque changement.
    Nouveau `frontend/src/components/ThemeToggle.tsx` : bouton rond
    soleil/lune (icônes SVG inline, même style que les autres boutons
    d'icône du repo), pas encore monté dans une page — ce sera fait dans
    `AppShell.tsx` à l'Unit 35. `frontend/src/main.tsx` : `<App/>` enveloppé
    dans `<ThemeProvider>`.
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` (outDir hors
    dépôt) transforme les 60 modules (2 nouveaux fichiers) sans erreur.
  - Non vérifié : bascule réelle du toggle en navigateur (pas de Chromium
    dans cette session) — le bouton n'est monté nulle part encore
    (prévu Unit 35), donc pas testable manuellement avant cette étape de
    toute façon. Recommandé à l'utilisateur, une fois l'Unit 35 livrée :
    `npm run dev`, cliquer le toggle, vérifier `document.documentElement`
    dans les devtools (`data-theme` change), recharger la page (le choix
    doit persister), et effacer `localStorage` pour vérifier le
    fallback sur `prefers-color-scheme`.

- **Unit 35 — `AppShell.tsx` prêt pour le mode legacy + montage `ThemeToggle`**
  (2026-07-16) :
  - Suite de l'Unit 34. `frontend/src/components/AppShell.tsx` : prop
    `me: Me` → `me: Me | null` (les 7 appelants actuels — Brief, Projets,
    Agenda, Actions, Paramètres, Assistant, détail projet — passent tous
    un `Me` non-null, donc compatibles sans modification, `Me` étant
    assignable à `Me | null`) ; nouvelle prop optionnelle
    `saasEnabled = true` (défaut `true` pour ne pas devoir toucher les 7
    appelants existants dans cette unité — seul le futur `ChatHome`,
    Unit 36+, passera `false`). Item de nav "Brief" renommé "Accueil".
    Nav filtrée : liste complète en SaaS, **accueil seul** en legacy
    puisque `/projects`, `/agenda`, `/actions`, `/settings` redirigent
    déjà vers `/` sans SaaS (vérifié dans les gardes `Navigate` de ces
    pages) — pas de liens morts affichés. `SidebarSearch` (recherche en
    langage naturel → `/assistant?q=…`) et les items "Bientôt disponible"
    masqués en mode legacy pour la même raison. `ThemeToggle` (Unit 34)
    monté dans le pied de la sidebar (desktop) et à côté de l'email en
    haut (mobile), sans dupliquer l'email dans les deux vues.
  - Restylage volontairement conservateur : la mise en page originale
    (largeur, labels visibles, recherche pleine largeur) est conservée
    pour ne pas casser les pages secondaires (Projets/Agenda/Actions/
    Paramètres) qui ne sont pas la cible de la refonte "chat-first" —
    seul le futur `ChatHome` (accueil) portera le look inspiré de l'image
    de référence.
  - Vérifié : `npx tsc --noEmit` sans erreur (widening de `me` compatible
    avec les 7 appelants existants sans les modifier) ; `vite build`
    transforme les 61 modules sans erreur.
  - Non vérifié : rendu visuel réel en navigateur (pas de Chromium dans
    cette session) — recommandé à l'utilisateur de lancer `npm run dev`
    et visiter `/projects` (rail complet inchangé) une fois l'Unit 36
    livrée et `ChatHome` montée avec `saasEnabled=false`, pour comparer
    visuellement le rail réduit à "Accueil" en mode legacy.

- **Unit 36 — Squelette statique du shell de chat (`ChatHome`)** (2026-07-16) :
  - Nouveaux fichiers : `frontend/src/components/chat/messageTypes.ts`
    (`HomeMessage`, pour l'instant uniquement `kind: "text"` — les variantes
    `status`/`empty-report`/`analysis` seront ajoutées aux unités qui en ont
    réellement besoin, pas par anticipation) ; `components/chat/ChatThread.tsx`
    (liste de bulles + zone de saisie, style porté à l'identique de
    `ConversationalAssistant.tsx` : bulles `rounded-2xl`, utilisateur
    `bg-accent-primary text-white`, assistant `bg-bg-tertiary` bordé,
    `TypingDots`, auto-scroll) ; `pages/ChatHome.tsx` (salutation centrée +
    avatar "orbe" en dégradé `ai-primary`→`ai-secondary` avec halo
    pulsant — nouveau `@keyframes orb-glow`/`.animate-orb-glow` dans
    `index.css`, pas un asset copié de l'image de référence — champ de
    saisie, 3 cartes d'action rapide dont le contenu dépend de `mode`
    ("legacy" vs "saas"), le tout enveloppé dans `AppShell` (Unit 35).
    Messages en `useState` local pour l'instant (aucun appel réseau).
  - Écart assumé par rapport au découpage initial du plan : `ChatHome`
    n'est **pas encore montée** sur la route `/` dans cette unité. Le plan
    prévoyait un montage "temporaire" derrière `HomePage.tsx` pour une
    vérification visuelle manuelle, mais cette session n'a pas de
    Chromium fonctionnel (contrainte documentée depuis les unités
    précédentes) — un montage sans câblage réel n'aurait donc apporté
    aucune vérification supplémentaire, tout en obligeant à retirer
    prématurément l'ancien JSX/état de `HomePage.tsx` (qui doit encore
    fonctionner en attendant l'Unit 37). Le montage réel sur `/`
    (remplacement de l'ancien formulaire ET de la redirection vers
    `BriefPage`) se fera à l'Unit 37, en même temps que le branchement du
    hook de données legacy, pour éviter tout état intermédiaire cassé.
  - Vérifié : `npx tsc --noEmit` sans erreur sur les 3 nouveaux fichiers ;
    `vite build` toujours propre (61 modules — les 3 nouveaux fichiers ne
    sont pas encore dans le graphe de dépendances depuis `main.tsx`, donc
    absents du bundle tant qu'ils ne sont montés nulle part, ce qui est
    attendu).
  - Non vérifié : rendu visuel réel (voir écart ci-dessus) — à faire une
    fois l'Unit 37 livrée.

- **Unit 37+38+39 — Câblage backend legacy, carte d'analyse inline, retrait
  du formulaire classique** (2026-07-16) :
  - Livrées ensemble (au lieu des 3 unités séparées prévues au plan) : le
    hook legacy produit des messages de type `analysis`/`empty-report`,
    qui n'ont de sens que rendus par une carte inline — impossible de
    vérifier l'un sans l'autre, et laisser l'ancien JSX de `HomePage.tsx`
    vivre à côté du nouveau hook aurait dupliqué toute la logique
    d'analyse (deux implémentations du même appel `/api/analyze` en
    parallèle).
  - Nouveau `frontend/src/analysisTypes.ts` : `ProjectAnalysis`,
    `DraftResult`, `riskBadgeClass`, `buildChecklist` extraits tels quels
    de l'ancien `AnalysisDashboard.tsx`.
  - Nouveau `frontend/src/components/chat/AnalysisCard.tsx` : même contrat
    de données que l'ancien composant (projectName, analysis,
    onRequestDraft?, onSelectThread?), restylé largeur-bulle, sections
    secondaires (sentiment/tech/checklist) repliées dans un `<details>`.
  - `messageTypes.ts` complété avec les variantes `status` (progression +
    barre), `empty-report`, `error`, et `analysis` (+ `allowThreadDrilldown`,
    même règle qu'avant : uniquement sur le résultat d'une analyse sans
    filtre). `ChatThread.tsx` sait désormais rendre les 5 variantes.
  - Nouveau `frontend/src/hooks/useLegacyHomeChat.ts` : porte
    `launchAnalysis`/`pollAnalysis`/`pickProjectBlock`/`NO_FILTER_RESULT_KEY`
    de l'ancien `HomePage.tsx` et le flux `/api/chat` de l'ancien
    `ConversationalAssistant.tsx`, tels quels côté endpoints/payloads.
    Premier message envoyé sans carte d'analyse active → lance
    `/api/analyze` (filtre = texte du message) ; une fois une carte
    présente, les messages suivants continuent via `/api/chat` sur ce
    projet. Relances (bouton "élargir" dans la carte `empty-report`, clic
    sur un fil détecté) **empilent** une nouvelle carte sous les
    précédentes (décision utilisateur actée dans le plan) plutôt que de
    remplacer le panneau comme avant. La mécanique de progression pendant
    le chargement (ticks du job, `applyAnalyzeTickToStore`/`useSyncStore`)
    est conservée à l'identique ; le rendu "partiel" pendant le chargement
    (bloc d'analyse encore incomplet) est conservé aussi, mis à jour en
    place (même id de message) plutôt que d'empiler un message par tick.
  - Non porté délibérément : la barre "Réponse rapide" de
    `ConversationalAssistant.tsx` (génération de brouillon simulée par
    `setTimeout` + template statique, déjà signalée comme non réelle par
    l'exploration préalable) — le vrai bouton "Générer brouillon de
    réponse" de la carte d'analyse (`/api/draft`, réel) couvre le même
    besoin sans logique fictive.
  - `frontend/src/pages/ChatHome.tsx` : en mode legacy, câblé sur
    `useLegacyHomeChat` ; panneau "Outils" repliable (période, LLM,
    modèles avancés — reprise directe des contrôles de l'ancien
    formulaire) ; 2 cartes d'action rapide réellement fonctionnelles
    ("Analyser tous mes emails" lance `send("")` sans filtre, "Choisir une
    période" ouvre le panneau Outils) — une 3ᵉ carte "Analyser un projet"
    a été délibérément omise plutôt que rendue décorative sans action
    réelle derrière (cf. `ai-workflow-rules.md`, pas d'affordance
    simulée). Mode SaaS : toujours en state local temporaire (inchangé
    depuis l'Unit 36), câblage réel à l'Unit 40.
  - `frontend/src/pages/HomePage.tsx` : réduit aux gardes de session
    (chargement du compte, redirection `/login`) puis rendu unique de
    `<ChatHome mode={saasEnabled && me ? "saas" : "legacy"} .../>` — tout
    l'ancien JSX (formulaire, panneau de résultat, bascule
    classique/conversationnel) et son state (plus de 15 `useState`)
    supprimés. La branche `if (saasEnabled && me) return <BriefPage .../>`
    a aussi disparu : `BriefPage.tsx`/`PortfolioAssistant.tsx` n'ont donc
    plus aucun appelant (orphelins assumés jusqu'à l'Unit 40/41, qui les
    remplacent puis les suppriment — pas supprimés maintenant pour ne pas
    perdre le mode SaaS avant d'avoir sa nouvelle implémentation).
  - Fichiers supprimés : `AnalysisDashboard.tsx`, `ConversationalAssistant.tsx`
    (plus aucune référence après grep, vérifié avant suppression).
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` (outDir hors
    dépôt) transforme 61 modules sans erreur.
  - Non vérifié : exécution réelle contre une vraie boîte IMAP — aucun
    `.env` avec identifiants IMAP n'existe dans cet environnement (aucun
    fichier `.env*` trouvé sous `services/email_analyzer/`), et pas de
    Chromium fonctionnel pour un test navigateur (contrainte déjà
    documentée). Recommandé à l'utilisateur : renseigner `.env`
    (`IMAP_USER`/`IMAP_PASSWORD`), lancer `uvicorn` + `npm run dev`,
    visiter `/`, taper un nom de projet ou envoyer vide via la carte
    "Analyser tous mes emails", vérifier la progression puis la carte
    d'analyse, le bouton brouillon (`/api/draft`) et l'enchaînement en
    chat une fois la carte affichée.

- **Unit 40 — Câblage SaaS de l'accueil chat-first** (2026-07-16) :
  - Nouveau `frontend/src/hooks/useSaasHomeChat.ts` : porte le chargement
    (`GET /api/assistant/messages`, repli sur un message d'accueil si vide
    ou en erreur) et l'envoi (`POST /api/assistant/chat`) de l'ancien
    `PortfolioAssistant.tsx`, en `HomeMessage[]` (uniquement `kind:"text"`,
    ce endpoint n'a pas de notion de carte d'analyse). Support de
    `initialQuestion` (ex-`?q=` de `/assistant`) porté à l'identique,
    envoyé une seule fois une fois l'historique chargé.
  - Nouveau `frontend/src/hooks/useBrief.ts` : porte `GET /api/brief`,
    `greetingName`/`formatSince` de l'ancien `BriefPage.tsx::loadBrief`.
  - Piège évité : les deux nouveaux hooks sont **toujours instanciés**
    dans `ChatHome.tsx` quel que soit le mode (les hooks React ne peuvent
    pas être conditionnels), ce qui aurait déclenché des appels
    `/api/brief`/`/api/assistant/*` (503 sans SaaS,
    `_require_saas()`/`assert_saas_enabled`) même en mode legacy. Les deux
    hooks acceptent donc un paramètre `enabled: boolean` (`mode ===
    "saas"`) qui court-circuite tout fetch/POST quand `false`.
  - `frontend/src/pages/ChatHome.tsx` : branche SaaS complète — salutation
    `Bonjour {greetingName(me.email)}` + "Voici ce qui a changé depuis
    {formatSince(...)}", 3 cartes d'action réellement fonctionnelles
    ("Analyser mes emails" ouvre `DiscoverProjectsModal`, réutilisé
    inchangé, avec le même callback `handleProjectsProcessed` que l'ancien
    `BriefPage` ; "Actions en attente"/"Projets à risque" affichent le
    compteur du Brief en badge — navigation vers `/actions`/`/projects`
    pas encore ajoutée, badges informatifs seulement pour cette unité).
    Lecture de `?q=` via `useSearchParams` (react-router), effacé après
    envoi.
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` transforme 65
    modules sans erreur.
  - Non vérifié : exécution réelle contre un tenant SaaS (pas de Postgres
    ni de session JWT réelle dans cet environnement) — recommandé à
    l'utilisateur de se connecter à un tenant existant, visiter `/`,
    vérifier le chargement de l'historique persistant, l'envoi d'un
    message, le badge des cartes d'action et l'ouverture du modal
    "Analyser mes emails".

- **Unit 41 — Retrait de l'ancienne page Brief/assistant plein écran**
  (2026-07-16) :
  - Fichiers supprimés (plus aucun appelant depuis l'Unit 40 :
    `HomePage.tsx` rend `ChatHome` dans les deux modes depuis l'Unit
    37+38+39, `App.tsx` routait encore `/assistant` vers `AssistantPage`) :
    `pages/BriefPage.tsx`, `PortfolioAssistant.tsx`, `pages/AssistantPage.tsx`.
  - `App.tsx` : route `/assistant` remplacée par un composant
    `AssistantRedirect` (`useLocation` + `Navigate`) qui redirige vers `/`
    en conservant `?q=` — pour les anciens liens/favoris, puisque `/assistant`
    n'existe plus en tant que page.
  - `components/AppShell.tsx` : `SidebarSearch` repointée de
    `/assistant?q=…` vers `/?q=…`, lu par `ChatHome` (mode SaaS,
    `useSaasHomeChat`) au montage. L'item de nav "Assistant" avait déjà
    disparu du rail à l'Unit 35 (accueil = assistant portefeuille
    persistant, plus besoin d'une page séparée).
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` transforme 63
    modules sans erreur ; grep sur les 3 noms de fichiers supprimés et sur
    tout import résiduel → aucune référence restante.
  - Non vérifié : navigation réelle d'un ancien lien `/assistant?q=…` en
    navigateur (pas de Chromium dans cette session) — recommandé à
    l'utilisateur de vérifier que la redirection atterrit bien sur `/`
    avec la question envoyée automatiquement.

- **Unit 42 — Synchronisation de la documentation (refonte chat-first)**
  (2026-07-16) :
  - `context/ui-context.md` : retiré l'affirmation "thème unique, pas de
    variante sombre" (§1) — documenté le mécanisme réel (variables CSS
    `--color-*-rgb`, blocs `:root`/`[data-theme="dark"]`, `ThemeProvider`,
    aucune classe `dark:`), palette sombre (valeurs clés), et la nouvelle
    disposition de l'accueil/rail (§4) : item "Assistant" retiré, rail
    réduit à l'accueil en mode legacy, `ChatHome`/`ChatThread`/
    `AnalysisCard` comme nouvelle structure de page.
  - `context/architecture.md` : nouvel addendum daté qui remplace
    explicitement l'affirmation de l'Unit 24 ("le mode legacy garde
    l'ancien comportement inchangé", devenue fausse) — liste les fichiers
    retirés/ajoutés et rappelle que les contrats d'API n'ont pas changé.
  - `context/project-overview.md` : nouvel addendum daté qui remplace la
    description du split Brief/formulaire legacy par l'accueil unique.
  - `context/progress-tracker.md` : "Current Phase" mis à jour pour
    pointer vers cette refonte plutôt que vers l'Unit 24.
  - Vérifié : relecture des 3 fichiers modifiés, cohérence avec le code
    réel (noms de fichiers, chemins) confirmée par grep avant rédaction.
  - Ceci clôt le plan `refait-le-design-de-lucky-comet` (Units 33-42).
    Écarts assumés par rapport au plan initial, résumés ici pour
    référence : (1) la 3ᵉ carte d'action rapide "Analyser un projet" en
    mode legacy a été omise (pas d'action réelle à lui donner sans
    dupliquer les 2 autres) ; (2) les badges "Actions en attente"/"Projets
    à risque" en mode SaaS affichent un compteur mais ne naviguent pas
    encore vers `/actions`/`/projects` ; (3) le rendu "partiel" pendant le
    chargement d'une analyse legacy est conservé mais simplifié (mise à
    jour en place d'une seule carte, pas de striation par tick) ; (4) la
    barre "Réponse rapide" (brouillon simulé) de l'ancien
    `ConversationalAssistant.tsx` n'a pas été portée, déjà signalée comme
    non réelle avant la refonte. Aucune de ces unités n'a pu être vérifiée
    en navigateur réel ni contre un vrai tenant SaaS/boîte IMAP (pas de
    Chromium fonctionnel ni de `.env`/Postgres configurés dans cette
    session) — recommandé à l'utilisateur de tester manuellement les
    parcours legacy et SaaS avant de considérer le chantier terminé.

- **Unit 43 — Parallélisation de `generate_intelligent_summary` (fix
  timeout `run_analysis_saas`)** (2026-07-17) :
  - Contexte : job arq `run_analysis_saas` observé en échec
    (`TimeoutError`) après avoir atteint le `job_timeout` de 1800s
    (`analysis_tasks.py:880`). Cause identifiée : `generate_intelligent_summary`
    (`project_mail.py`) traitait les projets d'un tenant en boucle
    strictement séquentielle, avec 1-2 appels LLM bloquants par projet
    (jusqu'à ~120s au pire cas avec retries Gemini) — un tenant à plusieurs
    projets dépassait donc facilement les 1800s, la charge étant
    entièrement additive.
  - Fix choisi (option "paralléliser les appels LLM", décision utilisateur
    explicite parmi 4 options proposées — pas une décision autonome) : le
    corps de boucle par projet a été extrait en une méthode
    `_generate_project_summary` (retourne `(project_name, summary)`),
    exécutée via `ThreadPoolExecutor(max_workers=min(5, nb_projets))` dans
    `generate_intelligent_summary`. `self.record_timing` (incréments
    `defaultdict` non-atomiques) a été protégé par un `threading.Lock`
    (`self._timing_lock`, posé dans `__init__`) pour éviter une perte de
    mises à jour sous concurrence.
  - Vérifié avant modification (recherche dédiée) : les 4 fonctions LLM
    (`generate_openai_assistant_summary`, `generate_gemini_assistant_summary`,
    `extract_structured_project_summary_openai/gemini`, `llm.py`)
    instancient un client local par appel — pas d'état module partagé
    dangereux ; `self.ai_analyzer` (singleton partagé, `ai_intelligent.py`)
    ne mute aucun état pendant ces méthodes ; `self.mail` (connexion IMAP)
    n'est pas touché dans cette boucle (déjà terminée avant la phase de
    résumé).
  - Vérifié après modification : `python3 -m py_compile` OK ; test manuel
    (script ad hoc, 6 projets factices, `assistant_provider=none`) confirme
    que les 6 clés attendues sont bien produites et que
    `step_counts["ia_sentiment"] == 6` exactement (aucune perte
    d'incrément sous les threads concurrents, donc le lock fonctionne).
    Non vérifié : exécution réelle contre un tenant SaaS multi-projets via
    arq (pas de `.env`/Postgres/IMAP configurés dans cette session) —
    recommandé à l'utilisateur de relancer une analyse SaaS réelle pour
    confirmer que le job termine désormais sous 1800s.
  - Non traité (hors scope de cette décision) : la phase de fetch IMAP
    reste séquentielle (un `fetch` par email, `project_mail.py:520-562`) —
    si des timeouts réapparaissent sur des tenants à très large fenêtre
    `days`, c'est le prochain goulot à adresser (options déjà notées :
    plafonner `days`/nb projets, ou paralléliser le fetch avec un pool de
    connexions IMAP dédié — non trivial, connexion IMAP non thread-safe en
    l'état).

- **Unit 44 — Pivot stratégique documentaire : "Personal Executive AI
  Assistant" (2026-07-17)** :
  - Contexte : l'utilisateur a défini une nouvelle vision long terme
    (document "Strategic Pivot Plan") repositionnant my-connector d'un
    "Email AI Assistant" vers une IA Exécutive Personnelle conversationnelle
    avec mémoire persistante, knowledge graph, context engine et
    architecture agent/tools. Reprend directement le chantier laissé en
    attente à l'entrée Unit 13 ("Doc d'implémentation en attente d'une
    session dédiée").
  - `context/project-overview.md` : réécriture complète. Retire le corps
    obsolète (Celery, React Native, cache vectoriel device 30 jours,
    "Zero-Touch Project Creation") et les deux notes datées Unit 24 /
    Units 33-41 (contenu absorbé). Nouvelle structure : Mission,
    positionnement avant/après, philosophie "conversation-first", 5 piliers
    produit (Conversation/Mémoire/Knowledge Graph/Agent/Tools) chacun
    annoté de son état réel actuel entre parenthèses, sources d'information
    par phase, scope actuel/hors-scope basés sur le vrai stack, roadmap IA
    en 4 phases, principes de développement, vision finale.
  - `context/architecture.md` : réécriture complète. Retire le corps
    français obsolète (stack Celery/RabbitMQ/Orama-Voy/SQLCipher,
    arborescence `apps/web`+`apps/mobile` fictive, invariants "Zero
    Clear-Text"/"30-Day Device Cache" jamais implémentés) et les deux notes
    datées. Nouvelles sections : stack technique réelle, arborescence réelle
    (`services/email_analyzer/{api,email_analyzer,frontend}`), modèle de
    données réel (tables SQLAlchemy listées une à une, avec mention
    explicite que `Memory`/`Fact`/`Entity`/`KnowledgeGraphEdge` etc.
    n'existent pas), architecture planifiée court/moyen terme (basée sur
    `evolution-plan.md`/`rfc-email-pipeline-v2.md` : pgvector, sync
    incrémental réel, threading JWZ), vision long terme Executive Assistant
    (diagramme Memory Engine/Knowledge Graph/Context Engine/Agent-Tools,
    marqué non implémenté), invariants réduits aux 4 qui s'appliquent
    réellement au code actuel.
  - `context/ai-workflow-rules.md` : mise à jour ciblée (pas de réécriture
    complète) — références Celery→`arq`, retrait React Native, nouvelle
    règle de scoping "conversation-first" (reprise des 5 critères du plan
    utilisateur), correction de l'item checklist "Data Encryption" qui
    affirmait à tort une garantie de chiffrement au repos non implémentée
    (reformulé en "Log Masking" + renvoi vers le statut réel documenté dans
    `architecture.md`).
  - Décision explicite (validée avec l'utilisateur avant exécution) :
    `context/code-standards.md` **non touché** dans cette session (hors
    scope explicite du pivot, malgré ses propres références obsolètes à
    Celery/React Native) — voir entrée "Next Up" ci-dessous pour le suivi.
    `context/ui-context.md` (déjà à jour depuis Units 33-42),
    `context/evolution-plan.md` et `context/rfc-email-pipeline-v2.md`
    également non touchés (déjà alignés sur le vrai stack, utilisés comme
    source de vérité pour la section "architecture planifiée" du nouvel
    `architecture.md`).
  - Vérifié : relecture manuelle de chaque fait technique cité (noms de
    tables, endpoints, chemins de fichiers) contre l'inventaire réel du
    code (`email_analyzer/db/models.py`, `api/routers/*.py`,
    `frontend/src/` — obtenu via exploration du code dans cette session).
    Aucun code modifié ; aucune fonctionnalité de la vision long terme
    (Memory Engine, Knowledge Graph, Agent/Tools, briefing quotidien
    complet) n'est implémentée par cette entrée — uniquement documentée
    comme vision/roadmap, avec état actuel clairement distingué.
  - Non vérifié : aucune vérification automatisée possible pour un
    changement purement documentaire (pas de build/test applicable).

- **Unit 45 — Conversation layer, fondation (2026-07-17)** :
  - Contexte : première brique de code de la roadmap Unit 44 (Phase 4 du
    plan de pivot — "créer la fondation de la couche conversation, sans
    raisonnement avancé"). Recherche préalable (inventaire complet du code
    via exploration dédiée) : deux chats existent déjà (legacy éphémère
    `/api/chat` scopé projet, et assistant portefeuille persistant
    `/api/assistant/chat` sur `AssistantMessage`) et aucune notion de
    `Conversation`/session séparée. Décision : **étendre** l'existant
    (`AssistantMessage`) plutôt que créer une nouvelle table ou dupliquer la
    logique — pas de `Conversation`/session multiple ajoutée dans cette
    unité (un seul historique continu par tenant+utilisateur, comme
    documenté sur le modèle existant ; ajouter du multi-conversation
    maintenant aurait été de la scope creep non demandée).
  - Ajout `email_analyzer/repositories/conversation_repository.py` :
    `ConversationRepository` — seul point d'accès SQL à `AssistantMessage`
    (`list_messages`, `append_turn`).
  - Ajout `email_analyzer/services/conversation_service.py` :
    `ConversationService` — orchestration (sélection fournisseur
    OpenAI/Gemini, construction du contexte portefeuille via
    `llm.build_portfolio_context`, appel LLM, persistance), avec logging
    explicite start/done/erreur (`logger.info`/`logger.error`). Reprend telle
    quelle la logique auparavant inline dans le routeur (`_select_provider`,
    `_pending_action_descriptions`, `_portfolio_projects_context`) — aucun
    changement de comportement.
  - Refactor `api/routers/assistant.py` : devient un contrôleur HTTP fin
    (auth, validation, mapping HTTP) qui délègue entièrement à
    `ConversationService`. Suit l'ordre Model → Repository → Service → API
    prescrit par les règles de développement.
  - Suit le pattern `Service ↓ Repository ↓ Model` demandé (pas
    d'abstraction supplémentaire, pas d'héritage).
  - Vérifié : syntaxe (`ast.parse`) des 3 fichiers modifiés/créés ; import
    réel des modules (`ConversationService`, `ConversationRepository`,
    routeur `assistant`) dans le venv du projet (`myenv`) ; `api.main:app`
    s'importe et expose toujours ses 22 routes (aucune route perdue/cassée).
  - Non vérifié : appel HTTP réel (`/api/assistant/chat`) contre une base
    Postgres + clé LLM réelle (pas d'environnement disponible dans cette
    session) — le comportement métier n'a pas changé (code déplacé, pas
    réécrit), le risque de régression est donc limité au déplacement lui-même.
  - Prochaine étape naturelle (Phase 5 du plan, non commencée) : Memory
    Engine (`MemoryService`/`MemoryRepository`/extraction de faits) —
    nécessitera de décider si la mémoire s'accroche à `ConversationService`
    existant ou reste un pilier séparé consommé par lui.

- **Unit 46 — Memory Engine, fondation (2026-07-17)** :
  - Contexte : suite directe d'Unit 45, Phase 5 du plan de pivot ("créer
    MemoryService/MemoryRepository/MemoryExtractor/MemoryStore/
    MemoryRetriever/MemoryScoring, PostgreSQL, pas de vector DB, rester
    simple/compréhensible"). Recherche préalable confirmée par l'inventaire
    d'Unit 44/45 : aucune notion de mémoire structurée n'existait, seul
    `AssistantMessage` (historique brut) s'en approchait.
  - Décision d'architecture (documentée pour transparence) : les 6 modules
    "suggérés" par le plan (Extractor/Store/Retriever/Scoring/Service/
    Repository) sont **collapsés en 2 classes** — `MemoryService`
    (orchestration + extraction + scoring simple = tri par confidence/
    récence) et `MemoryRepository` (accès SQL, incluant la déduplication).
    Créer 6 classes pour quelques méthodes chacune aurait été une
    abstraction prématurée, contraire à `ai-workflow-rules.md` ("avoid
    unnecessary abstractions") — à scinder plus tard seulement si une
    responsabilité grossit réellement.
  - `email_analyzer/db/models.py` : nouveau modèle `Memory` (`memories`) +
    enums `MemoryCategory` (preference/fact/context) et `MemorySource`
    (conversation pour l'instant — pas d'extraction depuis les emails dans
    cette unité). Colonnes : `tenant_id`, `user_id`, `content`, `category`,
    `source`, `confidence` (float), `created_at`/`updated_at`. Import
    `Float` ajouté à `sqlalchemy`.
  - `alembic/versions/011_memory.py` : migration `memories`
    (`010_agenda` → `011_memory`), même style que `008_assistant_messages.py`
    (index tenant_id/user_id, downgrade symétrique).
  - `email_analyzer/llm.py` : nouveau bloc "Memory extraction" —
    `MemoryFactLLM`/`MemoryExtractionLLM` (schémas Pydantic),
    `extract_memory_candidates_openai`/`_gemini` (même discipline que
    `extract_structured_project_summary_openai`/`_gemini` : jamais
    d'exception, toujours `{"data": ..., "erreur": ...}`). Ajout aussi
    `select_assistant_provider(preferred)` — extrait de la logique
    `_select_provider` qui était dupliquée dans
    `api/routers/assistant.py`/`ConversationService`, maintenant partagée
    entre `ConversationService` et `MemoryService` (évite la duplication
    signalée explicitement par l'utilisateur avant cette session).
    `ConversationService._select_provider` refactorée pour l'utiliser.
  - `email_analyzer/repositories/memory_repository.py` : `MemoryRepository`
    — `list_for_user` (tri confidence desc puis created_at desc = le
    "scoring" v1, pas de classe séparée), `find_by_content` (dédup exacte
    insensible casse/espaces, pas de similarité vectorielle), `create`,
    `delete`.
  - `email_analyzer/services/memory_service.py` : `MemoryService` —
    `list_memories`, `delete_memory`, `extract_and_store(tenant_id, user_id,
    turns, ...)` : sélectionne le fournisseur (`select_assistant_provider`,
    retourne `[]` silencieusement si aucune clé — pas d'exception, cohérent
    avec le reste de `llm.py`), appelle l'extraction LLM, filtre les
    candidats (`confidence < 0.5` ignorés, catégorie invalide repliée sur
    `fact`), déduplique via le repository, persiste. Logging explicite
    start/done/erreur (`logger.info`/`logger.warning`).
  - `api/routers/memory.py` (nouveau) : `GET /api/memory` (liste, triée),
    `POST /api/memory/extract` (déclenchement **explicite** — lit les N
    derniers tours de `ConversationService.list_history`, appelle
    `MemoryService.extract_and_store`, retourne les souvenirs créés),
    `DELETE /api/memory/{id}` (droit à l'oubli minimal). Décision
    volontaire : **pas d'extraction automatique après chaque message** de
    `/api/assistant/chat` — l'ajouter silencieusement aurait changé le
    coût/latence d'un endpoint existant sans le documenter comme tel
    (contraire à "avoid hidden behaviors"). Le branchement automatique est
    laissé à une décision produit explicite dans une session dédiée
    (Context Engine, Phase 7, où la mémoire doit de toute façon être
    injectée dans le prompt système du chat).
  - `api/main.py` : router `memory_router` importé et enregistré après
    `assistant_router`.
  - `context/project-overview.md` : pilier "Mémoire persistante" mis à jour
    (était "non implémenté", maintenant "fondation posée" avec détail de ce
    qui existe/n'existe pas encore).
  - Vérifié : syntaxe (`ast.parse`) de tous les fichiers créés/modifiés ;
    import réel dans le venv du projet (`myenv`) de `Memory`,
    `MemoryRepository`, `MemoryService`, du routeur `memory`, et de
    `api.main:app` — le nouveau router apparaît bien dans `app.routes`
    (10 routers montés désormais, contre 9 après Unit 45) ; chaîne de
    migrations Alembic cohérente (`011_memory.down_revision ==
    "010_agenda"`).
  - Non vérifié : migration réellement appliquée sur une base Postgres
    (`alembic upgrade head`), et appel HTTP réel des 3 nouveaux endpoints
    avec une vraie clé LLM (pas d'environnement disponible dans cette
    session — même limite que Unit 45).
  - Prochaine étape naturelle (Phase 6, non commencée) : Knowledge Layer
    (Entities/Facts/Relationships/Contacts/Organizations/Project Links,
    PostgreSQL, pas de graphe) — pourrait réutiliser `MemoryCategory` ou
    rester un modèle séparé selon ce que la Phase 6 exige réellement,
    décision à prendre au moment venu plutôt qu'anticipée ici.

- **Unit 47 — Knowledge Layer, fondation (2026-07-17)** :
  - Contexte : suite directe d'Unit 46, Phase 6 du plan de pivot ("Entities/
    Facts/Relationships/Contacts/Organizations/Project Links, pas de
    graphe, PostgreSQL, stocker les relations simplement").
  - Recherche préalable (obligatoire avant toute nouvelle brique, cf.
    consigne utilisateur) : `ProjectSummaryLLM` (`llm.py`) extrait déjà
    `participants: List[str]` et `companies: List[str]` par projet, persistés
    tels quels dans `ProjectSummary.structured_content` (JSONB, voir
    `project_mail.py:1055` et `analysis_tasks.py`) — mais uniquement comme
    JSON non normalisé par projet, sans identité partagée entre projets
    (le même contact apparaissant sur 2 projets = 2 chaînes de caractères
    sans lien). Décision : **ne pas ajouter de nouvel appel LLM** pour la
    Knowledge Layer — normaliser cette donnée déjà produite plutôt que la
    ré-extraire (évite un doublon de pipeline d'extraction et un coût LLM
    superflu).
  - Décision d'architecture (documentée pour transparence, même logique
    qu'Unit 46) : le plan nomme "Entities/Facts/Relationships" de façon
    générique — implémenté concrètement comme `Organization`/`Contact`/
    `ProjectContactLink` (le sous-ensemble pour lequel il existe déjà des
    données réelles). Pas de table `Fact` générique : aurait dupliqué
    `Memory` (Unit 46) — une seule notion de "fait durable" dans le code,
    pas deux.
  - `email_analyzer/db/models.py` : `Organization` (`organizations`, unique
    par tenant+nom), `Contact` (`contacts`, unique par tenant+nom, FK
    optionnelle vers `organizations`), `ProjectContactLink`
    (`project_contact_links`, association project↔contact, unique par
    couple). Dédup par nom exact, même discipline que `Memory`
    (`find_by_content`) — pas de résolution d'identité floue en v1.
  - `alembic/versions/012_knowledge.py` : migration des 3 tables
    (`011_memory` → `012_knowledge`), mêmes conventions que les migrations
    précédentes (index, contraintes uniques, downgrade symétrique).
  - `email_analyzer/repositories/knowledge_repository.py` :
    `KnowledgeRepository` — `get_or_create_organization`,
    `get_or_create_contact` (met à jour `organization_id` si l'organisation
    était inconnue lors de la première création), `link_project_contact`
    (idempotent), `list_contacts_for_tenant`, `list_contacts_for_project`,
    `list_organizations_for_tenant`.
  - `email_analyzer/services/knowledge_service.py` : `KnowledgeService` —
    `sync_from_project_summary(tenant_id, project_id)` relit
    `ProjectSummary.structured_content`, upsert une `Organization` par nom
    de société cité, lie tous les participants à la **première** société
    trouvée (limite explicitement documentée : le LLM ne fournit pas
    aujourd'hui de mapping fiable "quel participant appartient à quelle
    société" — pas d'invention de ce lien côté code), puis crée les
    `ProjectContactLink`. `list_contacts`/`list_organizations`/
    `list_project_contacts` pour la lecture. Logging explicite
    start/done. Suit le même choix qu'Unit 46 : pas de classes séparées
    par "verbe" (Extractor/Store/Retriever) pour une poignée de méthodes.
  - `api/routers/knowledge.py` (nouveau) : `GET /api/knowledge/contacts`
    (optionnellement filtré par `project_id`), `GET
    /api/knowledge/organizations`, `POST
    /api/knowledge/projects/{project_id}/sync` (déclenchement **explicite**
    — même raisonnement qu'Unit 46 pour `/api/memory/extract` : ne pas
    changer silencieusement le comportement du Fast-Track existant
    (`analysis_tasks.py`) en y greffant cette synchronisation sans décision
    produit explicite).
  - `api/main.py` : router `knowledge_router` importé et enregistré après
    `memory_router`.
  - `context/project-overview.md` : pilier "Knowledge Graph" mis à jour
    (était "non implémenté", maintenant "fondation posée" avec détail de ce
    qui existe/n'existe pas encore, notamment l'absence de résolution
    contact↔organisation fiable et l'absence d'adresses email normalisées
    sur `Email`).
  - Vérifié : syntaxe (`ast.parse`) de tous les fichiers créés/modifiés ;
    import réel dans le venv du projet (`myenv`) de `Organization`,
    `Contact`, `ProjectContactLink`, `KnowledgeRepository`,
    `KnowledgeService`, du routeur `knowledge`, et de `api.main:app` — 11
    routers montés désormais (10 après Unit 46, +1) ; chaîne de migrations
    Alembic cohérente (`012_knowledge.down_revision == "011_memory"`).
  - Non vérifié : migration réellement appliquée sur une base Postgres, et
    appel HTTP réel des 3 nouveaux endpoints sur un vrai projet avec
    `structured_content` rempli (pas d'environnement disponible dans cette
    session — même limite que Units 45/46). **Recommandé avant d'aller plus
    loin** : appliquer `alembic upgrade head` (couvre aussi 011) et vérifier
    en conditions réelles Units 45-47 ensemble plutôt que migration par
    migration.
  - Prochaine étape naturelle (Phase 7, non commencée) : Context Engine —
    le premier composant qui consommera réellement Memory (Unit 46) et
    Knowledge Layer (cette unité) pour construire le contexte unifié
    injecté dans `ConversationService`/le chat portefeuille ; c'est aussi
    le moment naturel pour décider si l'extraction mémoire/la synchronisation
    connaissance doivent devenir automatiques après un tour de chat ou un
    Fast-Track (actuellement explicites par choix, voir Units 46-47).

- **Unit 48 — Context Engine, fondation (2026-07-17)** :
  - Contexte : suite directe d'Unit 47, Phase 7 du plan de pivot ("Context
    Builder : Input = Conversation/Projects/Emails/Memory/Actions/Documents,
    Output = contexte IA unifié ; devient le seul fournisseur de contexte
    pour les requêtes LLM"). Recherche préalable : `ConversationService`
    construisait déjà un contexte "projets" en interne
    (`_build_portfolio_context`/`_pending_action_descriptions`, dupliquant
    des requêtes déjà écrites une fois) et ne consommait ni `Memory` (Unit
    46) ni la Knowledge Layer (Unit 47), posées mais jamais lues nulle part
    jusqu'ici. Décision : **déplacer** cette logique existante vers
    `ContextService` plutôt que la dupliquer, et y **brancher** Memory/
    Knowledge — pas de nouveau modèle de données.
  - `email_analyzer/llm.py` : `build_portfolio_chat_system_prompt` gagne un
    paramètre optionnel `extra_context_sections` (rétrocompatible, défaut
    `None`) qui insère des blocs de contexte additionnels déjà formatés
    avant la section "Règles". Deux nouveaux formatteurs suivant exactement
    la convention de `build_portfolio_context` (dicts simples en entrée, pas
    de dépendance DB dans `llm.py`) : `build_memory_context` (liste
    `{"category","content"}` → texte, chaîne vide si vide) et
    `build_knowledge_context` (liste `{"name","organization"}` → texte).
  - `email_analyzer/services/context_service.py` (nouveau) : `ContextService`
    — `build_portfolio_system_prompt(tenant_id, user_id)` orchestre
    `_build_projects_context` (reprise telle quelle de l'ancienne logique de
    `ConversationService`), `_build_memory_section` (appelle
    `MemoryService.list_memories`, ne déclenche **pas** d'extraction),
    `_build_knowledge_section` (appelle `KnowledgeService.list_contacts`/
    `list_organizations`, ne déclenche **pas** de sync), puis assemble le
    tout via `llm.build_portfolio_chat_system_prompt`. Documenté comme la
    seule source de vérité pour "quelles données entrent dans le prompt" —
    Emails/Documents explicitement exclus du scope v1 (pas de source de
    lecture dédiée pour les emails bruts, pas de modèle `Document`) plutôt
    que d'inventer une lecture partielle.
  - `email_analyzer/services/conversation_service.py` : supprime
    `_build_portfolio_context`/`_pending_action_descriptions` et les imports
    `Project`/`ProjectSummary`/`SuggestedAction`/`build_portfolio_context`
    devenus inutiles ; `send_message` appelle désormais
    `self._context.build_portfolio_system_prompt(tenant_id, user_id)`.
    Comportement observable inchangé tant que `memories`/`contacts` sont
    vides pour un tenant (nouveauté : dès qu'un utilisateur a des souvenirs
    ou des contacts synchronisés, ils apparaissent dans les réponses du chat
    portefeuille — c'est le but de cette unité).
  - Vérifié : syntaxe (`ast.parse`) de tous les fichiers modifiés/créés ;
    import réel dans le venv du projet (`myenv`) de `ContextService`,
    `ConversationService`, des nouveaux formatteurs `llm.py`, et de
    `api.main:app` (toujours 11 routers — Phase 7 n'ajoute aucun endpoint,
    uniquement du câblage interne) ; appel direct de
    `build_portfolio_chat_system_prompt("Aucun projet.", ["souvenir X"])`
    hors DB pour confirmer l'assemblage du prompt (bloc "souvenir X" bien
    inséré avant "Règles :").
  - Non vérifié : appel HTTP réel `/api/assistant/chat` sur un tenant ayant
    à la fois des `Memory` et des `Contact` réels, pour confirmer que le
    LLM reçoit et utilise correctement ces sections (pas d'environnement
    disponible dans cette session — même limite que Units 45-47).
    **Recommandé avant la suite** : appliquer les migrations 011/012,
    peupler un tenant de test (`POST /api/memory/extract` après quelques
    messages, `POST /api/knowledge/projects/{id}/sync` sur un projet ayant
    un résumé structuré), puis vérifier visuellement que
    `/api/assistant/chat` en tient compte.
  - Prochaine étape naturelle (Phase 8, non commencée) : Tool Architecture
    (remplacer les appels de service directs par des tools nommés —
    `SearchEmailsTool`, `SearchProjectsTool`, `SearchMemoryTool`,
    `DraftReplyTool`, etc.) — Phase 7 rend cette étape plus naturelle
    puisque `ContextService`/`MemoryService`/`KnowledgeService` exposent déjà
    des méthodes à granularité "tool-sized" ; il s'agira surtout de les
    envelopper avec un contrat entrée/sortie simple, pas de les réécrire.

- **Unit 49 — Vérification bout-en-bout réelle, Units 45-48 (2026-07-17)** :
  - Contexte : Units 45-48 avaient toutes été livrées avec une ligne "Non
    vérifié : appel HTTP réel / migration réellement appliquée" faute
    d'environnement disponible en session. L'utilisateur a demandé une
    vérification explicite avant de poursuivre vers la Phase 8 — traité
    comme une unité à part entière plutôt que rattaché rétroactivement aux
    précédentes, pour garder une trace propre de ce qui a été exercé et
    quand (cf. [[project-my-connector-verification-style]]).
  - Environnement trouvé disponible en session (absent lors des Units
    45-48) : Postgres local actif (`pg_isready` OK), `DATABASE_URL` réel
    dans `.env` racine pointant sur la base `connector`, `GEMINI_API_KEY`
    configurée (pas de clé OpenAI). Vérification donc faite avec le
    fournisseur Gemini.
  - `alembic upgrade head` exécuté depuis `services/email_analyzer` avec
    l'environnement du `.env` racine chargé : `010_agenda` →
    `011_memory` → `012_knowledge`, sans erreur. Confirmé par
    inspection directe (`psql \dt`) : les 4 tables (`memories`,
    `organizations`, `contacts`, `project_contact_links`) existent bien
    dans la base réelle.
  - Script de vérification jetable (créé dans le scratchpad de session,
    supprimé après usage — pas de framework de test dans ce repo, voir
    [[project-my-connector-verification-style]]) pilotant l'app FastAPI
    réelle via `starlette.testclient.TestClient` (pas de serveur uvicorn
    séparé nécessaire) :
    1. Seed direct via les modèles SQLAlchemy réels d'un tenant/user/
       membership/project/`ProjectSummary` de test (`structured_content`
       avec `participants`/`companies`, comme le produit vraiment le
       Fast-Track) — token JWT réel minté avec `create_access_token`.
    2. `POST /api/knowledge/projects/{id}/sync` → 200, contacts "Alice
       Dupont"/"Bob Martin" créés et liés à l'organisation "Acme Corp" —
       confirmé aucun appel LLM nécessaire pour cette route (lecture pure
       de `structured_content`).
    3. Deux `AssistantMessage` seedées directement (préférence utilisateur :
       "être appelé par mon prénom, Mohamed").
    4. `POST /api/memory/extract` → 200, **vrai appel Gemini** : le fait
       extrait correspond exactement à l'intention (`"L'utilisateur préfère
       être appelé par son prénom, Mohamed."`, catégorie `preference`,
       confiance 1.0) — confirme `extract_memory_candidates_gemini`/
       `MemoryService.extract_and_store` fonctionnels avec un vrai modèle,
       pas seulement en syntaxe.
    5. `ContextService.build_portfolio_system_prompt` appelé directement
       (hors HTTP) : le prompt assemblé contient bien les 3 sections
       (projet, souvenirs, contacts) dans l'ordre attendu — vérifié par
       inspection du texte complet, pas seulement par assertions
       partielles.
    6. `POST /api/assistant/chat` → 200, **second vrai appel Gemini** avec
       ce prompt enrichi : réponse obtenue — **"Mohamed, le projet de
       Refonte site vitrine avance bien..."** — la réponse commence
       spontanément par le prénom mémorisé, preuve comportementale (pas
       seulement structurelle) que Memory influence réellement une réponse
       réelle du chat portefeuille via le Context Engine.
    7. `GET /api/knowledge/contacts`, `GET /api/knowledge/organizations`,
       `GET /api/memory` relus après coup → cohérents avec ce qui précède.
    8. Toutes les données de test supprimées explicitement en fin de script
       (tenant, user, membership, project, summary, memories, contacts,
       organizations, liens) — base ramenée à son état initial, aucune
       donnée de test laissée dans `connector`.
  - Vérifié (récapitulatif, remplace les lignes "Non vérifié" des Units
    45-48 pour cette partie) : migrations 011/012 appliquées sur Postgres
    réel ; `POST /api/knowledge/projects/{id}/sync` réel (sans LLM) ;
    `POST /api/memory/extract` réel (avec Gemini) ; assemblage du prompt
    par `ContextService` avec données réelles ; `POST /api/assistant/chat`
    réel (avec Gemini) démontrant l'effet de la mémoire sur la réponse ;
    `GET /api/memory`, `GET /api/knowledge/contacts`,
    `GET /api/knowledge/organizations`.
  - Non vérifié (reste ouvert, hors scope de cette unité) : fournisseur
    OpenAI (pas de clé disponible en session — seul le chemin Gemini est
    exercé pour l'extraction/le chat) ; passage réel par le frontend/
    navigateur (aucune de ces routes n'est encore appelée depuis
    `ChatHome.tsx`/les hooks frontend — Memory/Knowledge restent
    entièrement backend à ce stade) ; comportement sous charge/concurrence ;
    `POST /api/knowledge/projects/{id}/sync` et `POST /api/memory/extract`
    testés isolément, pas encore dans le flux réel d'usage répété (plusieurs
    tours de chat, plusieurs projets).

- **Unit 50 — Tool Architecture, fondation (2026-07-17)** :
  - Contexte : Phase 8 du plan de pivot ("remplacer les appels de service
    directs par des tools nommés — SearchEmailsTool, SearchProjectsTool,
    SearchMemoryTool, DraftReplyTool, SummarizeProjectTool, FindContactTool
    — une responsabilité, entrée/sortie simples, facile à tester").
  - Recherche préalable : les services des Units 45-48
    (`ConversationService`, `MemoryService`, `KnowledgeService`,
    `ContextService`) exposent déjà des méthodes à granularité "tool-sized"
    (noté explicitement dans Unit 48). Le pattern de requête "emails
    importants d'un projet" existait déjà dans
    `api/routers/projects.py` (endpoint détail projet). Décision : chaque
    tool est une **enveloppe fine** — délégation pure vers un service
    existant (`SearchMemoryTool`→`MemoryService`, `FindContactTool`→
    `KnowledgeService`) ou reprise d'une requête déjà écrite ailleurs
    (`SearchEmailsTool` reprend exactement la requête de
    `projects.py::project_detail`) — aucune nouvelle logique métier créée
    à cette occasion, seulement un nouveau point d'entrée uniforme.
  - Décision d'architecture : pas de classe de base par héritage — suit
    explicitement `ai-workflow-rules.md` ("Avoid inheritance. Prefer
    composition."). `email_analyzer/tools/base.py` définit `Tool` comme un
    `typing.Protocol` (structural typing, documentation/vérification
    statique uniquement) plutôt qu'une classe abstraite à sous-classer. Pas
    de registre/dispatch dynamique par nom (aurait été de la
    métaprogrammation/réflexion, explicitement proscrites) — Phase 9
    (Agent) appellera ces classes directement et de façon séquentielle.
  - `email_analyzer/tools/search_projects_tool.py` : `SearchProjectsTool`
    — liste/filtre les projets d'un tenant par mot-clé sur le nom (nouveau
    filtre `ilike`, seule vraie logique ajoutée dans cette unité), retourne
    nom/statut/sentiment/résumé.
  - `email_analyzer/tools/summarize_project_tool.py` : `SummarizeProjectTool`
    — relit `ProjectSummary` existant pour un projet (ne relance jamais de
    calcul — le recalcul reste le rôle du Fast-Track,
    `POST /api/projects/{id}/refresh`).
  - `email_analyzer/tools/search_emails_tool.py` : `SearchEmailsTool` —
    reprend telle quelle la requête "top 10 emails par
    `importance_score`/`received_at`" déjà présente dans
    `projects.py::project_detail`, scopée à un projet. Pas de recherche
    plein texte ni cross-projet en v1 : `Email.body_encrypted` n'a pas
    vocation à être parcouru ainsi et il n'existe aucun index de recherche
    — documenté comme limite explicite plutôt que d'inventer une capacité
    absente.
  - `email_analyzer/tools/search_memory_tool.py` /
    `find_contact_tool.py` : délégation pure vers `MemoryService`/
    `KnowledgeService` (Units 46-47) avec un filtre par sous-chaîne
    optionnel côté tool (pas de nouvelle requête SQL).
  - Volontairement absents de cette unité : `DraftReplyTool` (le seul
    brouillon existant, `POST /api/draft` dans `api/main.py`, appartient au
    sous-système legacy IMAP sans tenant — l'envelopper maintenant aurait
    mélangé deux architectures différentes sans décision produit explicite
    sur laquelle des deux le futur Agent doit cibler) ; aucune exposition
    API (ces tools sont un usage interne, en préparation de l'Agent Phase
    9 — les exposer en REST maintenant dupliquerait `api/routers/*`
    existants sans consommateur réel).
  - Vérifié : syntaxe (`ast.parse`) de tous les fichiers ; **script de
    vérification réel** (scratchpad, supprimé après usage, pas de LLM
    nécessaire — ces 5 tools sont en lecture seule) : seed direct d'un
    tenant/projet/résumé/2 emails/organisation/contact/souvenir dans la
    vraie base Postgres locale, chaque tool exécuté et son retour comparé
    à la donnée seedée — `SearchProjectsTool` (filtre positif "ERP" trouve
    le projet, filtre négatif "nope" retourne `[]`), `SummarizeProjectTool`
    (résumé exact retourné), `SearchEmailsTool` (email à
    `importance_score=80` bien en tête devant celui à 40),
    `SearchMemoryTool` (filtre "lundi" trouve le souvenir, filtre "nope" ne
    trouve rien), `FindContactTool` (contact + organisation "Globex"
    retournés correctement). Toutes les données de test supprimées en fin
    de script.
  - Non vérifié : usage réel par un agent/LLM (Phase 9, non commencée — ces
    tools n'ont encore aucun appelant en dehors du script de vérification).
  - Prochaine étape naturelle (Phase 9, non commencée) : AI Agent — un
    Planner/Executor déterministe (pas de planification autonome, pas
    d'agents récursifs, per le plan) qui appelle ces 5 tools de façon
    séquentielle pour répondre à une requête, avant de basculer sur le
    chat portefeuille actuel (prompt unique, pas de sélection d'outils) ;
    décision à prendre à ce moment-là : le nouvel Agent remplace-t-il
    `ConversationService.send_message` ou coexiste-t-il comme un mode
    alternatif le temps d'être validé.

- **Unit 51 — AI Agent, fondation (2026-07-17)** :
  - Contexte : Phase 9 du plan de pivot ("Agent, Planner, Executor,
    Response Builder ; première version = exécution séquentielle de tools ;
    pas de planification autonome, pas d'agents récursifs, tout
    déterministe").
  - Décision explicite tranchée dans cette unité (laissée ouverte par Unit
    50) : l'Agent **coexiste** avec `ConversationService`/
    `/api/assistant/chat`, il ne le remplace pas. Raison : v1 est sans
    historique/persistance (une question isolée → une réponse), donc
    objectivement moins capable que le chat portefeuille existant pour un
    usage conversationnel réel ; le remplacer maintenant aurait cassé une
    fonctionnalité qui marche (violerait "Never break existing
    functionality") pour une version non encore éprouvée. Nouvel endpoint
    séparé `POST /api/agent/query`, pas de changement pour
    `/api/assistant/chat`.
  - `email_analyzer/llm.py` : `build_agent_context` (formate la liste de
    résultats d'outils — `{"tool","description","data"|"error"}` — en JSON
    lisible par bloc, signale explicitement un outil en échec plutôt que de
    l'omettre silencieusement) et `build_agent_chat_system_prompt` (prompt
    système : le LLM rédige uniquement à partir des données déjà
    récupérées, ne choisit aucun outil lui-même). Aucune nouvelle fonction
    d'appel LLM : `ResponseBuilder` réutilise
    `portfolio_assistant_chat_openai/_gemini` (Unit existante) — un prompt
    système + un thread de messages est un contrat assez générique pour ne
    pas justifier une troisième paire de fonctions dupliquée.
  - `email_analyzer/agent/planner.py` : `Planner.plan(query)` retourne
    **toujours** la même liste fixe (`search_projects`, `search_memory`,
    `find_contact`) — aucune analyse de la requête. C'est la lecture
    littérale de "no autonomous planning" pour une v1 : le module existe
    avec la bonne interface pour que Executor/ResponseBuilder n'aient pas à
    changer le jour où une vraie logique de planification (même simple,
    par mots-clés) sera ajoutée.
  - `email_analyzer/agent/executor.py` : `Executor` — mapping explicite
    `{nom_outil: instance_Tool}` construit dans `__init__` (pas de
    registre/découverte dynamique — proscrit par
    `ai-workflow-rules.md` "Avoid ... Reflection ... Dynamic imports").
    `execute(plan, tenant_id, user_id)` exécute chaque outil dans l'ordre,
    logge start/done/erreur avec durée en ms
    (`logger.info`/`logger.warning`), et **capture les exceptions par
    outil** — un outil cassé dégrade la réponse (résultat marqué `error`)
    plutôt que de faire échouer toute la requête (`ERROR HANDLING` du plan
    : erreurs explicites, jamais avalées silencieusement — ici elles sont
    capturées *et* propagées dans le résultat, pas juste avalées).
  - `email_analyzer/agent/response_builder.py` : `ResponseBuilder.build(...)`
    assemble le contexte, choisit `portfolio_assistant_chat_openai` ou
    `_gemini` selon le fournisseur déjà résolu par l'appelant.
  - `email_analyzer/agent/agent.py` : `Agent.answer(tenant_id, user_id,
    query, ...)` — orchestrateur unique : valide la requête, résout le
    fournisseur LLM (`select_assistant_provider`, Unit 46), appelle
    Planner → Executor → ResponseBuilder dans cet ordre, logge
    start/plan/done/erreur avec durée totale. Aucune boucle, aucune
    récursion, aucun appel LLM pour décider quoi faire ensuite — un seul
    passage séquentiel, conforme à "no recursive agents".
  - `api/routers/agent.py` (nouveau) : `POST /api/agent/query` —
    contrôleur HTTP fin (auth, validation, délégation à `Agent`), même
    structure que `assistant.py`/`memory.py`/`knowledge.py`.
  - `api/main.py` : router `agent_router` importé et enregistré.
  - Vérifié : syntaxe (`ast.parse`) de tous les fichiers créés/modifiés ;
    import réel dans le venv du projet (`myenv`) de `Agent` et de
    `api.main:app` — 12 routers montés désormais (11 après Unit 50, +1) ;
    **script de vérification réel** (scratchpad, supprimé après usage) :
    seed d'un tenant avec un projet ("Lancement app mobile"), un contact
    lié à une organisation, et un souvenir, dans la vraie base Postgres
    locale ; `POST /api/agent/query` avec la question "Quels sont mes
    projets en cours ?" → **vrai appel Gemini** → réponse 200 citant
    correctement le nom du projet, son statut et son résumé (donnée passée
    par les tools, pas inventée) ; requête à question vide → 422 propre
    (`Agent.answer` valide avant d'exécuter le plan). Toutes les données de
    test supprimées après coup.
  - Non vérifié : fournisseur OpenAI (même limite que Unit 49 — pas de clé
    disponible) ; comportement quand un outil échoue réellement en base
    (le chemin `except Exception` de `Executor` est écrit et raisonné mais
    pas déclenché par un scénario réel dans cette session) ; usage frontend
    (`/api/agent/query` n'est appelé depuis aucun composant React — c'est
    une API backend pure pour l'instant, à la manière de Memory/Knowledge).
  - Prochaine étape naturelle (Phase 10, non commencée) : Daily Brief
    (`DailyBriefService`, `DailySummaryGenerator`, `PriorityDetector`) —
    le plan demande de réutiliser les résumés existants ; `GET /api/brief`
    (Unit du pivot précédent) existe déjà et fait une partie de ce travail
    (compteurs + actions recommandées) sans être encore structuré comme un
    service dédié nommé — même geste que les Phases 4-7 : chercher
    l'existant (`api/routers/brief.py`) avant d'ajouter, l'étendre/le
    renommer plutôt que le dupliquer.

- **Unit 52 — Daily Brief, extraction (2026-07-17)** :
  - Contexte : Phase 10 du plan de pivot ("réutiliser les résumés
    existants" — `GET /api/brief` existait déjà et fonctionnait, faisait
    tout son travail directement dans le routeur). Entrée retroactive :
    résumée dans "Current Phase" au moment de sa livraison mais jamais
    détaillée ici — corrigé maintenant, pendant Unit 53, en relisant le
    code réel plutôt que de laisser le tracker incomplet.
  - `email_analyzer/services/brief_service.py` (nouveau) : `BriefService`
    — `build_daily_brief(tenant_id, previous_login_at)` calcule les
    compteurs (`BriefCounts` : nouveaux projets, actions en attente,
    échéances à venir sous `UPCOMING_DEADLINE_DAYS`=7 jours, emails
    importants, projets à risque) et les `RECOMMENDED_ACTIONS_LIMIT`=5
    actions recommandées, à partir de `Project`/`SuggestedAction` déjà
    persistés — aucune génération de résumé, comportement explicitement
    non un `DailySummaryGenerator`/`PriorityDetector` séparés (jugé
    prématuré, mêmes deux-trois lignes que Memory/Knowledge Units 46-47).
    Repli `FIRST_VISIT_FALLBACK_DAYS`=7 jours si `previous_login_at` est
    absent (premier login).
  - `email_analyzer/repositories/brief_repository.py` (nouveau) :
    `BriefRepository` — seul point d'accès SQL utilisé par `BriefService`.
  - `api/routers/brief.py` : `GET /api/brief` appelle désormais
    `BriefService(db).build_daily_brief(tenant.id, user.previous_login_at)`
    au lieu de la logique inline précédente — même forme de réponse,
    aucun changement de contrat pour `hooks/useBrief.ts`.
  - Vérifié bout-en-bout (base Postgres locale réelle, données seedées) :
    compteurs identiques à ceux seedés avant et après l'extraction, aucune
    régression de comportement.

- **Unit 53 — Planner v2 (query-aware) + première UI Memory/Knowledge
  (2026-07-17)** :
  - Contexte : reprise du pivot après Unit 52, en mode autonome
    (préférence utilisateur établie — décider et documenter plutôt que
    demander validation à chaque décision). Une question a été posée en
    amont, hors du périmètre "autonome" habituel, car génuinement
    ambiguë : le tracker référence trois numérotations de phases
    différentes selon le document (`project-overview.md` Roadmap IA,
    `evolution-plan.md` V2, et la numérotation du "Strategic Pivot Plan"
    utilisateur reprise par les Units 44-52 — ce dernier document n'est
    lui-même pas stocké dans le dépôt, seulement référencé). Réponse
    utilisateur : continuer la suite du pivot exécutif (troisième option).
  - **Partie 1 — Planner v2** (`email_analyzer/agent/planner.py`,
    `email_analyzer/agent/executor.py`, `email_analyzer/agent/agent.py`) :
    - Découverte avant modification : `Planner.plan(query)` ignorait
      totalement `query` (toujours `["search_projects", "search_memory",
      "find_contact"]`), et `Executor._run_tool` appelait chaque outil
      avec `tool.run(tenant_id)` seul — jamais de `keyword`/`name`, alors
      que `SearchProjectsTool`/`SearchMemoryTool`/`FindContactTool`
      acceptent tous un filtre optionnel depuis Phase 8 (jamais utilisé
      par l'Agent jusqu'ici).
    - `planner.py` : `_BASE_PLAN = ["search_projects", "search_memory"]`
      toujours inclus ; `find_contact` ajouté seulement si la requête
      contient un motif court et explicite ("contact", "qui est",
      "coordonnées", "numéro de", "email de", "joindre" — pas de NLP,
      substring matching sur la requête en minuscules).
    - `executor.py` : `execute(plan, tenant_id, user_id, query="")` —
      dérive un mot-clé déterministe via `_extract_keyword` (premier mot
      commençant par une majuscule après le premier mot de la phrase,
      regex simple ; volontairement pas de NER/LLM) et le transmet à
      `search_projects`(`keyword=`)/`search_memory`(`keyword=`)/
      `find_contact`(`name=`). `summarize_project`/`search_emails`
      restent hors du plan (nécessitent un `project_id` que rien ne
      résout depuis le texte libre — même limite que v1, non traitée
      dans cette unité).
    - `agent.py` : un seul changement, `self._executor.execute(plan,
      tenant_id, user_id, query=query)` (passage du paramètre manquant).
    - Vérifié en conditions réelles (Postgres local, tenant/utilisateur/
      2 projets/2 souvenirs/1 contact+organisation seedés puis
      entièrement supprimés après coup) : le plan varie bien selon la
      question ("Qui est le contact chez Ezikash ?" → inclut
      `find_contact`, "Quels sont mes projets en cours ?" → non) ; le
      mot-clé extrait filtre réellement les résultats (`search_projects`
      passe de 2 à 1 résultat sur "Que sais-tu du projet Lancement ?",
      motclé "Lancement") ; aucune exception levée sur les 4 requêtes
      testées. Limite connue et acceptée (repli sans erreur, documentée
      dans le docstring d'`_extract_keyword`) : une élision française
      ("d'Amina") fusionne l'apostrophe dans le token et empêche la
      détection de majuscule — le mot-clé reste alors `None`, pas de
      filtre, comportement identique à avant cette unité, pas un
      régression.
    - `python -m py_compile` OK sur les 3 fichiers modifiés.
  - **Partie 2 — UI Memory & Knowledge** (`frontend/src/`) :
    - Découverte avant modification (exploration dédiée) : `GET/DELETE
      /api/memory` et `GET /api/knowledge/contacts`/`/organizations`
      existent et fonctionnent depuis Units 46-47, mais aucune ligne de
      `frontend/src` ne les référence — confirmé par `grep -rn
      "api/agent\|api/memory\|api/knowledge" frontend/src` (aucune
      correspondance avant cette unité).
    - `memoryApi.ts`/`knowledgeApi.ts` (nouveaux) : suivent exactement le
      pattern déjà en place (`actionsApi.ts`/`agendaApi.ts` — type miroir
      du schéma Pydantic backend + fonctions `fetch*`/`delete*` via
      `apiFetch`+`parseResponseJson`/`parseApiError`).
    - `hooks/useMemoryAndKnowledge.ts` (nouveau) : pattern `useBrief.ts`
      (`useXxx(enabled) -> {data, loading, error, reload}`, gated par
      `saasEnabled`) — charge les 3 ressources en parallèle
      (`Promise.all`) ; `removeMemory(id)` appelle `DELETE
      /api/memory/{id}` puis retire la ligne localement (pas de refetch
      complet).
    - `pages/MemoryPage.tsx` (nouveau) : deux sections (souvenirs avec
      bouton supprimer par ligne ; contacts groupés par organisation).
      Mêmes garde-fous que `ProjectHubPage.tsx` (SaaS uniquement →
      `Navigate` vers `/` sinon, jeton absent → `/login`). Tokens
      réutilisés tels quels (`rounded-lg`, `bg-surface`,
      `border-border-default`, `bg-accent-soft`/`text-accent-primary`
      pour les badges de catégorie) — aucun nouveau token ajouté à
      `tailwind.config.js`.
    - `components/AppShell.tsx` : entrée "Mémoire" ajoutée à
      `SAAS_NAV_ITEMS` (icône 🧠, entre Actions et Paramètres).
    - `App.tsx` : route `/memory` → `MemoryPage`, même pattern que
      `/projects`/`/agenda`.
    - **Décision explicite de ne pas faire cette fois** : brancher `POST
      /api/agent/query` dans le frontend. Raison : `ConversationService`
      (le chat portefeuille réellement utilisé, `/api/assistant/chat`)
      injecte déjà Projects+Memory+Knowledge dans son prompt système à
      chaque message depuis Unit 48 (`ContextService`) — avant cette
      unité, l'Agent (plan fixe, sans historique) n'apportait donc aucune
      capacité que le chat portefeuille n'avait pas déjà en mieux
      (historique multi-tours). Lui donner une UI séparée aurait dupliqué
      une capacité sans l'améliorer. Le Planner v2 ci-dessus est un
      préalable raisonnable avant de rouvrir cette question, pas encore
      suffisant pour la trancher (le chat portefeuille n'utilise toujours
      aucun outil/plan, juste un prompt système enrichi).
    - Vérifié : `npx tsc --noEmit` sans erreur ; `npx vite build`
      transforme les 67 modules sans erreur (échec ensuite limité au même
      nettoyage de `dist/` pré-existant appartenant à `root`, problème
      d'environnement déjà documenté dans plusieurs entrées précédentes,
      sans rapport avec ce code). Types frontend vérifiés cohérents avec
      les schémas Pydantic réels des routeurs (lus directement, pas
      supposés).
    - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
      d'environnement que toutes les entrées précédentes — pas d'outil
      navigateur fonctionnel dans cette session) ; recommandé à
      l'utilisateur de lancer `npm run dev` localement pour un premier
      passage visuel de `/memory`.
  - Prochaine étape naturelle (non commencée) : maintenant que Memory et
    Knowledge ont une UI et que le Planner est query-aware, rouvrir la
    question laissée en suspens par Unit 51 (Agent vs. chat portefeuille)
    avec des éléments concrets en main plutôt qu'en théorie ; ou
    `DraftReplyTool`/`ListMeetingsTool` (Phase 8, toujours pas faits — le
    premier exigerait d'adapter la forme `analysis` de `POST /api/draft`
    au modèle `ProjectSummary` persisté, pas un simple wrapper ; le second
    est un wrapper propre autour de `api/routers/agenda.py`, plus proche
    du contrat "lecture seule" des tools existants).

- **Unit 54 — Objectives, fondation (2026-07-17)** :
  - Contexte : reprise du pivot en mode autonome (« continue
    l'implémentation des phases restantes »), Phase 11 du plan
    ("Objective/Milestone/Progress/Blocking Issue ; réutiliser `Project`
    autant que possible"). Recherche préalable : `grep -n
    "Objective\|Milestone\|Blocking"` sur `db/models.py` et
    `api/routers/*.py` → aucune correspondance, rien à étendre, concept
    réellement absent.
  - Décision d'architecture (documentée pour transparence, même logique
    que Units 46/47/52) : le plan nomme 4 concepts
    (Objective/Milestone/Progress/Blocking Issue) ; implémentés comme
    **2 tables seulement**. "Progress" n'est pas une table — c'est un
    calcul à la lecture (jalons terminés / total, voir
    `ObjectiveService._with_progress`), stocker une progression
    redondante avec les jalons aurait créé une resynchronisation à gérer
    pour rien. "Blocking Issue" n'est pas une table — c'est la colonne
    `Objective.blocking_reason` (texte, vidée automatiquement quand le
    statut quitte `blocked`), un objectif n'ayant qu'un seul blocage actif
    possible en v1, pas une liste.
  - `email_analyzer/db/models.py` : `ObjectiveStatus`
    (on_track/at_risk/blocked/done), `MilestoneStatus` (pending/done),
    `Objective` (`objectives` — `project_id` nullable, FK vers `Project`,
    "use existing projects whenever possible" du plan : un objectif peut
    soit prolonger un projet existant, soit rester un but personnel du
    tenant sans projet), `Milestone` (`milestones`, FK vers `objectives`).
  - `alembic/versions/013_objectives.py` : migration des 2 tables
    (`012_knowledge` → `013_objectives`), mêmes conventions (index,
    downgrade symétrique) — **appliquée réellement**
    (`alembic upgrade head`) sur la base Postgres locale `connector`.
  - `email_analyzer/repositories/objective_repository.py` :
    `ObjectiveRepository` — CRUD objectifs + jalons, `update_status` vide
    automatiquement `blocking_reason` si le nouveau statut n'est pas
    `blocked` (évite une raison de blocage orpheline après déblocage).
  - `email_analyzer/services/objective_service.py` : `ObjectiveService` —
    valide le titre (non vide), valide que `project_id` (si fourni)
    appartient bien au tenant (404 sinon, même garde-fou que les autres
    services), valide les statuts contre les enums (422 sinon), calcule la
    progression (`ObjectiveProgress` dataclass, pas de dépendance
    Pydantic/FastAPI dans le service). Logging explicite
    (`logger.info` sur create/update_status).
  - `api/routers/objectives.py` (nouveau) : `GET/POST /api/objectives`
    (filtre optionnel `project_id`), `PATCH /api/objectives/{id}` (statut +
    raison de blocage), `POST /api/objectives/{id}/milestones`,
    `PATCH /api/objectives/{id}/milestones/{milestone_id}` — CRUD complet,
    contrairement aux fondations précédentes (Memory/Knowledge) qui
    n'exposaient que lecture + une action d'écriture explicite ; ici
    justifié par la nature du concept (un objectif se construit
    incrémentalement, pas en un seul appel).
  - `api/main.py` : router `objectives_router` importé et enregistré.
  - Vérifié : syntaxe (`ast.parse`) de tous les fichiers créés/modifiés ;
    migration réellement appliquée (`012_knowledge` → `013_objectives`
    sans erreur) ; import réel dans le venv du projet (`myenv`) de
    `ObjectiveService` et de `api.main:app` — 13 routers montés désormais
    (12 après Unit 51, +1) ; **script de vérification réel** (scratchpad,
    supprimé après usage, pas de LLM nécessaire) contre la vraie base
    Postgres locale : création d'un objectif lié à un projet existant et
    d'un objectif personnel sans projet, 404 confirmé sur `project_id`
    inconnu, ajout de 2 jalons (0% → confirmé), complétion d'un jalon (50%
    exact), passage à `blocked` avec raison puis retour à `on_track`
    (`blocking_reason` bien remis à `None` automatiquement), 422 confirmé
    sur un statut invalide, filtre `project_id` sur `GET /api/objectives`
    confirmé (1 résultat vs 2 sans filtre). Toutes les données de test
    supprimées après coup.
  - Non vérifié : UI frontend (aucune page ne consomme encore
    `/api/objectives` — même situation que Memory/Knowledge avant Unit 53,
    documenté explicitement dans "Hors scope" de `project-overview.md`
    plutôt que laissé implicite) ; usage réel par l'Agent/le chat
    portefeuille (`ContextService` n'inclut pas encore les objectifs dans
    le prompt système — pas fait dans cette unité, resterait à décider si
    ça a sa place aux côtés de Projects/Memory/Knowledge ou si c'est trop
    tôt tant qu'il n'y a pas d'UI pour en créer).
  - Prochaine étape naturelle (Phase 12, non commencée) : Timeline unifiée
    — `GET /api/timeline` (`api/routers/brief.py`) existe déjà et fusionne
    emails + mises à jour de résumé par ordre chronologique, mais pas
    encore les actions suggérées ni les rendez-vous d'agenda (`Appointment`)
    que le plan demande d'inclure ("Merge Emails, Actions, Summaries,
    Future Meetings"). Même geste attendu : étendre `get_timeline`
    existant (l'extraire vers un `TimelineService` au passage, comme
    Unit 52 pour le Brief) plutôt qu'une nouvelle route.

- **Unit 55 — Timeline unifiée (2026-07-17)** :
  - Contexte : suite directe d'Unit 54 en mode autonome, Phase 12 du plan
    de pivot ("Merge Emails, Actions, Summaries, Future Meetings. Display
    everything chronologically."). `GET /api/timeline` existait déjà
    (`api/routers/brief.py`) et fusionnait 2 des 4 types demandés
    (emails, résumés) avec la logique inline dans le routeur — même
    situation que le Brief avant Unit 52. Décision : même geste, extraire
    puis étendre plutôt que dupliquer.
  - `email_analyzer/repositories/timeline_repository.py` (nouveau) :
    `TimelineRepository` — reprend telles quelles les 2 requêtes
    existantes (`list_email_events`, `list_summary_events`) et ajoute
    `list_action_events` (`SuggestedAction`) et `list_appointment_events`
    (`Appointment`), toutes acceptant le même filtre `project_id` optionnel
    que l'ancien code.
  - `email_analyzer/services/timeline_service.py` (nouveau) :
    `TimelineService.build_timeline(tenant_id, project_id, limit)` —
    assemble les 4 types d'événements (`TimelineEvent` dataclass : `type`
    ∈ {email, summary_update, action_suggested, meeting}), trie par `at`
    décroissant, tronque à `limit` — reprend exactement la logique de tri
    de l'ancien `get_timeline`, étendue aux 2 nouveaux types. Aucune
    nouvelle donnée : les 4 tables lues existaient déjà et étaient déjà
    peuplées par le Fast-Track/l'agenda IA.
  - `api/routers/brief.py` : `get_timeline` délègue à `TimelineService`
    (contrôleur HTTP fin) ; imports `Email`/`Project`/`ProjectSummary`
    retirés (plus utilisés directement dans ce fichier — la requête
    `discover_projects` n'en a jamais eu besoin) ; commentaire du type
    `TimelineEventOut.type` mis à jour pour lister les 4 valeurs possibles.
  - Vérifié : syntaxe (`ast.parse`) des 3 fichiers modifiés/créés ; import
    réel dans le venv du projet (`myenv`) de `TimelineService` et de
    `api.main:app` (toujours 13 routers — extension interne, pas de
    nouvelle route) ; **script de vérification réel** (scratchpad,
    supprimé après usage, pas de LLM nécessaire) contre la vraie base
    Postgres locale : un projet avec 1 email, 1 résumé, 1 action suggérée
    et 1 rendez-vous, chacun à une heure différente ; `GET /api/timeline`
    → les 4 types présents (`{"email", "summary_update",
    "action_suggested", "meeting"}`), triés du plus récent (email) au plus
    ancien (rendez-vous), exactement dans l'ordre attendu ; filtre
    `project_id` sur un projet inexistant → liste vide confirmée. Données
    de test supprimées après coup.
  - Non vérifié : usage frontend (aucune page ne consomme
    `GET /api/timeline` pour l'instant, à confirmer/vérifier séparément —
    pas exploré dans cette unité) ; comportement avec un grand volume
    d'événements mélangés (pas de pagination au-delà de `limit`, testé
    seulement avec 1 événement par type).
  - **Bilan à ce stade** : les 12 phases du plan de pivot ont chacune au
    moins une fondation implémentée et vérifiée avec de vraies données
    (Postgres local, et pour Conversation/Memory/Agent des appels Gemini
    réels). Ce qui reste ouvert, recensé pour référence plutôt que traité
    dans cette unité : UI frontend pour Objectives et Timeline ; decision
    Agent vs. chat portefeuille (Unit 51/53) ; `DraftReplyTool`/
    `ListMeetingsTool` (Phase 8) ; injection des Objectives dans
    `ContextService` ; extraction mémoire/sync connaissance automatiques
    (actuellement explicites par choix, Units 46-47) ; `code-standards.md`
    toujours non aligné (Unit 44).

- **Unit 56 — UI Objectifs & Timeline (2026-07-19)** :
  - Contexte : reprise du point suivant sur la liste laissée par Unit 55
    ("Attaque le point suivant" — premier de la liste : UI frontend pour
    Objectives et Timeline). Découverte avant modification (confirmée par
    grep) : `/api/objectives` et `/api/timeline` fonctionnels côté backend
    depuis Units 54-55, aucune référence dans `frontend/src` — même
    situation que Memory/Knowledge avant Unit 53, même geste appliqué.
  - `frontend/src/objectivesApi.ts` (nouveau) : miroir de
    `MilestoneOut`/`ObjectiveOut` (`api/routers/objectives.py`) —
    `fetchObjectives`, `createObjective`, `updateObjectiveStatus`,
    `addMilestone`, `setMilestoneStatus`. Suit le pattern déjà en place
    (`actionsApi.ts` pour le PATCH, `memoryApi.ts`/`knowledgeApi.ts` pour
    la forme générale).
  - `frontend/src/timelineApi.ts` (nouveau) : miroir de `TimelineEventOut`,
    une seule fonction `fetchTimeline(limit)`.
  - `frontend/src/hooks/useObjectivesAndTimeline.ts` (nouveau) : même
    pattern que `useMemoryAndKnowledge.ts` (Unit 53) —
    `useXxx(enabled) -> {data, loading, error, reload}`, chargement
    parallèle (`Promise.all`) des deux ressources ; actions optimistes
    côté client après chaque mutation (`addObjective`,
    `toggleObjectiveBlocked`, `addObjectiveMilestone`, `toggleMilestone`)
    qui remplacent l'objectif modifié dans le state local avec la réponse
    du serveur (le backend renvoie l'objectif complet à jour, y compris le
    pourcentage recalculé — pas de refetch complet nécessaire).
  - `frontend/src/pages/ObjectivesPage.tsx` (nouveau) : deux sections,
    même structure que `MemoryPage.tsx` — "Objectifs" (formulaire de
    création, liste avec barre de progression, badge de statut coloré,
    raison de blocage si présente, jalons à cocher, formulaire d'ajout de
    jalon inline, bouton bloquer/débloquer) et "Timeline" (flux
    chronologique en lecture seule, icône par type d'événement). Mêmes
    garde-fous que les autres pages SaaS (`Navigate` vers `/` si mode
    legacy, vers `/login` si jeton absent).
    - Correction en cours de rédaction : première version utilisait des
      classes Tailwind inventées (`bg-warning-soft`, `bg-danger-soft`) qui
      n'existent pas dans `tailwind.config.js` — repérée en vérifiant les
      tokens réels (`grep` sur les usages existants de `bg-warning`/
      `bg-danger` dans `AgendaPage.tsx`/`ProjectHub.tsx`) plutôt que
      supposés ; corrigée en `bg-warning/10`/`bg-danger/10` (modificateur
      d'opacité natif Tailwind, déjà la convention du projet pour les
      teintes sans variante "soft" dédiée).
  - `components/AppShell.tsx` : entrée "Objectifs" ajoutée à
    `SAAS_NAV_ITEMS` (icône 🎯, entre Mémoire et Paramètres).
  - `App.tsx` : route `/objectives` → `ObjectivesPage`, même pattern que
    `/memory`.
  - Volontairement pas dans cette unité : édition du titre d'un objectif,
    suppression d'un objectif ou d'un jalon, sélection d'un projet à la
    création (le formulaire ne crée que des objectifs personnels sans
    projet — lier un objectif à un projet reste possible via l'API mais
    pas encore depuis l'UI) — cohérent avec Memory/Knowledge (Unit 53) où
    la première UI était volontairement minimale (lecture + une action
    d'écriture), ici légèrement plus riche car le concept d'Objective
    l'exige (rien d'utile à afficher sans pouvoir en créer un).
  - Vérifié : `npx tsc --noEmit` sans erreur ; `npx vite build` transforme
    71 modules sans erreur (67 après Unit 53, +4 fichiers ajoutés dans
    cette unité — cohérent) ; échec ensuite limité au même nettoyage de
    `dist/` pré-existant appartenant à `root`, problème d'environnement
    déjà documenté dans plusieurs entrées précédentes, sans rapport avec ce
    code. Types frontend vérifiés cohérents avec les schémas Pydantic
    réels des routeurs (lus directement).
  - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
    d'environnement que toutes les entrées précédentes) ; recommandé à
    l'utilisateur de lancer `npm run dev` localement pour un premier
    passage visuel de `/objectives` — en particulier le calcul de largeur
    de la barre de progression (`style={{width: '${percent}%'}}`, jamais
    vérifié rendu, seulement vérifié par le type-check).
  - Prochaine étape naturelle (point suivant sur la liste laissée par Unit
    55) : la décision Agent vs. chat portefeuille (Unit 51/53), ou
    `DraftReplyTool`/`ListMeetingsTool` (Phase 8, toujours pas faits).

- **Unit 57 — Décision Agent/chat portefeuille + ListMeetingsTool
  (2026-07-19)** :
  - Contexte : reprise du point suivant sur la liste (« continue » après
    Unit 56) — deux items ouverts, traités ensemble car liés : la question
    Agent vs. chat (Unit 51/53) et `ListMeetingsTool` (Phase 8).
  - **Partie 1 — Décision Agent vs. chat portefeuille** (documentaire, pas
    de code) : la question restait ouverte depuis Unit 51 ("l'Agent
    remplace-t-il `ConversationService` ou coexiste-t-il ?"), rouverte
    explicitement par Unit 53 une fois le Planner devenu query-aware.
    Analyse : le chat portefeuille (`ContextService`) injecte
    *systématiquement* la totalité des projets/souvenirs/contacts du
    tenant dans le prompt système à chaque message — ça fonctionne mais ne
    passe pas à l'échelle indéfiniment. L'Agent (Planner+Executor) ne
    récupère que les données pertinentes à la question via des tools
    filtrés — ça passerait mieux à l'échelle mais n'a ni historique ni
    mémoire de conversation. Décision : **coexistence tranchée
    durablement**, pas de fusion. Raison explicite : fusionner
    maintenant (faire router `ConversationService` vers
    Planner/Executor pour construire son contexte au lieu de
    `ContextService`) serait une optimisation de la taille du contexte
    non justifiée par un problème réel observé — aucun signal actuel
    (nombre de projets/souvenirs) ne montre que le "tout injecter" de
    `ContextService` pose un problème de coût/latence/qualité. Contraire à
    "Ignore premature optimization. Correctness first." (règles de
    développement du pivot). Cette décision ferme la question tant qu'un
    signal concret (tenant avec beaucoup de données, plainte sur la
    qualité des réponses, coût token mesuré) ne la rouvre pas — documentée
    ici pour ne pas la rouvrir sans raison à chaque session.
  - **Partie 2 — `ListMeetingsTool`** (code) :
    `email_analyzer/tools/list_meetings_tool.py` (nouveau) — reprend
    exactement la requête déjà utilisée par `api/routers/agenda.py::
    get_agenda` (rendez-vous non annulés, `scheduled_at >= now`, tri
    ascendant), scopable par `project_id`. `DraftReplyTool` reste
    volontairement non fait (même raison que documentée depuis Unit 50 :
    `POST /api/draft` est un sous-système legacy IMAP non multi-tenant,
    l'adapter au modèle `ProjectSummary` persisté est un vrai chantier,
    pas un simple wrapper — pas traité dans cette unité, faute de
    justification suffisante pour l'ouvrir maintenant).
  - `email_analyzer/agent/planner.py` : nouveau `_MEETING_TRIGGERS`
    ("réunion", "rendez-vous", "rdv", "meeting", "calendrier", "agenda"),
    ajoute `list_meetings` au plan selon le même mécanisme substring que
    `_CONTACT_TRIGGERS` (Unit 53) — aucune nouvelle abstraction, extension
    directe du motif existant.
  - `email_analyzer/agent/executor.py` : `ListMeetingsTool` ajouté au
    mapping explicite `self._tools` ; passe par le chemin de repli
    `tool.run(tenant_id)` de `_run_tool` (le tool n'a pas besoin de
    `keyword`, seulement d'un `project_id` optionnel non résolu par le
    Planner — même limite que `summarize_project`/`search_emails`, mais
    ici sans bloquer l'inclusion dans le plan puisque le mode tenant-large
    a un sens pour ce tool, contrairement aux deux autres qui exigent un
    projet précis).
  - Vérifié : syntaxe (`ast.parse`) des 3 fichiers modifiés/créés ;
    **script de vérification réel** (scratchpad, supprimé après usage)
    contre la vraie base Postgres locale : `Planner().plan(...)` inclut
    bien `list_meetings` sur "Quels sont mes prochains rendez-vous ?" et
    pas sur "Comment vont mes projets ?" ; `ListMeetingsTool.run` direct
    sur 3 rendez-vous seedés (1 futur non annulé, 1 futur annulé, 1 passé)
    → ne retourne que le bon ; **vrai appel Gemini** via
    `POST /api/agent/query` avec la question "Quel est mon prochain
    rendez-vous ?" → réponse citant correctement la description, le
    projet, la date et le statut du rendez-vous seedé. Données de test
    supprimées après coup.
  - Non vérifié : `DraftReplyTool` (non fait, voir ci-dessus) ; usage
    frontend de l'Agent (toujours aucune UI, cohérent avec la décision de
    coexistence — pas de raison d'en ajouter une tant que l'Agent n'offre
    rien que le chat n'ait déjà).
  - Prochaine étape naturelle : plus d'item non traité sur la liste des 5
    laissée après Unit 55, à l'exception de `code-standards.md` (non
    aligné depuis Unit 44, référence toujours Celery/React Native) et de
    `DraftReplyTool` (délibérément non prioritaire, chantier plus lourd).
    Le pivot exécutif a maintenant une fondation vérifiée pour chacune des
    12 phases, une UI pour les 4 piliers qui en avaient besoin
    (Conversation, Memory, Knowledge, Objectives/Timeline), et une
    décision d'architecture documentée sur l'Agent — bon point d'arrêt
    naturel pour cette série de sessions autonomes.

- **Unit 58 — Rattrapage documentaire : architecture.md + code-standards.md
  (2026-07-19)** :
  - Contexte : en préparant `code-standards.md` (point suivant sur la
    liste après Unit 57), relecture de `architecture.md` pour vérifier la
    cohérence avant de le citer — découverte que §2 (arborescence), §3
    (modèle de données) et §7 (vision long terme) n'avaient **jamais été
    mis à jour depuis Unit 44** (2026-07-17) alors que 13 unités de code
    ont suivi (45-57) : §3 affirmait encore explicitement "il n'existe
    aucune des tables suivantes : Memory, ..., Contact, Organization,
    Objective, ..." — toutes existent depuis Units 46/47/54. §7 affirmait
    "Memory Engine : non commencé", "Knowledge Graph : non commencé",
    "Context Engine : non commencé", "Couche Agent/Tools : non commencé,
    aucune forme de tool-calling n'existe" — tout ceci était faux au
    moment de la lecture. Violation directe de la règle "Sync Specs
    Instantly" (`ai-workflow-rules.md` §5.1), passée inaperçue parce que
    chaque unité 45-57 mettait à jour `progress-tracker.md` et
    `project-overview.md` mais pas systématiquement `architecture.md`.
  - `context/architecture.md` :
    - §2 (arborescence) : ajout des dossiers `repositories/`, `services/`,
      `tools/`, `agent/`, des 4 nouveaux routeurs (`memory.py`,
      `knowledge.py`, `agent.py`, `objectives.py`), des nouvelles pages/
      hooks frontend (`MemoryPage.tsx`, `ObjectivesPage.tsx`,
      `useMemoryAndKnowledge.ts`, `useObjectivesAndTimeline.ts`).
    - §3 (modèle de données) : ajout de `Memory`, `Organization`/
      `Contact`/`ProjectContactLink`, `Objective`/`Milestone` avec leur
      rôle réel (une phrase chacun, cf. leurs docstrings dans
      `db/models.py`) ; la liste "n'existe pas" réduite à ce qui est
      encore vrai (`Fact` générique, `Entity` générique, `Document`,
      `Insight`, `Decision`, `ToolExecution`, `KnowledgeGraphEdge`).
    - §7 (vision long terme) : les 4 sous-sections (Memory Engine,
      Knowledge Graph, Context Engine, Agent/Tools) réécrites de "non
      commencé" à "fondation posée (Unit XX)" avec le détail réel de ce
      qui existe et n'existe toujours pas — même niveau de précision que
      `project-overview.md`, pour que les deux documents ne divergent
      plus sur le statut d'implémentation.
    - §8 (invariants) : ajout de l'invariant 5, "Model → Repository →
      Service → API", qui décrit une convention réellement suivie depuis
      Unit 45 mais jamais formalisée comme invariant (seulement respectée
      par habitude/répétition dans le code).
  - `context/code-standards.md` : réécriture complète. Contenu retiré
    (jamais implémenté dans ce dépôt, confirmé par recherche —
    `find`/`grep` sur `pyproject.toml`, `.pre-commit-config.yaml`,
    `mypy.ini`, `.eslintrc*`, `requirements.txt`, `package.json` : aucune
    trace) : Celery, React Native, `mypy`/`tsc` en CI, couverture de tests
    80%, `Bandit`, cache device 30 jours, index vectoriel Orama/Voy.
    Nouveau contenu factuel : stack réelle (renvoi à `architecture.md`),
    absence totale d'outillage automatisé assumée explicitement (section
    dédiée "Ce qui n'est délibérément pas en place" plutôt que silence sur
    le sujet), conventions Python réellement suivies (routes FastAPI
    synchrones — vérifié par grep sur `api/routers/objectives.py`/
    `memory.py` : aucun `async def` — Model→Repository→Service→API,
    `llm.py` seul point d'appel SDK LLM, jamais d'exception avalée
    silencieusement, logging `logger.info`/`logger.warning` par module,
    migrations Alembic avec `downgrade()` symétrique), conventions
    TypeScript (hooks personnalisés, types miroirs des schémas Pydantic,
    ne pas inventer de classe Tailwind sans vérifier son existence dans
    `tailwind.config.js` — référence explicite à l'erreur
    `bg-warning-soft` corrigée en Unit 56 comme exemple concret), protocole
    de vérification réel (scripts jetables contre une vraie base Postgres,
    pas de mocks — renvoi vers les Units 46-57 comme exemples).
  - Vérifié : relecture manuelle de chaque fait technique cité contre le
    code réel (arborescence via `ls`/`find`, absence d'outillage via
    `grep`/`find` ciblés, absence d'`async def` dans les routeurs récents
    via `grep -n "^def \|^async def "`). Aucun changement de code, aucune
    fonctionnalité modifiée — purement documentaire, comme Unit 44.
  - Non vérifié : aucune vérification automatisée possible pour un
    changement purement documentaire (même note qu'Unit 44).
  - Point encore ouvert (recensé, pas traité) : `DraftReplyTool`
    (délibérément non prioritaire depuis Unit 50, chantier plus lourd que
    les autres tools — nécessite d'adapter la forme `analysis` de
    `POST /api/draft` au modèle `ProjectSummary` persisté).

- **Unit 59 — DraftReplyTool (2026-07-19)** :
  - Contexte : dernier point sur la liste énumérée à l'utilisateur, demande
    explicite d'implémentation. Recherche préalable :
    `email_analyzer/templates.py::generate_response_draft` est déjà un
    moteur de templates **déterministe** (routine/alerte/kickoff choisi par
    heuristique sur risque/tendance/mots-clés, `format_template`
    remplit un gabarit) — pas d'appel LLM du tout, contrairement à
    l'hypothèse initiale. Utilisé aujourd'hui uniquement par le legacy
    `POST /api/draft` (`api/main.py`), qui attend un dict `analysis` à
    clés françaises (`évaluation_risque.niveau_risque`,
    `analyse_sentiment.tendance`, `emails_critiques`, `résumé_assistant.
    texte`) — la sortie de `EmailProcessor.generate_intelligent_summary`
    (mode IMAP), qui n'a rien à voir avec la forme SaaS persistée
    (`ProjectSummary`/`ProjectSummaryLLM.structured_content`, clés
    anglaises : `risques`, `next_steps`, `sentiment_trend`,
    `llm_risk_level`). Décision : **ne pas réécrire le moteur de
    templates** (fonctionne, testé de longue date par le legacy) — écrire
    uniquement l'adaptateur SaaS→legacy, une seule fois, dans le tool.
  - `email_analyzer/tools/draft_reply_tool.py` (nouveau) : `DraftReplyTool`
    — charge `Project`+`ProjectSummary` (404 si projet inconnu du tenant,
    422 si aucun résumé), construit le dict `analysis` legacy :
    `niveau_risque` = `ProjectSummary.llm_risk_level` (déjà au format
    attendu — FAIBLE/MODÉRÉ/CRITIQUE, confirmé dans le prompt système de
    `llm.py:366`), `recommandation` = description du premier `next_step`,
    `facteurs_risque` = descriptions de `structured_content["risques"]`,
    `tendance` = normalisée depuis `sentiment_trend` (contient
    "posit"/"nég"/"neg" → Positive/Négative/Neutre) avec repli sur
    `ProjectSummary.sentiment` (positif/under_tension/awaiting_feedback →
    Positive/Négative/Neutre) si `sentiment_trend` absent,
    `emails_critiques` = 3 emails les plus importants du projet (même
    requête que `SearchEmailsTool`, Unit 50), `résumé_assistant.texte` =
    `ProjectSummary.content`. Appelle ensuite
    `templates.generate_response_draft(project.name, analysis)` sans
    modification.
  - `api/routers/projects.py` : nouvel endpoint
    `POST /api/projects/{project_id}/draft-reply` (contrôleur HTTP fin,
    même style que le reste du routeur) — **pas** de nouveau routeur créé,
    ajouté à `projects_router` existant puisque l'action est scopée à un
    projet, comme `POST /api/projects/{id}/refresh`/`/merge`. Décision
    documentée : ce tool a été **exposé via l'API** (contrairement aux 6
    précédents, usage interne uniquement) parce que le Planner ne peut pas
    le déclencher automatiquement (pas de résolution de `project_id` depuis
    le texte libre, même limite que `summarize_project`/`search_emails`) —
    sans consommateur explicite, le tool serait du code mort, contraire à
    "Ban Speculative Coding" (`ai-workflow-rules.md` §1.3).
  - `email_analyzer/agent/executor.py` : `DraftReplyTool` ajouté au mapping
    `self._tools` (cohérence avec `summarize_project`/`search_emails`,
    déjà enregistrés bien que jamais inclus dans un plan) — mais **pas**
    ajouté aux déclencheurs du `Planner` (même raison que ci-dessus).
  - Vérifié : syntaxe (`ast.parse`) des 3 fichiers modifiés/créés ; import
    réel dans le venv du projet (`myenv`) et de `api.main:app` (toujours 13
    routers — endpoint ajouté à un routeur existant, pas de nouveau
    routeur) ; **script de vérification réel** (scratchpad, supprimé après
    usage, **aucun LLM nécessaire** — le moteur est déterministe) contre la
    vraie base Postgres locale : projet avec résumé `llm_risk_level=
    "CRITIQUE"`/risques/next_steps/email important → `template_id=
    "alerte"`, corps citant le nom du projet et la checklist contenant le
    facteur de risque et l'email critique ; projet avec résumé
    `llm_risk_level="FAIBLE"`/`sentiment_trend="Positive"` → `template_id=
    "routine"` ; projet sans résumé → 422 propre ; projet inconnu → 404.
    Les 4 scénarios confirment que le moteur de templates existant réagit
    correctement à des données mappées depuis la forme SaaS réelle, pas
    seulement à la forme legacy d'origine. Données de test supprimées
    après coup.
  - Non vérifié : usage frontend (`POST /api/projects/{id}/draft-reply`
    n'est appelé depuis aucune page — pas de bouton "Générer un brouillon"
    dans `ProjectDetailPage.tsx` à ce stade, cohérent avec le choix de ne
    pas construire d'UI spéculative pour un endpoint qui vient d'être créé).
  - **Bilan** : les 6 tools "core" nommés explicitement par le plan de
    pivot (SearchEmailsTool, SearchProjectsTool, SearchMemoryTool,
    DraftReplyTool, SummarizeProjectTool, FindContactTool) plus
    `ListMeetingsTool` (ajout justifié par une capacité agenda déjà
    existante) sont maintenant tous implémentés et vérifiés. Plus aucun
    point sur la liste énumérée précédemment, à l'exception de
    l'absence d'outillage automatisé (assumée, documentée dans
    `code-standards.md` Unit 58) et des vérifications humaines
    recommandées (rendu navigateur, fournisseur OpenAI).

- **Unit 60 — Chat rapide + analyse auto en arrière-plan avec indicateur
  (2026-07-19)** :
  - Contexte : trois demandes produit. Exploration préalable :
    `EmailProcessor.fetch_last_n_emails_for_chat` (`analyzer.py`, chat
    legacy `/api/chat`) ouvrait une connexion IMAP live à chaque message
    (`use_email_cache=False`) alors qu'un cache JSON par tenant existe
    déjà, rempli par `/api/analyze`/`/api/projects/{id}/refresh`
    (`project_mail.py::load_cache`) mais jamais lu par le chat. Le chat
    SaaS (`/api/assistant/chat` → `ConversationService`) lisait déjà
    uniquement la DB (jamais IMAP), donc conforme à la demande 1 côté
    source de données, mais avec un budget de latence non maîtrisé :
    historique non borné (défaut repository 100 messages),
    N+1 non borné dans `ContextService._build_projects_context`, et
    **aucun timeout** sur les deux appels Gemini du chat conversationnel
    (`project_assistant_chat_gemini` et `portfolio_assistant_chat_gemini`,
    `chat.send_message(...)` sans `request_options`). L'analyse
    automatique en arrière-plan existait déjà (auto-sync login 1x/24h +
    cron `run_agenda_refresh`/`run_scheduled_sync`, file `arq`/Redis,
    `jobs.py`) — il manquait seulement un indicateur global côté frontend.
  - **Demande 1 (chat → données disponibles, pas de relance d'analyse)** :
    `analyzer.py::fetch_last_n_emails_for_chat` construit maintenant
    l'analyzer avec `use_email_cache=True` et tente d'abord
    `_filter_cached_emails_for_project` (nouvelle méthode statique) —
    filtre `analyzer.email_cache` via `search_project_emails_from_list`
    (déjà existant dans `project_mail.py`, réutilisé tel quel) sans ouvrir
    de connexion IMAP. Repli sur la connexion IMAP live **uniquement** si
    le cache ne contient aucune correspondance (aucune analyse n'a encore
    tourné pour ce filtre/tenant).
  - **Demande 2 (indicateur "analyse en cours")** : `jobs.py` — nouvel
    index Redis par tenant (`SET myconnector:active_jobs:{tenant_id}`,
    peuplé dans `create_job`, purgé dans `set_status` au passage
    done/error) + `list_active_jobs(tenant_id)` (auto-nettoyant : ignore/
    retire les job_id dont l'état a expiré ou est déjà terminé). Nouvel
    endpoint `GET /api/jobs/active` (`api/main.py`) exposant
    `{"active": bool, "count": int}` pour le tenant authentifié. Frontend :
    `useActiveJobIndicator.ts` (polling 15s, détection de présence
    uniquement — pas de barre de progression) + badge discret
    "Analyse en cours…" dans `AppShell.tsx` (mobile et desktop), visible
    sur toutes les pages SaaS, déclenché indifféremment par une action
    manuelle, le cron ou l'auto-sync au login.
  - **Demande 3 (chat < 3s, best-effort + filet de sécurité)** :
    `ConversationService.send_message` borne l'historique à 20 messages
    (`_CHAT_HISTORY_LIMIT`, au lieu du défaut 100 du repository) ;
    `ContextService._build_projects_context`/`_pending_action_descriptions`
    plafonnés (15 projets les plus récemment actifs, 5 actions en attente
    par projet — le N+1 sous-jacent n'est pas éliminé, seulement borné,
    hors périmètre). Bug corrigé : les deux appels
    `chat.send_message(...)` Gemini (`project_assistant_chat_gemini` et
    `portfolio_assistant_chat_gemini`, `llm.py`) n'avaient aucun timeout
    (contrairement aux autres appels Gemini du fichier, qui passent tous
    `request_options={"timeout": ...}`) — corrigé. Nouvelle constante
    dédiée `chat_llm_timeout_seconds()` (`config.py`, défaut 8s, env
    `CHAT_LLM_TIMEOUT_SECONDS`), distincte des 60s de fond
    (`llm_timeout_seconds()`), utilisée par `ConversationService` et
    `agent/response_builder.py::ResponseBuilder.build` (paramètre
    `timeout` ajouté à `portfolio_assistant_chat_openai`/`_gemini`) ;
    `max_tokens` réduit à `CHAT_INTERACTIVE_MAX_OUTPUT_TOKENS=500` pour
    ces deux chemins (au lieu de `CHAT_MAX_OUTPUT_TOKENS=4096`, partagé
    avec le chat legacy, non touché). `agent/executor.py::Executor` :
    les appels aux 2-4 outils d'un plan tournent maintenant en parallèle
    (`ThreadPoolExecutor`) au lieu de séquentiellement — **point
    d'attention thread-safety** : une `Session` SQLAlchemy n'étant pas
    thread-safe, l'Executor ne partage plus une instance de tool par
    thread ; il stocke des **classes** de tool et construit, pour chaque
    appel parallèle, une `Session` dédiée via un `sessionmaker(bind=
    db.get_bind())`, fermée dans un `finally`. Filet de sécurité : sur un
    timeout détecté (mots-clés "timeout"/"timed out"/"deadline" dans le
    message d'erreur — aucun des deux SDK n'expose un signal dédié),
    `ConversationService` renvoie `ContextService.quick_fallback_summary`
    (nouveau, réutilise les mêmes requêtes bornées que le system prompt,
    formatées en texte court) et `Agent.answer` renvoie
    `build_agent_context(tool_results)` (déjà existant) — dans les deux
    cas, réponse "selon les données disponibles" au lieu d'un 502 brut.
  - **Limite assumée** : le budget de 3s est garanti côté temps de réponse
    utilisateur (timeout serré + filet), pas comme une promesse sur la
    latence réelle des fournisseurs LLM tiers — pas de mesure de latence
    réelle en production possible dans cette session (pas d'accès aux
    clés API OpenAI/Gemini réelles).
  - **Correctif frontend complémentaire (même Unit, suite à relecture
    demandée par l'utilisateur)** : le fix backend ci-dessus ne couvrait
    que `/api/chat` lui-même — `useLegacyHomeChat.ts::send` (câblage de
    `ChatHome.tsx` en mode legacy) lançait encore une vraie analyse de
    fond (`POST /api/analyze`, `launchAnalysis`) dès le **premier message
    tapé**, avant même qu'un projet actif existe, ce qui reproduisait côté
    frontend exactement le problème corrigé côté backend. `send()`
    retravaillé : le texte tapé est maintenant essayé comme filtre projet
    sur `/api/chat` (chemin cache-first) même sans projet actif ;
    `sendChatReply` renvoie désormais `false` uniquement sur un 404
    explicite ("aucune donnée disponible pour ce filtre") pour déclencher
    le repli sur `launchAnalysis` — seul cas où lancer une vraie analyse
    reste justifié (rien à répondre autrement). Les déclencheurs
    d'analyse volontaires (`widenDays`, `selectThread`, quick actions
    "Analyser tous mes emails"/"Choisir une période") restent inchangés.
    Mode SaaS (`useSaasHomeChat.ts`) non touché : appelait déjà uniquement
    `/api/assistant/chat`, jamais d'analyse depuis le chat. Vérifié :
    `npx tsc --noEmit` sans erreur, `vite build` transforme les 72
    modules sans erreur (même limite d'environnement `dist/` que
    ci-dessus). Non vérifié : passage réel dans le navigateur du nouveau
    chemin cache-first/repli 404 (même contrainte d'environnement).
  - Vérifié (script réel contre Postgres + Redis locaux, données de test
    nettoyées après coup) : `ast.parse` + import réel de
    `api.main:app` (27 routes, inchangé en dehors du nouvel endpoint) sur
    les 10 fichiers backend modifiés ; `jobs.py` — `create_job`/
    `set_status`/`list_active_jobs` testés en séquence contre le vrai
    Redis local (présence → retrait au fur et à mesure des `done`) ;
    `Executor.execute` sur 3 outils en parallèle contre un vrai tenant
    Postgres, aucune erreur de concurrence, résultats dans l'ordre du
    plan ; `ContextService.build_portfolio_system_prompt`/
    `quick_fallback_summary` exécutés contre un vrai tenant ; simulation
    d'un timeout LLM (mock sur `portfolio_assistant_chat_openai`) confirmant
    que `ConversationService.send_message` bascule bien sur le résumé de
    secours et persiste le tour ; `_filter_cached_emails_for_project`
    testé avec un cache synthétique (2/3 emails correctement filtrés par
    mot-clé projet, cache vide → repli correctement signalé par une liste
    vide). Frontend : `npx tsc --noEmit` sans erreur ; `vite build`
    transforme les 72 modules sans erreur (échec de build limité au
    nettoyage du dossier `dist/` pré-existant appartenant à `root`, même
    problème d'environnement déjà documenté dans les entrées précédentes,
    sans rapport avec ce correctif).
  - Non vérifié : rendu navigateur réel du badge "Analyse en cours…"
    (mêmes limites d'environnement que les entrées précédentes — pas
    d'outil navigateur fonctionnel dans cette session) ; latence réelle
    bout-en-bout d'un appel LLM production (OpenAI/Gemini réels) sous le
    budget de 8s/3s — recommandé à l'utilisateur de mesurer avec `curl`
    chronométré une fois les clés API configurées.

- Fix : l'assistant portefeuille répondait "je n'ai pas d'information sur un
  projet nommé X" pour des projets réels, avec emails, correctement nommés
  (2026-08-01, cas concret : tenant à 112 projets, projet `teyliom.com` réel,
  nom exact, classé 84e par `updated_at` — donc absent du contexte système
  qui ne montre que les 15 projets les plus récemment mis à jour,
  `_MAX_PROJECTS_IN_CONTEXT` dans `context_service.py`). Root cause : le
  system prompt (`llm.py::build_portfolio_chat_system_prompt`) présentait
  cette liste tronquée comme "l'état actuel de chaque projet du portefeuille"
  sans le signaler, et demandait au LLM de déclarer l'absence d'info sans
  jamais l'inciter à vérifier via l'outil `search_projects` avant de
  répondre. Correctif (minimal, pas de changement de schéma) :
  `ContextService._build_projects_context` calcule maintenant le nombre
  total de projets du tenant et le passe à `build_portfolio_context(...,
  total_count=...)`, qui ajoute une note explicite si la liste est tronquée ;
  la règle du system prompt exige désormais d'appeler `search_projects` pour
  tout projet nommé absent de la liste avant de conclure à une absence
  d'information. Non traité (hors scope de ce correctif, à surveiller si le
  même symptôme réapparaît avec un nom de projet qui ne correspond pas
  exactement au domaine cherché) : `search_projects` et
  `ToolExecutor._resolve_project_id` (`tool_executor.py`) ne matchent que sur
  `Project.name`, jamais sur `rules_matrix.sender_domains` — si un jour un
  projet est nommé différemment de son domaine expéditeur, la recherche par
  domaine échouera encore malgré ce correctif. Vérifié : requêtes SQL directes
  confirmant le rang 84/112 et le nom exact `teyliom.com` ; lecture complète
  du chemin contexte → prompt → outils. Non vérifié : test bout-en-bout réel
  du chat (pas de clé LLM/navigateur dans cette session) — recommandé à
  l'utilisateur de reposer la même question en conditions réelles pour
  confirmer que le modèle appelle bien `search_projects` en pratique.

- Feature : élargissement de la recherche de l'assistant chat (suite directe
  du correctif ci-dessus, sur demande explicite de l'utilisateur — 2026-08-01)
  à 4 axes : résolution de projet par `rules_matrix`, recherche par mot-clé
  dans le corps des emails, extraits de contenu des documents, historique des
  résumés de projet. Plan complet dans le transcript (mode plan, approuvé) ;
  décisions validées avec l'utilisateur : créer un historique des résumés
  (nouvelle table, archive à partir de maintenant seulement — résumés déjà
  écrasés dans le passé non récupérables) et ajouter une vraie recherche par
  mot-clé à `search_emails`.
  - **A. Résolution par rules_matrix** : nouvelle fonction pure
    `rules_matrix_field_matches` (`classification.py`, après `ProjectRules`)
    cherchant un terme dans `sender_domains`/`sender_emails`/`client_names`/
    `company_names`/`keywords`. Branchée dans `ToolExecutor._resolve_project_id`
    (`tool_executor.py`, nouveau palier 2 : égalité exacte sur un champ
    rules_matrix, avant le palier substring qui matche désormais aussi
    rules_matrix) et dans `SearchProjectsTool.run` (`search_projects_tool.py`,
    filtrage Python quand `keyword` est fourni — JSONB n'a pas d'ILIKE natif
    ici, coût négligeable au volume actuel ~115 projets/tenant).
  - **B. Recherche mot-clé emails** : `SearchEmailsTool.run`
    (`search_emails_tool.py`) accepte désormais un `keyword` optionnel (ILIKE
    sujet+corps, tri par date, `body_snippet` ajouté) ; sans `keyword`,
    comportement exactement inchangé (rétrocompatible). `project_id` reste
    obligatoire — pas de recherche tous-projets (hors scope, fonctionnalité
    distincte plus large).
  - **C. Extraits documents** : `SearchDocumentsTool.run`
    (`search_documents_tool.py`) ajoute une clé `snippet` (contenu autour du
    match) quand `keyword` est fourni ; sans `keyword`, forme du dict
    inchangée. Aucun changement dans `DocumentRepository`/`DocumentService`
    (le filtre ILIKE existait déjà, seul le formatage de sortie change).
  - **D. Historique des résumés** : nouveau modèle `ProjectSummaryHistory`
    (`db/models.py`, après `ProjectSummary`) + migration
    `019_project_summary_history.py` (`down_revision = 018_document_source`).
    Hook d'archivage dans `analysis_tasks.py::_run_fasttrack_sync` (juste
    avant l'écrasement en place de `ProjectSummary`, même transaction/commit,
    condition `new_emails and (content ou structured_content existant)` pour
    ne rien archiver sur une régénération sans changement ou un résumé tout
    juste créé). Nouvel outil `SearchProjectSummariesTool`
    (`tools/search_project_summaries_tool.py`, même patron que
    `SearchEmailsTool`) branché dans `tool_executor.py`
    (`_tool_classes`/`_PROJECT_NAME_TOOLS`/`TOOL_SPECS`/dispatch).
  - Helper partagé `extract_snippet` (`text_snippets.py`, nouveau fichier,
    module pur) utilisé par B/C/D pour ne jamais renvoyer un corps/contenu
    complet au LLM (fenêtre de contexte autour du match, ou début du texte
    sinon).
  - `llm.py::build_portfolio_chat_system_prompt` mis à jour : mentionne que
    l'utilisateur peut désigner un projet par domaine/client autant que par
    nom, et ajoute `search_project_summaries` à la liste des outils
    disponibles avec la consigne de l'utiliser pour l'état passé d'un projet.
  - Vérifié en conditions réelles contre la base Postgres locale (tenant
    `566b33cb-...`, 115+ projets) : migration upgrade/downgrade/upgrade
    propre ; `_resolve_project_id("Mediasoft")` résout désormais vers
    `Smartbanking` (`rules_matrix.client_names`, échouait avant) ;
    `SearchProjectsTool(keyword="neurotechnology")` retourne `Biometrics
    Service` ; `search_emails(keyword="harmonisation")` retourne des
    résultats triés par date avec `body_snippet` pertinent, et sans mot-clé
    la forme du dict est inchangée ; `search_documents(keyword="COPIL")`
    retourne un `snippet` pertinent extrait d'un vrai PDF ; insertion/lecture/
    suppression d'une ligne `ProjectSummaryHistory` de test via
    `SearchProjectSummariesTool` round-trip complet (avec et sans mot-clé).
  - **Non vérifié** : déclenchement réel de `_run_fasttrack_sync` en bout en
    bout (nécessite IMAP + appel LLM réel, non fait dans cette session pour
    éviter coût/temps — la logique du hook a été relue ligne à ligne contre
    le code réel, mais son déclenchement en conditions de production reste à
    confirmer par l'utilisateur après le prochain resync d'un projet, via
    `SELECT count(*) FROM project_summary_history`) ; test bout-en-bout du
    chat réel (pas de clé LLM/navigateur dans cette session) — recommandé de
    reposer une question du type "où en était le projet X la semaine
    dernière ?" une fois qu'au moins un resync aura eu lieu après ce
    déploiement.

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
- ~~**Doc d'implémentation en attente d'une session dédiée**~~ — **Traité par
  Unit 44** (2026-07-17) pour `project-overview.md`, `architecture.md` et
  `ai-workflow-rules.md`. Reste ouvert : **`context/code-standards.md`**
  référence toujours Celery et React Native (stack jamais implémentée) —
  non traité par décision utilisateur explicite lors d'Unit 44 (hors scope
  du pivot stratégique), à reprendre dans une session dédiée si besoin.

- **Fix — taille de la chatbox (`ChatThread.tsx`) et cohérence des rayons de
  bordure dans l'accueil chat-first** (2026-07-16) :
  - Cause : le conteneur de `ChatThread` posait `min-h-[min(70vh,560px)]`
    (hauteur **minimale**, pas de hauteur maximale) sur un `flex-col` sans
    hauteur bornée — la zone interne `overflow-y-auto` n'avait donc jamais
    de contrainte de hauteur réelle pour déclencher un scroll : la boîte de
    chat grossissait indéfiniment avec les messages, faisant défiler toute
    la page au lieu de rester une boîte de taille fixe avec défilement
    interne (comportement attendu d'une chatbox).
  - `frontend/src/components/chat/ChatThread.tsx` : `showThread = messages.length
    > 0 || busy` — tant qu'il n'y a ni message ni chargement en cours (état
    initial de l'accueil), seule la zone de saisie est rendue (pas de grande
    boîte vide) ; dès qu'il y a du contenu, le conteneur passe à une
    **hauteur fixe** `h-[min(70vh,560px)]` (au lieu de `min-h-`), qui borne
    réellement la boîte et active le défilement interne existant.
  - Cohérence des rayons de bordure : plusieurs éléments de l'accueil
    chat-first utilisaient `rounded-2xl` (16px, valeur Tailwind par défaut,
    non définie dans `tailwind.config.js::borderRadius`) au lieu du jeton
    documenté `rounded-lg` (14px, `ui-context.md` §3 : *"Cartes de projet,
    panneaux (sidebar, **assistant**)"* — correspond explicitement au
    panneau de chat). Alignés sur `rounded-lg` : le panneau `ChatThread`
    lui-même, la carte "Aucun email trouvé", `AnalysisCard.tsx`, les 3
    cartes d'action rapide et les 3 bandeaux d'état de `ChatHome.tsx`, et le
    panneau d'outils legacy (période/modèle). Laissés en `rounded-2xl` par
    choix délibéré : les bulles de message texte/erreur (forme de bulle de
    chat, pas une "carte" au sens du jeton) ; les champs de saisie et
    boutons restent en `rounded-xl` (déjà la convention réelle utilisée
    partout ailleurs dans le dépôt — ex. `AppShell.tsx` — pas touchés pour
    éviter une refonte plus large hors du périmètre de cette demande).
  - Vérifié : `tsc --noEmit` sans erreur ; `vite build` transforme les 63
    modules sans erreur (échec de build limité au nettoyage du dossier
    `dist/` pré-existant appartenant à `root`, problème d'environnement déjà
    documenté ci-dessus, sans rapport avec le code).
  - Non vérifié : rendu visuel réel dans un navigateur (même contrainte
    d'environnement que les entrées précédentes — pas d'outil navigateur
    fonctionnel dans cette session) ; recommandé à l'utilisateur de lancer
    `npm run dev` localement pour confirmer visuellement la taille de la
    boîte et le défilement interne avec une longue conversation.

- **Fix — `useLegacyHomeChat.ts` : budget de polling `/api/analyze` désaligné
  du `job_timeout` arq (récidive du bug d'Unit 26)** (2026-07-16) :
  - Contexte : signalé par l'utilisateur (capture d'écran) — le bouton
    "Analyser tous mes emails" de `ChatHome.tsx` (mode legacy) affiche
    `L'analyse prend trop de temps. Réessayez plus tard.` alors que le job
    backend tourne souvent encore sainement.
  - Cause racine : `ANALYZE_POLL_TIMEOUT_MS` (`useLegacyHomeChat.ts`) était
    resté à 5 min (valeur d'origine, portée depuis l'ancien `HomePage.tsx`,
    cf. fix du 2026-07-08 ci-dessus) alors que `WorkerSettings.job_timeout`
    (`analysis_tasks.py`) est passé à 1800 s (30 min) ce même jour-là pour
    `run_analysis_legacy`/`run_analysis_saas` — les deux jobs derrière
    `/api/analyze`. Le frontend abandonnait donc jusqu'à 6x plus tôt que le
    vrai timeout backend. **C'est la même classe de bug que celle corrigée
    pour la découverte de domaines en Unit 26** (`discoverApi.ts`,
    `DISCOVER_POLL_TIMEOUT_MS` aligné à 126 min sur le timeout arq dédié de
    7200 s) — la réconciliation équivalente n'avait jamais été appliquée à
    `useLegacyHomeChat.ts`.
  - Vérifié opérationnellement avant correctif : process `arq
    email_analyzer.analysis_tasks.WorkerSettings` et `uvicorn api.main:app`
    bien actifs sur la machine — écarte l'hypothèse "aucun worker ne
    consomme les jobs", confirme que le problème est côté budget client.
  - `services/email_analyzer/frontend/src/hooks/useLegacyHomeChat.ts` :
    `ANALYZE_POLL_TIMEOUT_MS` relevé de 5 à 35 min (marge au-dessus des 30
    min de `job_timeout`, même proportion de marge que
    `DISCOVER_POLL_TIMEOUT_MS`/discovery timeout) ; commentaire mis à jour
    pour documenter l'alignement et éviter une nouvelle récidive. Aucun
    changement backend (`job_timeout = 1800` déjà correct) ni sur
    `discoverApi.ts` (déjà aligné).
  - Hors périmètre (volontairement) : reprise de job via `localStorage`
    façon `discoverApi.ts`/Unit 26 — répond à un problème différent (survie
    à un refresh/remount de page), pas nécessaire pour corriger le faux
    timeout signalé ici.
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` transforme les 63
    modules sans erreur (échec de build limité au nettoyage du dossier
    `dist/` pré-existant appartenant à `root`, même problème d'environnement
    déjà documenté dans l'entrée précédente, sans rapport avec ce correctif).
  - Non vérifié : passage réel dans le navigateur avec un job d'analyse
    dépassant effectivement 5 min (même contrainte d'environnement que les
    entrées précédentes) — correctif vérifié par build/type-check
    uniquement, comme l'avait été le correctif équivalent d'Unit 26 côté
    frontend.

- **Fix — `ChatHome.tsx` : personnage animé remplace l'avatar "orbe", greeting
  évolutif (salutation + sous-titres)** (2026-07-27) :
  - Contexte : demande utilisateur (image de référence : personne aux
    cheveux bouclés, lunettes rondes, assise à un bureau devant un
    ordinateur) — remplacer l'avatar "orbe" (cercle dégradé `ai-primary`→
    `ai-secondary` avec halo pulsant, introduit en Unit 36 ci-dessus) par un
    personnage animé façon 3D, et faire évoluer le message de greeting
    (salutation + "dates" qui varient) plutôt qu'un "Bonjour" figé.
  - Aucun outil de génération d'image/3D disponible dans cet environnement
    et aucun asset image dans le repo (`public/`, `src/assets/` :
    inexistants) — décision validée avec l'utilisateur (`AskUserQuestion`) :
    personnage construit en SVG codé à la main + `framer-motion` (déjà en
    dépendance `^12.42.2`, jusqu'ici inutilisé nulle part dans le code —
    premier usage), plutôt qu'un asset externe à fournir.
  - Nouveau `frontend/src/components/GreetingCharacter.tsx` : illustration
    SVG (`viewBox 200x200`, bureau/chaise statiques, tête avec cheveux afro
    + lunettes rondes + clignement des yeux, torse chemise off-white, bras
    avec tapotement alterné sur un clavier si `active`, écran de laptop en
    dégradé `ai-primary`→`ai-secondary` reprenant l'identité visuelle IA
    portée par l'ancien orbe), animée via `motion.g`/`motion.svg`
    (`useReducedMotion()` désactive les animations si
    `prefers-reduced-motion`). Props `size` (défaut 128, vs 64 pour
    l'orbe), `active` (pose "au repos", mains immobiles, pas de pulse écran
    — utilisé pour l'état `!chat.loaded`).
  - Nouveau `frontend/src/greeting.ts` (racine `src/`, pas `src/lib/` — le
    repo garde une convention plate pour ses petits modules : `apiUtils.ts`,
    `timelineApi.ts`, etc.) : `getTimeOfDaySalutation()` (Bonjour 5h-18h,
    Bonsoir 18h-22h, Bonne nuit 22h-5h) et `pickVariant()` (rotation de
    sous-titres indexée par jour-de-l'année — délibérément pas aléatoire au
    montage, car `ChatHome` peut se remonter en naviguant vers une
    conversation puis en revenant à `/`, ce qui rendrait un tirage aléatoire
    instable ; l'index par jour reste stable dans la journée tout en variant
    réellement d'un jour à l'autre, sans état/`localStorage` additionnel).
    Pools de sous-titres séparés pour SaaS-avec-brief (le texte "depuis
    {formatSince(...)}" existant reste branché tel quel), SaaS-sans-brief,
    et legacy.
  - `frontend/src/hooks/useBrief.ts` non touché : `formatSince` a un
    consommateur indépendant (`components/chat/ConversationHistoryPanel.tsx`)
    donc reste en place tel quel.
  - `frontend/src/index.css` : suppression de `@keyframes orb-glow` /
    `.animate-orb-glow` (Unit 36) — plus aucun consommateur après le
    remplacement dans `ChatHome.tsx`.
  - Ajustement post-vérification visuelle : le torse (blanc cassé
    `#F7F5F1`) était quasiment invisible en thème clair (`stroke-border-default`
    trop proche de `bg-primary`) — contour renforcé (`stroke-text-secondary`,
    largeur 2 pour le torse ; ajout d'un contour `stroke-text-secondary/40`
    sous les bras) pour rester lisible dans les deux thèmes.
  - Vérifié : `npx tsc --noEmit` sans erreur ; `vite build` transforme les
    478 modules sans erreur (échec de build limité au nettoyage du dossier
    `dist/` pré-existant appartenant à `root`, même problème d'environnement
    déjà documenté dans les entrées précédentes, sans rapport avec ce
    changement). Rendu visuel réel vérifié cette fois (Playwright +
    Chromium, disponible dans cette session contrairement aux entrées
    précédentes) : capture d'écran de l'écran d'accueil en mode legacy,
    thème clair et sombre — personnage bien positionné au-dessus du texte
    de greeting, animations (bob, clignement, tapotement, pulse écran)
    fonctionnelles sans erreur console, lisible dans les deux thèmes après
    l'ajustement de contraste ci-dessus.

- **Feature — Knowledge Layer : sync automatique + bouton chat + avatar/cartes
  repositionnés en conversation** (2026-07-27) :
  - Contexte : demande utilisateur — le Knowledge Layer (`knowledge_service.py`,
    Unit 47/pivot Phase 6) restait alimenté uniquement par l'action manuelle
    `POST /api/knowledge/projects/{id}/sync`, sans aucun lien avec le pipeline
    d'analyse ni UI dans le chat ; l'avatar (`GreetingCharacter`, voir entrée
    ci-dessus) et les cartes `quickActions` (Analyser/Actions/Projets à risque)
    disparaissaient dès la première réponse du chat au lieu de rester visibles.
  - **Sync continue** : `_run_fasttrack_sync` (`analysis_tasks.py`) appelle
    désormais `KnowledgeService.sync_from_project_summary` juste après son
    `db.commit()`, gaté sur `if new_emails:` (seul cas où `structured_content`
    change) — couvre en un seul point tous les déclencheurs existants
    (rafraîchissement manuel, `run_scheduled_sync` 2x/jour, `run_agenda_refresh`).
    Échec isolé (try/except + log), n'invalide jamais l'analyse déjà commitée.
  - **Filet de rattrapage horaire** : nouveau job cron arq
    `run_knowledge_sync_scan` (`cron(..., minute=0, max_tries=1)` ajouté à
    `WorkerSettings.cron_jobs`, aux côtés des deux jobs existants) — reprend
    tout projet dont `ProjectSummary.updated_at` a bougé dans la dernière heure
    (colonne déjà existante, `onupdate=datetime.utcnow`, aucune migration
    nécessaire) sans table de suivi "dernier scan" dédiée : fenêtre = cadence
    du job, filet de sécurité assumé comme approximatif (pas le chemin
    principal). Aucune nouvelle infra de déploiement (même process arq unique).
  - **Bouton "Synchroniser" dans le chat** : nouvel endpoint bulk
    `POST /api/knowledge/sync` (sans `project_id`, scope tenant courant) →
    `KnowledgeService.sync_all_for_tenant`, synchrone (pas de job arq/polling —
    l'opération ne fait que relire des données déjà en base, aucun appel
    LLM/IMAP). Client `syncAllKnowledge()` ajouté à `knowledgeApi.ts` ; bouton
    dans la toolbar de `ChatHome.tsx` (à côté de "+ Nouvelle conversation" /
    "Historique"), état `syncing`/`syncError`/`syncMessage` avec bannières de
    résultat réutilisant les classes déjà en place pour `discoverSuccess`.
  - **Avatar + cartes en colonne gauche pendant la conversation** :
    `ChatHome.tsx` gagne un flag dérivé `showSidebar` (conversation démarrée +
    mode saas) ; si vrai, layout `lg:grid lg:grid-cols-[280px_1fr]` avec
    `GreetingCharacter size={64} active={false}` + `quickActions` (même markup,
    factorisé dans un helper `renderQuickActions` pour éviter la duplication)
    empilées verticalement à gauche, toolbar + `ChatThread` à droite. En
    dessous de `lg:`, empilement naturel (pas de logique responsive
    additionnelle). L'état pré-conversation (avatar 128px centré + grille
    horizontale) reste strictement inchangé — changement additif uniquement.
    Piège TypeScript rencontré et résolu : TS narrowe `mode` à `"saas"` à
    l'intérieur du bloc `{showSidebar ? ... }` (narrowing d'alias sur
    `const showSidebar = inConversation && mode === "saas"`), ce qui rendait
    invalides les `mode === "legacy" ? ... : ...` dupliqués dans les deux
    branches ; résolu en factorisant `ChatThread` dans une unique variable
    `chatThreadEl` calculée avant le ternaire (mode non re-narrowé à cet
    endroit), réutilisée telle quelle dans les deux branches.
  - Vérifié : import direct de `analysis_tasks`/`api.routers.knowledge`
    (aucune erreur, 3 cron jobs enregistrés, 4 routes Knowledge exposées sans
    conflit de path) ; `npx tsc --noEmit` sans erreur ; `vite build` transforme
    les modules sans erreur (échec limité au nettoyage de `dist/` pré-existant
    appartenant à `root`, même problème d'environnement déjà documenté
    ci-dessus, sans rapport avec ce changement). Non vérifié dans un navigateur
    réel (pas de session Playwright disponible dans cette exécution) : à
    confirmer visuellement par l'utilisateur (bouton Sync + repositionnement
    colonne gauche en desktop/mobile).

- **Feature — Contexte Personnel (Profil & Habitudes) + feuille de route des
  25 domaines de contexte** (2026-07-30) :
  - Contexte : l'utilisateur a fourni une vision produit en 25 domaines de
    contexte ("second cerveau") pour transformer my-connector en assistant
    exécutif complet. Vu l'ampleur (beaucoup de domaines nécessitent des
    comptes/API externes non disponibles, et `ai-workflow-rules.md` interdit
    le code spéculatif), portée clarifiée avec l'utilisateur
    (`AskUserQuestion`, plan mode) : construire uniquement le Domaine #1
    ("Contexte Personnel", explicitement "la base de tout" pour
    l'utilisateur) dans cette unité, et documenter les 24 autres domaines
    comme feuille de route (statut réel par domaine) plutôt que du code
    spéculatif — voir `context/project-overview.md` "Feuille de route — 25
    domaines de contexte" pour le détail domaine par domaine.
  - **Modèle** : nouvelle table `profiles` (migration `016_profile`,
    `db/models.py::Profile`) — un enregistrement par (tenant, user), unique
    (contrairement à `Memory`, plusieurs lignes par pair). Colonnes typées
    pour le profil (`display_name`, `languages: ARRAY(String)`, `timezone`,
    `location`) et les habitudes (`wake_time`/`sleep_time`/
    `work_start_time`/`work_end_time: Time`, `sport_days: ARRAY(String)`,
    `focus_hours`) ; JSONB pour `preferences` et `family_members` (listes
    ouvertes par nature, même logique que `Project.rules_matrix`/
    `Email.tags`) ; `favorite_contact_ids: ARRAY(UUID)` référence `Contact`
    (pas de duplication de données contact).
  - **Backend** : `ProfileRepository` (get/upsert, get-or-create + update
    partiel — pas de list/delete par id, c'est un singleton), `ProfileService`
    (validation minimale, pas d'appel LLM — le profil est saisi directement
    par l'utilisateur, contrairement à `Memory.extract_and_store`),
    `api/routers/profile.py` (`GET`/`PATCH /api/profile`, get-or-default,
    pas de 404), enregistré dans `api/main.py`.
  - **Context Engine** : `llm.py::build_profile_context` (même convention que
    `build_memory_context`/`build_knowledge_context` — chaîne vide si profil
    absent/vide) ; `ContextService._build_profile_section` ajouté à
    `extra_sections` de `build_portfolio_system_prompt`, aux côtés de Memory
    et Knowledge. C'est ce qui rend le cas d'usage demandé par l'utilisateur
    réel : un profil avec horaires de travail 8h-18h produit dans le prompt
    système "Ne propose pas de réunion en dehors de cette plage sauf demande
    explicite" — vérifié (voir ci-dessous), pas seulement documenté.
  - **Frontend** : `profileApi.ts`/`hooks/useProfile.ts` (même patron que
    `memoryApi.ts`/`useMemoryAndKnowledge.ts`), nouvelle section "Profil &
    habitudes" (`components/ProfileSettingsForm.tsx`) insérée dans
    `SettingsPage.tsx` entre "Compte et organisation" et "Boîte IMAP" — pas
    de nouvelle route (le profil personnel a sa place naturelle dans
    Paramètres, pas une page dédiée). Champs listes (langues, jours de
    sport) en saisie CSV, famille/préférences en zone de texte ligne par
    ligne (`Nom, relation` / `clé: valeur`) — pas d'éditeur structuré en v1.
    Volontairement pas de sélecteur de contacts favoris (`favorite_contact_ids`
    existe côté modèle/API sans UI — nécessiterait un multi-select croisant
    la liste de `/memory`, hors périmètre de ce formulaire).
  - **Feuille de route** : `AppShell.tsx::COMING_SOON_ITEMS` étendu de 2 à 8
    entrées (Documents, Clients, + Tâches, Communication, Finance, Santé,
    Développement, Automatisation) — regroupées par thème plutôt qu'une
    entrée par domaine (25 entrées grisées nuirait à la lisibilité de la nav
    sans servir la compréhension du travail de l'utilisateur). Détail complet
    des 25 domaines (statut, prérequis) dans `project-overview.md`, pas dans
    la nav.
  - Vérifié (Postgres réel jetable, `initdb`/`pg_ctl` local, pas de `sudo` sur
    l'instance système) : `alembic upgrade head` applique toute la chaîne de
    migrations jusqu'à `016_profile` sans erreur ; round-trip
    `downgrade -1` puis `upgrade head` confirmé symétrique (table
    supprimée puis recréée à l'identique). Script Python jetable :
    `ProfileService.update_profile` créé/mis à jour un profil réel en base,
    confirmé qu'une mise à jour partielle (`{"display_name": ...}` seul) ne
    touche pas les autres champs déjà enregistrés ; `ContextService.
    build_portfolio_system_prompt` relu et confirmé que le bloc "Contexte
    personnel de l'utilisateur" apparaît avec le texte attendu sur les
    horaires de travail. Round-trip HTTP réel via `TestClient` (utilisateur/
    tenant/token réels, pas de mock) : `GET /api/profile` sans profil
    (200, tous champs `null`), `PATCH` avec plusieurs champs (200, valeurs
    reflétées, heures sérialisées en `"HH:MM:SS"`), `GET` après `PATCH`
    (persistance confirmée), `GET` sans token (401). Import direct de
    `api.main` : route `/api/profile` (`GET`/`PATCH`) enregistrée sans
    conflit de path, 28 routes totales dans l'app (comptage via
    `original_router.routes`, l'introspection FastAPI 0.139 diffère des
    versions précédentes — `app.routes` seul ne suffit plus à lister les
    routes incluses via `include_router`). Cluster Postgres jetable arrêté
    et supprimé après vérification (aucun état laissé).
  - Vérifié (frontend) : `npx tsc --noEmit` sans erreur ; `vite build`
    transforme 483 modules sans erreur (échec limité au nettoyage du dossier
    `dist/` pré-existant appartenant à `root`, même problème d'environnement
    déjà documenté dans les entrées précédentes, sans rapport avec ce
    changement).
  - Non vérifié : rendu visuel réel dans un navigateur (pas de session
    Playwright lancée pour cette unité) — à confirmer par l'utilisateur
    (formulaire "Profil & habitudes" dans Paramètres, thème clair/sombre,
    nouvelles entrées "Bientôt disponible" dans la nav).

- **Rattrapage documentaire — Contexte Documents (domaine #7), backend
  construit hors-cycle sans entrée de suivi** (constaté et documenté le
  2026-08-01, code déjà présent avec dates de modification du 2026-07-31) :
  - Contexte : reprise du plan d'implémentation (« continue le plan
    d'implémentation ») — avant d'ajouter du code, exploration du dépôt
    (agent dédié) pour confirmer l'état réel plutôt que de faire confiance
    à ce fichier seul, qui n'affirmait nulle part que le domaine #7 avait
    une quelconque implémentation (`project-overview.md` le classait
    encore "À construire" jusqu'à cette révision). Constat : le backend
    complet existait déjà, non committé, jamais journalisé ici — cas
    concret de l'écart tracker/code déjà rencontré par le passé.
  - **Ce qui existait déjà** (aucun changement apporté par cette entrée,
    documentation a posteriori uniquement) : `Document` (`db/models.py`,
    une ligne par fichier — `tenant_id`, `user_id`, `project_id` nullable,
    `filename`, `content_type`, `size_bytes`, `extracted_text`,
    `created_at`) ; migration `017_documents.py` (chaîne `011→...→017`
    linéaire, cohérente avec le modèle) ; `email_analyzer/
    document_extraction.py` (extraction PDF via `pypdf` + texte brut
    `.txt`/`.md`/`.csv`, plafonnée à 200k caractères, ne lève jamais —
    Word/Excel/PowerPoint/images explicitement non supportés, par choix
    documenté, pas un oubli) ; `DocumentService`/`DocumentRepository`
    (upload avec validation taille/vide, stockage sur disque par
    `tenant_id/document_id` — jamais le nom de fichier original, pour
    éviter toute traversée de chemin —, liste avec filtre projet/mot-clé,
    téléchargement, suppression DB+disque) ; config
    `DOCUMENTS_STORAGE_DIR`/`DOCUMENT_MAX_SIZE_BYTES` (défaut 20 Mo) ;
    `api/routers/documents.py` (`GET/POST /upload/GET {id}/download/DELETE
    {id}`, déjà enregistré dans `api/main.py`) ; `tools/
    search_documents_tool.py` déjà câblé dans `tool_executor.py`
    (`TOOL_SPECS`, `_tool_classes`, projet optionnel, `_run_tool`) — donc
    déjà appelable par le LLM à la demande. `requirements.txt` a déjà
    `pypdf>=4,<6`.
  - **Ce qui manquait réellement** (confirmé par grep exhaustif sur
    `frontend/src`, pas seulement absence supposée) : aucune UI — l'entrée
    "Documents" de `AppShell.tsx::COMING_SOON_ITEMS` restait grisée alors
    que le backend fonctionnait ; et aucune injection dans le Context
    Engine (`context_service.py` documente explicitement que Documents
    n'y est pas encore ajouté, contrairement à Profile/Memory/Knowledge) —
    un document téléversé n'influence donc le chat que si le LLM appelle
    lui-même `search_documents`, jamais de façon ambiante. Voir l'unité
    suivante pour le comblement de la première lacune.

- **Feature — Contexte Documents (domaine #7) : UI frontend (upload, liste,
  téléchargement, suppression)** (2026-08-01) :
  - Contexte : suite directe de l'entrée précédente — le backend
    fonctionnait déjà en entier, seule l'UI manquait, même situation que
    Memory/Knowledge avant `MemoryPage.tsx` (Unit 53) ou
    Objectives/Timeline avant `ObjectivesPage.tsx` (Unit 56). Décision
    reprise du même patron plutôt que d'inventer une nouvelle convention :
    page dédiée `/documents` (pas une modale, pas intégrée à
    `ProjectDetailPage`), listée dans la nav (retirée de
    `COMING_SOON_ITEMS`).
  - `frontend/src/documentsApi.ts` (nouveau) : miroir de `DocumentOut`,
    `fetchDocuments`/`uploadDocument` (`FormData`, pas de `Content-Type`
    manuel — le navigateur fixe le boundary multipart)/`deleteDocument`,
    `listProjectOptions` (même convention que
    `discoverTicketsApi.ts::listProjectOptions`, dupliquée plutôt que
    partagée entre modules — convention déjà suivie par
    `discoverApi.ts`/`discoverTicketsApi.ts` eux-mêmes). `downloadDocument`
    matérialise un blob (`URL.createObjectURL` + `<a download>` synthétique)
    plutôt qu'un `<a href>` direct : la session vit dans un Bearer token
    (`apiFetch`), pas un cookie, donc un lien direct n'enverrait pas le
    jeton — premier consommateur de ce patron dans le repo (aucun
    téléchargement authentifié n'existait avant).
  - `frontend/src/hooks/useDocuments.ts` (nouveau) : même patron que
    `useMemoryAndKnowledge.ts` (liste + `reload(filters)` + mutation
    optimiste locale après upload/suppression, pas de re-fetch complet).
  - `frontend/src/pages/DocumentsPage.tsx` (nouveau) : formulaire de dépôt
    (fichier + sélecteur de projet optionnel), formulaire de filtre
    (mot-clé + projet), liste (nom, taille formatée, projet ou "Sans
    projet", "Texte extrait" si applicable, date, actions
    Télécharger/Supprimer). Pas d'éditeur de métadonnées, pas de
    prévisualisation du contenu extrait — hors périmètre de cette unité.
  - `App.tsx` (route `/documents`) et `AppShell.tsx`
    (`SAAS_NAV_ITEMS` gagne "Documents", retiré de `COMING_SOON_ITEMS`,
    seul groupe restant : Clients/Tâches/Communication/Finance/Santé/
    Développement/Automatisation).
  - Vérifié bout-en-bout contre un cluster Postgres jetable réel
    (`initdb`/`pg_ctl`, socket court `/tmp/pgsock_myconn`, supprimé après
    coup) : `alembic upgrade head` rejoue toute la chaîne `001→017` sans
    erreur (première vérification réelle des migrations 011-017, jamais
    exercées contre une vraie base dans une entrée précédente de ce
    fichier) ; utilisateur/tenant/token réels seedés directement via les
    modèles SQLAlchemy + `create_access_token` (contournement identique à
    celui documenté précédemment pour l'inscription) ; upload/liste réels
    via `curl` multipart contre un vrai `uvicorn`. Puis vérification
    navigateur réelle (Playwright + Chromium, backend réel + `vite` dev
    réel, pas de mock) : upload d'un fichier via l'UI, apparition dans la
    liste, téléchargement (nom de fichier confirmé via l'événement
    `download` de Playwright), suppression (disparition confirmée de la
    liste) — les trois flux rejoués en thème clair et sombre, aucune
    erreur console dans les deux cas. `npx tsc --noEmit` sans erreur ;
    `vite build` transforme 486 modules sans erreur (échec de build
    limité au nettoyage du dossier `dist/` pré-existant appartenant à
    `root`, même problème d'environnement déjà documenté dans les entrées
    précédentes, sans rapport avec ce changement).
  - Piège d'environnement rencontré et contourné (pas un bug du code) :
    `frontend/.env` définit `VITE_API_BASE=https://kckvfkftbdp4.shares.zrok.io`
    (un tunnel de développement de l'utilisateur, antérieur à cette
    session) — sous Playwright, cela déclenchait un échec CORS silencieux
    vers ce tunnel au lieu du backend local, faisant passer l'app entière
    en mode legacy (santé API injoignable ⇒ `saasEnabled=false` ⇒
    `/documents` redirige vers `/`). Contourné pour la seule durée de la
    vérification via `VITE_API_BASE="" npx vite ...` (une variable
    d'environnement de processus prime sur `.env` chez Vite) — fichier
    `.env` de l'utilisateur non modifié. À garder en tête pour toute
    future vérification Playwright dans cet environnement.
  - Non vérifié : upload d'un PDF réel (testé uniquement avec un fichier
    texte simple, `document_extraction.py` pour le chemin PDF n'a été
    revérifié que par lecture de code, pas par un upload réel de PDF dans
    cette unité) ; comportement au-delà de la limite de 20 Mo ; upload
    rattaché à un projet réel (testé sans projet uniquement, le sélecteur
    de projet n'a pas été exercé avec une vraie sélection).

- **Feature — Ingestion automatique des pièces jointes d'email en Documents,
  enrichissement de l'analyse et des réponses** (2026-08-01) :
  - Contexte : demande utilisateur explicite (« télécharger les documents
    des emails, les extraire pour enrichir l'analyse et les réponses »),
    suite directe de la mise en place du Contexte Documents (domaine #7,
    entrées précédentes du même jour). Périmètre choisi (décision
    documentée plutôt que question à l'utilisateur, cf. discipline
    "défaut minimal honnête") : brancher sur le pipeline Fast-Track déjà
    existant (seul chemin qui traite réellement les emails aujourd'hui
    pour un projet) plutôt qu'ajouter un scan rétroactif de tout
    l'historique — voir limites assumées ci-dessous.
  - **Portée volontairement restreinte à IMAP** : `process_delta`
    (Fast-Track, `analyzer.py`) reste IMAP-only (limite déjà documentée à
    l'Unit 15/17 — les tenants Gmail/Outlook-only sans IMAP configuré
    n'ont aujourd'hui aucun Fast-Track fonctionnel, pas seulement pas de
    pièces jointes). Cette unité n'élargit pas ce périmètre : elle branche
    l'extraction de pièces jointes exactement là où des emails sont déjà
    effectivement traités.
  - **Extraction** (`project_mail.py::extract_email_content`) : parcourt
    désormais aussi les parties du message pour capturer les pièces
    jointes nommées (`part.get_filename()`), filtrées via la nouvelle
    fonction partagée `document_extraction.is_supported_attachment`
    (réutilise exactement la même logique PDF/texte que `extract_text` —
    extraite en `_is_pdf`/`_is_plain_text` pour éviter la duplication du
    prédicat entre les deux modules). Décision : ne capturer que ce qui
    est réellement extractible/recherchable (PDF, `.txt`/`.md`/`.csv`) —
    les images/logos de signature (`image/png` etc.) ne sont jamais
    capturés, pour ne pas polluer `/documents` de pièces jointes inline
    sans valeur pour l'analyse (vérifié : un email de test avec un
    `.txt`, un `.csv` et un `.png` ne capture que les deux premiers).
    Stockées dans une clé `_attachments` (préfixe `_`, signale
    explicitement "éphémère, pas une métadonnée stable") **retirée avant
    toute écriture dans le cache JSON partagé** (`_process_one_email_id`)
    — les octets bruts ne sont ni sérialisables tels quels, ni
    souhaitables à dupliquer sur disque à chaque scan ; seule la copie en
    mémoire de l'exécution courante les porte. Conséquence assumée :
    seuls les emails traités pour la première fois par le Fast-Track
    (donc jamais présents dans le cache) auront leurs pièces jointes
    capturées — un email déjà en cache (rescan dans la même fenêtre) n'en
    aura pas, sans que cela soit un problème pratique puisque l'ingestion
    en base est de toute façon restreinte aux emails nouveaux (point
    suivant).
  - **Persistance** (`email_analyzer/services/document_service.py`) :
    nouvelle méthode `ingest_email_attachment` (à côté de `upload`,
    factorisées via un `_persist` commun) — jamais d'exception
    (contrairement à `upload`, pensé pour une requête HTTP) : fichier
    trop volumineux (`DOCUMENT_MAX_SIZE_BYTES`, 20 Mo) ou vide simplement
    ignoré. Dédoublonnée sur (tenant, projet, nom, taille) via la nouvelle
    `DocumentRepository.find_existing` — pas un identifiant stable par
    pièce jointe (hors périmètre v1, limite documentée dans le code) :
    suffisant pour éviter de recréer le même document à chaque re-scan
    (ex. bouton "chercher sur plus de jours") sans nouvelle table de
    suivi. Nouvelle colonne `Document.source` (migration
    `018_document_source`, nullable — même discipline que 004/005 sur
    `emails` : les lignes antérieures ne sont pas rétro-devinées) :
    `"upload"` (dépôt manuel) vs `"email"` (cette unité), exposée dans
    `DocumentOut`/`documentsApi.ts` et affichée en badge "Email" sur
    `/documents`.
  - **Câblage Fast-Track** (`analysis_tasks.py::_run_fasttrack_sync`) :
    nouvelle fonction `_ingest_new_email_attachments`, appelée juste après
    le hook `KnowledgeService.sync_from_project_summary` déjà existant
    (même bloc `if new_emails:`, même discipline best-effort — un échec
    n'invalide jamais l'analyse déjà commitée). Restreinte aux emails
    réellement nouveaux dans cette exécution via `seen_external_ids` (le
    même ensemble déjà utilisé pour dédupliquer l'insertion des `Email` —
    pas un recalcul indépendant) : aucun rattrapage rétroactif des emails
    déjà en base avant l'introduction de cette fonctionnalité. Utilisateur
    attribué au document (`Document.user_id`) : la première `Membership`
    "owner" du tenant (repli sur la première membership tout court) — un
    job Fast-Track (cron ou déclenchement manuel) n'a pas d'utilisateur
    acteur unique à qui attribuer l'ingestion, décision minimale assumée
    plutôt qu'une nouvelle notion de "compte système".
  - **Enrichissement de l'analyse et des réponses** — deux mécanismes
    distincts, décision explicite de ne pas aller au-delà :
    1. *Réponses* : un document ingéré devient immédiatement trouvable
       par l'outil agent `search_documents` (déjà câblé dans
       `tool_executor.py` depuis l'unité précédente) — donc par le chat
       portefeuille et, via lui, par `draft_reply`. Aucun changement de
       code nécessaire ici, c'est la conséquence directe de le stocker
       comme `Document`.
    2. *Analyse* : signal ambiant `document_count` par projet
       (`context_service.py::_document_counts`, une requête groupée —
       même principe que `_open_ticket_counts`), rendu dans
       `llm.py::build_portfolio_context` ("Documents disponibles : N").
       Décision explicite **contre** l'injection du texte extrait
       lui-même dans le prompt système (contrairement à Profile/Memory/
       Knowledge) : un document peut porter jusqu'à 200k caractères de
       texte extrait (`document_extraction._MAX_EXTRACTED_CHARS`), et il
       n'existe aujourd'hui aucune politique de troncature/sélection
       raisonnée à appliquer sans un nouvel arbitrage produit — documenté
       comme limite explicite dans le docstring de `context_service.py`
       plutôt que d'improviser une politique arbitraire. Génération LLM
       structurée du résumé de projet (`ProjectSummaryLLM`) non modifiée
       non plus, pour la même raison (pas de nouvel appel LLM ajouté par
       cette unité, cohérent avec la discipline de coût déjà suivie
       partout ailleurs dans ce fichier).
  - Vérifié bout-en-bout contre un second cluster Postgres jetable réel
    (`initdb`/`pg_ctl`, port/socket distincts du premier, supprimé après
    coup) : `alembic upgrade head` rejoue `001→018` sans erreur (première
    vérification réelle de la migration 018). Script Python jetable :
    email synthétique multipart (`email.message.EmailMessage`, 1 corps +
    3 pièces jointes `.txt`/`.png`/`.csv`) → `extract_email_content`
    capture bien exactement les 2 pièces supportées (le `.png` est
    correctement écarté) → `_ingest_new_email_attachments` appelée deux
    fois de suite (simulant un re-scan) ne crée que 2 `Document` (pas 4 —
    dédoublonnage confirmé), tous deux `source="email"` avec
    `extracted_text` non vide → `ContextService._project_rows` /
    `build_portfolio_context` affichent bien "Documents disponibles : 2"
    pour le projet. Vérifié séparément (script isolé) que le contenu mis
    en cache après extraction reste bien JSON-sérialisable (pas de
    `TypeError` sur des `bytes`) une fois `_attachments` retiré, alors que
    l'objet `email_content` en mémoire les conserve. `python -m
    py_compile` sur tous les fichiers modifiés + `import api.main` (29
    routes, aucune erreur) + `WorkerSettings.functions` inchangé (aucun
    nouveau job arq ajouté — cette unité s'exécute dans le job Fast-Track
    existant). `npx tsc --noEmit` sans erreur ; `vite build` transforme
    486 modules sans erreur (échec de build limité au nettoyage du
    dossier `dist/` pré-existant appartenant à `root`, même problème
    d'environnement déjà documenté dans les entrées précédentes).
  - Non vérifié : passage réel contre un vrai serveur IMAP (le test
    ci-dessus construit un message synthétique en mémoire, jamais fetché
    via `imaplib`) ; upload d'une pièce jointe PDF réelle par ce chemin
    (testé uniquement avec `.txt`/`.csv`, `document_extraction.py` pour
    le chemin PDF n'a été revérifié que par lecture de code) ; effet réel
    du signal "Documents disponibles" sur une vraie réponse d'un LLM
    (vérifié uniquement que le texte apparaît dans le contexte assemblé,
    pas qu'un modèle Gemini/OpenAI réel choisit d'appeler
    `search_documents` en conséquence) ; comportement sous forte
    volumétrie de pièces jointes (un email avec des dizaines de pièces
    jointes volumineuses dans un même delta Fast-Track).

- **Feature — Rattrapage rétroactif des pièces jointes d'email, en
  arrière-plan avec indication de progression** (2026-08-01) :
  - Contexte : demande utilisateur explicite (« faire un rattrapage
    rétroactif en background sans blocage du frontend avec une indication »)
    — l'ingestion automatique des pièces jointes (entrée précédente du même
    jour) ne couvre que les emails traités par le Fast-Track *à partir de
    maintenant* ; les emails déjà persistés avant cette fonctionnalité (des
    mois/années d'historique selon le tenant) n'ont jamais eu leurs pièces
    jointes capturées. Portée confirmée explicitement avec l'utilisateur
    (`AskUserQuestion`, en plan mode) : tout l'historique déjà en base, pas
    seulement une fenêtre récente. Plan conçu via un agent Plan après
    exploration dédiée (patterns de jobs arq + IMAP), approuvé avant
    implémentation.
  - **Trois contraintes structurantes découvertes en explorant le code**
    (pas supposées a priori) : (1) aucun moyen de relire un email par
    Message-ID en IMAP dans ce code — tout est basé sur un scan par fenêtre
    `SINCE`, comme `discover_sender_domains`/`discover_subject_ticket_keys` ;
    (2) le cache email JSON partagé ne porte jamais `_attachments` (retiré
    avant écriture, voir l'entrée précédente) — un re-scan avec cache activé
    sur des emails déjà connus renverrait donc silencieusement zéro pièce
    jointe, d'où `use_email_cache=False` obligatoire pour ce scan ; (3)
    réutiliser `search_project_emails` tel quel pour scanner tout
    l'historique accumulerait chaque corps + les octets bruts de chaque
    pièce jointe en mémoire simultanément avant de retourner — risque réel
    d'OOM sur une boîte multi-années, d'où un scan **streaming** dédié
    (callback par email, jamais d'accumulation).
  - `project_mail.py::EmailProjectAnalyzer.scan_attachments_by_message_id`
    (nouveau) : même squelette IMAP que `search_project_emails` (`select` +
    `SINCE`, sans `_uid_search_narrow_ids` puisqu'aucun filtre projet ici),
    mais appelle `on_email(message_id, attachments)` immédiatement pour
    chaque message puis abandonne la référence — jamais de liste accumulée.
    Toujours un fetch `RFC822` complet (jamais `headers_only`, il faut le
    corps pour les pièces jointes). Ne touche jamais `self.email_cache` (ni
    lecture ni écriture), quel que soit `use_email_cache` — documenté
    explicitement pourquoi (point 2 ci-dessus). `try/except` par email (log
    + continue), même arithmétique de chunking de progression que
    `search_project_emails` (`batch_threshold()`/`large_scan_threshold()`).
  - `analyzer.py::EmailProcessor.backfill_email_attachments` (nouveau) :
    façade IMAP-only (garde `email_address`/`password` en premier, comme
    `process_delta`) — **volontairement sans repli Gmail/Outlook**
    (contrairement à `_fetch_project_data`) : ces chemins renvoient des
    dicts déjà normalisés sans parties MIME, donc sans jamais
    `_attachments` — même limite préexistante que le Fast-Track
    (`process_delta` déjà IMAP-only, Open Question #1 / Units 15-17), non
    étendue par cette unité. Construit son propre `EmailProjectAnalyzer`
    avec `use_email_cache=False`.
  - `analysis_tasks.py` : `_attachment_backfill_days_back` (calcule la
    fenêtre depuis `MIN(Email.received_at)` des emails déjà rattachés à un
    projet du tenant — un email sans projet ne peut produire aucun
    `Document`, donc n'élargit pas la fenêtre ; gère un `received_at` naïf ;
    repli 30 jours si aucun email daté ; clamp à 3650 jours) ;
    `_run_email_attachment_backfill_sync` (précharge
    `{external_id: project_id}` pour tous les `Email` du tenant déjà
    rattachés à un projet, lance un seul scan IMAP complet via
    `backfill_email_attachments`, matche chaque email retrouvé contre cette
    map, ingère via `DocumentService.ingest_email_attachment` déjà
    idempotent — compteur avant/après sur `Document(source="email")` pour
    distinguer "créées" de "déjà présentes", le retour de
    `ingest_email_attachment` ne le permettant pas directement). Pas de
    `try/except: db.rollback(); raise` global comme `_run_fasttrack_sync` :
    chaque `Document` est commité individuellement par
    `DocumentRepository.create`, donc `db.rollback()` explicite dans le
    `except` par pièce jointe (sinon un insert en échec casse la session
    pour les pièces suivantes) — et un job interrompu (timeout, worker
    redémarré) a déjà persisté durablement tout ce qu'il a ingéré, un
    nouveau lancement reprend sans créer de doublon. Tenant sans IMAP
    configuré (Gmail/Outlook-only) : retourne `{"_error": ...}` (ne lève
    pas), même discipline que les jobs `discover_*`. `run_email_attachment_
    backfill` (wrapper arq, copié verbatim du patron `run_domain_discovery`,
    capture `CancelledError` explicitement) enregistré dans
    `WorkerSettings.functions` (`max_tries=1, timeout=7200` — limite
    assumée : une très grosse boîte peut ne pas finir dans ce délai, sans
    risque de corruption grâce aux commits par-document + dédoublonnage ;
    découpage multi-jobs hors périmètre de cette itération).
  - `api/routers/documents.py::POST /api/documents/backfill-from-emails`
    (nouveau, même squelette que `discover_projects`/`discover_tickets` de
    `brief.py` : `create_job`/`enqueue`) — placé ici plutôt que dans
    `brief.py` puisque le déclencheur vit sur la page Documents et que ce
    router porte déjà `_require_saas()`/`authenticate_bearer`. Aucune
    nouvelle route de polling : `GET /api/analyze/{job_id}` (générique,
    déjà existant) suffit. Aucune migration, aucun changement de modèle.
  - **Indication de progression sans blocage** : `frontend/src/
    attachmentBackfillApi.ts` (nouveau, clone structurel de
    `discoverApi.ts` — reprise via localStorage, polling 2.5s, budget aligné
    sur le timeout 7200s du worker) + `BackfillAttachmentsModal.tsx`
    (nouveau, version simplifiée de `DiscoverProjectsModal.tsx` : pas
    d'étape de sélection/confirmation puisque le rattrapage est entièrement
    automatique — juste démarrer, barre de progression, résumé final).
    Bouton "Rattraper depuis les emails" ajouté dans la section de dépôt de
    `DocumentsPage.tsx` (pas de changement à `useDocuments.ts`,
    `documentsApi.ts`, `App.tsx`, `AppShell.tsx`). L'indicateur ambiant
    "Analyse en cours…" (`useActiveJobIndicator`/`AppShell.tsx`) couvre déjà
    ce job gratuitement — `create_job(tenant_id=...)` l'enregistre
    automatiquement dans les jobs actifs du tenant, aucun code
    supplémentaire nécessaire pour ce signal ambiant.
  - Vérifié bout-en-bout contre un troisième cluster Postgres jetable réel
    (`initdb`/`pg_ctl`, port/socket dédiés, supprimé après coup) :
    `alembic upgrade head` rejoue `001→018` sans erreur (aucune nouvelle
    migration dans cette unité). Script Python jetable, `saas_logic.
    processor_from_tenant` monkeypatché avec un stub construisant des
    `email.message.EmailMessage` synthétiques passés par le **vrai**
    `extract_email_content` : (1) bornage de fenêtre — exclusion correcte
    d'un email sans projet même très ancien, repli 30 jours si aucun email
    daté, clamp à 3650 jours, `received_at` naïf géré sans lever ; (2)
    matching + ingestion — sur 3 emails synthétiques (1 rattaché à un
    projet avec `.txt`+`.csv`+`.png`, 1 sans projet, 1 inconnu en base),
    seul le premier compte "matched", seules les 2 pièces supportées sont
    ingérées (le `.png` correctement écarté), `Document.source == "email"`,
    `project_id`/`extracted_text` corrects, callback de progression appelé
    avec des paires `(processed, total)` croissantes jusqu'à `(3, 3)` ; (3)
    relance immédiate du même scénario → `attachments_created == 0`, aucun
    nouveau `Document` (idempotence confirmée) ; (4) tenant sans IMAP
    configuré → `_error` retourné (pas d'exception), aucun `Document` créé,
    vérifié à la fois au niveau du job et directement sur
    `EmailProcessor.backfill_email_attachments`. `python -m py_compile` sur
    tous les fichiers modifiés ; `import api.main` (61 chemins d'API dans le
    schéma OpenAPI, `POST /api/documents/backfill-from-emails` bien présent
    — le comptage brut de `app.routes` ne convient plus depuis FastAPI
    0.139, déjà noté dans une entrée antérieure) ; `run_email_attachment_
    backfill` confirmé dans `WorkerSettings.functions` avec `max_tries=1,
    timeout=7200`. `npx tsc --noEmit` sans erreur ; `vite build` transforme
    740 modules sans erreur (bond par rapport aux 486 précédents dû à des
    changements externes concurrents dans le dépôt pendant cette session,
    sans rapport avec cette unité — échec de build toujours limité au même
    problème d'environnement `dist/`/`root` déjà documenté).
  - Non vérifié : le vrai chemin IMAP (`scan_attachments_by_message_id`
    n'est jamais exercé contre un serveur réel, seul un stub simulant son
    contrat a été testé) ; le temps réel et la mémoire crête sur une boîte
    multi-années (le design streaming est *raisonné* à mémoire plate, pas
    mesuré en conditions réelles) ; le comportement d'un job réellement tué
    à 7200s en plein scan (durabilité déduite du commit par-document, pas
    observée sur un vrai kill) ; l'upload d'un PDF réel par ce chemin (même
    limite que l'entrée précédente, testé avec `.txt`/`.csv` uniquement) ;
    rendu visuel réel du modal dans un navigateur (pas de session Playwright
    lancée pour cette unité, contrairement à l'unité UI Documents du même
    jour).

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

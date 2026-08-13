# Project Overview: my-connector — Personal Executive AI Assistant

> **Pivot stratégique (2026-07-17)** — Ce document a été entièrement réécrit
> pour refléter le nouveau positionnement produit. Il remplace le texte
> précédent ("Intelligent Project Management & Email Assistant") et les
> notes datées Unit 24 / Units 33-41 qui documentaient au fil de l'eau
> l'écart entre la vision d'origine (stack Node.js/Celery/React Native,
> cache vectoriel device) et le code réel. Voir `context/progress-tracker.md`
> pour l'historique unité par unité.

## Mission

my-connector cesse d'être un simple analyseur d'emails. L'objectif long
terme est de devenir une **IA Exécutive Personnelle** : une interface
conversationnelle unique qui comprend la vie professionnelle de
l'utilisateur, mémorise le contexte dans la durée, et raisonne à travers
plusieurs sources d'information au lieu de forcer l'utilisateur à naviguer
entre boîtes mail, dossiers et outils.

Au lieu de parcourir des emails ou des projets, l'utilisateur doit pouvoir
simplement demander :

- « Qu'est-ce qui s'est passé aujourd'hui ? »
- « Quelles sont mes priorités ? »
- « Quel client attend une réponse ? »
- « Qu'est-ce qui bloque le projet BUMDA ? »
- « Prépare mon brief de demain. »

L'assistant rassemble l'information depuis chaque source connectée, raisonne
dessus, et produit une réponse contextuelle.

## Positionnement — avant / après

**Avant :** Email AI Assistant. Capacités principales : catégorisation
d'emails, détection de projet, résumés, actions suggérées, brouillons de
réponse.

**Après :** Personal Executive AI Assistant. Le connecteur email devient
**une source d'information parmi d'autres**. Sources futures envisagées :
Gmail, Outlook, Calendar, Drive, Slack, Teams, WhatsApp, CRM, GitHub,
Notion, fichiers locaux, API internes. L'IA doit fournir une compréhension
unifiée du travail de l'utilisateur.

## Philosophie — "conversation-first"

Le projet doit devenir conversation-first. Chaque fonctionnalité doit
répondre à une question :

> Est-ce que cela améliore la compréhension du travail de l'utilisateur
> par l'assistant ?

Si non, la fonctionnalité n'a probablement pas sa place dans le cœur du
produit.

## Piliers produit (vision)

Les cinq piliers ci-dessous définissent la direction long terme. Chaque
pilier précise son état réel actuel entre parenthèses — aucun ne doit être
présenté comme entièrement livré tant que ce n'est pas vrai dans le code.

1. **Conversation** — la conversation devient l'UI principale, sans
   navigation requise.
   *(Partiellement implémenté : accueil chat-first `frontend/src/pages/ChatHome.tsx`,
   backend `POST /api/assistant/conversations/{id}/chat` pour un chat
   portefeuille persistant côté SaaS, avec plusieurs conversations possibles
   par tenant/utilisateur (`Conversation`/`AssistantMessage`, migration
   `014_conversations`), et `POST /api/chat` pour un chat éphémère scoped à
   un projet (legacy IMAP, hors de cette unification). Depuis Unit 48
   (Context Engine), le prompt système du chat portefeuille intègre aussi
   les souvenirs et les contacts connus (voir piliers 2 et 3 ci-dessous).
   **Unification Agent/Conversation (2026-07-27)** : ce chat n'est plus
   limité au contexte résumé déposé une fois par message — le modèle peut
   désormais appeler des outils en cours de réponse (voir pilier 4) et la
   réponse est diffusée en flux réel token par token (SSE,
   `text/event-stream`), remplaçant le faux effet machine-à-écrire
   précédent. `POST /api/agent/query` (ancien endpoint séparé, jamais
   branché en prod) est retiré : ses capacités sont absorbées ici. Pas
   encore : rendu Markdown des réponses (texte brut `<pre>` côté frontend),
   compréhension de requêtes larges type "montre les projets bloqués" au
   niveau du Brief plutôt que du chat.)*

2. **Mémoire persistante** — l'assistant doit se souvenir d'informations
   dans la durée (pas seulement des emails) : préférences client, stack
   technique d'un projet, habitudes de l'utilisateur.
   *(Fondation posée (Unit 46, 2026-07-17) : table `memories`
   (`email_analyzer/db/models.py`), extraction structurée LLM
   (`llm.extract_memory_candidates_openai/_gemini`, catégories
   preference/fact/context) orchestrée par `MemoryService`
   (`email_analyzer/services/memory_service.py`) avec déduplication exacte
   côté `MemoryRepository`, exposée via `GET/DELETE /api/memory` et
   `POST /api/memory/extract`. **Injectée dans le chat portefeuille depuis
   Unit 48** (`ContextService`, `llm.build_memory_context`) — les souvenirs
   déjà extraits apparaissent maintenant dans le prompt système. **Première
   UI depuis Unit 53** (2026-07-17) : `frontend/src/pages/MemoryPage.tsx`
   (route `/memory`) liste les souvenirs et permet leur suppression —
   jusque-là les endpoints existaient sans être appelés depuis le
   frontend. Volontairement toujours pas : extraction automatique après
   chaque message (déclenchement explicite uniquement, pour ne pas
   changer le coût/latence de `/api/assistant/chat` sans le dire),
   recherche vectorielle/sémantique. L'historique brut `AssistantMessage`
   reste distinct et inchangé — la mémoire est un résumé durable extrait
   de cet historique, pas un remplacement.)*

3. **Knowledge Graph** — rendre explicites les relations entre entités
   (utilisateur, projets, contacts, documents) pour permettre un
   raisonnement relationnel plutôt qu'une recherche par mot-clé.
   *(Fondation posée (Unit 47, 2026-07-17) : tables relationnelles simples
   `organizations`/`contacts`/`project_contact_links`
   (`email_analyzer/db/models.py`), pas de graphe. Alimentées sans nouvel
   appel LLM — `KnowledgeService.sync_from_project_summary` relit
   `ProjectSummary.structured_content["participants"/"companies"]` (déjà
   extrait par le Fast-Track existant) et upsert les entités, exposées via
   `GET /api/knowledge/contacts`, `GET /api/knowledge/organizations`,
   `POST /api/knowledge/projects/{id}/sync` (déclenchement explicite, pas
   automatique après chaque Fast-Track). **Injectée dans le chat portefeuille
   depuis Unit 48** (`ContextService`, `llm.build_knowledge_context`).
   **Première UI depuis Unit 53** (2026-07-17) : `MemoryPage.tsx` affiche
   aussi les contacts groupés par organisation (lecture seule). Volontairement
   toujours pas : résolution contact↔organisation fiable (v1
   lie tous les participants d'un projet à la première société citée),
   documents, adresses email normalisées (`Email` ne stocke pas encore
   l'expéditeur en colonne dédiée), tout raisonnement relationnel multi-sauts
   côté chat (le LLM reçoit une liste plate, pas un graphe interrogeable).)*

**Context Engine** (Unit 48, 2026-07-17, pas un pilier produit du plan
initial mais le composant transverse qui les relie) : `ContextService`
(`email_analyzer/services/context_service.py`) assemble désormais le prompt
système unique du chat portefeuille à partir de Projects + Memory +
Knowledge + **Profile** (Contexte Personnel, ajouté 2026-07-30 — voir
section "Feuille de route — 25 domaines de contexte" ci-dessous, domaine
#1) — `ConversationService` ne construit plus son contexte lui-même, il
délègue entièrement à `ContextService`. Emails et Documents ne sont pas
(encore) des sources directes du Context Engine : les emails y entrent déjà
indirectement via les résumés de projet, et il n'existe pas de modèle
`Document`.

**Contexte Personnel** (2026-07-30, domaine #1 de la feuille de route
ci-dessous) : `Profile` (`db/models.py`), un enregistrement par
(tenant, user) — nom, langues, fuseau horaire, localisation, préférences,
famille, et habitudes (réveil, sommeil, horaires de travail, jours de
sport, plage de concentration). Saisi directement par l'utilisateur dans
Paramètres (`GET/PATCH /api/profile`), pas extrait par LLM. Injecté dans le
Context Engine : permet par exemple à l'assistant de ne pas proposer de
réunion en dehors des horaires de travail déclarés. Volontairement pas
encore : sélecteur de contacts favoris dans l'UI (`favorite_contact_ids`
existe côté modèle/API sans UI), raisonnement prévisionnel à partir des
habitudes (domaine #25 de la feuille de route, non commencé).

**Contexte Documents** (backend 2026-07-31, UI 2026-08-01, ingestion email
2026-08-01, domaine #7 de la feuille de route ci-dessous) : `Document`
(`db/models.py`) — fichiers locaux uniquement (pas de Drive/OneDrive/
Dropbox), stockés sur disque (`documents_storage_dir()`, configurable via
`DOCUMENTS_STORAGE_DIR`, défaut `<service>/data/documents`), texte extrait
pour PDF (`pypdf`) et texte brut (`.txt`/`.md`/`.csv`) uniquement — pas de
Word/Excel/PowerPoint/images en v1. `DocumentService`/`DocumentRepository`,
`GET/POST/DELETE /api/documents` (+ téléchargement), optionnellement
rattaché à un projet. Page `/documents` (upload, liste avec filtre
projet/mot-clé, téléchargement, suppression). Deux sources désormais
(colonne `Document.source`) : `"upload"` (dépôt manuel) et `"email"`
(pièces jointes PDF/texte des emails traités par le Fast-Track IMAP,
ingérées automatiquement — `analysis_tasks.py::_ingest_new_email_attachments`,
2026-08-01 ; images/logos de signature volontairement jamais capturés).
**Rattrapage rétroactif** (2026-08-01, bouton "Rattraper depuis les emails"
sur `/documents`) : job arq (`run_email_attachment_backfill`) qui rebalaye
tout l'historique déjà en base (depuis `MIN(Email.received_at)` des emails
déjà rattachés à un projet) pour les tenants IMAP — un seul scan `SINCE`
complet, matché en mémoire contre les emails déjà persistés, sans jamais
utiliser le cache email partagé (qui ne porte jamais les octets de pièces
jointes) ; idempotent (relancer ne crée pas de doublon), tourne en
arrière-plan avec barre de progression (`BackfillAttachmentsModal.tsx`) et
indicateur ambiant partagé avec les autres jobs. Limité aux tenants IMAP
(même limite que le Fast-Track, domaine #3) ; limite de temps de 7200s par
exécution sans découpage multi-jobs (boîtes très volumineuses : reprise
manuelle par un nouveau clic, pas de perte de données).
Exposé à l'agent via `search_documents_tool.py` (appelable à la demande par
le LLM, donc par le chat et `draft_reply`) et via un signal ambiant
`document_count` par projet dans le Context Engine ("Documents disponibles :
N") — mais **pas** le texte extrait lui-même, qui reste tool-call-only
(un document peut porter jusqu'à 200k caractères, sans politique de
troncature/sélection décidée à ce jour).

4. **Agent IA** — le LLM cesse d'être un simple générateur de texte pour
   devenir un orchestrateur qui décide quelles sources interroger avant de
   répondre.
   *(**Devenu réellement autonome (2026-07-27)** : le `Planner` déterministe
   à base de mots-clés et l'endpoint séparé `POST /api/agent/query`
   (`email_analyzer/agent/` — `Planner`, `Executor`, `ResponseBuilder`,
   `Agent`) sont supprimés. La décision Unit 57 ("coexistence durable, pas
   de fusion tant qu'aucun signal concret ne la justifie") est explicitement
   renversée : l'utilisateur veut une seule surface de chat, aussi capable
   que les assistants du marché. Le LLM choisit désormais lui-même ses
   outils via le tool-calling natif d'OpenAI/Gemini
   (`llm.stream_portfolio_assistant_chat_openai/_gemini`, paramètre
   `tools`), directement dans `ConversationService.stream_message` — plus
   de plan pré-calculé par motifs substring. Politique explicite donnée au
   modèle (dans le prompt système, `build_portfolio_chat_system_prompt`) :
   répondre depuis le contexte déjà résumé (`ContextService` — Projects +
   Memory + Knowledge, inchangé) quand il suffit, n'appeler un outil que
   si ce contexte est insuffisant (détail d'email récent, rendez-vous,
   résumé projet à jour, brouillon de réponse) — jusqu'à 4 allers-retours
   outils par message. C'est le sens nouveau de "Agent IA" pour cette
   version : le "no autonomous planning" du plan de pivot initial est
   levé.)*

5. **Tools** — chaque capacité devient un outil indépendant appelable par
   l'agent (`search_emails()`, `search_projects()`, `search_memory()`,
   `list_meetings()`, `draft_reply()`, etc.), sans génération SQL directe
   par le LLM.
   *(Fondation posée (Unit 50, 2026-07-17), complétée Units 57-59 : package
   `email_analyzer/tools/` — `SearchProjectsTool`, `SummarizeProjectTool`,
   `SearchEmailsTool`, `SearchMemoryTool`, `FindContactTool`,
   `ListMeetingsTool` (Unit 57, enveloppe la requête "prochains rendez-vous"
   déjà utilisée par `api/routers/agenda.py`), `DraftReplyTool` (Unit 59 —
   voir ci-dessous). Chaque tool a un nom, une description et une méthode
   `run(...)` typée à entrée/sortie simples (dicts), sans base commune par
   héritage (`Tool` est un `typing.Protocol`, pas une classe à hériter).
   Contrat de lecture/génération sans écriture en base : `DraftReplyTool`
   n'appelle pas non plus de LLM (contrairement à ce que son nom pourrait
   suggérer — voir détail ci-dessous), les autres n'appellent ni LLM ni
   n'écrivent en base — ils enveloppent des requêtes/services déjà
   existants (`MemoryService`, `KnowledgeService`, requêtes
   `Project`/`Email`/`Appointment`/`ProjectSummary` déjà utilisées
   ailleurs). **Orchestrateur remplacé (2026-07-27)** : les 7 tools sont
   désormais appelés par `ToolExecutor`
   (`email_analyzer/services/tool_executor.py`, successeur de l'ancien
   `Executor`), sur décision du LLM lui-même (voir pilier 4) plutôt que
   d'un plan déterministe — `summarize_project`/`search_emails`/
   `draft_reply` (qui exigent un `project_id`) sont désormais réellement
   atteignables : `ToolExecutor` résout un `project_name` en langage
   naturel fourni par le LLM vers un `project_id` réel. `draft_reply` garde
   aussi son consommateur direct existant :
   `POST /api/projects/{id}/draft-reply` (Unit 59).

   **`DraftReplyTool` (Unit 59, 2026-07-19)** : génère un brouillon de
   réponse (routine/alerte/kickoff) pour un projet SaaS persisté, en
   réutilisant tel quel le moteur de templates déterministe existant
   (`email_analyzer/templates.py::generate_response_draft`, utilisé sans
   modification depuis longtemps par le legacy `POST /api/draft`) — pas de
   nouvel appel LLM, pas de réécriture des templates. Le seul travail de
   cette unité est l'adaptateur : mapper `ProjectSummary`/
   `structured_content` (forme SaaS, clés anglaises) vers le dict français
   attendu par le moteur de templates (`évaluation_risque`,
   `analyse_sentiment`, `emails_critiques`, `résumé_assistant`) — le
   legacy `POST /api/draft` reste inchangé, toujours utilisé par le mode
   IMAP sans tenant.)*

## Feuille de route — 25 domaines de contexte (2026-07-30)

Vision fournie par l'utilisateur (2026-07-30) : 25 domaines de contexte
("second cerveau") à terme. Statut réel ci-dessous, pour éviter de
présenter comme livré ce qui ne l'est pas — voir aussi `architecture.md`
§7. Domaine #1 est le premier construit dans le cadre de cette feuille de
route (voir pilier "Contexte Personnel" ci-dessus, ajouté à cette révision).

| # | Domaine | Statut | Détail / prérequis |
|---|---|---|---|
| 1 | Contexte Personnel (profil, habitudes) | **Fait** (2026-07-30) | `Profile` (`db/models.py`), `ProfileService`, `GET/PATCH /api/profile`, injecté dans `ContextService` (`build_profile_context`), UI dans Paramètres. |
| 2 | Calendrier | Partiel (existant) | Agenda IA (`api/routers/agenda.py`, rendez-vous détectés par email) — pas de sync Google Calendar/Outlook Calendar/Apple Calendar dédiée (au-delà des scopes OAuth email déjà connectés). |
| 3 | Emails | Fait (existant) | Gmail OAuth2, Outlook OAuth2, IMAP legacy — scoring, classification, résumés. |
| 4 | Tâches (Todoist/Notion/Jira/ClickUp/Trello/Planner) | À construire | Aucun fournisseur choisi par l'utilisateur — bloqué tant qu'un outil précis et ses identifiants API ne sont pas fournis. |
| 5 | Contexte Projet | Fait (existant) | `Project`/`ProjectSummary`/`SuggestedAction`. |
| 6 | Contexte Professionnel (organigramme, CRM, pipeline, factures) | Partiel | `Organization`/`Contact` (Knowledge Layer) couvrent une base ; pipeline commercial/factures/contrats non modélisés — bloqué sur choix d'un CRM/outil de facturation externe ou décision de le construire en interne. |
| 7 | Contexte Documents (Drive/OneDrive/Dropbox/NAS/locaux) | Partiel (2026-08-01) | Fichiers locaux uniquement : `Document` (`db/models.py`), upload manuel + ingestion automatique des pièces jointes d'email (PDF/texte, Fast-Track IMAP), liste/téléchargement/suppression, page `/documents`, outil agent `search_documents`, signal ambiant `document_count` (pas le texte extrait) dans le Context Engine. Pas d'OAuth Drive/OneDrive/Dropbox — bloqué sur identifiants pour le stockage cloud ; ingestion email limitée aux tenants IMAP (Gmail/Outlook-only : Fast-Track non fonctionnel, limite préexistante, domaine #3). |
| 8 | Contexte Communication (WhatsApp/Slack/Teams/Discord/Telegram/SMS) | À construire | Bloqué sur choix de plateforme(s) + identifiants API (WhatsApp Business API, Slack app, etc.). |
| 9 | Contacts (CRM personnel) | Partiel (existant) | `Contact`/`Organization` liés aux projets (lecture seule dans `/memory`) ; pas encore de champs fonction/historique/dernière discussion/préférences/relation demandés ici. |
| 10 | Finances | À construire | Bloqué sur choix de banque(s)/Mobile Money et leurs API (Plaid, agrégateur local, etc.). |
| 11 | Santé | À construire | Bloqué sur choix Apple Health/Google Fit/Garmin/Fitbit et leurs API/OAuth. |
| 12 | Localisation (GPS, trafic, météo) | À construire | Bloqué sur choix de fournisseur (Google Maps, OpenWeather, etc.) et consentement de partage de localisation. |
| 13 | Contexte IA (historique, préférences, décisions passées) | Fait (existant) | `Memory` + `AssistantMessage` + `Conversation`. |
| 14 | Contexte Web (veille, actualités) | À construire | Bloqué sur choix de sources (flux RSS, API news) — pas de scraping non consenti. |
| 15 | Réseaux Sociaux (LinkedIn, Twitter, ...) | À construire | Bloqué sur choix de plateforme(s) + identifiants API. |
| 16 | Contexte Navigation (historique navigateur) | À construire | Nécessiterait une extension navigateur dédiée — hors périmètre actuel (web only, pas d'extension). |
| 17 | Contexte Développement (GitHub/GitLab/CI-CD) | À construire | Bloqué sur choix de fournisseur(s) + identifiants API/token. |
| 18 | Contexte Maison (domotique) | À construire | Bloqué sur choix de plateforme domotique (Home Assistant, etc.) + accès réseau local. |
| 19 | Contexte Voyage | À construire | Bloqué sur choix de fournisseur (emails de réservation déjà couverts indirectement par #3, mais pas de modélisation dédiée vols/hôtels/visa). |
| 20 | Contexte Objectifs | Fait (existant) | `Objective`/`Milestone` (Units 54-56). |
| 21 | Contexte Connaissances (second cerveau, notes) | À construire | Bloqué sur choix d'intégration (Notion/Obsidian/Evernote) ou décision de construire un module de notes interne. |
| 22 | Contexte Automatisation (Zapier/n8n/Make/Power Automate) | À construire | Bloqué sur choix de fournisseur + identifiants ; un récepteur de webhook générique interne est envisageable sans dépendance externe mais non commencé. |
| 23 | Contexte Temps Réel | Dépend des domaines amont | Suppose des sources déjà connectées (email, calendrier, IoT, ...) capables de pousser un événement — pas de socle de notifications proactives aujourd'hui. |
| 24 | Contexte Décisionnel (historique, pourquoi, risques, alternatives) | À construire | S'appuierait sur `Memory` (nouvelle catégorie "decision" ou table dédiée) — non commencé, pas de UI. |
| 25 | Contexte Prévisionnel | Dépend de #1 et #24 | Raisonnement sur habitudes (#1, fait) + historique de décisions (#24, non fait) pour anticiper — non commencé. |

Domaines marqués "À construire" : aucun code d'intégration écrit
spéculativement (`context/ai-workflow-rules.md`, "Ban Speculative
Coding") — chacun nécessite soit un fournisseur choisi par l'utilisateur
avec identifiants API réels, soit une décision produit explicite (ex.
construire un module de notes interne plutôt qu'intégrer Notion). Visibles
dans la navigation (`AppShell.tsx`, `COMING_SOON_ITEMS`) sous forme
d'entrées groupées par thème (pas une entrée par domaine, pour ne pas
noyer la nav) : Clients, Tâches, Communication, Finance, Santé,
Développement, Automatisation. (Documents, domaine #7, a quitté ce groupe
le 2026-08-01 pour devenir une entrée de nav réelle, `/documents`.)

## Sources d'information

**Actuel :** IMAP (mode legacy `.env`), Gmail OAuth2, Outlook OAuth2
(Microsoft Graph) — voir `api/routers/oauth.py`.

**Futur, par phases** (aucune non commencée) :

- Phase 1 — Gmail, IMAP *(fait)*
- Phase 2 — Outlook *(fait)*, Calendar, Drive, documents locaux
- Phase 3 — GitHub, Slack, Teams, WhatsApp, Notion
- Phase 4 — CRM, ERP, API internes

## Scope actuel (ce qui est réellement construit)

- Backend FastAPI + SQLAlchemy/PostgreSQL, jobs asynchrones via Redis +
  `arq` (voir `context/architecture.md`).
- Frontend web React/Vite/Tailwind (pas d'application mobile).
- Ingestion IMAP, Gmail OAuth2, Outlook OAuth2.
- Scoring d'importance des emails (règles + IA hybride), classification
  multi-critères, détection/déduplication de projets.
- Résumés de projet structurés par LLM (décisions, risques, échéances,
  prochaines étapes, rendez-vous probables), agenda IA, actions suggérées.
- Brief quotidien (`GET /api/brief`, hook `useBrief.ts`) : compteurs et
  actions recommandées depuis la dernière visite, affichés comme cartes
  rapides dans l'accueil chat-first. Organisé depuis Unit 52 (2026-07-17)
  en `BriefService`/`BriefRepository`
  (`email_analyzer/services/brief_service.py`) au lieu de logique inline
  dans le routeur — toujours aucune génération de résumé propre (réutilise
  `ProjectSummary`/`SuggestedAction` déjà produits par le Fast-Track).
- Chat portefeuille persistant (SaaS, streaming SSE + tool-calling natif
  LLM) et chat par projet (legacy, hors unification).
- Génération de brouillons de réponse (jamais d'envoi automatique) : legacy
  IMAP (`POST /api/draft`) et SaaS multi-tenant depuis un projet persisté
  (`POST /api/projects/{id}/draft-reply`, Unit 59), même moteur de
  templates déterministe (`templates.py`) sous les deux formes.
- Page "Mémoire & Contacts" (`/memory`, Unit 53) : consultation et
  suppression des souvenirs retenus, consultation des contacts détectés
  groupés par organisation.
- Objectifs (`Objective`/`Milestone`, Units 54-56, 2026-07-17) : CRUD
  complet côté backend (`POST/GET/PATCH /api/objectives`,
  `POST/PATCH /api/objectives/{id}/milestones/...`), optionnellement
  rattachés à un projet existant, progression calculée depuis les jalons,
  statut `blocked` avec raison. Page `/objectives` (Unit 56) : création
  d'objectif, ajout/complétion de jalons (case à cocher), bascule
  bloqué/débloqué — première UI, pas encore d'édition du titre ni de
  suppression d'objectif.
- Timeline unifiée (`GET /api/timeline`, Units 55-56, 2026-07-17) : fusionne
  4 types d'événements par ordre chronologique — emails, résumés mis à
  jour, actions suggérées, rendez-vous d'agenda. Affichée en lecture seule
  dans la page `/objectives` aux côtés des Objectifs.
- Contexte Personnel (`Profile`, 2026-07-30) : profil et habitudes déclarés
  par l'utilisateur, injectés dans le Context Engine. Page "Profil &
  habitudes" dans Paramètres.
- Contexte Documents (`Document`, backend 2026-07-31, UI 2026-08-01,
  ingestion email 2026-08-01) : upload manuel + ingestion automatique des
  pièces jointes PDF/texte des emails traités par le Fast-Track IMAP,
  liste/téléchargement/suppression, optionnellement rattachés à un projet,
  extraction de texte pour la recherche. Page `/documents`. Accessible à
  l'agent via l'outil `search_documents` (à la demande) et via un compte
  ambiant par projet dans le Context Engine (pas le texte extrait lui-même).
- Détection de tickets référencés en sujet d'email (`[libellé_numéro]`,
  Unit 63, 2026-07-28) : tag "Réclamation", rattachement automatique/manuel
  à un projet (modal "Détecter les tickets"), comptage exposé au chat
  (`open_ticket_count`) et outil agent `list_tickets`.

## Hors scope (actuel)

- Envoi automatique d'email : le système ne rédige que des brouillons
  nécessitant une relecture et un envoi manuel.
- Ingestion temps réel par webhook (push Gmail, etc.) : le système repose
  sur du polling planifié via des jobs `arq` en cron.
- Application mobile (React Native) : non construite, non planifiée à
  court terme.
- Exécution locale de LLM sur l'appareil de l'utilisateur : toute
  inférence passe par des API LLM cloud (OpenAI/Gemini).
- Mémoire long terme structurée (recherche vectorielle/sémantique), knowledge
  graph avec raisonnement relationnel multi-sauts : voir section "Piliers
  produit" ci-dessus — vision, non implémentés. (L'agent à tool-calling,
  lui, est implémenté depuis 2026-07-27 — voir pilier 4.)

## Roadmap IA

### Phase 1 — Conversation, projets, emails (quasi terminée)

Conversation de base, projets, emails, résumés, actions suggérées,
brouillons de réponse. Voir `context/progress-tracker.md` pour le détail
unité par unité de ce qui est fait.

### Phase 2 — Mémoire persistante, Knowledge Graph, recherche, timeline

Non commencée. `context/evolution-plan.md` (Phase 2) et
`context/rfc-email-pipeline-v2.md` documentent déjà une étape technique
préalable concrète : recherche sémantique centralisée via `pgvector` dans
PostgreSQL (pas de base vectorielle côté client), sync incrémental réel,
threading des conversations. C'est un prérequis technique pour la mémoire
et le knowledge graph décrits ici, pas encore la mémoire elle-même.

### Phase 3 — Architecture agent, tool-calling, planification, raisonnement multi-étapes

Tool-calling natif LLM démarré en avance de phase (2026-07-27, voir pilier 4)
dans le cadre de l'unification du chat portefeuille — le modèle choisit et
enchaîne réellement ses outils (jusqu'à 4 allers-retours par message), mais
sans planification multi-étapes explicite au-delà de cette boucle simple
(pas de décomposition de tâche, pas de mémoire d'exécution entre messages
distincts). Le reste de cette phase (raisonnement multi-étapes plus
avancé) reste non commencé.

### Phase 4 — Voix, mobile, notifications proactives, intégrations
cross-plateformes

Non commencée.

## Principes de développement

Toute nouvelle fonctionnalité doit satisfaire au moins un de ces
objectifs :

- Améliore la compréhension du travail de l'utilisateur.
- Améliore la mémoire.
- Améliore le raisonnement.
- Améliore le contexte disponible.
- Améliore la conversation.

Éviter les fonctionnalités qui n'augmentent que la complexité de l'UI sans
servir un de ces objectifs.

## Vision finale

L'objectif long terme est de construire une IA Exécutive qui agit comme le
second cerveau de l'utilisateur. Les emails ne sont qu'une source de
connaissance parmi d'autres. L'assistant doit comprendre les personnes, les
projets, les objectifs, les documents, les réunions, les tâches, les
conversations, les décisions et les relations entre eux, et permettre à
l'utilisateur de tout gérer en langage naturel.

La métrique de succès n'est plus « L'IA peut-elle résumer des emails ? »
mais :

> « L'utilisateur peut-il gérer toute sa vie professionnelle simplement en
> parlant à l'assistant ? »

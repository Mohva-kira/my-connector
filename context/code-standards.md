# Code Standards & Engineering Guidelines

> **Réécrit le 2026-07-19** (avec les corrections de `architecture.md`
> du même jour). Remplace le texte précédent, qui décrivait des outils et
> une pile jamais mis en place dans ce dépôt (Celery, React Native, mypy à
> 100%, couverture de tests à 80%, `Bandit`, cache device 30 jours, index
> vectoriel Orama/Voy). Ce document décrit les conventions réellement
> suivies dans le code depuis le pivot exécutif (Units 45-57, voir
> `context/progress-tracker.md`) — pas un objectif aspirationnel.
>
> Il n'y a **aucun outillage automatisé** dans ce dépôt aujourd'hui :
> ni CI, ni pre-commit hook, ni configuration `mypy`/`ruff`/`eslint`, ni
> suite de tests unitaires (confirmé : aucun `pyproject.toml`,
> `.pre-commit-config.yaml`, `mypy.ini`, `.eslintrc*` dans le repo). Les
> règles ci-dessous sont donc des conventions de code appliquées
> manuellement — un futur ajout d'outillage réel doit d'abord être
> installé/configuré avant d'être cité ici comme "en place".

## 1. Stack réelle

Backend Python (FastAPI, SQLAlchemy/PostgreSQL, Redis + `arq`), frontend
web React + Vite + Tailwind (pas d'application mobile). Détail complet :
`context/architecture.md` §1-2. Pas de Celery, pas de React Native, pas de
base vectorielle côté client.

## 2. Flux de développement réel

Pas de nomenclature de branche imposée, pas de pipeline CI, pas de revue
par pair obligatoire outillée — ce dépôt est développé en session assistée
par IA, une unité de travail à la fois, chaque unité documentée dans
`context/progress-tracker.md` (contexte, décision, ce qui a été vérifié et
comment, ce qui ne l'a pas été). C'est ce journal, pas un pipeline CI, qui
tient lieu d'historique de revue.

## 3. Standards Python (FastAPI)

* **Formatage :** PEP 8 de fait (pas de `black`/`ruff` configuré à ce
  jour) — cohérence avec le style déjà présent dans le fichier édité, pas
  de réindentation globale à l'occasion d'un patch ciblé.
* **Type hints :** systématiques sur les signatures de fonctions
  publiques (paramètres + retour). Aucune vérification `mypy` automatisée
  aujourd'hui — les types sont un contrat de lecture, pas un gate de CI.
* **Routes FastAPI synchrones par défaut :** la quasi-totalité des
  endpoints sont des `def` classiques (pas `async def`) — SQLAlchemy est
  utilisé en mode synchrone (`Session`), FastAPI les exécute dans un
  threadpool. N'utiliser `async def` que si un appel réellement
  asynchrone le justifie (aucun cas de ce type actuellement dans
  `api/routers/`).
* **Model → Repository → Service → API :** pour toute nouvelle
  fonctionnalité, la requête SQL vit dans `email_analyzer/repositories/
  *_repository.py`, l'orchestration/validation dans
  `email_analyzer/services/*_service.py`, le routeur (`api/routers/*.py`)
  reste un contrôleur HTTP fin (auth, parsing du body, appel du service,
  mapping vers le schéma Pydantic de réponse). Voir
  `architecture.md` §8 invariant 5. Les routeurs plus anciens
  (`projects.py`, `agenda.py`) gardent encore des requêtes inline — à
  migrer seulement à l'occasion d'un changement réel de leur logique.
* **Services indépendants de FastAPI :** un `*_service.py` ne doit pas
  dépendre de types Pydantic/FastAPI dans sa signature publique (utiliser
  des dataclasses simples, voir `BriefService`/`ObjectiveService`) — il
  doit rester appelable/testable sans lever un serveur HTTP.
* **`llm.py` reste le seul point d'appel SDK LLM :** aucun import direct
  `openai`/`google.generativeai` ailleurs dans le code — confirmé par
  grep à chaque nouvelle unité touchant l'IA (Units 46, 50-51, 57).
  Chaque fonction d'appel LLM retourne `{"data"|"erreur"}` ou
  `{"message"|"erreur"}`, jamais une exception qui remonte à l'appelant.
* **Gestion des exceptions :** pas de `try/except Exception: pass`
  silencieux. Un tool (`email_analyzer/tools/`) ou l'`Executor` de
  l'agent peuvent capturer une exception, mais toujours pour la logger et
  la renvoyer explicitement dans le résultat (`{"error": str(ex)}`),
  jamais pour l'avaler.
* **Logging :** `logging.getLogger(__name__)` par module, jamais
  `print`. Chemins d'exécution notables (services métier, `ToolExecutor`,
  `ConversationService`) loggent au moins un `start`/`done`/`error` — voir
  `conversation_service.py`/`tool_executor.py` pour le format
  (`logger.info("module.action start/done tenant=%s ...")`). Ne jamais
  logger un corps d'email, un token ou un mot de passe en clair.
* **Migrations :** toute modification de schéma passe par Alembic
  (`alembic/versions/`), jamais de DDL manuel. Une migration a toujours
  un `downgrade()` symétrique. Voir les migrations `011`-`013` pour le
  gabarit à suivre (docstring expliquant le "pourquoi", index sur les
  clés étrangères, contraintes uniques nommées explicitement).
* **Streaming SSE (depuis 2026-07-27, `POST /api/assistant/conversations/{id}/chat`,
  première route de streaming du repo) :** `StreamingResponse` +
  générateur Python synchrone (pas `async def` — cohérent avec la
  convention "routes synchrones" ci-dessus), `media_type="text/event-stream"`,
  formaté par `api/sse.py::format_sse_event`. Chaque événement est un dict
  `{"type": ..., ...}` ; les générateurs de `llm.py`
  (`stream_portfolio_assistant_chat_openai/_gemini`) et de
  `ConversationService.stream_message` ne lèvent jamais d'exception —
  toute erreur devient un événement terminal `{"type": "error", ...}`
  (même discipline que le reste de `llm.py`, reformulée en flux). Toute
  validation de pré-requête (message vide, ressource introuvable,
  fournisseur non configuré) doit lever une vraie `HTTPException` *avant*
  le premier `yield` — le routeur amorce donc le générateur (un premier
  `next()` explicite) avant de le passer à `StreamingResponse`, sinon
  l'exception surviendrait après l'envoi des entêtes HTTP 200, trop tard
  pour renvoyer le bon status code (voir `api/routers/assistant.py::chat`
  pour le patron à suivre pour une future 2e route de streaming).

## 4. Standards TypeScript (React web)

* **Composants fonctionnels + hooks personnalisés :** logique de
  chargement de données extraite dans `frontend/src/hooks/*.ts` (ex.
  `useMemoryAndKnowledge.ts`, `useObjectivesAndTimeline.ts`), pas inline
  dans le composant de page.
* **Typage strict des payloads API :** chaque module `*Api.ts`
  (`memoryApi.ts`, `objectivesApi.ts`, `timelineApi.ts`, ...) définit un
  type miroir exact du schéma Pydantic du routeur correspondant — à
  vérifier en lisant le routeur, pas en supposant la forme. Pas de type
  `any`.
* **Classes Tailwind :** réutiliser les tokens déjà définis dans
  `tailwind.config.js` (`bg-surface`, `border-border-default`,
  `text-text-primary/secondary/muted`, `bg-accent-soft`/
  `text-accent-primary`, et les modificateurs d'opacité natifs
  `bg-danger/10`/`bg-warning/10` pour les teintes sans variante "soft"
  dédiée) — ne jamais inventer un nom de classe sans vérifier son
  existence dans la config (erreur commise et corrigée en Unit 56 :
  `bg-warning-soft` n'existe pas).
* **Vérification :** `npx tsc --noEmit` doit passer sans erreur avant de
  considérer une unité frontend terminée. `npx vite build` est informatif
  (le nombre de modules transformés doit rester cohérent avec les fichiers
  ajoutés) mais échoue systématiquement dans cet environnement de
  développement sur le nettoyage du dossier `dist/` préexistant
  (appartient à `root`, EACCES) — échec connu, sans rapport avec le code,
  ne pas le traiter comme un signal de régression.

## 5. Sécurité

* **Secrets :** aucune clé en dur dans le code, lues via variables
  d'environnement (`os.environ`, jamais de valeur par défaut secrète).
* **Isolation multi-tenant :** toute requête sur une table liée à un
  tenant filtre explicitement sur `tenant_id` (et `user_id` quand
  pertinent) — voir `architecture.md` §8 invariant 3.
* **Chiffrement au repos :** `Email.body_encrypted` n'est **pas**
  chiffré aujourd'hui malgré son nom de colonne — ne pas prétendre le
  contraire dans le code ou la documentation (voir
  `architecture.md` §4/§6 pour le statut réel et le plan).

## 6. Ce qui n'est délibérément pas en place

Cette section remplace l'ancienne "Definition of Done" (couverture 80%,
`mypy`/`tsc` en CI, `Bandit`, nettoyage cache 30 jours) qui décrivait un
outillage inexistant. Statut réel :

* **Pas de suite de tests automatisée.** La vérification d'une unité de
  travail se fait par un script Python jetable (créé dans le scratchpad,
  supprimé après usage) qui exerce le code réel contre une vraie base
  Postgres locale (voire un vrai appel LLM quand pertinent), pas par des
  mocks — voir `context/progress-tracker.md`, n'importe quelle unité
  récente (46 à 57) pour des exemples concrets de ce protocole.
* **Pas de CI/CD.** Aucun pipeline GitHub Actions ou équivalent.
* **Pas de linter/formatter configuré.** Ni `ruff`/`black` côté Python,
  ni `eslint`/`prettier` côté frontend (`package.json` ne les liste pas).
* **Pas de scanner de sécurité statique.** Pas de `Bandit` ni équivalent.

Si l'un de ces outils est mis en place un jour, ce document doit être mis
à jour dans la même unité de travail que son installation — pas avant.

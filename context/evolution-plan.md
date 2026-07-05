# Plan d'Évolution Produit & Technique : my-connector

## Contexte

Un écart majeur existe entre la documentation d'architecture et le code réel :

- **`architecture.md` décrit une cible non implémentée** : Node.js/NestJS, BullMQ/Redis, mono-tenant strict, Fast-Track (`project-fasttrack-queue`), Historical Clustering Worker.
- **Le code réel (`services/email_analyzer/`, cf. `progress-tracker.md`) est** : Python FastAPI + SQLAlchemy/Alembic, ingestion **IMAP** (OAuth Gmail livré en Unit 1 mais **non branché** sur `/api/analyze` — Open Question #1 du tracker), jobs **en mémoire** via `ThreadPoolExecutor(max_workers=2)` contraint à **1 seul worker uvicorn**, chiffrement Fernet des identifiants IMAP/OAuth, facturation CinetPay, **multi-tenant déjà en production** (contredit la ligne "Out of Scope: multi-tenant" de `project-overview.md`), UI de chargement gamifiée (Unit 7, étape 1/8 seulement, React + Zustand + React Query + Framer Motion — pas Phaser).
- **Le Fast-Track et le clustering historique n'existent pas encore en code** : ce sont des specs d'architecture, pas des acquis à consolider.

Décision produit : ce plan s'ancre sur **l'état réel du code**, pas sur la spec Node/BullMQ. La Phase 1 traite donc la résolution de cette divergence documentaire, la construction réelle du Fast-Track/clustering, et le remplacement du job store en mémoire comme des priorités de stabilisation — pas des optimisations d'un existant.

But recherché : produire une feuille de route crédible en 3 phases qui ferme l'écart doc/code, industrialise l'asynchrone, puis ouvre la voie à l'intelligence proactive — sans jamais introduire Node.js/NestJS/BullMQ, qui ne correspondent pas à la stack réelle.

---

## 🚀 Plan d'Évolution Technologique & Produit : my-connector

### 📌 Phase 1 : Consolidation & Stabilisation du MVP (0 - 3 mois)

**Objectif principal :** Fermer l'écart entre `architecture.md` et le code réel, rendre les flux asynchrones robustes et partagés (fin de la contrainte 1-worker), et livrer pour de vrai les deux mécanismes cœur de produit qui n'existent pas encore : le clustering historique à l'onboarding et le Fast-Track par projet.

**Évolutions Techniques / Architecture :**
- **Réconciliation stack** : mettre à jour `architecture.md` pour refléter Python/FastAPI comme stack de référence (remplacer les sections NestJS/BullMQ par FastAPI + Redis + `arq` — worker asyncio natif, cohérent avec le code async existant, contrairement à Celery qui imposerait un runtime sync parallèle). Documenter la décision dans `progress-tracker.md` (`Architecture Decisions`).
- **Remplacement du job store en mémoire** (`email_analyzer/jobs.py`) : migrer de `ThreadPoolExecutor` + dict en mémoire vers **Redis + `arq`**, avec deux files distinctes — `batch-sync-queue` (sync 2x/jour) et `fasttrack-queue` (haute priorité, un seul projet à la fois, jamais bloquée par le batch). Ceci lève la contrainte actuelle « 1 seul worker uvicorn » (store non partagé) et permet le scaling horizontal des workers, conformément à `code-standards.md §8` (idempotence, dédup par `(user_id, email_external_id)`, retry avec backoff exponentiel max 5).
- **Implémentation réelle du Fast-Track** : `POST /api/projects/{id}/refresh` → `202 {job_id}` immédiat, lecture de `last_processed_email_timestamp`, fetch du delta uniquement (IMAP et/ou Gmail API selon connexion active), regénération ciblée du résumé/sentiment/actions, `GET /api/jobs/{id}` pour le polling (réutilise le pattern de `jobs.py`/`HomePage.tsx` déjà validé en Unit 6).
- **Implémentation réelle du Historical Clustering Worker** (onboarding) : job dédié qui regroupe les threads historiques par similarité de sujet + relations expéditeur/destinataire + analyse sémantique légère, retourne les `proposed_projects` au format JSON strict (`architecture.md` Process 1), avant persistance validée par l'utilisateur (Validation Board).
- **Fermeture du gap OAuth** (Open Question #1) : brancher `gmail_oauth.fetch_emails()` dans `EmailProcessor` pour que `/api/analyze` utilise Gmail OAuth quand IMAP n'est pas configuré — actuellement les deux chemins coexistent sans bascule automatique.
- **Chiffrement `body_encrypted`** : le Fernet actuel ne couvre que les identifiants OAuth/IMAP (`gmail_oauth.py`, credentials IMAP) — étendre le chiffrement au **corps des emails stockés** (colonne `body_encrypted` du schéma cible), avec clé dérivée par tenant et rotation planifiée, sans jamais logguer le corps en clair (déjà respecté par `code-standards.md §12`).
- **Renforcement du JSON strict des prompts** : ajouter une validation de schéma systématique (pydantic) sur chaque sortie LLM (schéma `type/importance_score/action_required/deadline/category/confidence` de `code-standards.md §7`), avec re-tentative automatique en cas de sortie non conforme et flag humain si `confidence < 0.5`.

**Améliorations UX / UI :**
- **État `color-fasttrack-active` pulsé** : appliqué **localement à la carte projet concernée** dans le Project Hub pendant le traitement (pas d'overlay plein écran) — respecte la règle `architecture.md` *« Only the affected project card is updated »*. Le composant `SyncGame`/`useSyncGame` (Unit 7) reste réservé au sync global initial ; un état pulsé distinct et minimal est nécessaire pour le Fast-Track par carte.
- **Toast `color-discovery-banner`** : n'existe pas encore dans le code — l'introduire comme notification non bloquante, auto-dismiss, un seul toast visible à la fois (conforme à la hiérarchie de notification de `ui-context.md`, où *Discovery Notification* est le niveau le plus bas et ne doit jamais interrompre le flux).

---

### 📈 Phase 2 : Industrialisation & Extension Fonctionnelle (3 - 6 mois)

**Objectif principal :** Officialiser l'architecture multi-tenant déjà en production (au lieu de la documenter comme hors-scope), industrialiser les coûts/quotas API, et enrichir la recherche au-delà du FTS.

**Évolutions Techniques / Architecture :**
- **FTS PostgreSQL → recherche sémantique via `pgvector`** : rester dans PostgreSQL (conforme à `code-standards.md §6` — *« PostgreSQL only, no other databases in the primary data path »* — donc pas de base vectorielle externe type Pinecone/Weaviate). Embeddings générés via l'API OpenAI (`text-embedding-3-small`), stockés en colonne `vector`, recherche cosinus scoping strict par `tenant_id`, permettant une recherche sémantique **inter-projets**.
- **Rate-limiting & quotas Gmail/Outlook/OpenAI** : implémenter un limiteur token-bucket par tenant dans Redis, avec circuit breaker sur les 429/quota Gmail/Graph API, et cache de réponses LLM par hash de prompt pour réduire les coûts tokens sur les requêtes répétées (résumés inchangés, chat sur contexte identique).
- **Formalisation du multi-tenant** : corriger `project-overview.md` (retirer la mention "Out of Scope: multi-tenant") et documenter dans `architecture.md` le modèle `Tenant` déjà utilisé en production (JWT + bcrypt + CinetPay), pour que la doc cesse de contredire le code.

**Améliorations UX / UI :**
- **Filtres avancés dans le Project Hub** : par santé (`color-sentiment-*`), par date de dernière activité, par tag/projet archivé — progressive disclosure conforme à `ui-context.md`.
- **Gestion des sentiments critiques** : notification prioritaire dès qu'un projet bascule vers `color-sentiment-critical` (#EF4444), positionnée juste sous *Deadline Today* dans la hiérarchie de notification — jamais au-dessus des alertes critiques système.
- **Actions rapides 1-clic** sur le Validation Board (onboarding) : valider/fusionner/renommer un projet proposé sans changer d'écran, avec confirmation optimiste + rollback si l'action échoue côté serveur.

---

### 🔮 Phase 3 : Intelligence Avancée & Écosystème (6 mois +)

**Objectif principal :** Rendre l'assistant proactif sans jamais franchir la limite « aucune action automatique » posée par `project-overview.md` et `ai-workflow-rules.md §8`.

**Évolutions Techniques / Architecture :**
- **Agents IA limités pour brouillons locaux** : génération de brouillons de réponse stockés en base avec statut `draft`, jamais transmis à l'API Gmail/Graph sans validation explicite de l'utilisateur — respecte strictement *« AI must NEVER send emails »* (`code-standards.md §7`, `ai-workflow-rules.md §8`).
- **Webhooks Slack/Teams** : notifications opt-in par tenant pour les événements de priorité haute (deadline critique, sentiment critique), avec scoping strict des données transmises (jamais le corps d'email, uniquement résumé/métadonnées).

**Améliorations UX / UI :**
- **Widget d'action contextuelle** : mini-panneau flottant proposant l'action recommandée du moment (suivre X, valider Y) sans quitter la vue courante.
- **Métriques de productivité** : tableau « temps gagné » basé sur le delta entre le temps de lecture email estimé (avant) et le temps de scan du briefing (après), aligné sur le succès criterion *« 2-minute rule »* déjà défini dans `ui-context.md`/`project-overview.md`.

---

### 🛠️ Matrice des Risques & Facteurs Clés de Succès (KPIs)

**Risques Techniques majeurs :**
1. **Coût tokens LLM incontrôlé** (clustering historique + résumés + chat, sur potentiellement des dizaines de milliers d'emails par tenant, cf. le cas réel à 15 749 candidats déjà observé) → Contingence : cache par hash de prompt (Phase 2), plafond de tokens par tenant avec dégradation gracieuse (résumé plus court plutôt qu'échec), usage de `gpt-4o-mini` par défaut et `gpt-4o` seulement sur re-génération demandée explicitement.
2. **Latence Fast-Track dégradée par la contrainte 1-worker actuelle** → Contingence : priorité absolue à la migration Redis/`arq` de la Phase 1 ; sans elle, tout ajout de charge (multi-tenant, clustering) aggrave le risque de 504 déjà rencontré une fois (Unit 6).
3. **Rate-limiting Gmail/Microsoft Graph** sur la synchronisation batch multi-tenant → Contingence : token-bucket Redis par tenant (Phase 2), backoff exponentiel déjà en place au niveau job (`code-standards.md §8`) à étendre au niveau appel API externe.

**KPIs de performance UX :**
- **Règle des 2 minutes** : temps médian entre ouverture du dashboard et dernier scroll/clic de lecture du briefing < 2 min (mesuré côté frontend, événement `briefing_viewed` → `first_action_taken`).
- **Adoption Fast-Track** : % de projets actifs ayant au moins un `refresh` manuel par semaine, et latence P95 du job Fast-Track (cible < 30s une fois la migration Redis/`arq` livrée, contre un comportement non mesurable aujourd'hui puisque le mécanisme n'existe pas).
- **Précision du clustering à l'onboarding** : % de projets proposés conservés tels quels (ni renommés, ni fusionnés, ni supprimés) par l'utilisateur sur le Validation Board — proxy direct de la qualité de l'IA de découverte.

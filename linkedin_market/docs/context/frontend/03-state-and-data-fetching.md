# Gestion d'État & Data Fetching

> Vérifié dans `src/main.tsx`, `src/hooks.ts`, `src/api.ts`, `src/components/TaskDrawer.tsx` et les pages.
> Versions installées : `@tanstack/react-query` **5.104.1**, React **18.3.1**, `react-router-dom` **6.30.6**.

## 1. Répartition de l'État

| Catégorie | Outil | Contenu `[ACTUEL]` |
| :--- | :--- | :--- |
| **État serveur** | **TanStack Query 5** (`QueryClient` unique dans `main.tsx`) | Statut, monitoring, réglages, profil, offres, rapports, candidatures, tâches. Seule source de vérité des données métier : aucune copie dans un store global. |
| **État d'interface global** | **Contexte React** `TaskProvider` (`useTaskUi`) — pas de Zustand / Redux | Tiroir des tâches ouvert / fermé, tâche sélectionnée, tâches déjà « signalées » (`useRef<Set>`), statuts déjà vus (`useRef<Map>`). |
| **Préférence persistée** | `localStorage["lm-theme"]` + `document.documentElement.dataset.theme` | Thème clair / sombre (lu dans `main.tsx`, écrit par `toggleTheme()`). Aucune autre donnée dans le stockage du navigateur. |
| **État d'URL** | React Router (`useParams`) | Candidature sélectionnée (`/candidatures/:jobId`). Les filtres de la page Offres et le rapport choisi dans Marché **ne sont pas** dans l'URL. |
| **État local** | `useState` | Formulaires (brouillons), onglet actif, modales ouvertes, filtres, sélection de rapport, compétences ajoutées depuis Marché (`added`). |
| **Secret** | Constante module `token` dans `api.ts` | Jeton local lu une fois dans `<meta name="lm-token">`, jamais stocké ailleurs. |

`[ACTUEL]` (vagues 2 à 7, revue du 2026-10-09) :
- filtres de la page Offres (`/offres?q=&min_score=&label=&status=&source=`) et rapport de Marché (`/marche?rapport=`) dans
  l'URL (`useSearchParams`) ;
- pas de store global supplémentaire ni de `SourceConsentProvider` : la popup de risques est portée par
  `ConsentGate` (état local de `App.tsx`, abonné à `onConsentRequired()` d'`api.ts`, pour les 428) et par
  `useConsentGuard()` (état local de l'appelant, pour la popup **préventive**) ;
- jeton périmé (401) : drapeau module `tokenExpired` de `hooks.ts` lu par `useSyncExternalStore`
  (`useTokenExpired`) ;
- mémoire d'un message interrompu par un 428 (`ComposeInitial` : canal, destinataire, objet, corps, pièce jointe ;
  **jamais** `confirm`) dans l'état de `ConsentGate`, pour rouvrir la relecture.

## 2. Stratégie de Requêtage (Query & Mutations)

### 2.1 Configuration globale `[ACTUEL]`

```ts
new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: false, staleTime: 5000 } } })
```
- `retry: false` : une erreur (401, 409…) s'affiche immédiatement, sans trois tentatives — adapté à une API
  locale.
- `refetchOnWindowFocus: false` : le rafraîchissement repose sur les intervalles et les invalidations.
- `staleTime: 5000` : 5 s de fraîcheur, évite les requêtes en double entre composants montés ensemble.
- Mutations : options par défaut (pas de `retry`).

### 2.2 Clés de cache et intervalles `[ACTUEL]`

> **Vague 3 (frontend)** : clé `sources` (`GET /api/sources`, sans intervalle, invalidée après acceptation,
> révocation et `block/clear`) ; bandeaux de blocage par source lus dans `sources[].blocked`. Repli temporaire sur
> `/api/status.blocked` si `GET /api/sources` répond 404 — **à retirer en vague 4** (le backend de la vague 3 sera
> livré).


> **Vague 7, partie 1 `[ACTUEL]`** (le tableau ci-dessous décrit l'état d'avant) :
> - `useTask` s'arrête quand la tâche est finale ; `useTasks` interroge toutes les 2 s s'il existe une tâche non
>   finale, **sinon toutes les 15 s** (décision 2026-10-08 : capte une tâche lancée depuis un autre onglet) ;
> - `invalidateAfterTask(client, task)` invalide selon `task.kind` (§ 2.3) ;
> - `useJobs` : anti-rebond de 300 ms sur `q` (`useDebouncedValue`) et `placeholderData: keepPreviousData` ;
> - filtres de la page Offres dans l'URL (`useSearchParams`).
> Vague 7, partie 2 `[ACTUEL]` : `useLabelJob()` invalide `["jobs"]`, `["report"]`, `["monitor"]` (Offres et Marché) ;
> `/marche?rapport=`. Vagues 3-4 `[ACTUEL]` (UI et backend) : `useSources()` `["sources"]` sans
> intervalle ; `ComposeModal` lit `["message-draft", app.id, to]` (`staleTime: 0`, `gcTime: 0`) à chaque relecture.
> Tous les hooks passent par `useLiveQuery` : requête et intervalle suspendus après un 401.


Toutes les fonctions passent par `api<T>(path)` (préfixe `/api`). Les clés sont **plates** (pas de hiérarchie
`[domaine, "list", filtres]`).

| Hook (`hooks.ts`) | Clé | Endpoint | `refetchInterval` | `enabled` |
| :--- | :--- | :--- | :--- | :--- |
| `useStatus()` | `["status"]` | `GET /api/status` | **15 s** | — |
| `useMonitor()` | `["monitor"]` | `GET /api/monitor` | **10 s** | — |
| `useSettings()` | `["settings"]` | `GET /api/settings` | — | — |
| `useProfile()` | `["profile"]` | `GET /api/profile` | — | — |
| `useApplications()` | `["applications"]` | `GET /api/applications` | — | — |
| `useReports()` | `["reports"]` | `GET /api/reports` | — | — |
| `useReport(name)` | `["report", name]` | `GET /api/reports/{name}` | — | `!!name` |
| `useJobs(filters)` | `["jobs", querystring]` | `GET /api/jobs?q=&min_score=&label=&status=&source=` (paramètres vides retirés) | — | — |
| `useJob(id)` | `["job", id]` | `GET /api/jobs/{id}` | — | `!!id` |
| `useApplication(id)` | `["application", id]` | `GET /api/applications/{id}` | — | `!!id` |
| `useTasks()` | `["tasks"]` | `GET /api/tasks` (30 dernières, 5 dernières lignes de journal) | **2 s** | — |
| `useTask(id)` | `["task", id]` | `GET /api/tasks/{id}` (journal complet) | **1,5 s** | `!!id` |

Requête hors cache : `ComposeModal` charge le brouillon par `api(…/message-draft)` dans un `useEffect`
(erreur ignorée silencieusement) `[ACTUEL]`.

Écarts d'interrogation et de recherche : **corrigés** `[ACTUEL]` (vague 7). `useTask` s'arrête dès que la tâche est finale
(`refetchInterval: false`, 1,5 s sinon, conformément au lot 1 `02`) ; `useTasks` interroge toutes les 2 s tant qu'une tâche
n'est pas finale, 15 s sinon ; la recherche texte d'Offres a un anti-rebond de 300 ms (`useDebouncedValue`) et
`placeholderData: keepPreviousData`.

`[ACTUEL]` (UI et backend, vagues 3-4) : `["sources"]` (`GET /api/sources`, sans intervalle, invalidée
avec `["status"]` après acceptation, révocation ou levée d'un blocage) ; `["message-draft", appId, to]`
(`GET …/message-draft?to=`, toujours relu). Pas de clé `["suppression"]` : aucune liste d'oppositions n'est
affichée (minimisation).

### 2.3 Mutations et invalidations `[ACTUEL]`

> Vagues 2-4 `[ACTUEL]` (UI et backend) — en plus du tableau ci-dessous :
> - importer une offre Indeed (`useImportItem`, vague 5) `POST /items/import` : `jobs`, `report`, `monitor` ; pas de
>   `track` (réponse `201` synchrone, aucune tâche) ;
> - fin de tâche : `invalidateAfterTask(client, task)` selon `task.kind` (table `INVALIDATE_BY_KIND`, type inconnu ⇒
>   tout le cache), et non plus `invalidateQueries()` sans clé ;
> - annoter (Offres ou Marché) : `jobs`, `report`, `monitor` ;
> - accepter les risques `POST /sources/{code}/consent`, révoquer `DELETE /sources/{code}/consent`, lever un
>   blocage `POST /block/clear {source}` : `sources`, `status` ;
> - passer la prudence à « prudent » depuis la popup (`useSwitchToPrudent`, local à `RiskConsentDialog`,
>   `PUT /settings {caution: "prudent"}`) : `settings`, `status` ;
> - opposition `POST /suppressions` (une requête par coordonnée, 201 idempotent) : `application/{id}`,
>   `applications` ;
> - relance explicite après un 428 (`ConsentGate`) : appel direct `api` puis `track` si la réponse est une tâche ;
>   pour `/runs`, si la prudence vient d'être passée à « prudent », corps relancé avec `caution: "prudent"` et sans
>   `max_items` > 15 ; pour `…/messages`, aucune requête : la relecture se rouvre.

Toutes les mutations passent par `useAction(fn, invalidate)` : `useMutation` qui, **en cas de succès**,
appelle `invalidateQueries({ queryKey })` pour chaque clé listée (correspondance par **préfixe** : `["jobs"]`
invalide toutes les variantes filtrées, `["report"]` tous les rapports). Pas de mise à jour optimiste.

| Action | Requête | Clés invalidées | Suite |
| :--- | :--- | :--- | :--- |
| Lever le blocage (`App`) | `POST /block/clear` | `status` | — |
| Lancer une analyse (`RunModal`) | `POST /runs` | — (appel direct `api`) | `track(task)` |
| Annoter (Offres) | `PUT /jobs/{id}/label` | `jobs`, `monitor` | — |
| Annoter (Marché, `OfferCard`) | `PUT /jobs/{id}/label` | `report`, `jobs`, `monitor` | — |
| Entraîner | `POST /model/train` | `tasks` | `track` |
| Recalculer les scores | `POST /runs/rescore` | `tasks` | `track` (Marché vide aussi `added`) |
| Préparer | `POST /jobs/{id}/prepare` | `tasks` | `track` |
| Ajouter une compétence (Compétences) | `POST /profile/skills` | `profile`, `jobs`, `monitor`, `tasks` | `track(linkedin_sync.task)` si présent |
| Ajouter une compétence (Marché) | `POST /profile/skills` | `profile`, `tasks` | idem |
| Retirer une compétence | `DELETE /profile/skills/{skill}` | `profile`, `jobs`, `monitor` | — |
| Enregistrer l'identité | `PUT /profile` | `profile`, `jobs`, `monitor` | — |
| Annuler les ajustements | `POST /profile/reset-overrides` | `profile`, `jobs`, `monitor` | — |
| Importer depuis LinkedIn | `POST /profile/import-linkedin` | `tasks` | `track` |
| Enregistrer les paramètres / oublier une réponse | `PUT /settings` | `settings`, `status`, `monitor` | Notice « Paramètres enregistrés. » |
| Email de test | `POST /smtp/test` | — | « Envoyé à {to}. » |
| Enregistrer message / CV | `PATCH /applications/{id}` | `application/{id}`, `applications` | — |
| Changer le statut | `PATCH /applications/{id}` `{status}` | `application/{id}`, `applications`, `jobs`, `monitor` | — |
| Enregistrer les contacts | `PATCH /applications/{id}` `{contacts}` | `application/{id}` | — |
| Lancer la candidature | `POST /applications/{id}/apply` | `tasks`, `application/{id}` | `track` |
| Chercher les recruteurs | `POST /applications/{id}/contacts/lookup` | `tasks` | `track` |
| Envoyer un message | `POST /applications/{id}/messages` (`confirm: true`) | `tasks`, `application/{id}` | `track`, fermeture |
| Répondre / annuler une tâche (`TaskDetail`) | `POST /tasks/{id}/answer`, `/cancel` | — (appel direct, le polling rafraîchit) | — |
| Confirmer `confirm_*` | `POST /tasks/{id}/answer` (`remember: false`) | `tasks` | — |

**Invalidation de fin de tâche** `[ACTUEL]` : `TaskProvider` compare le statut de chaque tâche de `["tasks"]` à la valeur
précédente ; à la première transition vers `terminée` / `échec` / `annulée`, il appelle `invalidateAfterTask(client, task)`
(`hooks.ts`). Cette fonction invalide toujours `["tasks"]` et `["task", id]`, puis les clés de la table `INVALIDATE_BY_KIND`
ci-dessous ; un `kind` absent de la table invalide **tout** le cache (sûr par défaut).

| `task.kind` | Clés invalidées en plus de `tasks` / `task` |
| :--- | :--- |
| `run` | `reports`, `report`, `jobs`, `monitor`, `profile`, `status`, `sources` |
| `rescore`, `train` | `reports`, `report`, `jobs`, `monitor`, `profile` |
| `prepare`, `apply`, `contacts`, `message` | `applications`, `application`, `jobs`, `monitor`, `status`, `sources` |
| `import_profile`, `profile_skill` | `profile`, `status`, `sources` |
| `purge` (kind interne, jamais lancé par l'interface, `backend/05` § 1.3) | `jobs`, `job`, `monitor` |

`sources` figure là où une tâche navigateur peut avoir posé un repos de 6 h (I-06) lu par le bandeau de blocage ;
`reports` / `report` là où un nouveau rapport est écrit.

Écarts relevés :
- Annoter depuis la page Offres n'invalide pas `["report"]` : Marché peut afficher une annotation périmée
  jusqu'au prochain chargement (5 s de `staleTime`). **Corrigé (vague 7, `[ACTUEL]`)** : `useLabelJob` invalide `report`.
- Enregistrer les contacts n'invalide pas `["applications"]` (sans effet visible aujourd'hui, la liste n'affiche
  pas les contacts).
- `[CIBLE]` (non livré) mise à jour optimiste admise pour l'**annotation** uniquement (réversible, fréquente) avec
  restauration sur erreur ; jamais pour un envoi, un statut ou un consentement.

### 2.4 Gestion des requêtes en cours `[ACTUEL]`
- Bouton déclencheur en `loading={mutation.isPending}` (libellé « … », désactivé) ; le reste de la page reste
  utilisable.
- Pages : « Chargement… » tant que `data` est absent ; Marché affiche un squelette pendant le chargement d'un
  rapport ; pas d'indicateur `isFetching` lors des rafraîchissements périodiques (volontaire, pas de clignotement).
- Actions longues : jamais attendues dans la page ; elles deviennent des **tâches** suivies par polling.

## 3. Gestion des Erreurs Réseau

### 3.1 Client `api<T>()` `[ACTUEL]`
> **Vague 2 `[ACTUEL]`** : `api()` lève `ApiError(status, detail, body, retryAfter)` ; `detail` est toujours une chaîne
> (une liste Pydantic est normalisée en « champ : message », G-1) ; `retryAfter` vient de l'en-tête `Retry-After`.
> Panne réseau ⇒ `ApiError(0, "API locale injoignable : l'application est-elle lancée ?")`.
> **Vague 3 `[ACTUEL]` côté UI** : sur un 428, `ApiError` porte `source`, `riskVersion` et `request` (chemin,
> méthode, corps d'origine) ; `api()` notifie les abonnés de `onConsentRequired()` **et** lève l'erreur (l'appelant
> affiche `detail`). Aucun rejeu dans `api()`.

- En-têtes : `Content-Type: application/json` et `X-Local-Token` si un jeton est présent (absent en
  `--dev`, le proxy Vite transmet vers 8765 sans jeton).
- Corps lu en texte puis `JSON.parse` tolérant (corps non JSON ⇒ `null`).
- Si `!response.ok` : `throw new ApiError(message, status)` avec `message` = `detail` si chaîne, sinon
  concaténation des `msg` d'un tableau Pydantic (séparateur « · »), sinon `Erreur {status}`.
- **Aucun traitement spécifique par code** : 401, 403, 409, 422, 429, 500, 502, 503 sont affichés tels quels
  dans l'`ErrorBox` du composant appelant. Une panne réseau (API arrêtée) lève un `TypeError` natif
  (« Failed to fetch » / « NetworkError when attempting to fetch resource ») affiché brut, en anglais.
- `authedUrl(path)` : URL `/api…?token=…` pour les contenus ouverts dans un nouvel onglet (rapport HTML, export
  JSON, CV imprimable).

### 3.2 Codes renvoyés par l'API et traitement attendu
> **Vague 3 `[ACTUEL]` (UI et backend)** : 428 ⇒ `ConsentGate` ouvre `RiskConsentDialog` pour
> `body.source` (construite depuis le 428 si `GET /api/sources` n'est pas chargée) ; aucun rejeu ; bandeau
> « Relancer : {action} » + « Ignorer ». 409 `Ce destinataire a demandé à ne plus être contacté.` ⇒ bandeau
> d'opposition de `ComposeModal` (vague 4).

> **Vague 2 `[ACTUEL]`** : 401 ⇒ bandeau global « recharge la page » **et arrêt de tout polling** (décision
> 2026-10-09 : le jeton ne redeviendra pas valide sans rechargement) ; 409 / 422 ⇒ `detail` tel quel ; 428 ⇒
> `detail` affiché en attendant la popup (vague 3) ; 429 ⇒ `detail` + « Réessaie dans X h ».


| Code | Origine (vérifiée) | Message serveur (exemples) | Traitement `[ACTUEL]` | Traitement `[CIBLE]` |
| :--- | :--- | :--- | :--- | :--- |
| **401** | middleware `guard` : jeton absent / invalide | « Jeton local manquant ou invalide. » | `ErrorBox` | Le jeton change **à chaque lancement** : un onglet resté ouvert après redémarrage reçoit 401 partout. Bandeau global unique « L'application a redémarré : recharge la page » + bouton « Recharger » (`location.reload()`), requêtes suspendues. |
| **403** | `guard` : `Host` non local ou `Origin` étrangère | « Hôte non autorisé. » / « Origine non autorisée. » | `ErrorBox` | Bandeau global « Ouvre l'interface via http://127.0.0.1:8765 ». |
| **404** | ressource absente | « Rapport introuvable. », « Aucune candidature préparée pour cette offre. » | `ErrorBox` | Idem + lien de retour (liste). |
| **409** | conflit d'état : tâche déjà en cours, identifiants absents, blocage actif (`require_browser_ready`), statut déjà `postulée`, tâche n'attendant plus de réponse, pas d'analyse, destinataire en opposition (vague 4) | « Une analyse est déjà en cours. », « Identifiants LinkedIn absents : … » | `ErrorBox` | `ErrorBox` + action : « Voir la tâche » (ouvre le tiroir), rafraîchir `status` (bandeau de blocage). |
| **422** | validation : `SettingsError`, Pydantic, `confirm` absent, email / objet invalide | « apply_daily_max doit être compris entre 0 et 200. » | `ErrorBox` | Évité par la validation client (cf. `05`) ; sinon message rattaché au champ. |
| **428** `[ACTUEL]` | consentement de source absent ou d'une autre `risk_version` (I-16) | corps : `{"detail": "...", "source": "linkedin", "risk_version": "2026-10"}` | `ConsentGate` ouvre `RiskConsentDialog` ; aucun rejeu | Ouvrir `RiskConsentDialog` pour `source` ; **aucun rejeu automatique** (décision du 2026-10-09) : bandeau « Relancer : {action} », clic explicite ; un message rouvre la relecture (I-04). |
| **429** | plafond quotidien (`check_quota`) | « Plafond quotidien atteint (10 candidatures par jour) : réessaie demain ou relève-le dans les paramètres. » | `ErrorBox` ; boutons déjà désactivés par anticipation via `status.quotas` | Message + lien « Régler les plafonds » (`/parametres`), rappel que le plafond n'est jamais dépassé (I-07). |
| **500** | `read_files` d'une candidature | message d'exception | `ErrorBox` | Message générique « Erreur interne de l'application locale » + détail repliable. |
| **502** | échec SMTP du test | erreur SMTP | `ErrorBox` | Inchangé. |
| **503** | SMTP non configuré ; interface non compilée (HTML) | « SMTP non configuré : … » | `ErrorBox` (+ bouton désactivé par anticipation via `status.smtp_configured`) | Inchangé. |
| Réseau | API arrêtée | `TypeError` | texte anglais brut | `ApiError(status 0)` « API locale injoignable : l'application est-elle lancée ? » |

Implémentation `[ACTUEL]` : `api()` lève `ApiError` ; `useWatchUnauthorized()` (abonnement aux `QueryCache` /
`MutationCache`) traite le 401 (bandeau unique dans `App`, requêtes suspendues) ; 409, 422 et 429 restent traités
par le composant appelant ; le 428 passe par `onConsentRequired()` → `ConsentGate`. Restent `[CIBLE]` : bandeau
global 403, lien « Régler les plafonds » sur 429, message générique 500.

### 3.3 Erreurs de rendu
> **Vague 2 `[ACTUEL]`** : `ErrorBoundary` autour de `<Routes>` avec `key = pathname` (changer de page réinitialise
> l'erreur) ; le tiroir des tâches reste hors frontière. Les rapports passent par `reportSchema` (Zod, `.catch`
> sur `weekly`, `rising`… : un rapport ancien ou incomplet ne fait plus planter Marché).

Avant la vague 2 : aucune *Error Boundary*. Depuis : `components/ErrorBoundary.tsx` (« Cette page a rencontré une
erreur », « Revenir au tableau de bord », « Réessayer ») autour de `<Routes>`. Reste `[CIBLE]` : une frontière
propre à `MarketBody`.

### 3.4 Bug bloquant relevé — corrigé en vague 0 `[ACTUEL]`
`SettingsPage` appelle `useAction(...)` (hook `testMail`) **après** le `return` anticipé
`if (!form) return <ErrorBox …/>` : le premier rendu exécute 6 hooks, le suivant 7 → React lève « Rendered more
hooks than during the previous render » et la page Paramètres plante dès que les réglages arrivent. Correction :
déclarer `testMail` avant le `return` anticipé ; fait ; la règle ESLint `react-hooks/rules-of-hooks`
(en `error`, cf. `04-routing-and-pages.md` § 4.3) empêche toute récidive.

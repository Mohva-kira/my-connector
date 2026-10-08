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

`[CIBLE]` :
- filtres de la page Offres et rapport de Marché dans les **paramètres de recherche d'URL** (`useSearchParams`)
  pour survivre au rechargement et pouvoir être partagés entre onglets ;
- pas de store global supplémentaire : le contexte `TaskProvider` suffit ; un second contexte
  `SourceConsentProvider` porte la file d'attente de la popup de risques (action en attente + source).

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
| `useJobs(filters)` | `["jobs", querystring]` | `GET /api/jobs?q=&min_score=&label=&status=` (paramètres vides retirés) | — | — |
| `useJob(id)` | `["job", id]` | `GET /api/jobs/{id}` | — | `!!id` |
| `useApplication(id)` | `["application", id]` | `GET /api/applications/{id}` | — | `!!id` |
| `useTasks()` | `["tasks"]` | `GET /api/tasks` (30 dernières, 5 dernières lignes de journal) | **2 s** | — |
| `useTask(id)` | `["task", id]` | `GET /api/tasks/{id}` (journal complet) | **1,5 s** | `!!id` |

Requête hors cache : `ComposeModal` charge le brouillon par `api(…/message-draft)` dans un `useEffect`
(erreur ignorée silencieusement) `[ACTUEL]`.

Écarts relevés :
- `useTask` continue d'interroger **toutes les 1,5 s même quand la tâche est finale**, tant que le détail est
  ouvert ; `useTasks` interroge toutes les 2 s en permanence, même sans tâche active. `[CIBLE]` :
  ```ts
  refetchInterval: (query) => (query.state.data && FINAL_STATUSES.includes(query.state.data.status) ? false : 1500)
  ```
  et pour `useTasks` : 2 s s'il existe une tâche non finale, sinon 15 s. Le lot 1 (`02-data-flow-and-events.md`)
  fixe 1,5 s « tant que le statut n'est pas final » : la cible s'y conforme.
- La recherche texte de la page Offres déclenche une requête **à chaque frappe** (clé `["jobs", query]`).
  `[CIBLE]` anti-rebond de 300 ms sur `q` et `placeholderData: keepPreviousData` pour éviter le clignotement.

`[CIBLE]` nouvelles clés : `["sources"]` (`GET /api/sources`, sans intervalle, invalidée après acceptation /
révocation), `["suppression"]` si une liste d'oppositions est affichée.

### 2.3 Mutations et invalidations `[ACTUEL]`

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

**Invalidation de fin de tâche** : `TaskProvider` compare le statut de chaque tâche de `["tasks"]` à la valeur
précédente ; à la première transition vers `terminée` / `échec` / `annulée`, il appelle
`client.invalidateQueries()` **sans clé** (tout le cache). Simple et sûr pour un utilisateur unique ; coût :
une rafale de requêtes à chaque fin de tâche. `[CIBLE]` invalider selon `task.kind` : `run` → `reports`,
`jobs`, `monitor`, `profile`, `status` ; `rescore` / `train` → `jobs`, `report`, `monitor` ; `prepare` /
`apply` / `contacts` / `message` → `applications`, `application`, `jobs`, `monitor`, `status` ;
`import_profile` / `profile_skill` → `profile`.

Écarts relevés :
- Annoter depuis la page Offres n'invalide pas `["report"]` : Marché peut afficher une annotation périmée
  jusqu'au prochain chargement (5 s de `staleTime`). `[CIBLE]` invalider `report` aussi.
- Enregistrer les contacts n'invalide pas `["applications"]` (sans effet visible aujourd'hui, la liste n'affiche
  pas les contacts).
- `[CIBLE]` mise à jour optimiste admise pour l'**annotation** uniquement (réversible, fréquente) avec
  restauration sur erreur ; jamais pour un envoi, un statut ou un consentement.

### 2.4 Gestion des requêtes en cours `[ACTUEL]`
- Bouton déclencheur en `loading={mutation.isPending}` (libellé « … », désactivé) ; le reste de la page reste
  utilisable.
- Pages : « Chargement… » tant que `data` est absent ; Marché affiche un squelette pendant le chargement d'un
  rapport ; pas d'indicateur `isFetching` lors des rafraîchissements périodiques (volontaire, pas de clignotement).
- Actions longues : jamais attendues dans la page ; elles deviennent des **tâches** suivies par polling.

## 3. Gestion des Erreurs Réseau

### 3.1 Client `api<T>()` `[ACTUEL]`
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

| Code | Origine (vérifiée) | Message serveur (exemples) | Traitement `[ACTUEL]` | Traitement `[CIBLE]` |
| :--- | :--- | :--- | :--- | :--- |
| **401** | middleware `guard` : jeton absent / invalide | « Jeton local manquant ou invalide. » | `ErrorBox` | Le jeton change **à chaque lancement** : un onglet resté ouvert après redémarrage reçoit 401 partout. Bandeau global unique « L'application a redémarré : recharge la page » + bouton « Recharger » (`location.reload()`), requêtes suspendues. |
| **403** | `guard` : `Host` non local ou `Origin` étrangère | « Hôte non autorisé. » / « Origine non autorisée. » | `ErrorBox` | Bandeau global « Ouvre l'interface via http://127.0.0.1:8765 ». |
| **404** | ressource absente | « Rapport introuvable. », « Aucune candidature préparée pour cette offre. » | `ErrorBox` | Idem + lien de retour (liste). |
| **409** | conflit d'état : tâche déjà en cours, identifiants absents, blocage actif (`require_browser_ready`), statut déjà `postulée`, tâche n'attendant plus de réponse, pas d'analyse ; `[CIBLE]` destinataire en opposition | « Une analyse est déjà en cours. », « Identifiants LinkedIn absents : … » | `ErrorBox` | `ErrorBox` + action : « Voir la tâche » (ouvre le tiroir), rafraîchir `status` (bandeau de blocage). |
| **422** | validation : `SettingsError`, Pydantic, `confirm` absent, email / objet invalide | « apply_daily_max doit être compris entre 0 et 200. » | `ErrorBox` | Évité par la validation client (cf. `05`) ; sinon message rattaché au champ. |
| **428** `[CIBLE]` | consentement de source absent ou d'une autre `risk_version` (I-16) | corps proposé : `{"detail": "...", "source": "linkedin", "risk_version": "2026-10"}` | — | Ouvrir `RiskConsentDialog` pour `source`, puis rejouer la requête **une fois** après acceptation. |
| **429** | plafond quotidien (`check_quota`) | « Plafond quotidien atteint (10 candidatures par jour) : réessaie demain ou relève-le dans les paramètres. » | `ErrorBox` ; boutons déjà désactivés par anticipation via `status.quotas` | Message + lien « Régler les plafonds » (`/parametres`), rappel que le plafond n'est jamais dépassé (I-07). |
| **500** | `read_files` d'une candidature | message d'exception | `ErrorBox` | Message générique « Erreur interne de l'application locale » + détail repliable. |
| **502** | échec SMTP du test | erreur SMTP | `ErrorBox` | Inchangé. |
| **503** | SMTP non configuré ; interface non compilée (HTML) | « SMTP non configuré : … » | `ErrorBox` (+ bouton désactivé par anticipation via `status.smtp_configured`) | Inchangé. |
| Réseau | API arrêtée | `TypeError` | texte anglais brut | `ApiError(status 0)` « API locale injoignable : l'application est-elle lancée ? » |

`[CIBLE]` implémentation : `api()` lève `ApiError` enrichie (`status`, `detail`, `body`) ; un `QueryCache` /
`MutationCache` global (`onError`) gère 401, 403 et réseau (bandeau unique dans `App`), tandis que 409, 422, 428
et 429 restent traités localement par le composant appelant (le 428 via `SourceConsentProvider`).

### 3.3 Erreurs de rendu
`[ACTUEL]` aucune *Error Boundary* : une exception de rendu (ex. donnée inattendue d'un rapport ancien) vide toute
l'application. `[CIBLE]` `ErrorBoundary` autour de `<Routes>` (message « Cette page a rencontré une erreur » +
« Revenir au tableau de bord ») et autour de `MarketBody` ; le tiroir des tâches reste hors de la frontière pour
rester utilisable.

### 3.4 Bug bloquant relevé — corrigé en vague 0 `[ACTUEL]`
`SettingsPage` appelle `useAction(...)` (hook `testMail`) **après** le `return` anticipé
`if (!form) return <ErrorBox …/>` : le premier rendu exécute 6 hooks, le suivant 7 → React lève « Rendered more
hooks than during the previous render » et la page Paramètres plante dès que les réglages arrivent. Correction :
déclarer `testMail` avant le `return` anticipé ; fait ; la règle ESLint `react-hooks/rules-of-hooks`
(en `error`, cf. `04-routing-and-pages.md` § 4.3) empêche toute récidive.

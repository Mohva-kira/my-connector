# Contrats d'API & Conventions

> API **locale** de `linkedin_market` (`web/app.py`, routeurs `web/routers/core.py`, `jobs.py`,
> `applications.py`). Tout ce qui n'est pas marqué `[CIBLE]` est vérifié dans le code et les tests
> (`tests/test_api.py`, `test_messages_api.py`, `test_apply_task.py`). Les messages `detail` sont cités tels quels.

## 1. Standards d'API

| Règle | Valeur | Justification |
| :--- | :--- | :--- |
| Style | REST JSON sur HTTP **local** (`http://127.0.0.1:8765`, port modifiable par `--port`). Pas de HTTPS : le trafic ne quitte pas la boucle locale. | Auto-hébergé, cf. lot 1 `04` § 1 |
| Préfixe | `/api` **sans numéro de version**. Le frontend est compilé et livré avec le backend (même version, même dépôt) : il n'existe pas de client tiers à garder compatible. | Une version `/api/v2` ne serait introduite que si une API publique était un jour exposée (hors périmètre). |
| URLs | Noms de ressources au pluriel et en anglais (`/runs`, `/jobs`, `/applications`, `/tasks`, `/reports`), segments composés en **kebab-case** (`/message-draft`, `/reset-overrides`, `/import-linkedin`, `/contacts/lookup`). Identifiants d'offre avec préfixe (`job:4012345678`) passés tels quels dans le chemin. | `[ACTUEL]` |
| Clés JSON | **snake_case**, identiques aux noms Python et aux colonnes de l'ERD (`apply_daily_max`, `needs_input`, `created_at`). Pas de camelCase : le glossaire impose un seul nom par concept (lot 1 `00` § 1). | `[ACTUEL]` |
| Valeurs d'énumération | En français, sans majuscule, accents conservés (`en_attente`, `terminée`, `postulée`, `déjà_postulée`). | Identiques aux constantes du code et aux `CHECK` de l'ERD |
| Dates | `[ACTUEL]` ISO 8601 **local sans fuseau** à la seconde (`2026-10-08T14:32:05`) ; `generated_at` d'un rapport en microsecondes. `[CIBLE]` ISO 8601 UTC avec décalage (`2026-10-08T12:32:05Z`), sérialisé depuis `timestamptz`. | Écart E-9 (§ 4 de `02`) |
| Documentation | OpenAPI/Swagger uniquement avec `--dev` (`/api/docs`, `/api/openapi.json`) ; désactivée sinon. | Surface minimale |

### 1.1 Changements de contrat figés pour la vague 1 (PostgreSQL) `[CIBLE]`
Seuls ces points changent ; routes, corps de requête, codes et messages `detail` sont inchangés.

| # | Champ / ressource | Avant `[ACTUEL]` | Après la vague 1 |
| :--- | :--- | :--- | :--- |
| C-1 | **Tout horodatage** (`created_at`, `started_at`, `finished_at`, `prepared_at`, `updated_at`, `generated_at`, `trained_at`, `at` des historiques, d'`activity`, de `blocked`, `blocked.until`, `errors[].at`, `recent_events[].at`, `last_analysis.generated_at`, `GET /api/reports[].generated_at`) | ISO local naïf (`2026-10-08T14:32:05`, parfois µs) | ISO 8601 **UTC avec suffixe `Z`**, à la seconde : `2026-10-08T12:32:05Z` (helper unique `db.utc_iso()` côté serveur). Les rapports **importés** gardent leur `generated_at` d'origine dans le JSON (I-02), mais la liste `GET /api/reports` expose `created_at` de la base converti en UTC. |
| C-2 | Horodatage absent (`started_at`, `finished_at` d'une tâche non démarrée / non finie, `updated_at` jamais modifié) | `""` | `null` |
| C-3 | Dates sans heure (`posted_at`, jours de `per_day`) | `AAAA-MM-JJ` | inchangé (`posted_at` = date de la source ; jours de `per_day` = jour **local** de la machine) |
| C-4 | `Contact.name` (lecture **et** `PATCH /api/applications/{id}` `contacts`) | `name` | **`full_name`** (E-15). `name` n'est plus accepté en écriture (Pydantic `extra="ignore"` ⇒ un client obsolète perd le nom : le frontend doit migrer dans la même vague). |
| C-5 | `Contact.source` | `linkedin` \| `manuel` | `linkedin_equipe_recrutement` \| `manuel` (E-15) ; défaut `manuel` |
| C-6 | `history[].kind` des candidatures | `statut`, `candidature`, `contacts`, `message` | `status`, `apply`, `contacts`, `message` (E-15) ; détails inchangés (`status`, `state`/`detail`, `found`/`emails`, `channel`/`to`/`extrait`…) |
| C-7 | `Task.id` | 12 caractères hexadécimaux | UUID v7 en texte (`0199c4a2-…`), généré par Python (`uuid.uuid7()`) ; toujours une chaîne opaque |
| C-8 | Nom d'un **nouveau** rapport (`result.report`, `GET /api/reports[].name`) | `data-engineer-20261008-1432` | `data-engineer-2026-10-08`, puis `-2`, `-3`… le même jour local (E-1, E-2). Les rapports importés gardent leur nom. |
| C-9 | `folder` d'une candidature | chemin absolu | inchangé (absolu, calculé depuis `folder_path` relatif à `DATA_DIR`) |
| C-10 | `score` (candidature) | nombre | toujours un nombre JSON (le `Decimal` de `numeric(4,1)` est converti en `float`) |

Les identifiants d'élément gardent la forme `<kind>:<external_id>` (`job:4012345678`) : la base stocke
`kind` + `external_id` et le dépôt recompose l'identifiant ; la question de l'unicité entre sources est reportée à
la vague 5.

## 2. Format Standard des Réponses

### Réponse de Succès
Pas d'enveloppe `success/data/meta` : la ressource est renvoyée directement (objet ou liste). Il n'y a pas de
pagination côté serveur — les volumes sont plafonnés par construction (≤ 60 offres par type et par collecte,
`limit ≤ 100` pour les tâches).

```json
{"id": "job:4012345678", "label": 1, "n_labels": 7, "min_labels": 6}
```

### Réponse d'Erreur
Format natif FastAPI, message **en français** destiné à être affiché tel quel :

```json
{"detail": "Une analyse est déjà en cours."}
```

Exception : les erreurs de **validation de schéma** Pydantic (types, champ manquant, `Literal`, validateurs)
renvoient `422` avec une **liste** :

```json
{
  "detail": [
    {"type": "literal_error", "loc": ["body", "channel"], "msg": "Input should be 'email' or 'linkedin'",
     "input": "sms", "ctx": {"expected": "'email' or 'linkedin'"}}
  ]
}
```

Pour un `field_validator`, `msg` vaut `"Value error, Adresse email invalide."`. Ces messages Pydantic sont en
anglais (écart E-10) ; `[CIBLE]` : un gestionnaire `RequestValidationError` les traduit en
`{"detail": "<champ> : <message français>"}` en gardant le code 422. Le frontend doit accepter les deux formes
(`typeof detail === "string"`).

### Objet Tâche (`Task.public()`)
Toute route `202` renvoie l'objet tâche complet ; `GET /api/tasks` et les champs `active_tasks` / `recent_tasks`
de `/api/monitor` utilisent `public(with_logs=False)` (5 dernières lignes de `logs`).

```json
{
  "id": "a1b2c3d4e5f6",
  "kind": "apply",
  "title": "Postuler : Senior Python Developer",
  "status": "attente_saisie",
  "logs": ["14:32:05 Navigateur démarré : chrome", "14:32:19 Ouverture de la candidature simplifiée…",
           "14:32:31 Étape 1 remplie.", "14:32:40 Question : How many years of experience do you have with Spark?"],
  "result": null,
  "error": "",
  "needs_input": {"question": "How many years of experience do you have with Spark?", "field": "s2_f1",
                  "options": [], "suggestion": "10", "review": ""},
  "meta": {"key": "job:4012345678", "job_id": "job:4012345678"},
  "browser": true,
  "created_at": "2026-10-08T14:32:04",
  "started_at": "2026-10-08T14:32:04",
  "finished_at": ""
}
```

| Champ | Type | Règle |
| :--- | :--- | :--- |
| `status` | `en_attente` \| `en_cours` \| `attente_saisie` \| `attente_verification_manuelle` \| `terminée` \| `échec` \| `annulée` | Le client interroge toutes les 1,5 s tant que le statut n'est pas final. |
| `result` | objet propre au `kind` ou `null` | Rempli seulement en `terminée` (cf. `05` § 1.3). |
| `error` | chaîne | Message de l'exception (`str(exc)`) en `échec` ; `"Tâche annulée."` en `annulée`. |
| `needs_input` | `null` ou `{question, field, options, suggestion, review}` | Non nul uniquement en `attente_saisie`. `field` commençant par `confirm` = confirmation d'une action irréversible ; `review` = texte à relire (≤ 3 500 caractères). |
| `logs` | liste de chaînes `HH:MM:SS message` | 400 lignes au plus (`MAX_LOG_LINES`). |
| `created_at`, `started_at`, `finished_at` | chaîne ISO ou `""` | `""` tant que l'étape n'est pas atteinte. |

## 3. Headers HTTP Standards

| En-tête | Sens | Règle |
| :--- | :--- | :--- |
| `X-Local-Token: <jeton>` | Requête | Obligatoire sur tout `/api/*` sauf `/api/health`. Alternative `?token=<jeton>` pour les liens ouverts dans un nouvel onglet (rapport HTML, `cv.html`). Absent/faux ⇒ `401`. Désactivé si `create_app(token=None)` (tests, `--dev`). |
| `Host` | Requête | Doit correspondre à `^(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$`, sinon `403 {"detail": "Hôte non autorisé."}` — vérifié sur **toutes** les routes, SPA comprise. |
| `Origin` | Requête | Pour `POST/PUT/PATCH/DELETE` : si présent, doit égaler `Host` (ou `http://127.0.0.1:5174` / `http://localhost:5174` en `--dev`), sinon `403 {"detail": "Origine non autorisée."}`. |
| `Content-Type: application/json` | Requête | Pour tout corps. |
| `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` | Réponse | Ajoutés à toutes les réponses (`setdefault`). `no-referrer` évite la fuite de `?token=` vers les sites ouverts depuis un rapport. |
| `Cache-Control: no-store` | Réponse | Sur `index.html` (contient le jeton). |
| `Content-Security-Policy: sandbox allow-scripts allow-popups` | Réponse | `GET /api/reports/{name}/html`. Pour `cv.html` : `sandbox allow-scripts allow-modals allow-popups`. Origine opaque : ces pages ne peuvent ni lire le jeton ni appeler l'API. |
| `Authorization`, `Idempotency-Key`, `X-Correlation-Id` | — | **Non utilisés.** Pas de compte (jeton local, cf. `04`) ; idempotence par refus des doublons de tâche (409) ; la corrélation se fait par l'`id` de tâche repris dans les logs. |
| `Retry-After` | Réponse | `[CIBLE]` sur `429` : secondes jusqu'à minuit (heure locale), début du jour suivant pour les plafonds. |

## 4. Codes de Statut Utilisés

| Code | Usage réel |
| :--- | :--- |
| `200 OK` | Lecture, mise à jour synchrone (`PUT /api/settings`, `PATCH /api/applications/{id}`, `PUT /api/jobs/{id}/label`…), action synchrone (`POST /api/block/clear`, `/api/smtp/test`, `/api/tasks/{id}/cancel`, `/api/tasks/{id}/answer`, `/api/profile/skills`). |
| `201 Created` | `[CIBLE]` `POST /api/sources/{code}/consent`, `POST /api/suppressions`, `POST /api/items/import`. Aucune route actuelle ne renvoie 201. |
| `202 Accepted` | Tâche créée : `POST /api/runs`, `/api/runs/rescore`, `/api/model/train`, `/api/jobs/{id}/prepare`, `/api/profile/import-linkedin`, `/api/applications/{id}/apply`, `/contacts/lookup`, `/messages`. |
| `204 No Content` | `[CIBLE]` `DELETE /api/sources/{code}/consent`. |
| `400 Bad Request` | Non utilisé (FastAPI renvoie 422 pour un corps invalide). |
| `401 Unauthorized` | Jeton local manquant ou invalide. |
| `403 Forbidden` | `Host` non local ou `Origin` étrangère. |
| `404 Not Found` | Rapport, tâche, offre, candidature, CV introuvable ; toute route `GET /api/…` inconnue (`{"detail": "Introuvable."}`). |
| `405 Method Not Allowed` | Méthode non `GET` sur une route `/api/…` inexistante (capturée par la route SPA `GET /{path:path}`). |
| `409 Conflict` | Tâche du même type déjà active ; identifiants LinkedIn absents ; blocage en cours ; aucune analyse disponible ; candidature déjà postulée ; tâche qui n'attend pas de réponse ou déjà terminée. `[CIBLE]` destinataire en liste d'opposition, `risk_version` périmée. |
| `422 Unprocessable Entity` | Corps invalide (Pydantic) **et** règle métier de validation : réglage hors bornes, statut inconnu, `value` d'annotation, confirmation absente, adresse/objet invalides, test SMTP vers un tiers, mot-clé vide. |
| `428 Precondition Required` | `[CIBLE]` action navigateur sans consentement actif à la version courante des risques de la source (I-16). Corps : `{"detail": "…", "source": "linkedin", "risk_version": "2026-10"}` — le frontend ouvre la popup de cette source puis rejoue la requête une fois. |
| `429 Too Many Requests` | Plafond quotidien atteint (`apply_daily_max`, `message_daily_max`). |
| `500 Internal Server Error` | Dossier de candidature hors du répertoire de données (`ValueError` de `folder_of`) ; toute exception non rattrapée. |
| `502 Bad Gateway` | `POST /api/smtp/test` : le serveur SMTP refuse ou échoue (`OutreachError`). |
| `503 Service Unavailable` | SMTP non configuré (`/messages` canal `email`, `/smtp/test`) ; frontend non compilé (route SPA, page HTML). |

## 5. Catalogue des Endpoints

Notation : **Corps** = modèle Pydantic ou type ; **Erreurs** = code et `detail` exact.

### 5.1 Santé, état, monitoring (`core.py`, `app.py`)

| Méthode & chemin | Corps / paramètres | Succès | Erreurs |
| :--- | :--- | :--- | :--- |
| `GET /api/health` | — (sans jeton) | `200 {"ok": true}` | 403 hôte |
| `GET /api/status` | — | `200` objet ci-dessous | 401, 403 |
| `POST /api/block/clear` | — | `200 {"ok": true}` (événement `unblocked`) | — |
| `GET /api/monitor` | — | `200` objet ci-dessous | — |

`GET /api/status` :
```json
{
  "session": {"age_hours": 3.2, "ttl_hours": 72, "valid": true},
  "credentials_configured": true,
  "smtp_configured": true,
  "anthropic_configured": false,
  "interpreter": "claude",
  "browser": "auto",
  "headless": false,
  "profile_file": true,
  "has_analysis": true,
  "quotas": {"apply": {"used": 2, "limit": 10}, "messages": {"used": 0, "limit": 15}},
  "active_tasks": 1,
  "blocked": {"at": "2026-10-08T09:14:02", "where": "collecte", "until": "2026-10-08T15:14"}
}
```
`session.age_hours` vaut `null` sans session enregistrée ; `blocked` vaut `null` hors période de repos. Aucun
secret n'est exposé (test `test_monitor_and_status_shape`). `[CIBLE]` ajout de
`"sources": [{"code": "linkedin", "consent_required": true}]`.

`GET /api/monitor` :
```json
{
  "generated_at": "2026-10-08T14:40:00",
  "by_status": {"préparée": 3, "postulée": 2, "contactée": 1},
  "funnel": [{"step": "Offres analysées", "count": 30}, {"step": "Jugées pertinentes", "count": 7},
             {"step": "Candidatures préparées", "count": 6}, {"step": "Postulées", "count": 2},
             {"step": "Entretiens", "count": 0}],
  "quotas": {"apply": {"used": 2, "limit": 10}, "messages": {"used": 0, "limit": 15}},
  "activity": [{"date": "2026-09-25"}, {"date": "2026-10-08", "run": 1, "apply": 2}],
  "recent_events": [{"at": "2026-10-08T14:20:11", "kind": "apply", "job": "job:4012345678", "title": "Senior Python Developer"}],
  "model": {"trained": true, "n_labels": 14, "train_accuracy": 0.86, "importance": [{"feature": "couverture_competences", "weight": 1.92}], "loss_history": [0.69, 0.41], "trained_at": "2026-10-07T18:02:44"},
  "last_analysis": {"report": "data-engineer-20261008-1432", "generated_at": "2026-10-08T14:32:58.120331", "keyword": "Data Engineer", "counts": {"job": 30, "post": 30, "person": 30}},
  "active_tasks": [],
  "recent_tasks": [],
  "errors": [{"id": "0f9e8d7c6b5a", "title": "Collecte et analyse — Data Engineer", "error": "Aucun navigateur utilisable (LINKEDIN_BROWSER=auto) : …", "at": "2026-10-07T21:03:10"}]
}
```
`activity` contient 14 jours (jours vides inclus) ; `recent_events` les 15 derniers événements, du plus récent au
plus ancien ; `errors` les 5 dernières tâches en `échec` parmi les 30 plus récentes.

### 5.2 Réglages (`core.py`)

| Méthode & chemin | Corps | Succès | Erreurs |
| :--- | :--- | :--- | :--- |
| `GET /api/settings` | — | `200` réglages complets | — |
| `PUT /api/settings` | objet JSON libre (`dict`) : **mise à jour partielle**, clés inconnues ignorées | `200` réglages fusionnés et enregistrés | `422 {"detail": "<message SettingsError>"}` ; `422` liste si le corps n'est pas un objet |

Réglages et règles de `settings.validate_settings()` :

| Clé | Défaut | Validation | Message d'erreur exact |
| :--- | :--- | :--- | :--- |
| `keyword` | `""` | chaîne, `strip()` | — |
| `location` | `""` | chaîne (vide = monde entier) | — |
| `remote_only` | `true` | `bool(value)` | — |
| `caution` | `"normal"` (`[CIBLE]` décidé : `"prudent"`) | ∈ `prudent`, `normal`, `agressif` | `caution doit être l'un de : prudent, normal, agressif.` |
| `max_items` | `null` | `[ACTUEL]` `null`/`""` ou entier 1–200 ; `[CIBLE]` décidé : entier 1 à `Caution.max_items` de la prudence choisie (15 / 30 / 60) | `max_items doit être un entier.` / `max_items doit être compris entre 1 et 200.` ; `[CIBLE]` `max_items ne peut pas dépasser 15 en prudence « prudent ».` |
| `profile_source` | `"auto"` | ∈ `auto`, `file`, `linkedin` | `profile_source doit être l'un de : auto, file, linkedin.` |
| `interpreter` | `"claude"` | ∈ `claude`, `local`, `none` (`[CIBLE]` `none` refusé à l'écriture et lu comme `local`, I-11) | `interpreter doit être l'un de : claude, local, none.` |
| `apply_min_score` | `70.0` | nombre 0–100 | `apply_min_score doit être un nombre.` / `apply_min_score doit être compris entre 0 et 100.` |
| `apply_daily_max` | `10` | entier 0–200 | `apply_daily_max doit être compris entre 0 et 200.` |
| `message_daily_max` | `15` | entier 0–200 | `message_daily_max doit être compris entre 0 et 200.` |
| `contact_email`, `contact_phone`, `signature` | `""` | chaîne, `strip()` | — |
| `linkedin_skill_sync` | `true` | `bool(value)` | — |
| `answers` | `{}` | objet ; clé = `nlp.normalize(question)[:200]`, valeur `str[:500]` | `answers doit être un objet question -> réponse.` |

Attention : `PUT` avec `answers` **remplace** l'objet entier (fusion au niveau des clés de premier niveau).

### 5.3 Profil et compétences (`core.py`)

| Méthode & chemin | Corps | Succès | Erreurs |
| :--- | :--- | :--- | :--- |
| `GET /api/profile` | — | `200` profil effectif | — |
| `PUT /api/profile` | `ProfileUpdate` | `200` profil effectif | `422 seniority doit être l'un de : junior, confirme, senior, lead.` ; `422 years_experience doit être compris entre 0 et 60.` |
| `POST /api/profile/skills` | `SkillBody {skill: str}` | `200` profil + `linkedin_sync` | `422 {"detail": "Compétence vide."}` |
| `DELETE /api/profile/skills/{skill}` | — (`{skill:path}` accepte `/`, ex. `ci/cd`) | `200` profil effectif | — |
| `POST /api/profile/reset-overrides` | — | `200` profil (surcharges vidées) | — |
| `POST /api/profile/import-linkedin` | — | `202` tâche `import_profile` | `409 Un import est déjà en cours.` ; `409` identifiants ; `409` blocage |

`ProfileUpdate` : `title: str | None`, `years_experience: int | None`, `seniority: str | None`,
`clear_years: bool = false` (met `years_experience` à `null`). Les `None` sont ignorés ; `title` tronqué à 120
caractères ; une compétence normalisée de plus de 60 caractères est ignorée.

`GET /api/profile` :
```json
{
  "available": true,
  "origin": "/home/me/linkedin_market/profil.md",
  "title": "Data Engineer freelance",
  "skills": ["airflow", "python", "spark"],
  "base_skills": ["python", "spark"],
  "years_experience": 10,
  "seniority": "senior",
  "overrides": {"title": "", "years_experience": null, "seniority": "", "added_skills": ["airflow"], "removed_skills": []},
  "suggestions": {"missing": [{"skill": "aws", "weight": 0.5}], "rising": [{"skill": "docker", "growth": 1.5}]},
  "scores_outdated": true
}
```
Sans profil : `{"available": false, "message": "Aucun profil : dépose profil.md / profil.json / cv.pdf à la racine du projet, …", "overrides": {…}, "skills": [], "base_skills": [], "suggestions": {…}, "scores_outdated": false}`.

`POST /api/profile/skills` ajoute `linkedin_sync` (l'ajout local n'est **jamais** annulé) :

| `linkedin_sync.status` | Condition | Contenu |
| :--- | :--- | :--- |
| `off` | `linkedin_skill_sync = false` | `message: "Synchronisation LinkedIn désactivée dans les paramètres."` |
| `skipped` | compétence déjà présente, ou tâche `profile_skill` active pour elle | `message` |
| `unavailable` | identifiants absents ou blocage | `message: "Compétence ajoutée ici, mais pas sur LinkedIn : <detail>"` |
| `queued` | tâche créée | `task: Task.public()` |

### 5.4 Rapports (`core.py`)

| Méthode & chemin | Succès | Erreurs |
| :--- | :--- | :--- |
| `GET /api/reports` | `200 [{"name": "data-engineer-20261008-1432", "keyword": "Data Engineer", "generated_at": "2026-10-08T14:32:58.120331", "jobs": 30, "html": true}]` (plus récent d'abord) | — |
| `GET /api/reports/{name}` | `200` contenu JSON du rapport + `top_jobs` enrichis (`label`, `application_status`, identité réparée) + `is_latest` | `404 Rapport introuvable.` (nom contenant `/`, `\` ou commençant par `.` refusé) |
| `GET /api/reports/{name}/html` | `200 text/html` sandboxé | `404 Rapport introuvable.` |

Clés d'un rapport : `generated_at`, `window_days` (30), `request`, `profile`, `counts`, `top_terms`, `skill_demand`,
`rising`, `weekly`, `gap`, `top_jobs`, `model`, `min_labels`, `interpretation`, `interpreter` (`claude` \| `local`
\| `none`), `interpreter_model`. Aucune route n'écrit un rapport existant (I-02).

### 5.5 Tâches (`core.py`)

| Méthode & chemin | Corps / paramètres | Succès | Erreurs |
| :--- | :--- | :--- | :--- |
| `GET /api/tasks` | `limit: int = 30` (plafonné à 100) | `200` liste `public(with_logs=False)`, plus récente d'abord | — |
| `GET /api/tasks/{task_id}` | — | `200 Task.public()` | `404 Tâche introuvable.` |
| `POST /api/tasks/{task_id}/cancel` | — | `200 {"ok": true}` | `409 Tâche introuvable ou déjà terminée.` |
| `POST /api/tasks/{task_id}/answer` | `Answer {value: str, remember: bool = true}` | `200 {"ok": true}` | `409 Cette tâche n'attend pas de réponse.` ; `409 Cette tâche n'attend plus de réponse.` |

`remember=true` mémorise la réponse (`settings.remember_answer`) **sauf** si `needs_input.field` commence par
`confirm` (I-04). Pour `confirm_submit`, seule la valeur exacte `"Soumettre"` envoie la candidature ; toute autre
valeur (`"Annuler"`) donne `result.state = "annulée"`. Une tâche chargée depuis `tasks.jsonl` après redémarrage
ne peut pas être annulée (pas de contexte) ⇒ 409.

### 5.6 Analyses, offres, annotations, modèle (`jobs.py`)

| Méthode & chemin | Corps / paramètres | Succès | Erreurs |
| :--- | :--- | :--- | :--- |
| `POST /api/runs` | `RunBody` | `202` tâche `run` (`browser = scrape`) | `409 Une analyse est déjà en cours.` ; `422 Renseigne le mot-clé du poste visé (Paramètres ou lancement d'analyse).` ; `422` réglage invalide ; si `scrape` : `409` identifiants, `409` blocage ; `[CIBLE]` `428` |
| `POST /api/runs/rescore` | — | `202` tâche `rescore` | `409 Un recalcul est déjà en cours.` ; `409 Aucune analyse disponible : lance d'abord une analyse.` |
| `GET /api/jobs` | `min_score: float = 0`, `label: pertinente\|non_pertinente\|non_annotee`, `status: aucune\|<statut>`, `q: str` (recherche normalisée dans titre, entreprise, lieu) | `200` voir ci-dessous ; sans analyse `{"report": null, "jobs": []}` | — |
| `GET /api/jobs/{job_id}` | — | `200` ligne d'offre + `text` (8 000 caractères max) + `owned_skills` | `404 Aucune analyse disponible : lance d'abord une analyse.` ; `404 Offre introuvable.` |
| `PUT /api/jobs/{job_id}/label` | `LabelBody {value: int \| null}` (champ requis) | `200 {"id", "label", "n_labels", "min_labels"}` | `422 value doit valoir 1, 0 ou null.` |
| `POST /api/model/train` | — | `202` tâche `train` | `409 Un entraînement est déjà en cours.` ; `409` aucune analyse |
| `POST /api/jobs/{job_id}/prepare` | — | `202` tâche `prepare` (`meta.key = job_id`) | `409 La préparation de cette offre est déjà en cours.` ; `409` aucune analyse ; `404 Offre introuvable.` |

`RunBody` (tous facultatifs, fusionnés avec `GET /api/settings`) :

| Champ | Type | Défaut | Règle |
| :--- | :--- | :--- | :--- |
| `scrape` | `bool` | `true` | `false` = analyse de l'historique sans navigateur. |
| `keyword` | `str \| null` | réglage | Obligatoire au final (422). |
| `location` | `str \| null` | réglage | — |
| `remote_only` | `bool \| null` | réglage | Ajoute `f_WT=2` à la recherche d'offres LinkedIn. |
| `caution` | `str \| null` | réglage | `prudent`/`normal`/`agressif`. |
| `max_items` | `int \| null` | `Caution.max_items` | `[ACTUEL]` 1–200 ; `[CIBLE]` décidé : 1 à `Caution.max_items`, valeur supérieure refusée (422) (E-3). |
| `interpreter` | `str \| null` | réglage | `claude`/`local`/`none` (`[CIBLE]` `none` ⇒ 422). |
| `sources` `[CIBLE]` | `list[str]` | `["linkedin"]` | Codes de `GET /api/sources` ; `indeed` refusé (import manuel). |

Exemple :
```http
POST /api/runs HTTP/1.1
Host: 127.0.0.1:8765
X-Local-Token: 6c3n…
Content-Type: application/json

{"keyword": "Data Engineer", "caution": "prudent", "scrape": true}
```
```json
{"id": "a1b2c3d4e5f6", "kind": "run", "title": "Collecte et analyse — Data Engineer", "status": "en_attente",
 "logs": [], "result": null, "error": "", "needs_input": null, "meta": {}, "browser": true,
 "created_at": "2026-10-08T14:30:01", "started_at": "", "finished_at": ""}
```
Résultat final : `{"report": "data-engineer-20261008-1432", "jobs": 30, "counts": {"job": 30, "post": 30, "person": 30}}`.

`GET /api/jobs` :
```json
{
  "report": "data-engineer-20261008-1432",
  "generated_at": "2026-10-08T14:32:58.120331",
  "keyword": "Data Engineer",
  "model": {"trained": false, "n_labels": 3, "train_accuracy": null, "importance": [], "loss_history": [], "trained_at": null},
  "min_labels": 6,
  "jobs": [{
    "id": "job:4012345678", "title": "Senior Data Engineer", "company": "Acme", "url": "https://www.linkedin.com/jobs/view/4012345678/",
    "location": "France (télétravail)", "posted_at": "2026-10-05", "score": 82.4,
    "criteria": {"couverture_competences": 0.75, "similarite_texte": 0.31, "similarite_titre": 0.5, "seniorite": 1.0,
                 "experience": 1.0, "teletravail": 1.0, "localisation_compatible": 1.0, "freelance": 0.0},
    "job_skills": ["airflow", "python", "spark", "aws"], "missing_skills": ["aws"], "matched_skills": ["airflow", "python", "spark"],
    "years_required": 5, "seniority_required": "senior", "remote": true, "freelance": false,
    "label": null, "application_status": null
  }]
}
```
`summary` (phrases clés) est retiré de la liste et présent dans `GET /api/jobs/{id}`. Les offres d'anciennes
collectes peuvent porter `posted_label` et `applicants` (identité réparée par `services.repair_identity`).

Incohérence notée : « aucune analyse » renvoie `404` sur `GET /api/jobs/{id}` mais `409` sur les routes d'action
(`rescore`, `train`, `prepare`). Conservé : lecture d'une ressource absente (404) vs précondition d'action non
remplie (409).

### 5.7 Candidatures, contacts, messages (`applications.py`)

| Méthode & chemin | Corps | Succès | Erreurs |
| :--- | :--- | :--- | :--- |
| `GET /api/applications` | — | `200` liste (plus récente préparation d'abord) | — |
| `GET /api/applications/{job_id}` | — | `200` entrée + `cv`, `message`, `advice`, `offer` + `statuses` | `404 Aucune candidature préparée pour cette offre.` ; `500 Dossier de candidature hors du répertoire de données.` |
| `PATCH /api/applications/{job_id}` | `ApplicationPatch` | `200` comme `GET` | `404` ; `422 Statut inconnu. Valeurs : préparée, contactée, postulée, entretien, refusée, abandonnée, externe, échec.` ; `422` liste (contact invalide) |
| `GET /api/applications/{job_id}/cv.html` | — | `200 text/html` sandboxé | `404` candidature ; `404 CV introuvable.` |
| `POST /api/applications/{job_id}/apply` | — | `202` tâche `apply` | `404` ; `409 Une candidature est déjà en cours pour cette offre.` ; `409 Cette candidature est déjà au statut « postulée ».` ; `409` identifiants / blocage ; `429 Plafond quotidien atteint (10 candidatures par jour) : réessaie demain ou relève-le dans les paramètres.` ; `[CIBLE]` `428` |
| `POST /api/applications/{job_id}/contacts/lookup` | — | `202` tâche `contacts` | `404` ; `409 Une recherche de contacts est déjà en cours pour cette offre.` ; `409` identifiants / blocage ; `[CIBLE]` `428` |
| `GET /api/applications/{job_id}/message-draft` | — | `200 {"subject": "Candidature — <titre ≤ 100>", "body": "<message.txt>\n\n<signature>"}` | `404` |
| `POST /api/applications/{job_id}/messages` | `MessageBody` | `202` tâche `message` (`browser` si canal `linkedin`) | voir ordre ci-dessous |
| `POST /api/smtp/test` | `TestMail {to: str \| null}` | `200 {"channel": "email", "to": "me@example.com", "subject": "Test my-connector", "attachments": []}` | `503 SMTP non configuré : renseigne SMTP_HOST et SMTP_FROM dans le fichier .env.` ; `422 Le test n'est autorisé que vers ton adresse de contact ou l'expéditeur SMTP.` ; `502 <OutreachError>` |

`ApplicationPatch` (tous facultatifs) : `cv`, `message`, `advice` (`str`, réécrivent `cv.md`/`message.txt`/
`conseils.md` ; changer `cv` régénère `cv.html` et supprime `cv.pdf`), `status` (statut libre, **seule** voie de
régression, I-03 ; ajoute un historique `kind="statut"`), `contacts: list[Contact]` (remplace la liste).

`Contact` (`[CIBLE]` vague 1 : `full_name`, cf. § 1.1 C-4/C-5) : `name: str = ""`, `profile_url: str = ""` (vide ou `^https://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+`,
sinon « L'URL doit être un profil LinkedIn (https://www.linkedin.com/in/...). »), `email: str | null`
(`^[^@\s]+@[^@\s]+\.[^@\s]+$`, sinon « Adresse email invalide. »), `headline: str = ""`, `source: str = "manuel"`.

`MessageBody` :

| Champ | Type | Défaut | Règle |
| :--- | :--- | :--- | :--- |
| `channel` | `Literal["email", "linkedin"]` | requis | — |
| `to` | `str` | requis | Email (`outreach._EMAIL`, sans `,;<>`) ou URL de profil LinkedIn. |
| `subject` | `str` | `""` | Email : non vide, sans saut de ligne. Ignoré pour LinkedIn. |
| `body` | `str` | requis | Non vide (vérifié à l'envoi). LinkedIn hors relation de 1er degré : tronqué à 300 caractères (note de connexion). |
| `attach_cv` | `bool` | `true` | Email : joint `cv.pdf` (généré si absent). |
| `confirm` | `bool` | `false` | Doit valoir `true` (I-04). |

Ordre des contrôles de `POST …/messages` (le premier qui échoue répond) :
1. `404 Aucune candidature préparée pour cette offre.`
2. `422 Confirmation explicite requise : relis le message puis confirme l'envoi.`
3. Canal `email` : `503 SMTP non configuré : …` → `422 Adresse email invalide : « pas-un-mail ».` →
   `422 Objet requis, sur une seule ligne.` → `429 Plafond quotidien atteint (15 messages par jour) : …`.
   Canal `linkedin` : `409` identifiants → `429` plafond → `409` blocage.
4. `409 Un envoi est déjà en cours pour cette candidature.`
5. `[CIBLE]` dans la transaction : `409 Ce destinataire a demandé à ne plus être contacté.` ;
   `428` (canal `linkedin`). Le pied de conformité (origine de l'adresse + désinscription) est **ajouté par le serveur** après le corps saisi (cf. `05-queues-and-integrations.md`) : il n'est jamais exigé dans `body`, donc pas de `422` à ce titre.

Résultat de tâche : email `{"channel": "email", "to": "jane@acme.com", "subject": "Candidature — Senior Python Developer", "attachments": ["cv.pdf"]}` ;
LinkedIn `{"channel": "linkedin", "mode": "message" | "connexion", "to": "https://www.linkedin.com/in/jane-doe/", "truncated": false}`.

Entrée de candidature (`GET /api/applications`) :
```json
{
  "id": "job:4012345678", "title": "Senior Python Developer", "company": "Acme",
  "url": "https://www.linkedin.com/jobs/view/4012345678/", "score": 82.4,
  "folder": "/home/me/linkedin_market_data/applications/20261008-acme-senior-python-developer",
  "generator": "claude", "status": "contactée", "apply_state": "",
  "prepared_at": "2026-10-08T14:35:12", "updated_at": "2026-10-08T15:02:40",
  "contacts": [{"name": "Jane Doe", "profile_url": "https://www.linkedin.com/in/jane-doe/", "email": "jane@acme.com",
                "headline": "Talent Acquisition Lead", "source": "linkedin"}],
  "history": [{"at": "2026-10-08T15:02:40", "kind": "message", "channel": "email", "to": "jane@acme.com", "extrait": "Bonjour Jane, …"}]
}
```
Types d'historique écrits par le code : `statut`, `candidature`, `contacts`, `message` (l'ERD cible les renomme
`status`, `apply`, `contacts`, `message` — conversion par l'import JSON, cf. § 1.1 C-6).

### 5.8 Sources et consentements `[CIBLE]`

| Méthode & chemin | Corps | Succès | Erreurs |
| :--- | :--- | :--- | :--- |
| `GET /api/sources` | — | `200` liste | — |
| `POST /api/sources/{code}/consent` | `ConsentBody {risk_version: str, accepted: Literal[True]}` | `201` consentement | `404 Source inconnue.` ; `422 Cette source ne demande pas de consentement.` (mode `api_officielle`/`import_manuel`) ; `409 Le texte des risques a changé : relis la nouvelle version.` |
| `DELETE /api/sources/{code}/consent` | — | `204` (révocation : `revoked_at = now()`) | `404 Aucun consentement actif pour cette source.` |
| `POST /api/items/import` | `ManualItem {source_code: Literal["indeed"], url: HttpUrl, title: str, company: str = "", location: str = "", text: str}` | `201` élément créé | `422` ; `409 Cette offre est déjà dans l'historique.` |
| `POST /api/suppressions` | `SuppressionBody {recipient: str, reason: Literal["désinscription", "refus_oral", "plainte"]}` | `201 {"ok": true}` (aucune donnée en clair renvoyée) | `422` |

`GET /api/sources` :
```json
[
  {"code": "linkedin", "label": "LinkedIn", "access_mode": "navigateur", "enabled": true, "risk_version": "2026-10",
   "consent": {"risk_version": "2026-10", "accepted_at": "2026-10-08T12:30:00Z"}, "consent_required": false},
  {"code": "france_travail", "label": "France Travail", "access_mode": "api_officielle", "enabled": true,
   "risk_version": null, "consent": null, "consent_required": false, "configured": true},
  {"code": "malt", "label": "Malt", "access_mode": "navigateur", "enabled": true, "risk_version": "2026-10",
   "consent": null, "consent_required": true},
  {"code": "indeed", "label": "Indeed", "access_mode": "import_manuel", "enabled": true, "risk_version": null,
   "consent": null, "consent_required": false}
]
```
Le texte des risques lui-même est versionné dans le frontend (popup) ; le backend ne stocke que `risk_version`.
`consent_required` = mode `navigateur` et aucun consentement actif à la version courante.

Réponse `428` (sur `POST /api/runs` avec `scrape=true`, `/apply`, `/contacts/lookup`, `/messages` canal
`linkedin`, `/profile/import-linkedin`, synchronisation de compétence) :
```json
{"detail": "Accepte d'abord les risques liés à LinkedIn (version 2026-10) avant toute action dans le navigateur.", "source": "linkedin", "risk_version": "2026-10"}
```
Pour `POST /api/profile/skills`, l'absence de consentement ne lève pas d'erreur : `linkedin_sync.status = "unavailable"`.

### 5.9 Route SPA

`GET /{path}` hors `/api/` : fichier de `web/frontend/dist` s'il existe (résolution vérifiée dans le dossier),
sinon `index.html` avec `<!--LM_TOKEN-->` remplacé par `<meta name="lm-token" content="…">` et
`Cache-Control: no-store` ; `503` HTML si le frontend n'est pas compilé.

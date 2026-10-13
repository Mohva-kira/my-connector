# Logique Métier & Architecture des Services

> Complète `schemas-architecture/01-system-overview.md` § 4 (découpage en modules). Ici : ce que fait chaque
> couche **dans le code actuel**, la cible après migration PostgreSQL, les transactions et verrous, la
> journalisation, et la liste des **écarts code ↔ documentation** (§ 4) référencés ailleurs par `E-n`.

## 1. Découpage en Couches (Clean / Hexagonal)

Règle de dépendance (lot 1) : `Entrées → Orchestration → Domaine / Intégrations → Persistance`. Le domaine
n'importe ni FastAPI ni Selenium ; les routeurs n'importent jamais Selenium (`browser.create_driver` est importé
**dans** les fonctions de `web/services.py`, au moment de l'exécution de la tâche).

### 1.1 Controller / Handler Layer — `web/app.py`, `web/routers/*.py`, `__main__.py`
- Valide le corps (modèles Pydantic `RunBody`, `ProfileUpdate`, `SkillBody`, `Answer`, `LabelBody`,
  `ApplicationPatch`, `Contact`, `MessageBody`, `TestMail`), applique les **préconditions** (`require_browser_ready`,
  `has_active`), soumet la tâche, traduit les exceptions métier en HTTP (§ 1.5).
- `[ACTUEL]` quelques règles vivent encore dans les routeurs : filtres de `GET /api/jobs`, bornes de `LabelBody`,
  contrôles d'objet et de destinataire de `send_message`, liste blanche du test SMTP. `[CIBLE]` les déplacer dans
  les services (`JobQueryService.list()`, `OutreachService.check_request()`), le routeur ne gardant que le mapping
  HTTP. Le CLI (`__main__.py`) doit appeler les mêmes services, pas dupliquer les règles.

### 1.2 Service / Use-Case Layer — `web/services.py`, `pipeline.py`, `web/guards.py`
Fonctions de cas d'usage, toutes de signature `(ctx: TaskContext, …) -> dict` quand elles tournent en tâche :

| Fonction | Cas d'usage | Préconditions revérifiées dans la tâche | Écritures |
| :--- | :--- | :--- | :--- |
| `run_analysis(ctx, request, scrape, interpreter_name)` | `LancerAnalyse` / `ReanalyserHistorique` | `guard_not_blocked()` (dans `pipeline.scrape`) | `collected_items`, `candidate_profiles.text`, `reports` (+ HTML), `activity run` |
| `rescore(ctx)` | `RecalculerScores` | analyse existante | nouveau rapport (reprend l'interprétation précédente) |
| `train_model(ctx)` | `EntrainerModele` | ≥ 6 annotations, deux classes | `matching_models` (version + 1, active), nouveau rapport |
| `prepare_application(ctx, job_id)` | `PreparerCandidature` | offre dans le dernier rapport **et** dans l'historique | dossier, `applications` |
| `apply_job(ctx, job_id)` | `PostulerEasyApply` | plafond `apply`, blocage | `cv.pdf`, `applications` + `application_events`, `labels`, `activity apply` |
| `lookup_contacts(ctx, job_id)` | `ChercherRecruteurs` | blocage | `company_contacts` fusionnés, historique |
| `send_message(ctx, job_id, …)` | `EnvoyerMessage` | plafond messages ; blocage (canal LinkedIn) | `activity message_*`, historique, statut `contactée` |
| `import_linkedin_profile(ctx)` | `ImporterProfilLinkedIn` | blocage | `candidate_profiles.text` |
| `sync_skill_to_linkedin(ctx, skill)` | `ModifierProfil` (volet LinkedIn) | blocage | `activity profile_skill` si `ajoutée` |
| `status_snapshot()`, `monitor_snapshot()`, `effective_profile()`, `jobs_overview()` | lectures | — | — |

Règle : **toute précondition vérifiée par le routeur est revérifiée au début de la tâche** (la tâche peut attendre
le verrou navigateur plusieurs minutes, l'état a pu changer). C'est déjà le cas pour les plafonds et le blocage ;
`[ACTUEL]` (vague 3) idem pour le consentement (I-16, `consent.require`) ; `[ACTUEL]` (vague 4) pour l'opposition (I-13, `suppression_list`).

Cible : un module par cas d'usage sous `linkedin_market/services/` (`analysis.py`, `applications.py`,
`outreach.py`, `sources.py`, `retention.py`), chacun recevant une `Session` SQLAlchemy et un `TaskContext`.

### 1.3 Domaine — `nlp.py`, `trends.py`, `matching.py`, `profile.py`, `application.py`, `report.py`, `interpreter.py`, `settings.py`
Calculs purs (sans réseau) :
- **Matching** : `build_features()` (8 critères ∈ [0, 1]), `MatchingModel.fit()` (descente de gradient,
  `lr=0.3`, `epochs=3000`, `l2=0.05`, classes repondérées, normalisation moyenne/écart-type), `score_jobs()`
  (score = probabilité × 100, arrondi à 0,1), `skill_gap()`. Poids a priori `PRIOR_WEIGHTS =
  [2.5, 2.0, 1.2, 0.6, 0.5, 0.6, 0.8, 0.4]`, biais `-4.3`.
- **Tendances** : fenêtre `TREND_WINDOW_DAYS = 30` appliquée à la lecture (`storage.load_items` filtre
  `posted_at >= aujourd'hui − 30 j`, la dernière version d'un `id` gagne).
- **Statuts de candidature** : `STATUSES`, `_PROGRESS`, `advance_status()` (I-03).
- **Interprète local** : `LocalInterpreter.interpret(summary)` — texte à base de règles, sections « Lecture du
  marché », « Tendances », « Écart de profil », « Ce qui pèse dans ton matching », « Offres à cibler ».
- Écart de couche : `application.generate()` (domaine) appelle `interpreter.ClaudeInterpreter` (intégration).
  `[CIBLE]` : injecter un `Writer` (`ClaudeWriter` | `LocalWriter`) depuis le service.

### 1.4 Integration / Gateway Layer
| Module | Système externe | Exceptions levées |
| :--- | :--- | :--- |
| `browser.py` | Chrome / Firefox via Selenium Manager | `RuntimeError("Aucun navigateur utilisable (LINKEDIN_BROWSER=…) : …")` |
| `session.py` | LinkedIn (connexion, cookies 72 h) | `MissingCredentialsError`, `TimeoutException` (vérification manuelle non faite en 180 s) |
| `scraper.py` | LinkedIn (recherche offres, posts, personnes, profil propre) | aucune : `_blocked()` journalise et la collecte s'arrête |
| `apply_bot.py` | LinkedIn Easy Apply | `ApplyError`, `BlockedError(ApplyError)` |
| `contacts.py`, `profile_sync.py`, `outreach.send_linkedin_message` | LinkedIn | `PermissionError` (blocage), `OutreachError`, `ProfileSyncError` |
| `outreach.send_email` | SMTP de l'utilisateur | `OutreachError` |
| `interpreter.ClaudeInterpreter` | API Anthropic | toute exception du SDK, `RuntimeError("Claude a refusé la requête.")` |
| `cv_pdf.py` | — (génération locale de `cv.pdf`) | — |
| `sources/france_travail.py` `[ACTUEL]` | API Offres d'emploi v2 | `SourceFetchError` (l'analyse continue sans cette source ; distincte de `consent.SourceUnavailable`) |
| `sources/malt.py`, `sources/wttj.py`, `sources/base.py` `[ACTUEL]` | Malt, WTTJ (navigateur, lecture seule) | blocage : `BrowserJobSource.blocked` ⇒ source arrêtée, événement `blocked` ; `pipeline.collect` orchestre les sources, `pipeline.guard_source_rest` nomme la source en repos |

`[CIBLE]` unifier la détection de blocage : une seule exception `BlockedError` (aujourd'hui `PermissionError`,
`BlockedError` et un simple booléen selon le module) et un seul helper `browser.assert_not_blocked(driver, where)`
qui enregistre l'événement `blocked` puis lève.

### 1.5 Exceptions métier → HTTP
| Exception | Origine | HTTP |
| :--- | :--- | :--- |
| `settings.SettingsError(ValueError)` | validation des réglages / profil | 422 |
| `ValueError` de `build_request` (mot-clé vide) | `POST /api/runs` | 422 |
| `activity.QuotaExceeded` | `check_quota` | 429 |
| `activity.BlockedByLinkedIn(RuntimeError)` | `guard_not_blocked` | 409 |
| identifiants absents (`require_browser_ready`) | `.env` | 409 |
| `services.NoAnalysisError(LookupError)` | `require_results` | 409 (action) / 404 (lecture d'offre) |
| `application.ApplicationNotFound(KeyError)` | `_entry` | 404 |
| `outreach.OutreachError` | validation du destinataire / test SMTP | 422 / 502 |
| `ValueError` de `folder_of` | dossier hors `applications/` | 500 |
| `consent.SourceConsentRequired` `[ACTUEL]` | `consent.require` (gestionnaire `web/app.py`) | 428 |
| `outreach.RecipientSuppressed(OutreachError)` `[ACTUEL]` (vague 4) | `services.reserve_message` / `_confirm_reservation` (`suppression_list`, contact `opposed_at`) | 409 (API) ; réservation `échec` (tâche) |
| `suppressions.OpposedContact(ValueError)` `[ACTUEL]` (vague 4, alias `application.ContactOpposed`) | `replace_contacts` (coordonnée opposée) | 422 |
| Toute exception **dans une tâche** | — | tâche `échec`, `error = str(exc)` (3 lignes de traceback dans `logs`) |

### 1.6 Repository / Data Access Layer
- `[ACTUEL]` (vague 1) façades inchangées pour `pipeline.py` : `storage.py` (historique, annotations, rapports),
  `application.py` (suivi + fichiers), `settings.py`, `activity.py`, `session.py`, `matching.py` (persistance du
  modèle), `TaskManager` ; elles ouvrent `user_transaction()` et délèguent à `linkedin_market/db/` (`models.py`,
  `session.py`, `users.py`, `bootstrap.py`, `import_json.py`, `crypto.py`) et `linkedin_market/db/repositories/`
  (modules de fonctions par agrégat : `items`, `applications`, `activity`, `tasks`, `settings`, `profile`, `reports`,
  `models`, `platform_sessions`, **`outreach`** `[ACTUEL]` (vague 2, G-6/E-11 : réservation des messages)). Chaque
  fonction prend `session, user_id` en premiers arguments (I-01) ; aucune requête SQL hors de ces modules. Les
  fichiers produits (rapports HTML, dossiers de candidature) restent sur disque, seuls leurs chemins relatifs sont
  en base.
- `[ACTUEL]` (vague 3) dépôt `sources` (`db/repositories/sources.py` : sources, consentement actif, acceptation = révocation + insertion, révocation) et domaine `consent.py` ; `activity.last_block_event(source)` filtre le repos par source (`data.source`, absent = `linkedin`).

## 2. Gestion des Transactions & Verrous

### 2.1 Existant `[ACTUEL]`
| Mécanisme | Portée | Limite |
| :--- | :--- | :--- |
| `TaskManager._lock` (`RLock`) | état des tâches en mémoire | — |
| `TaskManager._browser_lock` (`Lock`) | un seul Selenium à la fois ; attente par tranches de 0,5 s avec `ctx.check()` (annulable) | — |
| Transaction PostgreSQL par appel de façade (`user_transaction()`) | chaque lecture / écriture | une transaction par cas d'usage complet reste `[CIBLE]` pour le reste des services (§ 2.2) ; livrée pour la réservation de message et le plafond `apply` (vague 2) |
| `SELECT … FOR UPDATE` sur la candidature (`application.py`) et sur `user_settings` (`db/repositories/settings.py`, `models.py`) | mise à jour d'une candidature ; nouvelle version du modèle ; réservation d'un message (`services.reserve_message`) ; plafond `apply` revérifié au début de la tâche (`activity.check_quota_locked`) | — |
| `has_active(kind, key)` | une tâche par type et par offre | vérifié puis `submit()` sans verrou commun : deux requêtes simultanées peuvent passer (fenêtre de quelques ms, acceptable en local) |

E-11 résolu (vague 2) : la réservation transactionnelle ci-dessous empêche tout dépassement du plafond de
messages, y compris en cas d'envois simultanés.

### 2.2 Réservation d'un message `[ACTUEL]` (vague 2, G-6/E-11) et cible plus large `[CIBLE]`
- **Livré** : `services.reserve_message()` ouvre **une** transaction (`user_transaction()`), verrouille la
  candidature (`FOR UPDATE`) puis `user_settings` (`FOR UPDATE`), compte les messages du jour
  (`db/repositories/outreach.count_day`) et insère `outreach_messages(status='confirmé')` si le plafond n'est pas
  atteint (`backend/03` § 5.1). Le plafond `apply` suit le même principe (`activity.check_quota_locked`, `FOR
  UPDATE` sur `user_settings` seul, pas de ligne à réserver). Effets externes (SMTP, LinkedIn) **hors**
  transaction ; une seconde transaction courte écrit `envoyé`/`échec`/l'erreur « à vérifier » après coup (§ 5.1 de
  `backend/03`).
- **Cible restante** `[CIBLE]`, pour les autres cas d'usage (préparation, recherche de contacts, import de
  profil…) :
  - **Une transaction par cas d'usage**, ouverte par le service (`with session.begin():`), jamais par le routeur ni
    le repository. Une tâche longue ouvre des transactions **courtes** autour de chaque écriture ; aucune
    transaction ne reste ouverte pendant une navigation Selenium, un appel Claude ou un envoi SMTP.
  - **Contraintes plutôt que contrôles applicatifs** quand c'est possible : `UNIQUE` partiels (`source_consents`,
    `matching_models.is_active`), `ON CONFLICT` pour l'historique, trigger d'immuabilité des rapports.
- **Pas de verrou optimiste** (`version`) : un seul utilisateur, conflits rares ; les écritures concurrentes
  proviennent de tâches du même processus, couvertes par les verrous ci-dessus.

Implémentation livrée (`linkedin_market/web/services.py::reserve_message`, cf. `03-database-and-migrations.md`
§ 5.1 pour le détail des verrous) :

```python
def reserve_message(job_id: str, *, channel: str, to: str, subject: str, body: str, attach_cv: bool) -> UUID:
    start, end = activity.day_bounds()
    with user_transaction() as (s, uid):
        found = applications_repo.get(s, uid, job_id, lock=True)          # FOR UPDATE candidature
        if found is None:
            raise application.ApplicationNotFound(job_id)
        limit = settings_repo.get_settings(s, uid, lock=True)["message_daily_max"]  # FOR UPDATE user_settings
        if outreach_repo.count_day(s, uid, start, end) >= limit:
            raise activity.QuotaExceeded("messages", limit)
        return outreach_repo.insert_confirmed(s, uid, found[0].id, {...})  # status="confirmé"
```

Le contrôle `suppression_list` (`RecipientSuppressed`, I-13) s'ajoute à cette même transaction en vague 4 ; il n'y
est pas encore.

## 3. Logging & Traçabilité

### 3.1 Existant `[ACTUEL]`
- Module `logging` standard ; logger racine du paquet `linkedin_market` (niveau `INFO`). Format console de
  `web/__main__.py` : `%(asctime)s - %(levelname)s - %(message)s` ; Uvicorn en `warning`.
- `_TaskLogHandler` recopie dans `Task.logs` les messages émis **par le thread de la tâche**
  (`record.thread`), préfixés `HH:MM:SS`, 400 lignes maximum ; `ctx.log()` écrit directement. Ces logs sont
  visibles dans l'interface et **persistés** dans `tasks.logs` (table `tasks`).
- Le journal d'activité (`activity_events`) est un journal **métier** (plafonds, repos, monitoring), distinct des
  logs techniques.

### 3.2 Règles
- **Langue** : messages en français, tutoiement, actionnables (« renseigne SMTP_HOST… »), comme le code actuel.
- **Corrélation** : l'`id` de tâche est la clé de corrélation (pas de `traceId`). `[CIBLE]` le `_TaskLogHandler`
  préfixe aussi la ligne console par `[<task_id>]` et `activity_events.data.task_id` est renseigné.
- **Format** : texte lisible pour la console et l'interface (outil local, lu par un humain). `[CIBLE]` option
  `LM_LOG_JSON=1` pour un format JSON une ligne (`{"at", "level", "task_id", "logger", "message"}`) à destination
  des personnes qui centralisent leurs logs ; non activée par défaut.
- **Interdits** (I-15, RGPD art. 5(1)(c) et 32) : mots de passe, `LINKEDIN_PASSWORD`, `SMTP_PASSWORD`,
  `ANTHROPIC_API_KEY`, `FRANCE_TRAVAIL_CLIENT_SECRET`, jeton d'accès France Travail, cookies, jeton local,
  `LM_SECRET_KEY`, corps complet d'un message, texte d'un profil tiers.
- **Données de tiers dans les logs** `[ACTUEL]` (vague 2, E-12) : module `logmask.py` (`mask_email`, `mask_name`,
  `mask_profile_url`, `mask_recipient`, `mask_text`). `services.send_message` / `_send_linkedin` journalisent le
  destinataire masqué (`j***@acme.com` ou `linkedin.com/in/j***`) ; `contacts.find_hiring_team` journalise les
  noms en initiales et les emails masqués. `TaskManager._append_log` applique `mask_text` en filet de sécurité à
  **toute** ligne de log (détection par motif), et `task.error` est aussi masqué (`TaskManager._run`). L'information
  complète reste dans l'historique de candidature, `outreach_messages` et `company_contacts`, soumis à la
  rétention (I-14). L'événement d'activité `message_*` (`activity_events`, lu par `/api/monitor`) porte lui aussi
  le champ `to` **masqué** (`services.send_message` passe `logmask.mask_recipient(to)` à `activity.record()`,
  qui n'applique aucun masquage lui-même) ; seul l'historique de candidature garde l'adresse complète.
- **Rétention des logs** : table `tasks` `[ACTUEL]` ; purge des tâches finales à 12 mois par la tâche `purge` `[ACTUEL]`
  (vague 6, `03` § 6).

## 4. Écarts Code ↔ Documentation (constatés le 2026-10-08)

Résolus et retirés à l'acceptation de la vague 1 (2026-10-09) : E-1, E-2 (slug `mot-cle-AAAA-MM-JJ[-n]`,
`UNIQUE (user_id, slug)`), E-8 (écritures en base), E-9 (`timestamptz` UTC, C-1), E-15 (noms de l'ERD), E-16
(`person` sans texte, `CHECK`), E-19 (404 sur HTML absent). Résolus et retirés à l'acceptation de la vague 2
(2026-10-09) : E-3, E-4, E-5, E-7, E-10, E-11, E-12, E-20, E-22 (détail de chaque correction dans `01` § 1.2,
G-1 à G-12, et `05` §§ 1, 2). Résolu et retiré à l'acceptation de la vague 4 (2026-10-12) : E-17 (`01` § 5.7.1, `05` § 2.4).
Résolus et retirés à l'acceptation des vagues 5-6 (2026-10-13) : E-6 (`nlp.redact_third_parties`, `05` § 2.3) et E-18
(réglages `home_location` / `timezone`, `01` § 5.2).
Écart « 3 ans » résolu (vague 6) : l'intervalle calendaire SQL `interval '3 years'` fait foi (`03` § 6), pas 1095 jours. Les numéros ne sont pas réattribués.

| # | Écart | Référence | Correction proposée |
| :--- | :--- | :--- | :--- |
| E-21 | Lot 1 `02` § 6 : l'événement `blocked` porterait `until` ; le code n'enregistre que `where` (+ `job`) et calcule `until` à la lecture. | lot 1 `02` § 6 | Corriger la table du lot 1 (calcul à la lecture conservé). Toujours ouvert : `until` reste calculé à la lecture (`activity.py`), pas stocké. |

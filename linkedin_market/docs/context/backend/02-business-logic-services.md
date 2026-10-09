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
`[CIBLE]` idem pour le consentement (I-16) et l'opposition (I-13).

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
| `sources/france_travail.py` `[CIBLE]` | API Offres d'emploi v2 | `SourceUnavailable` (l'analyse continue sans cette source) |
| `sources/malt.py`, `sources/wttj.py` `[CIBLE]` | Malt, WTTJ (navigateur) | `BlockedError` partagée |

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
| `SourceConsentRequired` `[CIBLE]` | `SourceConsentService.require` | 428 |
| `RecipientSuppressed` `[CIBLE]` | contrôle `suppression_list` | 409 |
| Toute exception **dans une tâche** | — | tâche `échec`, `error = str(exc)` (3 lignes de traceback dans `logs`) |

### 1.6 Repository / Data Access Layer
- `[ACTUEL]` (vague 1) façades inchangées pour `pipeline.py` : `storage.py` (historique, annotations, rapports),
  `application.py` (suivi + fichiers), `settings.py`, `activity.py`, `session.py`, `matching.py` (persistance du
  modèle), `TaskManager` ; elles ouvrent `user_transaction()` et délèguent à `linkedin_market/db/` (`models.py`,
  `session.py`, `users.py`, `bootstrap.py`, `import_json.py`, `crypto.py`) et `linkedin_market/db/repositories/`
  (modules de fonctions par agrégat : `items`, `applications`, `activity`, `tasks`, `settings`, `profile`, `reports`,
  `models`, `platform_sessions`). Chaque fonction prend `session, user_id` en premiers arguments (I-01) ; aucune
  requête SQL hors de ces modules. Les fichiers produits (rapports HTML, dossiers de candidature) restent sur
  disque, seuls leurs chemins relatifs sont en base.
- `[CIBLE]` dépôts `outreach` (vague 4) et `consents` (vague 3).

## 2. Gestion des Transactions & Verrous

### 2.1 Existant `[ACTUEL]`
| Mécanisme | Portée | Limite |
| :--- | :--- | :--- |
| `TaskManager._lock` (`RLock`) | état des tâches en mémoire | — |
| `TaskManager._browser_lock` (`Lock`) | un seul Selenium à la fois ; attente par tranches de 0,5 s avec `ctx.check()` (annulable) | — |
| Transaction PostgreSQL par appel de façade (`user_transaction()`) | chaque lecture / écriture | une transaction par cas d'usage complet reste `[CIBLE]` (§ 2.2) |
| `SELECT … FOR UPDATE` sur la candidature (`application.py`) et sur `user_settings` (`db/repositories/settings.py`, `models.py`) | mise à jour d'une candidature ; nouvelle version du modèle | le couple « compter puis enregistrer » des plafonds n'est pas encore sérialisé (E-11) |
| `has_active(kind, key)` | une tâche par type et par offre | vérifié puis `submit()` sans verrou commun : deux requêtes simultanées peuvent passer (fenêtre de quelques ms, acceptable en local) |

Conséquence restante : dépassement possible du plafond de messages email (E-11, vague 2).

### 2.2 Cible `[CIBLE]`
- **Une transaction par cas d'usage**, ouverte par le service (`with session.begin():`), jamais par le routeur ni
  le repository. Une tâche longue ouvre des transactions **courtes** autour de chaque écriture ; aucune
  transaction ne reste ouverte pendant une navigation Selenium, un appel Claude ou un envoi SMTP.
- **Verrous pessimistes** (`SELECT … FOR UPDATE`) :
  - `applications` (ligne de la candidature) pendant la réservation d'un message et pendant un changement de
    statut + insertion `application_events` ;
  - `user_settings` (ligne de l'utilisateur) pendant tout contrôle de plafond quotidien suivi d'une insertion
    (`apply`, `message_*`) : sérialise les compteurs entre candidatures différentes, ce que le verrou sur
    `applications` seul ne fait pas.
- **Pas de verrou optimiste** (`version`) : un seul utilisateur, conflits rares ; les écritures concurrentes
  proviennent de tâches du même processus, couvertes par les verrous ci-dessus.
- **Contraintes plutôt que contrôles applicatifs** quand c'est possible : `UNIQUE` partiels (`source_consents`,
  `matching_models.is_active`), `ON CONFLICT` pour l'historique, trigger d'immuabilité des rapports.
- **Effets externes hors transaction, état avant/après en base** : réservation `outreach_messages(status='confirmé')`
  commitée **avant** l'envoi ; après l'envoi, seconde transaction `envoyé` / `échec`. Un redémarrage entre les deux
  laisse une ligne `confirmé` sans `sent_at` : elle est affichée « à vérifier », jamais renvoyée automatiquement.

Exemple (réservation d'un message, cf. `03-database-and-migrations.md` § 5 pour le SQL) :

```python
def reserve_message(db: Session, user_id: UUID, application_id: UUID, req: MessageRequest) -> OutreachMessage:
    with db.begin():
        app = db.execute(select(Application).where(Application.id == application_id, Application.user_id == user_id)
                         .with_for_update()).scalar_one_or_none()
        if app is None:
            raise ApplicationNotFound(application_id)
        settings = db.execute(select(UserSettings).where(UserSettings.user_id == user_id).with_for_update()).scalar_one()
        if suppression_repo.contains(db, user_id, recipient_hash(req.to)):
            raise RecipientSuppressed("Ce destinataire a demandé à ne plus être contacté.")
        if outreach_repo.count_today(db, user_id) >= settings.message_daily_max:
            raise QuotaExceeded("messages", settings.message_daily_max)
        msg = OutreachMessage(user_id=user_id, application_id=app.id, channel=req.channel, recipient=req.to,
                              subject=req.subject, body=req.body, cv_attached=req.attach_cv, status="confirmé")
        db.add(msg)
    return msg
```

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
- **Données de tiers dans les logs** : aujourd'hui `services.send_message` journalise l'adresse du destinataire
  et `contacts.find_hiring_team` le nom et l'email des recruteurs ; ces lignes sont persistées dans `tasks.logs`
  (E-12). `[CIBLE]` : masquer (`j***@acme.com`, initiales) dans les logs ; l'information complète reste dans
  `outreach_messages` / `company_contacts`, soumis à la rétention (I-14).
- **Rétention des logs** : table `tasks` `[ACTUEL]` ; purge à 12 mois (lot 1 ERD § 3) `[CIBLE]` (tâche `purge`,
  vague 6).

## 4. Écarts Code ↔ Documentation (constatés le 2026-10-08)

Résolus et retirés à l'acceptation de la vague 1 (2026-10-09) : E-1, E-2 (slug `mot-cle-AAAA-MM-JJ[-n]`,
`UNIQUE (user_id, slug)`), E-8 (écritures en base), E-9 (`timestamptz` UTC, C-1), E-15 (noms de l'ERD), E-16
(`person` sans texte, `CHECK`), E-19 (404 sur HTML absent). Les numéros ne sont pas réattribués.

| # | Écart | Référence | Correction proposée |
| :--- | :--- | :--- | :--- |
| E-3 | `max_items` (réglage ou `RunBody`) accepte 1–200 et **remplace** `Caution.max_items` : 200 éléments par type en `prudent`. | I-07, R-2 | **Décidé (2026-10-08)** : `max_items` ≤ `CAUTION_LEVELS[caution].max_items` (422 au-delà côté API/réglages, `min()` défensif dans `pipeline`) ; `DEFAULT_CAUTION = "prudent"`. **Partiellement livré en vague 1** : `DEFAULT_CAUTION = "prudent"`, contrôle dans `settings.validate_settings` (message `max_items ne peut pas dépasser N en prudence « … ».`), `CHECK` en base, import qui ramène une valeur hors plafond à `NULL`. **Reste en vague 2** : `RunBody` (422) et `min()` défensif dans `pipeline`. |
| E-4 | Blocage pendant la collecte : la tâche finit `terminée` sans signaler le blocage. | I-06 | **Tranchée le 2026-10-08 (option A)** : rapport partiel conservé, tâche `terminée`, `result` = `{report, jobs, counts, blocked: true, blocked_where: "collecte"}` (`blocked: false` sans blocage) ; bandeau « Collecte interrompue par LinkedIn » ; lot 1 `02` flux 3 corrigé. À livrer en vague 2. |
| E-5 | Vérification à la connexion : `session._login_with_password` attend 180 s (`MANUAL_CHECK_TIMEOUT`) puis lève `TimeoutException` **sans** événement `blocked` ; la reprise de session par cookies ne détecte pas de checkpoint. | I-06 | Enregistrer `blocked(where="connexion")` sur délai dépassé. |
| E-6 | La génération de CV par Claude envoie le **profil source complet** et le **texte intégral de l'offre** (`application._generate_with_claude`), qui peut contenir des noms de personnes. | I-11 | **Tranchée le 2026-10-09 (option A)** : I-11 = « jamais de donnée de tiers identifiable » ; profil de l'utilisateur admis ; texte de l'offre passé par `nlp.redact_third_parties(text, known_names)` avant envoi (emails, téléphones, URL `linkedin.com/in/…`, noms des contacts de la candidature, motifs LinkedIn « X et des millions… », « relations travaillent ici » ⇒ `[personne]`). Pas de nouvelle version de la popup de risques. À livrer en vague 6. |
| E-7 | `interpreter="none"` accepté (réglages, CLI `--interpreter none`, `LINKEDIN_INTERPRETER`) et produit un rapport **sans** interprétation. | I-11 | Refuser `none` à l'écriture (422), le lire comme `local`. **Partiellement livré en vague 1** : réglages limités à `claude`/`local` (`CHECK` en base), `none` lu comme `local` (`settings.py`), import `none` ⇒ `local`. **Reste en vague 2** : `RunBody` (422), CLI `--interpreter none`, `LINKEDIN_INTERPRETER=none`, branche `none` de `interpreter.py`. |
| E-10 | Messages de validation Pydantic en anglais (`Input should be …`). | Convention française | Gestionnaire `RequestValidationError` (§ 2 de `01`). |
| E-11 | Plafond de messages email : tâches `message` email non `browser` ⇒ jusqu'à 4 en parallèle (`max_workers=4`) ; le compteur n'est incrémenté qu'après l'envoi ⇒ dépassement possible de 3. | I-07 | Réservation transactionnelle (§ 2.2). |
| E-12 | Destinataires, noms et emails de recruteurs écrits dans les logs de tâche persistés. | I-15, RGPD art. 5(1)(c) | Masquage (§ 3.2). |
| E-17 | Le brouillon de message (`message_draft`) n'ajoute ni origine de l'adresse ni mention de désinscription ; aucune `suppression_list`. | I-12, I-13 | Pied de message imposé + contrôle (cf. `05` § 2.4). |
| E-18 | Les prompts (`interpreter.SYSTEM_PROMPT`, `application.CV_SYSTEM_PROMPT`), les libellés (`FEATURE_LABELS`) et le CV local codent en dur « freelance basé à Abidjan », « GMT ». | Outil open source multi-utilisateurs | Réglages `home_location` / `timezone` injectés dans les prompts. |
| E-20 | `SMTP_SECURITY` autre que `ssl`/`starttls` ⇒ port 25 **sans chiffrement** (utilisé par les tests avec `none`). | RGPD art. 32 | **Tranchée le 2026-10-08** : `none` admis seulement si `SMTP_HOST` ∈ `127.0.0.1`, `localhost`, `::1` ; sinon `503` (message `01` § 1.2 G-5), à l'API et au début de la tâche. À livrer en vague 2. |
| E-21 | Lot 1 `02` § 6 : l'événement `blocked` porterait `until` ; le code n'enregistre que `where` (+ `job`) et calcule `until` à la lecture. | lot 1 `02` § 6 | Corriger la table du lot 1 (calcul à la lecture conservé). |
| E-22 | Annulation pendant la collecte : `pipeline.scrape` lève `pipeline.Cancelled` (`RuntimeError`), pas `TaskCancelled` ⇒ la tâche finit `échec` (« Collecte annulée. ») au lieu de `annulée` ; l'annulation n'est testée qu'entre deux types d'éléments. | Statuts de tâche (lot 1 `02`) | Lever `TaskCancelled` (ou l'intercepter dans `TaskManager._run`) ; tester `should_stop` à chaque page. |

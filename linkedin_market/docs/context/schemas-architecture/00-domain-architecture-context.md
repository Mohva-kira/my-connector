# Contexte Architectural du Domaine & Glossaire

> **Périmètre** : `linkedin_market` — outil **open source auto-hébergé** d'analyse du marché freelance
> et d'aide à la candidature. Chaque utilisateur installe l'outil sur sa machine, avec **son** compte
> LinkedIn et **sa** base PostgreSQL. L'éditeur du logiciel n'héberge aucune donnée (pas de SaaS).
>
> **Statut des sections** : `[ACTUEL]` = implémenté dans le code ; `[CIBLE]` = décidé, pas encore
> implémenté (migration PostgreSQL, sources Malt / WTTJ / France Travail, popup de risques).

## 1. Glossaire Métier Partagé (Ubiquitous Language)

Les termes français sont ceux de l'interface et de la documentation ; l'identifiant entre parenthèses est
le seul nom autorisé dans le code (classes, colonnes, clés JSON). Ne jamais introduire de synonyme
(`offer`, `candidate`, `scrape_run`…).

| Terme | Identifiant code | Définition exacte |
| :--- | :--- | :--- |
| **Utilisateur** | `user` | Le freelance qui exploite son instance. En auto-hébergement il y en a un seul, mais toute donnée porte `user_id` (cf. I-01). |
| **Profil candidat** | `Profile` / `candidate_profiles` | Titre, compétences, années d'expérience, séniorité et texte libre de l'utilisateur. Origine : `profil.json/.md/.txt`, `cv.pdf` ou lecture du profil LinkedIn. Les **surcharges** (compétences ajoutées / retirées dans l'UI) s'appliquent par-dessus sans jamais réécrire le fichier source. |
| **Séniorité** | `seniority` | `junior`, `confirme`, `senior`, `lead` (chaîne vide = inconnue). |
| **Source** | `source_code` | Plateforme d'origine d'un élément : `linkedin`, `france_travail`, `malt`, `wttj`, `indeed`. Chaque source a un **mode d'accès** : `api_officielle`, `navigateur`, `import_manuel`. |
| **Consentement aux risques** | `source_consents` | Acceptation explicite, versionnée et révocable par l'utilisateur des risques d'une source en mode `navigateur` (popup obligatoire, cf. I-16). |
| **Niveau de prudence** | `caution` | Plafond de volume et rythme de navigation : `prudent` (15 éléments/type, pause 6–14 s), `normal` (30, 4–9 s), `agressif` (60, 2,5–6 s). |
| **Recherche** | `SearchRequest` | Mot-clé de poste + filtres (localisation, `remote_only`, prudence, `max_items`, source du profil, compétences additionnelles). |
| **Tâche** | `Task` / `tasks` | Exécution asynchrone d'une action longue. Types (`kind`) : `run`, `rescore`, `train`, `prepare`, `apply`, `contacts`, `message`, `import_profile`, `profile_skill`, `purge` `[ACTUEL]` (vague 6, `CHECK ck_tasks_kind`, migration `0004`). |
| **Élément collecté** | `Item` / `collected_items` | Une unité récupérée sur une source, de type (`kind`) `job` (offre), `post` ou `person` (profil tiers). |
| **Offre** | `Item(kind="job")` | Élément scorable et éligible à une candidature. |
| **Historique** | `history` | Cumul des éléments collectés d'une exécution à l'autre ; base du calcul des tendances. |
| **Tendance** | `trends` | Agrégats sur la **fenêtre de 30 jours** : termes fréquents (TF-IDF), compétences par type d'élément, croissance J0–15 vs J15–30, série hebdomadaire. |
| **Écart de compétences** | `skill_gap` | Compétences demandées par le marché absentes du profil candidat. |
| **Annotation** | `label` | Jugement binaire de l'utilisateur sur une offre : `1` pertinente (« p »), `0` non pertinente. |
| **Modèle de matching** | `matching_models` | Régression logistique (descente de gradient, L2, classes repondérées) sur **8 critères**. Poids *a priori* tant qu'il y a moins de 6 annotations de **chaque** classe, puis modèle appris. |
| **Score** | `score` | Probabilité (0–100 %) qu'une offre corresponde au profil. Seuil de proposition : `apply_min_score` (70 % par défaut). |
| **Interprétation** | `interpretation` | Texte de synthèse du rapport. Produit par Claude **ou** par l'interprète local à base de règles — toujours présent dans le rapport final (cf. I-11). |
| **Rapport** | `reports` | Instantané immuable d'une analyse : JSON (agrégats, offres scorées, interprétation) + rendu HTML. |
| **Dossier de candidature** | `Application` | CV adapté (`cv.md` / `cv.html`), message, conseils et texte de l'offre, préparés pour une offre. |
| **Candidature** | `applications` | Suivi d'une offre. Statuts : `préparée`, `contactée`, `postulée`, `entretien`, `refusée`, `abandonnée`, `externe`, `échec`. |
| **Candidature simplifiée** | `apply_bot` | Formulaire *Easy Apply* LinkedIn piloté par le navigateur jusqu'à l'écran de relecture. |
| **Réponse mémorisée** | `answers` | Réponse que l'utilisateur a déjà donnée à une question de formulaire (clé = question normalisée). Seule source de remplissage automatique avec email et téléphone. |
| **Contact entreprise** | `company_contacts` | Recruteur affiché dans « Rencontrez l'équipe de recrutement » d'une offre (3 max). Son email n'est conservé que s'il est **affiché** par LinkedIn — jamais deviné ni reconstitué. |
| **Message de prospection** | `outreach_messages` | Email (SMTP) ou message LinkedIn envoyé à un contact entreprise au sujet d'une offre. |
| **Opposition** | `suppression_list` | Demande d'un destinataire de ne plus être contacté. Définitive ; conservée sous forme d'empreinte (hash) après purge du contact. `[ACTUEL]` (vague 4). |
| **Contact opposé** (`opposed`) | `company_contacts.opposed_at` | Contact dont l'opposition a été saisie : `email` effacé, `profile_url` vide, la ligne reste ; ses coordonnées ne peuvent plus être saisies (422). |
| **RecipientSuppressed** | `outreach.RecipientSuppressed` | Exception (409) levée quand l'empreinte du destinataire est dans `suppression_list` ou que son contact porte `opposed_at`. |
| **Pied imposé** | `outreach.compose_footer` | Mention ajoutée par le serveur après le corps de chaque message (origine de l'adresse, objet lié à l'offre, désinscription « STOP »). Non modifiable, jamais saisie par l'utilisateur (I-12). |
| **Purge** | `purge` / `RetentionService.purge` | Suppression, par la tâche `purge` (non navigateur, au démarrage puis toutes les 24 h), des données arrivées à échéance : éléments `post`/`person` à 30 jours et offres sans candidature ni annotation à 90 jours (`collected_items.purge_after`, I-10), contacts entreprise 3 ans après le dernier contact (`company_contacts.purge_after`, I-14), sessions de plateforme expirées, événements d'activité et tâches terminées de plus de 12 mois. Ne touche jamais `suppression_list`, `reports`, `applications`. Journalise des compteurs par table, jamais de contenu. `[ACTUEL]` (vague 6). |
| **Rédaction de tiers** | `nlp.redact_third_parties` | Remplacement par `[personne]` des emails, téléphones, URL `linkedin.com/in/…`, noms de contacts connus et motifs LinkedIn (« X et des millions… », « relations travaillent ici ») dans le texte d'une offre avant son envoi à Claude pour la génération de CV (I-11, E-6). `[ACTUEL]` (vague 6) ; les noms connus sont ceux de tous les contacts de l'utilisateur. |
| **Lieu de résidence / fuseau** | `home_location`, `timezone` (`user_settings`) | Lieu et fuseau horaire IANA de l'utilisateur (ex. `Africa/Abidjan`), vides par défaut ; injectés dans les prompts Claude, les libellés de l'interprète local et le CV local à la place de valeurs codées en dur (E-18). `[ACTUEL]` (vague 6) ; à saisir dans Paramètres après la migration `0004`. |
| **Blocage** | `activity_events(kind="blocked")` | Checkpoint, authwall, CAPTCHA ou perte de session détecté. Ouvre une **période de repos** de 6 h pendant laquelle aucune action navigateur n'est permise. |
| **Plafond quotidien** | `apply_daily_max`, `message_daily_max` | Nombre maximal de candidatures (10 par défaut) et de messages (15 par défaut) par jour. |

## 2. Invariants Métier Non Négociables

Ces règles s'appliquent à **tout** point d'entrée : CLI (`python -m linkedin_market`), API web, tests, scripts.

| ID | Invariant | Garde-fou technique |
| :--- | :--- | :--- |
| **I-01** | Toute ligne persistée appartient à un utilisateur ; toute requête filtre par `user_id`. | Colonne `user_id NOT NULL` + FK sur chaque table métier ; filtre dans chaque dépôt `db/repositories/*` `[ACTUEL]` (vague 1). |
| **I-02** | Un rapport est immuable après production. | Aucun endpoint `PUT/PATCH/DELETE` sur `/api/reports` ; trigger `trg_reports_immutable` refuse tout `UPDATE` sur `reports` `[ACTUEL]`. |
| **I-03** | Le statut d'une candidature ne régresse jamais automatiquement (`_PROGRESS` : `préparée/externe/échec`=0 < `contactée`=1 < `postulée`=2 < `entretien`=3). Seule une action explicite (`PATCH /api/applications/{id}`) peut le changer librement. | `application.advance_status()`. |
| **I-04** | Aucune candidature ni aucun message n'est soumis sans confirmation explicite de l'utilisateur à l'instant de l'envoi. | Candidature : arrêt à l'écran de relecture puis question `confirm_*` posée pendant la tâche (jamais mémorisée). Message : `confirm: true` exigé dans le corps de `POST /api/applications/{id}/messages` (HTTP 422 sinon). |
| **I-05** | Aucune réponse de formulaire n'est inventée : seuls `contact_email`, `contact_phone` et les `answers` mémorisées sont remplis ; toute autre question est posée à l'utilisateur (statut de tâche `attente_saisie`, délai 600 s). | `apply_bot`, `TaskContext.ask()`. |
| **I-06** | Checkpoint / authwall / CAPTCHA / perte de session ⇒ arrêt immédiat de l'action, enregistrement d'un **blocage**. Aucun contournement automatisé n'est tenté ; seule la vérification **manuelle** par l'utilisateur (`attente_verification_manuelle`) est permise. | `BlockedError`, `activity.guard_not_blocked()` (HTTP 409). |
| **I-07** | Les plafonds de la prudence et les plafonds quotidiens ne sont jamais dépassés, même sur demande. `max_items` ne peut que réduire le volume de la prudence choisie. Relever un plafond quotidien dans les Paramètres est permis ; le dépasser ne l'est pas. | `activity.check_quota()` (HTTP 429). |
| **I-08** | Un utilisateur n'utilise que **son propre** compte sur chaque source : ni faux compte, ni compte partagé, ni rotation d'identités. | Un seul jeu d'identifiants par source et par utilisateur. |
| **I-09** | Minimisation : pour un profil tiers (`kind="person"`), seuls le titre et les compétences extraites sont conservés ; aucune donnée sensible (art. 9 RGPD : santé, opinions, religion, syndicat…) n'est stockée. | Filtre à l'ingestion. |
| **I-10** | Les éléments `post` et `person` sont purgés à **30 jours** (`purge_after`) ; les offres non liées à une candidature à **90 jours**. | `purge_after` posé à l'ingestion `[ACTUEL]` ; tâche de purge quotidienne `[ACTUEL]` (vague 6). |
| **I-11** | Claude ne reçoit **jamais de donnée de tiers identifiable**. L'**interprétation** ne lui envoie que des agrégats. La **génération de CV** (si l'utilisateur choisit Claude) lui envoie le profil de l'utilisateur et le texte de l'offre, **après retrait** par `nlp.redact_third_parties` des noms, emails, téléphones et URL de profils de personnes (décidé le 2026-10-09, E-6 option A). Le **mode local** doit toujours produire un rapport **avec interprétation** sans aucun service IA : si Claude est indisponible ou échoue, bascule automatique sur l'interprète local. | `interpreter.interpret()` ; `nlp.redact_third_parties` `[ACTUEL]` (vague 6) ; la valeur `none` n'existe plus que sur d'anciens rapports. |
| **I-12** | Prospection B2B uniquement si : destinataire joint **dans le cadre de ses fonctions**, objet **en rapport avec sa fonction** (une offre qu'il publie), information sur l'origine de l'adresse, **lien/mention de désinscription simple et gratuite** dans chaque message. | Gabarit de message imposé par `outreach` `[ACTUEL]` (vague 4 : `compose_footer`, toujours ajouté par le serveur). |
| **I-13** | Chaque contact porte sa base légale (`interet_legitime`) et sa source ; une adresse présente dans `suppression_list` bloque tout envoi, définitivement. | Vérification avant chaque envoi (HTTP 409) `[ACTUEL]` (vague 4 : dans la transaction de réservation, revérifiée au début de la tâche). |
| **I-14** | Un contact entreprise sans réponse est supprimé **3 ans** après le dernier contact. | `purge_after` sur `company_contacts`. |
| **I-15** | Identifiants, mots de passe SMTP, clés API et cookies de session ne sont jamais versionnés ni journalisés. Les cookies de session expirent **72 h** après la dernière connexion par identifiants. | `.env` hors Git ; cookies chiffrés Fernet en base (`platform_sessions`) `[ACTUEL]`. |
| **I-16** | Une source en mode `navigateur` (LinkedIn, Malt, WTTJ) ne peut être utilisée qu'après acceptation, dans une **popup de risques**, de la version courante du texte de risques ; un changement de version redemande l'acceptation. | `source_consents` ; HTTP 428 sinon `[ACTUEL]` (vague 3, `consent.py`, revérifié au début de chaque tâche navigateur ; repos de 6 h par source ; CLI avec collecte refusé sans consentement). Items/import, nouvelles sources `[CIBLE]`. |

## 3. Matrice Actions Métier → Impact Système

| Action Métier | Entrée Frontend | Services Backend Impactés | Tables Modifiées `[CIBLE]` | Tâche / Événement |
| :--- | :--- | :--- | :--- | :--- |
| `AccepterRisquesSource` | Popup « Risques LinkedIn » → « J'accepte » | `SourceConsentService` | `source_consents` | — |
| `LancerAnalyse` | Dashboard → « Lancer » (`POST /api/runs`) | `pipeline.scrape`, `nlp`, `trends`, `matching`, `interpreter`, `report` | `collected_items`, `reports`, `tasks`, `activity_events` | `Task(kind="run", browser=scrape)` |
| `ReanalyserHistorique` | Dashboard → « Analyser sans collecte » | `pipeline.analyse`, `interpreter` | `reports`, `tasks` | `Task(kind="run", browser=False)` |
| `RecalculerScores` | Offres → « Recalculer » (`POST /api/runs/rescore`) | `matching` | `tasks` | `Task(kind="rescore")` |
| `AnnoterOffre` | Offres → 👍 / 👎 (`PUT /api/jobs/{id}/label`) | `storage.save_labels` | `labels` | — |
| `EntrainerModele` | Offres → « Entraîner » (`POST /api/model/train`) | `matching.train` | `matching_models` | `Task(kind="train")` |
| `PreparerCandidature` | Offre → « Préparer » (`POST /api/jobs/{id}/prepare`) | `application.build/write`, `interpreter` (CV via Claude ou gabarit local) | `applications` | `Task(kind="prepare")` |
| `PostulerEasyApply` | Candidature → « Postuler » (`POST /api/applications/{id}/apply`) | `apply_bot`, `session`, `guards` | `applications`, `activity_events` (`apply`) | `Task(kind="apply", browser=True)` |
| `ChercherRecruteurs` | Candidature → « Trouver les recruteurs » | `contacts` | `company_contacts` | `Task(kind="contacts", browser=True)` |
| `EnvoyerMessage` | Candidature → « Envoyer » après relecture | `outreach` (SMTP ou LinkedIn), contrôle `suppression_list` | `outreach_messages`, `applications` (→ `contactée`), `activity_events` | `Task(kind="message")` |
| `EnregistrerOpposition` | Candidature → « Ce contact refuse d'être recontacté » | `SuppressionService` | `suppression_list`, `company_contacts` | — |
| `ModifierProfil` | Compétences → ajouter / retirer | `settings`, `profile_sync` (si `linkedin_skill_sync`) | `profile_skill_overrides` | `Task(kind="profile_skill", browser=True)` |
| `ImporterProfilLinkedIn` | Compétences → « Importer depuis LinkedIn » | `profile`, `session` | `candidate_profiles` | `Task(kind="import_profile", browser=True)` |
| `LeverBlocage` | Dashboard → « J'ai vérifié mon compte » (`POST /api/block/clear`) | `activity` | `activity_events` (`unblocked`) | — |
| `PurgerDonneesTiers` | Automatique (quotidien) | `RetentionService` | `collected_items`, `company_contacts` | Tâche planifiée `[ACTUEL]` (vague 6) |

## 4. Niveaux de Consistance & Traçabilité

- **Consistance forte (transaction ACID PostgreSQL)** — `[ACTUEL]` pour le changement de statut et l'activation
  d'un modèle, `[ACTUEL]` pour le consentement (vague 3), `[CIBLE]` pour l'envoi de message :
  - changement de statut de candidature + entrée d'historique ;
  - envoi d'un message : contrôle `suppression_list` + contrôle de plafond + insertion `outreach_messages` dans
    la même transaction (verrou `SELECT … FOR UPDATE` sur la ligne `applications`) ;
  - acceptation / révocation d'un consentement de source ;
  - activation d'un nouveau modèle de matching (un seul `is_active = true` par utilisateur).
- **Consistance éventuelle** : tendances et scores (recalculés à la demande ou après entraînement), compteurs du
  tableau de bord (`/api/monitor`), purge de rétention (quotidienne).
- **Sérialisation des actions navigateur** : une seule session Selenium à la fois par instance ; les tâches
  `browser=True` attendent en `en_attente` (verrou du `TaskManager`).
- **Traçabilité** : chaque tâche a un `id` (UUID v7) repris dans ses logs et, `[CIBLE]`, dans
  `activity_events.data.task_id`. Toutes les dates sont stockées en **UTC** (`timestamptz`) et exposées en ISO 8601
  suffixé `Z` `[ACTUEL]` (vague 1).

## 5. Risques Connus & Mesures d'Atténuation

> Ces risques sont **assumés par l'utilisateur** qui les accepte dans la popup (I-16). La documentation décrit
> comment **réduire** chaque risque ; elle n'autorise **aucune** technique de dissimulation ou de contournement
> des protections des plateformes (CAPTCHA, empreinte navigateur maquillée, proxies tournants, faux comptes).

| # | Risque | Référence | Probabilité / impact | Atténuation retenue |
| :--- | :--- | :--- | :--- | :--- |
| R-1 | **Restriction ou suspension du compte LinkedIn** : l'*User Agreement* (version du 3 nov. 2025, § 8.2 « Don'ts », points 2, 3 et 13) interdit les logiciels de scraping, le contournement des limites d'usage et les bots pour accéder au service, envoyer des messages ou ajouter des contacts. | LinkedIn User Agreement ; contentieux LinkedIn c. Proxycurl (fermeture, juil. 2025) et c. ProAPIs (oct. 2025) | Élevée / élevé pour l'utilisateur | Popup de risques versionnée (I-16) ; prudence `prudent` par défaut, `max_items` jamais supérieur au plafond de la prudence choisie ; plafonds (I-07) ; arrêt + repos de 6 h au premier blocage (I-06) ; navigateur visible par défaut (`LINKEDIN_HEADLESS=0`) ; aucune action sans clic (I-04). |
| R-2 | **Responsabilité de l'éditeur** si l'outil distribué facilite une collecte massive. | Mêmes contentieux ; RGPD art. 25 (*privacy by design*) | Moyenne / élevé | Pas de mode serveur, pas de revente ni d'export de profils tiers ; volumes plafonnés dans le code (non configurables au-delà de `agressif`) ; README et popup explicites. |
| R-3 | **Traitement de données personnelles de tiers** (profils, posts, recruteurs) sans consentement. | RGPD art. 6(1)(f), 5(1)(c), 5(1)(e), 14 ; CNIL, recommandations sur l'intérêt légitime et le moissonnage (19 juin 2025) | Moyenne / moyen | Minimisation (I-09), finalité limitée à l'analyse personnelle du marché, purge à 30 j (I-10), aucun envoi de données brutes à un tiers (I-11). Chaque utilisateur auto-hébergé est responsable de traitement : un modèle de registre est fourni dans le README `[CIBLE]`. |
| R-4 | **Prospection B2B non conforme**. | CPCE art. L.34-5 (transposition ePrivacy 2002/58/CE) ; fiche CNIL « prospection commerciale par courrier électronique » mise à jour le 14 juin 2026 | Moyenne / moyen (sanction CNIL) | I-12, I-13, I-14 ; plafond de 15 messages/jour ; pas d'envoi en masse ni de séquence automatique de relances. |
| R-5 | **Rupture des sélecteurs** à chaque évolution de l'interface LinkedIn / Malt / WTTJ. | — | Élevée / moyen | Repérage par texte et `aria-label` (FR + EN), jamais par classes CSS générées ; fixtures HTML dans `tests/fixtures/` ; échec lisible (statut `échec`) plutôt que comportement imprévisible. |
| R-6 | **Conditions d'utilisation des autres sources**. Indeed : API *Job Sync* réservée aux partenaires approuvés (employeurs, agences, plateformes) et contrat développeur interdisant le scraping. Malt / WTTJ : pas d'API publique de lecture des offres. | Indeed Developer Agreement / Job Sync API terms | — | `indeed` : **import manuel uniquement** (URL ou texte d'offre collé). `france_travail` : **API officielle** « Offres d'emploi v2 » (francetravail.io, OAuth2 *client credentials*) — source sans risque, recommandée. `malt`, `wttj` : mode `navigateur` soumis aux mêmes règles que LinkedIn (I-06, I-07, I-16). |
| R-7 | **Dissimulation de l'automatisation** : options Chrome comme `--disable-blink-features=AutomationControlled` ou `excludeSwitches=["enable-automation"]`. | Contraire à la ligne directrice ci-dessus | — | Retirées de `browser.py` le 2026-10-08 (vague 0) `[ACTUEL]`, garanti par `tests/test_browser.py`. Ne jamais ajouter d'option de ce type. |
| R-8 | **Fuite de secrets locaux** (cookies LinkedIn, mot de passe SMTP). | RGPD art. 32 | Faible / élevé | API liée à `127.0.0.1` uniquement, vérification `Host`/`Origin`, jeton local `X-Local-Token` ; cookies chiffrés en base avec `LM_SECRET_KEY` `[ACTUEL]`. |

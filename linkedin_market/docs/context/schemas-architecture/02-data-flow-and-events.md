# Flux de Données & Diagrammes de Séquence

Conventions communes à tous les flux :
- Toute requête mutative porte l'en-tête `X-Local-Token` (ou `?token=` pour les liens ouverts dans un nouvel
  onglet). Absent ou invalide ⇒ `401`. `Host` non local ⇒ `403`. `Origin` étrangère ⇒ `403`.
- Une action longue renvoie `202 Accepted` + l'objet tâche ; le frontend interroge `GET /api/tasks/{id}` toutes les
  1,5 s tant que le statut n'est pas final (`terminée`, `échec`, `annulée`).
- Statuts de tâche : `en_attente` → `en_cours` ⇄ `attente_saisie` / `attente_verification_manuelle` →
  `terminée` | `échec` | `annulée`.

## 1. Flux Critique 1 : Accès à l'Interface Locale

```mermaid
sequenceDiagram
    autonumber
    actor U as Utilisateur
    participant CLI as python -m linkedin_market.web
    participant API as API FastAPI (127.0.0.1:8765)
    participant SPA as Interface React

    U->>CLI: Lance l'interface
    CLI->>API: create_app(token=secrets.token_urlsafe(24))
    CLI->>U: Ouvre http://127.0.0.1:8765/
    U->>API: GET /
    API-->>SPA: index.html avec le jeton injecté (lisible uniquement par la même origine)
    SPA->>API: GET /api/status (X-Local-Token)
    alt Host non local ou Origin étrangère
        API-->>SPA: 403 {"detail": "Hôte non autorisé."}
    else Jeton invalide
        API-->>SPA: 401 {"detail": "Jeton local manquant ou invalide."}
    else OK
        API-->>SPA: 200 {"session":{"age_hours":3.2,"ttl_hours":72,"valid":true},"credentials_configured":true,"smtp_configured":true,"anthropic_configured":true,…}
    end
```

## 2. Flux Critique 2 : Acceptation des Risques d'une Source `[CIBLE]`

```mermaid
sequenceDiagram
    autonumber
    actor U as Utilisateur
    participant SPA as Interface
    participant API as API
    participant DB as PostgreSQL

    U->>SPA: Clique « Lancer l'analyse » (sources : linkedin)
    SPA->>API: GET /api/sources
    API->>DB: SELECT consent courant (user_id, source_code='linkedin', risk_version)
    API-->>SPA: 200 [{"code":"linkedin","mode":"navigateur","risk_version":"2026-10","consent":null}]
    SPA->>U: Popup « Risques LinkedIn » (suspension de compte, CGU § 8.2, RGPD, prudence)
    alt Refus
        U->>SPA: « Annuler »
        SPA-->>U: Analyse lancée sans LinkedIn ou abandon
    else Acceptation (case cochée + bouton)
        U->>SPA: « J'accepte les risques »
        SPA->>API: POST /api/sources/linkedin/consent {"risk_version":"2026-10","accepted":true}
        API->>DB: INSERT source_consents (accepted_at = now())
        API-->>SPA: 201
        SPA->>API: POST /api/runs {...}
    end
    Note over API: Sans consentement valide, POST /api/runs avec scrape=true ⇒ 428 Precondition Required
```

## 3. Flux Critique 3 : Analyse du Marché (`kind="run"`)

```mermaid
sequenceDiagram
    autonumber
    actor U as Utilisateur
    participant SPA as Interface
    participant API as API / guards
    participant TM as TaskManager
    participant SEL as Navigateur (Selenium)
    participant SRC as LinkedIn / Malt / WTTJ
    participant FT as API France Travail
    participant CORE as nlp · trends · matching
    participant IA as Claude (optionnel)
    participant DB as Stockage

    U->>SPA: Mot-clé + prudence + « Lancer »
    SPA->>API: POST /api/runs {"keyword":"Data Engineer","caution":"prudent","scrape":true}
    API->>API: guards : identifiants ? consentement ? blocage < 6 h ?
    API->>TM: submit("run", browser=true)
    API-->>SPA: 202 {"id":"a1b2c3d4e5f6","status":"en_attente",...}
    TM->>SEL: create_driver() puis login (cookies < 72 h sinon identifiants)
    alt Checkpoint / authwall à la connexion
        SEL-->>TM: page /checkpoint
        TM->>TM: statut attente_verification_manuelle
        U->>SEL: Résout la vérification lui-même dans la fenêtre visible
        TM->>TM: reprise (ou échec + blocage si délai dépassé)
    end
    loop Pour chaque type (job, post, person) jusqu'à max_items de la prudence
        SEL->>SRC: Page de recherche (f_WT=2, 30 derniers jours)
        SRC-->>SEL: Résultats
        SEL->>SEL: human_pause(min_delay..max_delay)
        alt Blocage détecté
            TM->>DB: activity_events(kind="blocked", where="collecte")
            TM->>TM: arrêt immédiat de la collecte (aucune relance), analyse sur les éléments déjà collectés
        end
    end
    TM->>FT: GET /offresdemploi/v2/offres/search?motsCles=… (Bearer)
    FT-->>TM: 200/206 offres JSON
    TM->>DB: Ajout à l'historique (dédoublonnage source + id, minimisation des person)
    TM->>CORE: Extraction, tendances 30 j, scoring 8 critères
    CORE-->>TM: résultats + summary (agrégats uniquement)
    alt interpreter = claude et clé disponible
        TM->>IA: messages.create(summary agrégé)
        IA-->>TM: interprétation
    else Claude absent ou en erreur
        TM->>CORE: LocalInterpreter.interpret(summary)
    end
    TM->>DB: Rapport immuable (JSON + HTML) + activity_events(kind="run")
    SPA->>API: GET /api/tasks/a1b2c3d4e5f6 (polling)
    API-->>SPA: 200 {"status":"terminée","result":{"report":"data-engineer-2026-10-08","blocked":false,…}}
```

Décision E-4 (2026-10-08, option A) : un blocage pendant la collecte n'empêche pas le rapport. La tâche finit
`terminée` avec un **rapport partiel** (données déjà collectées + historique de 30 jours) et
`result.blocked = true`, `result.blocked_where = "collecte"` ; l'interface affiche « Collecte interrompue par
LinkedIn ». I-06 reste entier : arrêt immédiat, événement `blocked`, repos de 6 h, aucune relance.

## 4. Flux Critique 4 : Candidature Simplifiée (`kind="apply"`)

```mermaid
sequenceDiagram
    autonumber
    actor U as Utilisateur
    participant SPA as Interface
    participant API as API
    participant TM as TaskManager
    participant BOT as apply_bot (Selenium)
    participant LI as LinkedIn

    U->>SPA: « Postuler » sur une candidature préparée
    SPA->>API: POST /api/applications/{job_id}/apply
    API->>API: plafond apply_daily_max (sinon 429), blocage (sinon 409)
    API->>TM: submit("apply", browser=true)
    API-->>SPA: 202 tâche
    TM->>BOT: Ouvre l'offre
    alt Pas de bouton Easy Apply
        BOT-->>TM: état "externe" (candidature sur site tiers, aucun remplissage)
    else Easy Apply
        loop Chaque étape du formulaire
            BOT->>BOT: Remplit email, téléphone, réponses mémorisées
            opt Question inconnue
                TM->>SPA: statut attente_saisie {"question":"Années d'expérience en Spark ?","field":"…"}
                U->>SPA: Saisit la réponse (+ « mémoriser »)
                SPA->>API: POST /api/tasks/{id}/answer {"value":"5","remember":true}
            end
        end
        BOT->>LI: Arrive à l'écran de relecture
        TM->>SPA: attente_saisie {"field":"confirm_submit","question":"Soumettre la candidature « titre » ?","options":["Soumettre","Annuler"]}
        U->>SPA: Confirme explicitement
        SPA->>API: POST /api/tasks/{id}/answer {"value":"Soumettre"}
        BOT->>LI: Clic « Envoyer la candidature »
        alt Envoi non vérifiable
            BOT-->>TM: état "incertaine" (vérifier sur LinkedIn avant de réessayer)
        end
        TM->>TM: advance_status(job_id, "postulée") + activity_events(kind="apply")
    end
```

## 5. Flux Critique 5 : Message de Prospection (`kind="message"`)

```mermaid
sequenceDiagram
    autonumber
    actor U as Utilisateur
    participant SPA as Interface
    participant API as API
    participant DB as PostgreSQL
    participant TM as TaskManager
    participant OUT as outreach
    participant SMTP as SMTP utilisateur

    U->>SPA: Ouvre « Contacter l'entreprise »
    SPA->>API: GET /api/applications/{job_id}/message-draft
    API-->>SPA: 200 {"subject":"…","body":"… [origine de l'adresse] … [mention de désinscription]"}
    U->>SPA: Relit, modifie, coche « J'ai relu » puis « Envoyer »
    SPA->>API: POST /api/applications/{job_id}/messages {"channel":"email","to":"…","confirm":true}
    API->>API: confirm=false ⇒ 422, message déjà en cours ⇒ 409
    API->>DB: BEGIN, adresse dans suppression_list ? (⇒ 409), message_daily_max atteint ? (⇒ 429)
    API->>DB: INSERT outreach_messages(status='confirmé'), COMMIT
    API->>TM: submit("message")
    API-->>SPA: 202 tâche
    TM->>OUT: send_email(...)
    OUT->>SMTP: STARTTLS + AUTH + DATA (Message-ID, en-tête List-Unsubscribe [CIBLE])
    alt Succès
        TM->>DB: status='envoyé', application → contactée (sans régression), activity_events(kind="message_email")
    else OutreachError
        TM->>DB: status='échec', error=…
    end
```

## 6. Événements du Journal d'Activité

Le journal (`activity.jsonl` aujourd'hui, table `activity_events` demain) sert au monitoring, aux plafonds
quotidiens et aux périodes de repos. Un événement = `{at, kind, ...data}`.

| `kind` | Émis par | Données | Consommé par |
| :--- | :--- | :--- | :--- |
| `run` | `services.run_analysis` | `keyword`, `scrape`, `jobs` | `/api/monitor` |
| `apply` | `services.apply_job` | `job`, `title` | plafond `apply_daily_max` |
| `message_email`, `message_linkedin` | `services.send_message` | `job`, `to` | plafond `message_daily_max` |
| `profile_skill` | `services.add_skill` | `skill`, `linkedin_name` | `/api/monitor` |
| `blocked` | toute action navigateur | `where`, `job?` (la fin du repos `until` est calculée à la lecture par `block_status()`) | `guard_not_blocked()` (repos 6 h) |
| `unblocked` | `POST /api/block/clear` | — | lève le repos en cours |
| `error` | erreurs non métier | `message` | `/api/monitor` |

## 7. Résilience & Gestion des Pannes

| Situation | Comportement |
| :--- | :--- |
| Chargement de page navigateur | `set_page_load_timeout(60)` ; échec ⇒ tâche `échec`, message lisible. |
| Saisie utilisateur attendue | `INPUT_TIMEOUT_SECONDS = 600` ; au-delà ⇒ tâche `échec`. |
| Blocage plateforme | Arrêt immédiat, événement `blocked`, repos de **6 h** (`BLOCK_COOLDOWN_HOURS`) ; levée manuelle uniquement après vérification du compte dans un navigateur normal. |
| Claude indisponible / erreur | Bascule automatique sur `LocalInterpreter` ; le rapport indique `interpreter_used="local"`. |
| API France Travail | Timeout 10 s, 3 tentatives avec backoff exponentiel (1 s, 2 s, 4 s) sur `429` / `5xx` ; respect de `Retry-After` ; en échec, l'analyse continue avec les autres sources. `[CIBLE]` |
| SMTP | Timeout 25 s (`SMTP_TIMEOUT`), **aucune** relance automatique (éviter les doublons) ; statut `échec` et nouvel envoi manuel. |
| Arrêt de l'application pendant une tâche | `[ACTUEL]` (vague 1) : au redémarrage, les tâches non finales de la table `tasks` sont marquées `échec` (« Tâche interrompue par l'arrêt de l'application. »). |
| Annulation | `POST /api/tasks/{id}/cancel` ; vérifiée entre deux sources / deux étapes (`Cancelled`). |

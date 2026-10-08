# Authentification & Matrice de Permissions

> Modèle : **une instance auto-hébergée = un utilisateur**, API sur la boucle locale. Il n'y a ni compte, ni mot
> de passe applicatif, ni JWT, ni rôle. La sécurité repose sur trois barrières cumulées appliquées par le
> middleware `guard` de `web/app.py` (Host, Origin, jeton local), puis sur les **garde-fous métier** (§ 2) qui
> tiennent lieu de permissions. Références : RGPD art. 32 (sécurité du traitement), lot 1 I-15, R-8.

## 1. Stratégie d'Authentification

### 1.1 Mécanisme `[ACTUEL]`
| Élément | Valeur |
| :--- | :--- |
| Écoute | `uvicorn.run(app, host="127.0.0.1", port=args.port)` — adresse non configurable ; seul `--port` l'est (défaut 8765). Aucune option `0.0.0.0`, aucune configuration de proxy (`forwarded_allow_ips`) : à ne jamais ajouter. |
| Jeton | `secrets.token_urlsafe(24)` (24 octets aléatoires ⇒ 32 caractères base64url, 192 bits) **régénéré à chaque lancement** ; ou valeur fixe `LINKEDIN_UI_TOKEN` (`.env`) pour qui veut garder un onglet ouvert entre deux redémarrages. |
| Transport du jeton | En-tête `X-Local-Token` (appels `fetch` de la SPA) ; paramètre `?token=` (liens ouverts dans un nouvel onglet : rapport HTML, `cv.html`). |
| Vérification | `hmac.compare_digest(given, expected)` (temps constant) sur tout chemin `/api/*` sauf `/api/health` ⇒ `401 {"detail": "Jeton local manquant ou invalide."}`. |
| Distribution au frontend | Le serveur remplace `<!--LM_TOKEN-->` de `index.html` par `<meta name="lm-token" content="…">` ; réponse `Cache-Control: no-store`. Seule une page servie par la même origine peut lire ce `<meta>`. |
| Durée de vie | Celle du processus. Pas de rafraîchissement ni de révocation : arrêter l'application invalide le jeton (sauf `LINKEDIN_UI_TOKEN`). |
| Désactivation | `create_app(token=None)` : tests (`TestClient`) et `--dev` (proxy Vite sur `:5174`). En `--dev`, Swagger est exposé et les origines `http://127.0.0.1:5174` / `http://localhost:5174` acceptées. **Le mode `--dev` ne doit pas être utilisé avec de vrais identifiants.** |
| Cookies HTTP | Aucun : pas de session navigateur, donc pas de surface CSRF par cookie. |

### 1.2 Pourquoi pas de JWT ni de login
- Un seul utilisateur, sur sa machine : un mot de passe applicatif protégerait l'API contre… l'utilisateur lui-même,
  et serait stocké sur la même machine que `.env`.
- Le jeton éphémère suffit contre les **autres pages web** ouvertes dans le navigateur (cible réaliste d'attaque) ;
  les **processus locaux** du même compte système peuvent de toute façon lire `.env` et `linkedin_market_data/` :
  ils sont hors du modèle de menace (§ 1.3).
- Un JWT (signature, expiration, rafraîchissement, révocation) n'apporterait que de la complexité pour le même
  niveau de protection.

### 1.3 Modèle de menace local

| Menace | Scénario | Défense `[ACTUEL]` | Complément `[CIBLE]` |
| :--- | :--- | :--- | :--- |
| **DNS rebinding** | Un site `evil.example` résout son nom vers `127.0.0.1` ; ses scripts deviennent « même origine » que l'API. | `Host` doit valoir `127.0.0.1`, `localhost` ou `[::1]` (port quelconque) sur **toutes** les routes ⇒ `403 Hôte non autorisé.` (le navigateur envoie `Host: evil.example`). | — |
| **CSRF** (requête forgée depuis une autre page) | Formulaire ou `fetch` vers `http://127.0.0.1:8765/api/applications/…/apply`. | Mutations : `Origin` présent ≠ `Host` ⇒ `403 Origine non autorisée.` ; et dans tous les cas jeton exigé, qu'une page tierce ne connaît pas. Pas de CORS configuré ⇒ aucune réponse lisible cross-origin. | Refuser aussi les mutations **sans** `Origin` quand `Sec-Fetch-Site` vaut `cross-site`. |
| **Lecture du jeton par une autre page** | Page tierce qui charge `/` dans une iframe ou via `fetch`. | Politique de même origine : le `<meta>` n'est lisible que par une page de l'origine `127.0.0.1:8765`. | `X-Frame-Options: DENY` / `frame-ancestors 'none'` sur la SPA. |
| **Contenu HTML non fiable servi par l'API** | Texte d'offre ou de profil collecté injecté dans un rapport ou un CV. | `report.html` et `cv.html` servis avec `Content-Security-Policy: sandbox …` (sans `allow-same-origin` ⇒ origine opaque, aucun accès au jeton ni à l'API) ; données échappées (`html.escape`, `</` neutralisé) ; CV rendu via DOMPurify. | CSP de la SPA : `default-src 'self'`. |
| **Fuite du jeton par l'URL** | `?token=` dans l'historique ou dans un `Referer`. | `Referrer-Policy: no-referrer` sur toutes les réponses ; jeton régénéré à chaque lancement. | Préférer un jeton à usage unique pour l'ouverture d'onglets. |
| **Traversée de chemin** | `GET /api/reports/../../.env`, dossier de candidature modifié à la main. | `storage.load_report` refuse `/`, `\`, `.` initial ; `application.folder_of` refuse un dossier hors `applications/` (500) ; route SPA : fichier servi seulement s'il est **dans** `dist/` après `resolve()`. | — |
| **Injection d'en-tête SMTP** | Objet `"a\nBcc: x@y.z"`. | 422 « Objet requis, sur une seule ligne. » + `OutreachError` dans `build_email` ; destinataire sans `,;<>`. | — |
| **Exposition réseau** | Accès depuis le LAN. | Écoute `127.0.0.1` uniquement ; PostgreSQL publié sur `127.0.0.1:5432` (`compose.yaml`). | — |
| **Autre utilisateur du même poste** | Session système différente. | Hors périmètre de l'API ; `session.json` en `chmod 0o600`. | Droits `0700` sur `linkedin_market_data/` ; cookies chiffrés (§ 1.4). |
| **Logiciel malveillant du même compte** | Lit `.env`, la base, les fichiers. | Hors périmètre (aucune défense applicative possible). | Documenté dans le README. |

### 1.4 Secrets (I-15)

| Secret | Stockage `[ACTUEL]` | Stockage `[CIBLE]` | Jamais |
| :--- | :--- | :--- | :--- |
| `LINKEDIN_EMAIL` / `LINKEDIN_PASSWORD` | `.env` (hors Git), lus par `config.linkedin_credentials()` ; en CLI sans `.env`, saisis via `getpass` | inchangé | journalisés, renvoyés par l'API (`/api/status` n'expose que `credentials_configured`) |
| Cookies LinkedIn | `linkedin_market_data/session.json` en clair, `chmod 0o600`, `logged_in_at` UTC, supprimé après 72 h | `platform_sessions.cookies_encrypted` = `Fernet(LM_SECRET_KEY).encrypt(json(cookies))`, `expires_at = last_login_at + 72 h` ; la réutilisation ne prolonge pas la session | versionnés, journalisés, renvoyés par l'API |
| `SMTP_USER` / `SMTP_PASSWORD` | `.env` | inchangé | journalisés (seul `smtp_configured` est exposé) |
| `ANTHROPIC_API_KEY` | `.env`, lu par le SDK `anthropic` | inchangé | envoyés ailleurs qu'à `api.anthropic.com` |
| `FRANCE_TRAVAIL_CLIENT_ID` / `_SECRET` | — | `.env` ; jeton d'accès en **mémoire** seulement (jamais en base) | journalisés |
| `LM_SECRET_KEY` | — | `.env` ; si absente au premier lancement, `Fernet.generate_key()` est ajoutée à `.env` (droits `0600`) avec un message invitant à la sauvegarder | dans la base ou les sauvegardes `pg_dump` |
| `DATABASE_URL` / `LM_DB_PASSWORD` | — | `.env` | journalisés (masquer le mot de passe dans tout message d'erreur de connexion) |
| Jeton local | mémoire du processus (`app.state.token`) | inchangé | écrit sur disque (sauf choix explicite `LINKEDIN_UI_TOKEN`) |

Rotation de `LM_SECRET_KEY` : `MultiFernet([nouvelle, ancienne])` pendant une version, réchiffrement des lignes
`platform_sessions` au démarrage, puis retrait de l'ancienne. Une clé perdue ne fait perdre que les cookies (nouvelle
connexion LinkedIn). Cookies expirés (> 72 h) : ligne supprimée à la lecture et par la purge.

## 2. Matrice RBAC (Rôles & Permissions)

### 2.1 Pas de rôles — justification
- **Un seul utilisateur par instance** (décision de cadrage) : pas d'administrateur, de manager ni de lecteur.
- I-08 impose que chaque utilisateur n'utilise **que son propre compte** de source : un mode multi-utilisateur
  partageant une instance mènerait à des sessions LinkedIn mutualisées, exclues.
- Un mode multi-utilisateur ferait de l'éditeur ou de l'hébergeur un responsable de traitement de données de tiers
  à grande échelle (R-2, R-3) : hors périmètre.
- `user_id` (I-01) existe en base pour la cohérence du modèle et l'export, **pas** comme mécanisme d'autorisation.

### 2.2 Matrice des garde-fous par action (les vraies « permissions »)

Toutes les routes exigent Host local + jeton (sauf `/api/health`, sans jeton). Colonnes : **Id.** = identifiants
LinkedIn présents ; **Cons.** = consentement aux risques de la source `[CIBLE]` ; **Bloc.** = aucun blocage depuis
6 h ; **Plaf.** = plafond quotidien ; **Unic.** = pas de tâche identique active ; **Conf.** = confirmation explicite.

| Action | Route | Id. | Cons. | Bloc. | Plaf. | Unic. | Conf. | Code si refus |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Lecture (état, réglages, profil, offres, rapports, tâches, candidatures) | `GET …` | | | | | | | — |
| Modifier réglages / profil / annotation / dossier / statut | `PUT`/`PATCH`/`DELETE`/`POST /profile/skills` | | | | | | | 422 |
| Lever un blocage | `POST /api/block/clear` | | | | | | (UI) | — |
| Analyse sans collecte | `POST /api/runs` `scrape=false` | | | | | ✓ | | 409 |
| Collecte navigateur | `POST /api/runs` `scrape=true` | ✓ | ✓ | ✓ | prudence | ✓ | | 409 / 428 |
| Collecte France Travail `[CIBLE]` | `POST /api/runs` `sources` ∋ `france_travail` | | | | prudence | ✓ | | 409 ; source ignorée si non configurée |
| Import manuel Indeed `[CIBLE]` | `POST /api/items/import` | | | | | | | 409 / 422 |
| Recalcul, entraînement, préparation | `/runs/rescore`, `/model/train`, `/jobs/{id}/prepare` | | | | | ✓ | | 409 |
| Importer le profil LinkedIn | `POST /api/profile/import-linkedin` | ✓ | ✓ | ✓ | | ✓ | | 409 / 428 |
| Ajouter une compétence sur LinkedIn | effet de `POST /api/profile/skills` | ✓ | ✓ | ✓ | | ✓ | | `linkedin_sync.status` |
| Postuler (Easy Apply) | `POST /api/applications/{id}/apply` | ✓ | ✓ | ✓ | `apply_daily_max` | ✓ | `confirm_submit` | 409 / 428 / 429 |
| Chercher les recruteurs | `POST …/contacts/lookup` | ✓ | ✓ | ✓ | | ✓ | | 409 / 428 |
| Email de prospection | `POST …/messages` `channel=email` | | | | `message_daily_max` | ✓ | `confirm: true` + opposition `[CIBLE]` | 409 / 422 / 429 / 503 |
| Message LinkedIn | `POST …/messages` `channel=linkedin` | ✓ | ✓ | ✓ | `message_daily_max` | ✓ | `confirm: true` + opposition `[CIBLE]` | 409 / 422 / 428 / 429 |
| Email de test | `POST /api/smtp/test` | | | | | | destinataire = soi | 422 / 502 / 503 |
| Accepter / révoquer des risques `[CIBLE]` | `POST`/`DELETE /api/sources/{code}/consent` | | | | | | case cochée (UI) + `accepted: true` | 409 / 422 |
| Enregistrer une opposition `[CIBLE]` | `POST /api/suppressions` | | | | | | | 422 |

## 3. Rate Limiting & Protection Anti-Abus

### 3.1 Limitation HTTP
Aucune limitation de débit HTTP : le seul client est l'interface locale de l'utilisateur ; le polling des tâches
(toutes les 1,5 s) et les lectures sont bon marché. Pas d'en-têtes `X-RateLimit-*`.

### 3.2 Limitation métier (protège le compte de l'utilisateur et les tiers)
| Mécanisme | Valeur | Code | Réf. |
| :--- | :--- | :--- | :--- |
| Volume par collecte | `prudent` 15, `normal` 30, `agressif` 60 éléments par type (offres, posts, profils) | — | I-07, R-1 (E-3 : `max_items` à plafonner) |
| Rythme de navigation | pause `random.uniform(min_delay, max_delay) × facteur` : 6–14 s, 4–9 s, 2,5–6 s | — | `browser.human_pause` |
| Candidatures par jour | `apply_daily_max` = 10 (0–200) | 429 | I-07 |
| Messages par jour (email + LinkedIn cumulés) | `message_daily_max` = 15 (0–200) | 429 | I-07, R-4 |
| Période de repos après blocage | 6 h (`BLOCK_COOLDOWN_HOURS`) | 409 | I-06 |
| Un seul navigateur à la fois | verrou `TaskManager._browser_lock` | tâche `en_attente` | lot 1 `00` § 4 |
| Une tâche par type / clé | `has_active` | 409 | — |
| API France Travail `[CIBLE]` | ≤ 10 appels/s documentés par l'éditeur ; le client se limite à 2 appels/s et respecte `Retry-After` | — | `05` § 2.5 |

Message 429 actuel : `Plafond quotidien atteint (<limite> <candidatures|messages> par jour) : réessaie demain ou
relève-le dans les paramètres.` `[CIBLE]` ajouter `Retry-After` (secondes jusqu'à minuit local).

Relever un plafond reste possible dans les réglages (0–200) : c'est un choix conscient de l'utilisateur, affiché
avec l'avertissement R-1 / R-4. Les volumes de prudence, eux, ne sont **pas** configurables au-delà de `agressif`
(R-2).

### 3.3 Ce qui est exclu
Aucune technique de contournement des protections des plateformes n'est implémentée ni documentée : pas de
résolution de CAPTCHA, pas de maquillage d'empreinte navigateur (cf. R-7 : options retirées de `browser.py` en vague 0), pas de
proxies ni de rotation d'IP, pas de comptes multiples. Un blocage arrête l'action ; seule l'intervention manuelle
de l'utilisateur dans la fenêtre visible du navigateur est attendue (`attente_verification_manuelle`).

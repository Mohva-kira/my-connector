# linkedin_market — directives pour Claude Code

## Le projet en bref
Outil **open source auto-hébergé** d'analyse du marché freelance et d'aide à la candidature : collecte d'offres,
posts et profils (LinkedIn aujourd'hui ; France Travail, Malt, WTTJ, import manuel Indeed en `[CIBLE]`),
tendances sur 30 jours, matching appris sur les annotations, dossiers de candidature, prospection B2B confirmée.
**Une instance = un utilisateur**, sur son poste, avec **son** compte et ses données. Deux points d'entrée qui
appellent les mêmes fonctions de `pipeline.py` : CLI `python -m linkedin_market` et interface web locale
`python -m linkedin_market.web` (FastAPI + SPA React sur **http://127.0.0.1:8765**, jamais exposée ailleurs).

Ce dossier est un sous-projet du dépôt `my-connector`. Le `CLAUDE.md` et le dossier `context/` de la racine
concernent `services/email_analyzer`, **pas** ce projet : pour tout fichier sous `linkedin_market/`, ce fichier et
`linkedin_market/docs/context/` font foi.

## Documentation de contexte (à lire avant de coder)
Chemins relatifs à `linkedin_market/docs/context/`. Le lot 1 (`schemas-architecture/`) est la **source de vérité**
(glossaire, invariants I-01…I-16, risques R-1…R-8) ; les lots backend et frontend précisent où et comment.

| Tâche | Lire, dans l'ordre |
| :--- | :--- |
| Toute tâche | `schemas-architecture/00-domain-architecture-context.md` (glossaire, invariants, risques) |
| Architecture, nouveau module, dépendance | `schemas-architecture/01-system-overview.md` → `02-data-flow-and-events.md` → `04-infrastructure-and-deployment.md` |
| Backend (routeur, service, tâche) | `backend/00-backend-business-context.md` → `01-api-contracts.md` → `02-business-logic-services.md` (§ 4 : écarts **E-1…E-22**) → `05-queues-and-integrations.md` |
| Sécurité, jeton, plafonds, secrets | `backend/04-auth-and-permissions.md` |
| Données, PostgreSQL, migrations, rétention | `schemas-architecture/03-database-erd.md` → `backend/03-database-and-migrations.md` |
| Frontend | `frontend/00-frontend-business-context.md` → `01-ui-design-system.md` → `02-components-catalog.md` → `03-state-and-data-fetching.md` → `04-routing-and-pages.md` → `05-forms-and-validation.md` |

Avant de corriger un comportement, cherche s'il est déjà listé comme écart `E-n` (`backend/02` § 4) ou comme
écart frontend : la correction proposée y figure souvent.

## Commandes (vérifiées depuis la racine du dépôt `my-connector/`)
Le `.venv` et le `.env` sont à la **racine du dépôt** (`config.PROJECT_ROOT`), pas dans `linkedin_market/`. Les
fichiers `profil.*` / `cv.pdf` et `linkedin_market_data/` y sont aussi.

```bash
# Environnement Python (3.14 minimum)
python3.14 -m venv .venv                                   # Windows : .venv\Scripts\python
/usr/bin/python3 -m pip --python .venv/bin/python install -r linkedin_market/requirements-dev.txt   # .venv sans pip ; sinon .venv/bin/pip install -r …
# Dépendances d'exécution : linkedin_market/requirements.txt (incluses par requirements-dev.txt). Le requirements.txt
# de la racine sert à email_analyzer : ne pas l'installer dans ce .venv (il rétrograderait selenium).

# CLI
.venv/bin/python -m linkedin_market -k "Data Engineer" --caution prudent
.venv/bin/python -m linkedin_market -k "Data Engineer" --no-scrape --interpreter local   # sans navigateur

# Interface web
.venv/bin/python -m linkedin_market.web                    # http://127.0.0.1:8765, ouvre le navigateur
.venv/bin/python -m linkedin_market.web --dev --no-open    # sans jeton, Swagger /api/docs, origines :5174 admises

# Frontend (dans linkedin_market/web/frontend)
npm ci                 # Node 22 LTS minimum (vérifié aussi avec Node 24)
npm run dev            # Vite, port 5174, proxy /api -> 127.0.0.1:8765 (lancer l'API en --dev)
npm run build          # tsc --noEmit && vite build -> dist/ servi par FastAPI
npx tsc --noEmit       # contrôle de types seul

# PostgreSQL 18 (base de l'application ; .env racine : LM_DATABASE_URL, LM_DB_PASSWORD, LM_SECRET_KEY)
docker compose --env-file .env -f linkedin_market/compose.yaml up -d db
# LM_DATABASE_URL (et non DATABASE_URL, déjà lu par email_analyzer dans le même .env)

# Tests (hors ligne, ~2 min). Base jetable dont le nom finit par _test, différente de LM_DATABASE_URL ;
# sans LM_TEST_DATABASE_URL, les tests marqués db sont ignorés (la porte qualité exige 0 ignoré).
# Démarrage (CLI et web) : alembic upgrade head -> utilisateur unique -> import des JSON historiques -> tâches non finales en échec.
# LM_DATABASE_URL=<url> .venv/bin/alembic -c linkedin_market/alembic.ini check    # modèles == migrations
LM_TEST_DATABASE_URL=postgresql+psycopg://<user>:<mdp>@127.0.0.1:<port>/linkedin_market_test .venv/bin/python -m pytest linkedin_market/tests
.venv/bin/python -m pytest linkedin_market/tests/test_messages_api.py -q

# Lint Python : config linkedin_market/ruff.toml (py314, 120 colonnes, règles E F W I UP B),
# outils dans linkedin_market/requirements-dev.txt (hors image Docker d'email_analyzer).
# Dette existante au 2026-10-08 : 224 avertissements, 40 fichiers non formatés : ne corrige que ce que tu touches.
.venv/bin/ruff check linkedin_market
.venv/bin/ruff format --check linkedin_market
# Lint front (dans linkedin_market/web/frontend) : ESLint 9 en configuration plate (eslint.config.js).
# Dette : aucune (0 erreur, 0 avertissement depuis la vague 2) ; tout avertissement nouveau bloque.
npm run lint
```
Ne jamais utiliser `--dev` avec de vrais identifiants LinkedIn.

## Invariants non négociables (détail : `schemas-architecture/00` § 2, application : `backend/00` § 2)
Décidé le 2026-10-08 (E-4, option A) : un `run` bloqué pendant la collecte finit `terminée` avec un rapport
partiel et `result.blocked = true` (`backend/01` § 1.2 G-10).

Ils s'appliquent à **tout** point d'entrée : CLI, API, tâches, scripts, tests.
- **I-01** Toute donnée persistée porte `user_id` ; toute requête filtre dessus (PostgreSQL 18, `db/repositories/*`).
- **I-02** Un rapport est immuable : aucune route ni code ne réécrit un rapport existant.
- **I-03** Le statut d'une candidature ne régresse jamais automatiquement (`application.advance_status()`) ;
  seul `PATCH /api/applications/{id}` le change librement.
- **I-04** **Aucune candidature ni aucun message n'est envoyé sans confirmation explicite de l'utilisateur au
  moment de l'envoi** : `confirm_submit` posé pendant la tâche (seule la réponse `Soumettre` envoie), `confirm: true`
  exigé par `POST …/messages`. Une confirmation n'est jamais mémorisée, pré-cochée ni simulée.
- **I-05** **Aucune réponse inventée** : seuls `contact_email`, `contact_phone` et les `answers` mémorisées sont
  remplis ; toute autre question passe par `TaskContext.ask()`. Cases à cocher jamais cochées d'office.
- **I-06** Checkpoint, authwall, CAPTCHA, perte de session ⇒ **arrêt immédiat**, événement `blocked`, repos de 6 h.
  **Aucun contournement** : seule la vérification manuelle par l'utilisateur dans la fenêtre visible est permise.
- **I-07** Plafonds de prudence (`prudent` 15 / `normal` 30 / `agressif` 60 éléments par type) et plafonds
  quotidiens (`apply_daily_max`, `message_daily_max`) jamais dépassés ; contrôle à l'API **et** au début de la tâche.
  Décidé (E-3) : `max_items` ≤ plafond de la prudence choisie, prudence par défaut `prudent` : `CHECK` en
  base et défaut `[ACTUEL]` (vague 1), contrôle 422 à l'API et `min()` dans `pipeline` `[CIBLE]` (vague 2).
- **I-08** Uniquement le propre compte de l'utilisateur : ni faux compte, ni compte partagé, ni rotation.
- **I-09** Minimisation : un profil tiers (`person`) ne garde que titre + compétences ; aucune donnée sensible
  (art. 9 RGPD).
- **I-10** Purge : `post`/`person` à 30 jours, offres sans candidature à 90 jours (`purge_after` posé à l'ingestion ;
  tâche `purge` `[CIBLE]`, vague 6).
- **I-11** Claude ne reçoit **jamais de donnée de tiers identifiable** (profils, posts, recruteurs, contacts) ;
  l'interprétation n'envoie que des agrégats ; le mode local produit toujours une interprétation. Génération de CV
  par Claude (E-6, décidé le 2026-10-09) : profil de l'utilisateur + texte de l'offre **après**
  `nlp.redact_third_parties` (noms, emails, téléphones, URL de profils de personnes remplacés par `[personne]`) `[CIBLE]`.
- **I-12** Prospection B2B (CPCE L.34-5, CNIL) : destinataire joint dans ses fonctions, objet lié à son offre,
  origine de l'adresse et moyen de désinscription simple et gratuit dans chaque message.
- **I-13** Base légale `interet_legitime` + source sur chaque contact ; une entrée de `suppression_list` bloque
  tout envoi, définitivement.
- **I-14** Contact entreprise supprimé 3 ans après le dernier contact.
- **I-15** Identifiants, mots de passe SMTP, clés API, cookies, jeton local : jamais versionnés ni journalisés ;
  cookies valables 72 h.
- **I-16** Source en mode `navigateur` (LinkedIn, Malt, WTTJ) utilisable seulement après acceptation de la version
  courante du texte de risques dans la **popup de risques** (HTTP 428 sinon) `[CIBLE]`. Le CLI avec collecte refuse
  de démarrer sans ce consentement et renvoie vers Paramètres → Sources et consentements (décidé le 2026-10-09).

### Points en arbitrage
Aucun au 2026-10-09 (E-4, désinscription, E-6 tranchés). Désinscription : réponse « STOP » + en-tête
`List-Unsubscribe` mailto, opposition saisie par l'utilisateur (`backend/05` § 2.4). Tout nouveau point qui
touche un invariant ou un risque se soumet à l'utilisateur avant le code.

## Interdits absolus
- N'ajouter **aucune** technique de dissimulation ou de contournement des protections des plateformes :
  résolution de CAPTCHA, maquillage d'empreinte navigateur, options type `--disable-blink-features=AutomationControlled`
  ou `excludeSwitches=["enable-automation"]` (retirées de `browser.py` en vague 0 : R-7 ; ne jamais les réintroduire), proxies ou
  rotation d'IP, comptes multiples, relance automatique après blocage.
- Pas de collecte ni d'appel réseau vers Indeed (import manuel uniquement, R-6) ; pas d'écriture automatisée sur
  Malt / WTTJ.
- Pas d'écoute autre que `127.0.0.1`, pas d'option `0.0.0.0` ni de proxy public ; pas de mode SaaS ni
  multi-utilisateur ; pas d'export de profils tiers.
- Pas de relance automatique d'un envoi (SMTP, message LinkedIn, Easy Apply) : un échec ou un état `incertaine`
  revient à l'utilisateur.
- Sélecteurs Selenium par **texte visible et `aria-label` (FR + EN)**, jamais par classes CSS générées (R-5).

## Conventions de code
**Python**
- Python 3.14, typage complet, ruff (py314, 120 colonnes). Docstrings, commentaires, logs et messages en
  **français** (tutoiement, actionnables : « renseigne SMTP_HOST… »). Les `detail` d'erreur HTTP sont en français
  et affichés tels quels par l'UI ; une exception levée dans une tâche doit porter un message lisible.
- `pathlib` partout, aucun chemin de driver codé en dur (Selenium Manager) : code portable Linux / Windows / macOS.
- Couches : `Entrées → Orchestration → Domaine / Intégrations → Persistance` (`schemas-architecture/01` § 4). Le
  domaine n'importe ni FastAPI ni Selenium ; les routeurs n'importent jamais Selenium et ne portent pas de règle
  métier ; seuls les modules de persistance touchent `DATA_DIR` et la base.
- Noms : uniquement ceux du **glossaire** (`job`, `application`, `source_code`, `caution`, `label`…), jamais
  `offer`, `candidate`, `scrape_run`. Clés JSON en snake_case, énumérations en français sans majuscule.
- Dates : UTC `timestamptz` en base ; à l'API, ISO 8601 UTC suffixé `Z` via `db.utc_iso` (C-1). Jamais de date naïve.
- Persistance : PostgreSQL 18 (SQLAlchemy 2.x, psycopg 3, Alembic). SQL uniquement dans `db/repositories/*` ; toute
  évolution de schéma = nouvelle révision Alembic (`migrations/versions/`), vérifiée par `alembic check`.
- Logs : jamais de secret, de cookie, de corps de message complet ni de texte de profil tiers ; masquer les
  adresses de tiers (E-12).
- Toute précondition vérifiée par un routeur est revérifiée au début de la tâche (plafond, blocage, consentement).

**Frontend** (`linkedin_market/web/frontend/src`)
- TypeScript `strict`, pas de `any`, types d'API dans `api.ts`. État serveur via **TanStack Query** (`hooks.ts`,
  `useAction`), pas de store global supplémentaire.
- Tailwind avec les **jetons sémantiques existants** (`bg-surface`, `text-text-secondary`, `bg-danger/15`…), jamais
  de couleur brute ni d'hexadécimal ; nouvelle couleur = variable dans les deux thèmes de `index.css`.
- Interface en français, tutoiement. Validation client = confort ; le serveur reste seul garant. Schémas Zod dans
  `src/schemas.ts` `[ACTUEL]`, avec les bornes et messages exacts du serveur.
- Toute action d'envoi passe par un écran de relecture ; aucune action irréversible sans clic explicite.

## Tests
- Tout changement de comportement s'accompagne d'un test **pytest** dans `linkedin_market/tests/`.
- Tests **hors ligne** : fixtures HTML (`tests/fixtures/`), doublures ; aucun appel réel à LinkedIn, France
  Travail, Claude ou SMTP. La fixture `conftest.data_dir` isole les données dans `tmp_path`.
- Lancer la suite complète avant de conclure ; `npx tsc --noEmit` (ou `npm run build`) après toute modification du front.

## Marqueurs `[ACTUEL]` / `[CIBLE]`
`[ACTUEL]` = vérifié dans le code ; `[CIBLE]` = décidé, pas encore implémenté (nouvelles sources, popup de risques,
opposition, purge…). PostgreSQL 18 est `[ACTUEL]` depuis la vague 1. Avant d'implémenter un `[CIBLE]`, vérifie
l'existant (le code a pu évoluer) ;
une fois livré, passe la mention en `[ACTUEL]` dans les docs concernées et retire l'écart `E-n` correspondant.

## Maintenance de la documentation
Si une implémentation change l'architecture, un contrat d'API, un schéma, un invariant ou une convention, mets à
jour le fichier de `linkedin_market/docs/context/` concerné **dans le même changement** (lot 1 d'abord s'il s'agit
du glossaire ou d'un invariant). Un nouveau terme métier entre au glossaire avant d'apparaître dans le code.
Garde `linkedin_market/.cursor/rules/linkedin-market.mdc` (règles Cursor) cohérent avec ce fichier.

# Infrastructure, Déploiement & Environnements

## 1. Modèle de Distribution

`linkedin_market` est publié en **open source** et **auto-hébergé** : chaque utilisateur l'installe sur son propre
poste et devient responsable de traitement de ses données (RGPD art. 4(7)). L'éditeur ne fournit **aucun**
serveur, aucun compte, aucune télémétrie. Conséquences d'architecture :
- l'API n'écoute **que** sur `127.0.0.1` ; aucune option ne doit permettre `0.0.0.0` ni l'exposition derrière un
  proxy public ;
- pas d'authentification par mot de passe : le jeton local aléatoire (`secrets.token_urlsafe(24)`, régénéré à
  chaque lancement) suffit, car seul le navigateur de l'utilisateur atteint l'API ;
- le navigateur piloté tourne sur la machine de l'utilisateur, avec son IP et son compte (I-08) ;
- la base PostgreSQL est locale ; aucune donnée ne sort de la machine hormis vers les services choisis
  (sources, Claude si activé, SMTP de l'utilisateur).

Plateformes supportées : Linux (dont Firefox snap sous Ubuntu, via `geckodriver` dans le `PATH`), Windows 10/11,
macOS. Chemins et commandes restent portables (`pathlib`, aucun chemin de driver codé en dur, Selenium Manager).

## 2. Environnements

| Environnement | Lancement | Particularités |
| :--- | :--- | :--- |
| **Développement backend** | `.venv/bin/python -m linkedin_market.web --dev --no-open` | Jeton désactivé, Swagger sur `/api/docs`, origines `http://127.0.0.1:5174` et `http://localhost:5174` autorisées. |
| **Développement frontend** | `cd linkedin_market/web/frontend && npm run dev` (Vite, port 5174, proxy `/api` → 8765) | Rechargement à chaud. |
| **Tests** | `.venv/bin/python -m pytest linkedin_market/tests` | `create_app(token=None)` ; fixtures HTML hors ligne (`tests/fixtures/`) ; aucun appel réseau réel ; base PostgreSQL de test jetable (`LM_TEST_DATABASE_URL`, nom en `_test`, 0 test ignoré exigé) `[ACTUEL]`. |
| **Utilisation (production locale)** | `python -m linkedin_market.web` (ouvre le navigateur) ou `python -m linkedin_market -k "Data Engineer"` | Frontend compilé servi depuis `web/frontend/dist`. |

Il n'y a ni *staging* ni production hébergée : la « mise en production » est la publication d'une version taguée.

## 3. Composants à Installer

| Composant | Version | Installation |
| :--- | :--- | :--- |
| Python | **3.14** minimum | `python3.14 -m venv .venv` puis `/usr/bin/python3 -m pip --python .venv/bin/python install -r linkedin_market/requirements-dev.txt` (ou `.venv/bin/pip install -r linkedin_market/requirements.txt` si le venv a pip) |
| Node.js | 22 LTS minimum, 24 vérifié (build du frontend uniquement) | `npm ci && npm run build` dans `web/frontend` (`tsc --noEmit && vite build`) |
| Navigateur | Chrome / Chromium ou Firefox récent | Selenium Manager télécharge le driver adapté. |
| PostgreSQL `[ACTUEL]` | **18** | Paquet système **ou** `docker compose --env-file .env -f linkedin_market/compose.yaml up -d db` depuis la racine. |
| Dépendances Python clés | `selenium>=4.50,<5`, `fastapi>=0.115,<1`, `uvicorn[standard]>=0.32,<1`, `anthropic>=1.9,<2`, `numpy>=1.26,<3`, `pypdf>=4,<6`, `fpdf2>=2.8,<3`, `python-dotenv>=1,<2`, `sqlalchemy>=2.0,<2.2`, `psycopg[binary]>=3.2,<4`, `alembic>=1.18,<2`, `cryptography>=44,<51` (Fernet) ; dev : `ruff`, `pytest`, `httpx` | `linkedin_market/requirements.txt` `[ACTUEL]` (dev : `linkedin_market/requirements-dev.txt`, qui inclut le premier) |

`compose.yaml` cible (base seule, l'application reste lancée par Python pour garder le navigateur visible) :

```yaml
services:
  db:
    image: postgres:18
    restart: unless-stopped
    environment:
      POSTGRES_DB: linkedin_market
      POSTGRES_USER: linkedin_market
      POSTGRES_PASSWORD: ${LM_DB_PASSWORD:?définir LM_DB_PASSWORD dans .env}
    ports:
      - "127.0.0.1:5432:5432"   # jamais exposé hors de la machine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U linkedin_market -d linkedin_market"]
    volumes:
      - lm_pgdata:/var/lib/postgresql
volumes:
  lm_pgdata:
```

Au démarrage, l'application exécute `alembic upgrade head`, puis la migration d'import JSON si des fichiers
`linkedin_market_data/*.json(l)` non migrés existent (cf. `03-database-erd.md` § 4).

## 4. Variables d'Environnement

Lues depuis `.env` à la racine du dépôt (jamais versionné ; `.env.example` versionné sans valeurs).

| Variable | Description | Exemple / Format | Obligatoire |
| :--- | :--- | :--- | :--- |
| `LINKEDIN_EMAIL` | Identifiant LinkedIn de l'utilisateur | `prenom.nom@exemple.fr` | Pour les actions navigateur (sinon demandé en CLI ; API ⇒ 409) |
| `LINKEDIN_PASSWORD` | Mot de passe LinkedIn | — | Idem |
| `LINKEDIN_BROWSER` | Navigateur piloté | `auto` \| `chrome` \| `firefox` (défaut `auto`) | Non |
| `LINKEDIN_BROWSER_BINARY` | Chemin d'un navigateur non standard | `/usr/bin/chromium` | Non |
| `LINKEDIN_HEADLESS` | Masquer la fenêtre | `0` (défaut, recommandé) \| `1` | Non |
| `LINKEDIN_INTERPRETER` | Interprète par défaut (CLI) | `claude` (défaut) \| `local` | Non |
| `ANTHROPIC_API_KEY` | Clé API Claude | `sk-ant-…` | Non (sinon interprète local) |
| `ANTHROPIC_MODEL` | Modèle Claude | `claude-opus-5-5` (défaut) | Non |
| `LINKEDIN_MARKET_DATA_DIR` | Dossier des données locales | `linkedin_market_data/` (défaut) | Non |
| `LINKEDIN_UI_TOKEN` | Jeton local fixe (sinon aléatoire à chaque lancement) | 32+ caractères aléatoires | Non |
| `SMTP_HOST` | Serveur SMTP | `smtp.exemple.fr` | Pour l'envoi d'emails |
| `SMTP_PORT` | Port | `587` (STARTTLS) / `465` (SSL) | Non (déduit de `SMTP_SECURITY`) |
| `SMTP_SECURITY` | Chiffrement | `starttls` (défaut) \| `ssl` ; `none` accepté aujourd'hui mais à supprimer (E-20) | Non |
| `SMTP_USER`, `SMTP_PASSWORD` | Authentification SMTP | — | Selon le serveur |
| `SMTP_FROM` | Expéditeur | `Prénom Nom <prenom@exemple.fr>` | Non (défaut `SMTP_USER`) |
| `LM_DATABASE_URL` | Connexion PostgreSQL (préfixe `LM_` : `DATABASE_URL` est réservé à email_analyzer, même `.env`) | `postgresql+psycopg://linkedin_market:***@127.0.0.1:5432/linkedin_market` | Oui |
| `LM_DB_PASSWORD` | Mot de passe du conteneur `db` | Chaîne aléatoire | Avec `compose.yaml` |
| `LM_SECRET_KEY` | Clé Fernet de chiffrement des cookies de session | `Fernet.generate_key()` (44 caractères base64) | Oui ; générée au premier lancement si absente |
| `FRANCE_TRAVAIL_CLIENT_ID`, `FRANCE_TRAVAIL_CLIENT_SECRET` `[ACTUEL]` | Application déclarée sur francetravail.io, abonnée à l'API « Offres d'emploi » | — | Pour la source `france_travail` |

## 5. Sécurité du Poste

- **Réseau** : Uvicorn sur `127.0.0.1:8765` ; PostgreSQL publié sur `127.0.0.1` uniquement.
- **Requêtes** : middleware `guard` — `Host` ∈ {`127.0.0.1`, `localhost`, `[::1]`} sinon `403` ; `Origin` identique
  à `Host` pour les méthodes mutatives sinon `403` (protection DNS-rebinding / CSRF) ; jeton comparé en temps
  constant (`hmac.compare_digest`) sinon `401`.
- **Secrets** : `.env`, `linkedin_market_data/`, `web/frontend/node_modules/`, `web/frontend/dist/` et
  `tests/fixtures/_work/` sont dans `.gitignore`. Les journaux ne contiennent ni mot de passe, ni cookie, ni corps de
  message complet.
- **Sauvegarde** : `pg_dump -Fc linkedin_market > sauvegarde.dump` (documenté dans le README) ; le dossier
  `linkedin_market_data/` contient rapports et dossiers de candidature à sauvegarder avec.

## 6. Pipeline CI/CD (GitHub Actions `[CIBLE]`)

Déclenché sur chaque *pull request* et sur `main` :

1. **Lint & format Python** : `ruff check linkedin_market` puis `ruff format --check linkedin_market`
   (`target-version = "py314"`, `line-length = 120`).
2. **Lint & types frontend** : `npm ci`, `npm run lint` (ESLint 9, configuration plate), `tsc --noEmit`.
3. **Tests** : `pytest linkedin_market/tests` avec un service `postgres:18` ; tests hors ligne (aucun accès à
   LinkedIn, France Travail, Claude ou SMTP réels — doublures et fixtures HTML).
4. **Migrations** : `alembic upgrade head` puis `alembic check` (aucune différence entre modèles et migrations).
5. **Build** : `npm run build` (le `dist/` est joint aux artefacts de version, pas versionné).
6. **Publication** : sur tag `vX.Y.Z` (SemVer), création d'une *release* GitHub avec archive source + `dist/`
   compilé et notes de version. Aucun déploiement automatique : chaque utilisateur met à jour son instance
   (`git pull`, `pip install -r linkedin_market/requirements.txt`, redémarrage ⇒ migrations appliquées au lancement).

# linkedin_market — analyse du marché LinkedIn pour un poste

Outil autonome (indépendant de `services/email_analyzer`) pour un freelance
full remote : il collecte offres, posts et profils LinkedIn pour un mot-clé
de poste, en tire les tendances sur 30 jours, mesure l'écart avec ton profil
et classe les offres avec un modèle de matching entraîné par descente de
gradient.

## Lancer

```bash
.venv/bin/python -m linkedin_market            # prompt de saisie
.venv/bin/python -m linkedin_market --update          # met à jour la dernière recherche (nouvelles offres), sans question
.venv/bin/python -m linkedin_market -k "Data Engineer" --caution prudent
.venv/bin/python -m linkedin_market -k "Data Engineer" --no-scrape --label 10   # réanalyse + annotation
```

Sous Windows : `.venv\Scripts\python -m linkedin_market`.

## Configuration (`.env` à la racine, jamais versionné)

| Variable | Rôle | Défaut |
|---|---|---|
| `LINKEDIN_EMAIL`, `LINKEDIN_PASSWORD` | identifiants (sinon demandés, mot de passe masqué) | — |
| `LINKEDIN_BROWSER` | `auto`, `chrome`, `firefox` | `auto` |
| `LINKEDIN_BROWSER_BINARY` | chemin du navigateur s'il n'est pas standard | — |
| `LINKEDIN_HEADLESS` | `1` pour masquer le navigateur | `0` |
| `LINKEDIN_INTERPRETER` | `claude`, `local`, `none` | `claude` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | interprétation par Claude | `claude-opus-5-5` |
| `LINKEDIN_MARKET_DATA_DIR` | dossier des données | `linkedin_market_data/` |

## Profil

Dépose à la racine du projet l'un de : `profil.json`, `profil.md`,
`profil.txt`, `cv.pdf` — ou choisis la source `linkedin` (lecture de ton
profil et de `/details/skills/`). Format JSON :

```json
{"title": "Ingénieur logiciel freelance", "skills": ["Python", "FastAPI", "AWS"],
 "years_experience": 10, "seniority": "senior", "text": "Résumé libre..."}
```

## Fonctionnement

1. **Session** : cookies conservés 72 h après la dernière connexion par
   identifiants, puis supprimés et écrasés à la connexion suivante.
2. **Collecte** (`scraper.py`) : offres (`f_WT=2` full remote, 60 derniers
   jours, 100 par défaut, lues par lots de 15 : chaque lot est enregistré et scoré pendant une pause de 2 min, on attend la fin du traitement s'il est plus long), posts (dernier mois, triés par date), profils (recherche
   personnes). Volume et pauses selon le niveau `prudent` /
   `normal` / `agressif` (100 offres chacun ; la prudence règle les pauses entre pages). Arrêt immédiat sur checkpoint/authwall.
3. **Historique** (`history.jsonl`) : cumulé d'une exécution à l'autre, il
   rend la mesure des tendances plus fiable avec le temps.
4. **Extraction locale** (`nlp.py`) : n-grammes, TF-IDF NumPy, vocabulaire
   de compétences, séniorité, années, télétravail.
5. **Tendances** (`trends.py`) : termes fréquents, compétences par source,
   croissance 0-30 j vs 30-60 j, série hebdomadaire.
6. **Matching** (`matching.py`) : 8 critères par offre, régression
   logistique par descente de gradient (L2, classes repondérées). Poids a
   priori jusqu'à 6 annotations des deux types ; ensuite modèle appris,
   sauvegardé dans `matching_model.json`. On te fait annoter les offres
   les plus incertaines (score proche de 50 %).
7. **Interprétation** (`interpreter.py`) : Claude reçoit uniquement les
   agrégats ; `LINKEDIN_INTERPRETER=local` le remplace par un interprète à
   base de règles qui s'appuie sur les poids appris.
8. **Rapport** : `reports/<mot-cle>-<date>.html` (graphiques) et `.json`
   (même structure, prévu pour une future application).
9. **Candidature** (`application.py`) : après le rapport (ou directement
   avec `--apply`), les offres pertinentes — annotées « p », puis celles
   dont le score dépasse `--apply-min-score` (70 % par défaut), 10 au
   maximum — sont affichées en fiches courtes :

   ```
   [1]  86.4 %  Senior Python Developer (Freelance) — PayAfrica · Remote · Worldwide
                Freelance · Remote · Senior · 6+ ans · publiée le 2026-10-06
                ✓ aws, docker, fastapi, python, sql
                ✗ kubernetes, langchain, rag
                › You will design and build scalable REST APIs with FastAPI and PostgreSQL…
                › Requirements: 6+ years of experience with Python, Docker and Kubernetes…
                https://www.linkedin.com/jobs/view/999/
   ```

   Pour chaque offre choisie (`1,3`, `2-4`, `tous`), un dossier
   `applications/<date>-<entreprise>-<poste>/` contient `cv.html`
   (imprimable), `cv.md` (CV au format ATS : renseigne ton nom et ton
   profil LinkedIn dans Paramètres → Coordonnées du CV ; exports PDF et
   Word nommés `Prénom-Nom-CV-Poste` depuis l'interface), `message.txt` (message de candidature),
   `conseils.md` (écarts à traiter) et `offre.txt`. Le CV et l'offre
   s'ouvrent dans le navigateur ; **en ligne de commande, tu postules
   toi-même** (l'outil n'envoie rien). L'interface web, elle, peut postuler et
   contacter le recruteur sur ta demande (voir ci-dessous). Le statut (`préparée` / `postulée`) est suivi dans
   `applications.json`, et une offre où tu as postulé compte comme
   « pertinente » pour l'entraînement.

   - Avec Claude : CV réécrit pour l'offre (langue de l'offre, mots-clés
     ATS, compétences en hausse du profil mises en avant), sans rien
     inventer ; les compétences manquantes vont dans les conseils, pas
     dans le CV. **Ton profil complet et le texte de l'offre sont envoyés
     à l'API Anthropic** (contrairement à l'interprétation, qui n'envoie
     que des agrégats).
   - En local : CV assemblé (accroche, atouts ciblés, profil source
     inchangé), message en français.

## Limites

- Les sélecteurs LinkedIn changent : en cas d'échec, l'extraction se rabat
  sur le texte brut de la page, moins précis.
- Les dates sont approximées depuis « il y a 2 semaines ».
- Le scraping automatisé est contraire aux conditions d'utilisation de
  LinkedIn : garde des volumes faibles.

## Interface web (gestion)

```bash
# Dépendances Python : requirements.txt (FastAPI, uvicorn, fpdf2 inclus). Front :
cd linkedin_market/web/frontend && npm install && npm run build   # une fois, puis après chaque changement du front
.venv/bin/python -m linkedin_market.web                        # http://127.0.0.1:8765
```

Développement du front : `python -m linkedin_market.web --dev` puis `npm run dev` (port 5174).
Tests : `.venv/bin/python -m pytest linkedin_market/tests`.

Pages : **Tableau de bord** (quotas du jour, entonnoir, activité, tâches en direct), **Offres**
(bouton **Mettre à jour**, annotation 👍/👎, entraînement, recalcul des scores, préparation d'une candidature),
**Candidatures** (message, CV et conseils éditables, statut, **Postuler**, contacts, **écrire à
l'entreprise**), **Marché** (tendances et écart de compétences), **Compétences** (ajouter/retirer
des compétences, suggestions du marché, « Recalculer les scores » — tes fichiers `profil.*` ne sont
jamais modifiés), **Paramètres** (valeurs par défaut, plafonds, email/téléphone de candidature,
réponses mémorisées).

Une compétence ajoutée dans l'outil (« J'ai » sur la page Marché, page Compétences) est aussi
ajoutée à ton **profil LinkedIn** par une tâche de fond (`profile_sync.py`) : nom officiel proposé
par LinkedIn quand il correspond, rien n'est fait si la compétence y figure déjà, rien n'est jamais
retiré de LinkedIn. Désactivable dans les Paramètres (`linkedin_skill_sync`).

Variables `.env` en plus : `SMTP_HOST`, `SMTP_PORT`, `SMTP_SECURITY` (`starttls`/`ssl`/`none`),
`SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` (envoi d'emails) ; `LINKEDIN_EMAIL` et `LINKEDIN_PASSWORD`
sont obligatoires dans l'interface (aucune saisie dans le terminal). Une vérification manuelle
LinkedIn (captcha, 2FA) demande un navigateur visible (`LINKEDIN_HEADLESS=0`).

### Postuler et contacter

- **Postuler** : « Candidature simplifiée » uniquement, pilotée par Selenium. L'outil remplit l'email et
  le téléphone des paramètres, téléverse le CV adapté (PDF) et réutilise tes réponses mémorisées ;
  **toute autre question te est posée** (rien n'est deviné). À l'écran de relecture, tu vois ce qui part
  et tu confirmes ou tu annules. Offre sans candidature simplifiée : statut `externe`, tu postules sur
  le site de l'entreprise.
- **Contacts** : « Chercher les recruteurs » lit le bloc « équipe de recrutement » de l'offre et les
  coordonnées que LinkedIn t'affiche ; l'email n'est récupéré que s'il est visible pour toi.
- **Message** : email (SMTP, CV en pièce jointe) ou message LinkedIn (message direct si la personne est
  dans tes relations, sinon demande de connexion avec une note de 300 caractères max). Chaque envoi
  exige une relecture et une confirmation.
- **Garde-fous** : plafonds quotidiens (candidatures, messages), pauses aléatoires, arrêt et repos de
  6 h si LinkedIn impose un checkpoint, API accessible uniquement depuis `127.0.0.1`.

⚠️ L'automatisation de LinkedIn (collecte, candidatures, messages) est contraire à leurs conditions
d'utilisation : ton compte peut être restreint. Les sélecteurs de LinkedIn changent régulièrement ; les
tests de ce dépôt valident la logique sur des pages HTML locales, pas sur LinkedIn lui-même — fais une
première candidature et un premier message de contrôle à la main avant de t'y fier.

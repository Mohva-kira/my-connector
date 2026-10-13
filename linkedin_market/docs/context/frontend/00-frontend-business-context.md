# Contexte Métier Frontend

> **Périmètre** : interface web locale de `linkedin_market` (SPA React servie par FastAPI sur
> http://127.0.0.1:8765, code dans `web/frontend/`). Source de vérité du domaine : lot 1
> (`docs/context/schemas-architecture/00` à `04`) — glossaire, invariants **I-01…I-16**, risques **R-1…R-8**.
>
> **Marqueurs** : `[ACTUEL]` = vérifié dans le code de `web/frontend/src` ; `[CIBLE]` = décidé, pas encore
> implémenté (au 2026-10-13 : bandeau 403, lien « Régler les plafonds », message 500 générique ; popup de risques, sources
> France Travail / Malt / WTTJ / Indeed et opposition sont `[ACTUEL]`).
>
> **Langue** : toute l'interface est en **français** (`<html lang="fr">`, libellés, messages d'erreur) et tutoie
> l'utilisateur (« Lance une première analyse », « Vérifie ton compte ») `[ACTUEL]`. Ce ton est conservé.

## 1. Périmètre & Utilisateurs Cibles (Personas)

L'outil est **open source et auto-hébergé** : une instance = un utilisateur, sur son propre poste
(cf. `04-infrastructure-and-deployment.md` § 1). Il n'y a ni inscription, ni connexion, ni rôles : l'accès est
protégé par un **jeton local** (`secrets.token_urlsafe(24)`, régénéré à chaque lancement) que le serveur injecte
dans `index.html` (`<meta name="lm-token">`) et que `api.ts` renvoie dans l'en-tête `X-Local-Token` `[ACTUEL]`.

- **Persona unique — le freelance full remote (`user`)** :
  - **Profil & contexte d'usage** : développeur / data / consultant indépendant qui cherche des missions en
    télétravail complet. Utilise l'interface sur **ordinateur** (navigateur de bureau, souvent écran large), à côté
    de la fenêtre du navigateur **piloté** que l'outil ouvre pour LinkedIn (visible par défaut,
    `LINKEDIN_HEADLESS=0`). Sessions courtes et répétées : une analyse par jour ou par semaine, puis tri des offres
    et préparation de quelques candidatures.
  - **Objectifs principaux** :
    1. comprendre son marché (tendances sur 30 jours, compétences demandées, écart de compétences) — page **Marché** ;
    2. trier et **annoter** les offres (pertinente / non pertinente) pour entraîner le modèle de matching — page **Offres** ;
    3. ajuster son **profil candidat** (compétences ajoutées / retirées, titre, années, séniorité) — page **Compétences** ;
    4. **préparer** puis **envoyer** des candidatures et des messages B2B aux recruteurs, toujours après relecture
       (I-04) — page **Candidatures** ;
    5. surveiller quotas, blocages et tâches en cours — **Tableau de bord**.
  - **Contraintes d'ergonomie** :
    - doit **voir et décider** à chaque étape sensible : aucune soumission silencieuse (I-04), aucune réponse
      inventée (I-05) ;
    - doit pouvoir intervenir **dans la fenêtre du navigateur piloté** (vérification manuelle d'un checkpoint,
      statut `attente_verification_manuelle`) sans perdre le fil dans l'interface ;
    - accepte des tâches longues (plusieurs minutes, pauses de 6–14 s en `prudent`) : l'UI ne doit jamais bloquer,
      le suivi se fait dans le **tiroir des tâches** ;
    - mode **sombre** apprécié (sessions du soir) : bascule disponible `[ACTUEL]` ;
    - utilisation au clavier possible et contrastes conformes WCAG 2.2 AA (cf. `01-ui-design-system.md` § 4) ;
    - largeur mobile non prioritaire mais **tolérée** (navigation horizontale défilante sous `md`) `[ACTUEL]`.

- **Persona secondaire (indirect) — le recruteur / l'entreprise** : n'utilise pas l'interface, mais reçoit les
  messages. Ses droits imposent des contraintes visibles dans l'UI : origine de l'adresse, mention de
  désinscription, respect de l'opposition (I-12, I-13, R-4) `[ACTUEL]` (vague 4).

## 2. Parcours Clés & Actions Utilisateur

Les noms d'actions reprennent la matrice du lot 1 (`00-domain-architecture-context.md` § 3).

### Parcours 0 : Accepter les risques d'une source navigateur (`AccepterRisquesSource`) `[ACTUEL]` (vague 3)
- **Déclencheur** : première action qui utilise une source en mode `navigateur` (`linkedin`, `malt`, `wttj`) —
  « Lancer » une analyse avec collecte, « Lancer la candidature », « Chercher les recruteurs », message LinkedIn,
  « Importer depuis LinkedIn », ajout d'une compétence synchronisée ; ou réponse **HTTP 428** de l'API ; ou
  changement de `risk_version` (I-16).
- **Étapes** :
  1. l'UI lit `GET /api/sources` ; si `consent` est absent ou porte une autre `risk_version`, la **popup de
     risques** s'ouvre avant tout appel mutatif ;
  2. l'utilisateur lit le texte (suspension de compte, CGU LinkedIn § 8.2 version du 3 nov. 2025, RGPD, plafonds,
     conseil de commencer en `prudent`), coche « J'ai lu et j'accepte ces risques pour mon compte » ;
  3. clic sur « J'accepte les risques » → `POST /api/sources/{code}/consent` → l'action initiale reprend ;
  4. refus (bouton « Refuser » ou touche Échap) → l'action est abandonnée, aucune requête mutative n'est envoyée.
- **Feedback** : bouton d'acceptation désactivé tant que la case n'est pas cochée ; spinner sur le bouton pendant
  l'enregistrement ; erreur affichée dans la popup (la case reste cochée). Spécification complète :
  `02-components-catalog.md` § 4.1.

### Parcours 1 : Lancer une analyse du marché (`LancerAnalyse` / `ReanalyserHistorique`)
- **Déclencheur** : bouton « Lancer une analyse » (Tableau de bord, Offres) ou « Nouvelle analyse » (Marché) `[ACTUEL]`.
- **Étapes** :
  1. `RunModal` s'ouvre, pré-rempli avec les **Paramètres** (`GET /api/settings`) : mot-clé, localisation,
     prudence, « Full remote uniquement » `[ACTUEL]` ;
  2. case « Collecter de nouvelles données sur LinkedIn » cochée par défaut ; décochée = réanalyse de l'historique
     sans navigateur (`scrape: false`) `[ACTUEL]` ;
  3. `[ACTUEL]` choix des **sources** (`SourcePicker` : LinkedIn / Malt / WTTJ en `navigateur`, France Travail en
     `api_officielle`) avec l'état du consentement de chacune ; popup de risques si nécessaire (Parcours 0) ;
  4. « Lancer » → `POST /api/runs` → **202** + tâche → le **tiroir des tâches** s'ouvre sur cette tâche `[ACTUEL]`.
- **Feedback** : bouton « Lancer » désactivé si le mot-clé est vide, libellé remplacé par « … » pendant l'appel ;
  avertissement CGU visible quand la collecte est cochée `[ACTUEL]` ; suivi des statuts de tâche (polling 1,5 s)
  puis rafraîchissement global des données quand la tâche devient finale `[ACTUEL]`.

### Parcours 2 : Trier et annoter les offres (`AnnoterOffre`, `EntrainerModele`, `RecalculerScores`)
- **Déclencheur** : page **Offres** (tableau filtrable) ou section « Offres à cibler » de **Marché** (cartes).
- **Étapes** : filtrer (texte, score minimal par pas de 5, annotation, statut de candidature) → 👍 / 👎
  (`PUT /api/jobs/{id}/label`, re-clic = retrait de l'annotation) → « réentraîne le modèle »
  (`POST /api/model/train`) → « Recalculer les scores » (`POST /api/runs/rescore`) `[ACTUEL]`.
- **Feedback** : bouton d'annotation actif coloré (`primary` / `danger` dans Offres, `success` / `warning` dans
  Marché) ; bandeau « Poids a priori : N annotation(s), il en faut au moins 6 des deux types » ou « Modèle appris
  sur N annotations (précision X %) » `[ACTUEL]`.

### Parcours 3 : Ajuster le profil candidat (`ModifierProfil`, `ImporterProfilLinkedIn`)
- **Déclencheur** : page **Compétences**, ou bouton « J'ai » sur une compétence manquante de **Marché** `[ACTUEL]`.
- **Étapes** : ajouter / retirer / remettre une compétence ; modifier titre, années (0–60), séniorité ;
  « Annuler mes ajustements » ; « Importer depuis LinkedIn » (tâche navigateur) `[ACTUEL]`.
- **Feedback** : badge « ajoutée » sur les compétences de surcharge ; bandeau « Tes compétences ont changé depuis la
  dernière analyse : les scores affichés sont périmés » (`scores_outdated`) ; si `linkedin_skill_sync` est actif,
  une tâche `profile_skill` est suivie dans le tiroir, sinon un bandeau explique pourquoi LinkedIn n'est pas
  modifié `[ACTUEL]`.

### Parcours 4 : Préparer et envoyer une candidature simplifiée (`PreparerCandidature`, `PostulerEasyApply`)
- **Déclencheur** : « Préparer la candidature » (détail d'une offre) puis onglet **Postuler** de la candidature.
- **Étapes** :
  1. `POST /api/jobs/{id}/prepare` → tâche `prepare` → dossier (CV adapté, message, conseils) éditable dans
     l'onglet **Dossier** (`PATCH /api/applications/{id}`) `[ACTUEL]` ;
  2. « Lancer la candidature » → `POST /api/applications/{id}/apply` (désactivé si identifiants absents, blocage
     actif, plafond `apply_daily_max` atteint ou statut déjà `postulée` / `entretien`) `[ACTUEL]` ;
  3. chaque question inconnue arrive en `attente_saisie` : le tiroir s'ouvre, l'utilisateur répond (réponse
     **mémorisée**) — I-05 `[ACTUEL]` ;
  4. à l'écran de relecture LinkedIn, la tâche pose `confirm_submit` : la **modale « Relire avant l'envoi »**
     affiche le texte relu et les boutons « Soumettre » / « Annuler » ; Échap ou clic hors de la modale = « Annuler »
     ; la réponse est envoyée avec `remember: false` (jamais mémorisée) — I-04 `[ACTUEL]`.
- **Feedback** : état `apply_state` affiché (« Envoyée », « Envoi non confirmé », « Échec ») ; bandeau dédié pour
  `incertaine` (« Vérifie sur LinkedIn avant de réessayer ») et pour `externe` `[ACTUEL]`.

### Parcours 5 : Contacter l'entreprise (`ChercherRecruteurs`, `EnvoyerMessage`, `EnregistrerOpposition`)
- **Déclencheur** : onglet **Contacts** d'une candidature.
- **Étapes** :
  1. « Chercher les recruteurs de l'offre » (tâche `contacts`) ou « Ajouter un contact » à la main puis
     « Enregistrer les contacts » `[ACTUEL]` ;
  2. « Écrire » / « Écrire à l'entreprise » → `ComposeModal` : canal Email (SMTP) ou Message LinkedIn, brouillon
     chargé depuis `GET /api/applications/{id}/message-draft` `[ACTUEL]` ;
  3. « Relire avant l'envoi » → **écran de relecture** (canal, destinataire, objet, pièce jointe, message) →
     « Confirmer l'envoi » → `POST …/messages` avec `confirm: true` `[ACTUEL]` ;
  4. `[ACTUEL]` (vague 4) l'écran de relecture affiche aussi l'**origine de l'adresse**, la **mention de désinscription**
     imposée (non supprimable), exige la case « J'ai relu ce message et je confirme l'envoi », et **bloque** l'envoi
     vers une adresse présente dans la liste d'opposition (I-12, I-13) ;
  5. `[ACTUEL]` (vague 4) « Ce contact refuse d'être recontacté » enregistre une **opposition** définitive.
- **Feedback** : compteur de caractères, avertissement au-delà de 300 caractères en LinkedIn (note de connexion
  tronquée), bandeau « Plafond quotidien de messages atteint (x / y) », bandeau SMTP non configuré `[ACTUEL]`.

### Parcours 6 : Lever un blocage (`LeverBlocage`)
- **Déclencheur** : bandeau rouge global « LinkedIn a interrompu la navigation le … » (`status.blocked`) `[ACTUEL]`.
- **Étapes** : vérifier son compte dans un navigateur normal, puis lien « lève le blocage » → `POST /api/block/clear`.
- **Feedback** : le bandeau disparaît après rafraîchissement de `["status"]` `[ACTUEL]`. L'UI ne propose **aucun**
  moyen de contourner la vérification (I-06, R-1).

### Parcours 7 : Importer une offre Indeed à la main `[ACTUEL]` (vague 5)
- **Déclencheur** : bouton « Importer une offre » de la page **Offres**.
- **Étapes** : formulaire URL de l'offre + texte collé (+ titre, entreprise, localisation facultatifs) → création
  d'un élément `source_code = indeed`, mode `import_manuel` (R-6) → l'offre apparaît après « Recalculer les scores ».
- **Feedback** : validation du formulaire (cf. `05-forms-and-validation.md` § 3.7), notice rappelant qu'aucune
  collecte automatique n'est faite sur Indeed.

## 3. Résultats Finaux Attendus (Expected Client Outcomes)

- **États visuels de succès** :
  - `[ACTUEL]` pas de système de *toast* : les succès s'affichent par un `Notice` vert contextuel
    (« Paramètres enregistrés. », « Envoyé à {adresse}. », « Ajoutée à ton profil : … ») ou par le passage de la
    tâche au statut `terminée` dans le tiroir (badge vert, `result` JSON affiché).
  - `[ACTUEL]` à la fin d'une tâche, `TaskProvider` invalide les requêtes selon son type (`INVALIDATE_BY_KIND`,
    `03` § 2.3) : listes, profil, monitoring et rapports concernés se rafraîchissent sans rechargement.
  - `[ACTUEL]` navigation après action : « Voir la candidature » mène à `/candidatures/{jobId}` ; l'ouverture de
    `/candidatures` sélectionne automatiquement la première candidature.
  - `[ACTUEL]` (vague 6, `InterpreterBadge`) le rapport affiché indique toujours **quel interprète** a produit l'interprétation (« Lecture rédigée
    par Claude » / « Lecture rédigée par l'interprète local ») — l'interprétation n'est jamais vide (I-11).
- **Résilience & gestion des échecs** :
  - `[ACTUEL]` les erreurs API sont affichées via `ErrorBox` (`role="alert"`) avec le `detail` français renvoyé
    par FastAPI ; les erreurs de validation Pydantic (tableau) sont concaténées par « · ».
  - `[ACTUEL]` la saisie est préservée en cas d'erreur : formulaires en état local, non réinitialisés sur échec
    (`RunModal`, `ComposeModal`, Paramètres, Contacts).
  - `[ACTUEL]` partiel : 401 (jeton périmé → « Recharge la page », polling suspendu), 428 (popup de risques), 429
    (`detail` + « Réessaie dans X h ») et API injoignable (`ApiError` de statut 0) sont traités ; 403, lien « Régler les
    plafonds » et message 500 générique restent `[CIBLE]` — cf. `03-state-and-data-fetching.md` § 3.
  - `[ACTUEL]` (vague 2, `src/schemas.ts`) aucun texte technique brut : les messages `422` de `settings.py` (« apply_daily_max doit être compris
    entre 0 et 200. ») sont précédés d'une validation client qui affiche le libellé du champ.
- **Accessibilité & conformité** :
  - navigation complète au clavier sur les parcours critiques (popup de risques, relecture, réponse à une tâche) ;
    piège de focus et restitution du focus dans toutes les modales `[ACTUEL]` (`Modal` sur `<dialog>` + `showModal()`) ;
  - aucune action irréversible sans relecture (I-04) ; aucun envoi vers une adresse opposée (I-13) ;
  - aucune fonctionnalité d'UX ne propose de contourner une protection de plateforme (CAPTCHA, empreinte, proxy,
    comptes multiples — I-06, I-08, R-7).

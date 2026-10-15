# Formulaires & Validation

> Vérifié dans `pages/SettingsPage.tsx`, `pages/Skills.tsx`, `pages/Applications.tsx`, `pages/Jobs.tsx`,
> `components/RunModal.tsx`, `components/ComposeModal.tsx`, `components/TaskDrawer.tsx`, et côté serveur
> `settings.py` (`validate_settings`, `save_profile_settings`), `web/routers/*.py` (modèles Pydantic),
> `outreach.py`. Les bornes ci-dessous sont celles du code serveur, **seul garant** des règles (lot 1,
> `01-system-overview.md` § 5) ; la validation client n'est qu'un confort.

## 1. Bibliothèques et Standards

- **Gestionnaire de formulaire** `[ACTUEL]` : **aucun**. Formulaires contrôlés à la main (`useState` + `onChange`),
  soumission par bouton (`onClick`) ; un seul `<form onSubmit>` (ajout de compétence, page Compétences).
- **Schéma de validation** `[ACTUEL]` : **aucun** (ni Zod, ni Yup). Validation client limitée aux attributs HTML
  (`type="number" min max`, `type="email"`) — **non bloquants**, car la soumission ne passe pas par un `<form>`
  natif — et à quelques conditions `disabled` (champ requis vide). Toute autre erreur vient du serveur (422) et
  s'affiche dans l'`ErrorBox` globale du formulaire, avec le nom technique du champ
  (« apply_daily_max doit être compris entre 0 et 200. »).
- **Depuis la vague 2** `[ACTUEL]` : **Zod 4** (`src/schemas.ts`, § 4) ; erreurs par champ via `Field error` +
  `fieldErrors()`. Le hook `useZodForm` décrit ci-dessous n'a pas été retenu (validation au rendu par
  `safeParse`, bouton désactivé tant que le schéma échoue).
- **Conception d'origine** : **Zod 4** pour les schémas (fichier `src/schemas.ts`), types inférés
  (`type SettingsForm = z.infer<typeof settingsSchema>`), messages en français portant le **libellé du champ**.
  Pas de React Hook Form : les formulaires sont petits ; un hook maison `useZodForm(schema, initial)` (valeurs,
  erreurs par champ, `validate()` à la soumission, revalidation au changement d'un champ en erreur) suffit.
  Les schémas reprennent **exactement** les bornes serveur ; si une borne change dans `settings.py`, elle change
  ici dans la même PR.

## 2. Règles de Validation UX

- **Validation à la première soumission** : aucune erreur affichée avant le premier clic sur « Enregistrer » /
  « Lancer » / « Relire avant l'envoi » ; les boutons restent toutefois désactivés si un champ **requis** est vide
  (comportement actuel conservé : `disabled={!value("keyword").trim()}`, `disabled={!ready}`).
- **Revalidation en direct** : après une première tentative, un champ en erreur est revalidé à chaque changement
  (`onChange`) ; les autres au `onBlur`.
- **Affichage** : message sous le champ (`Field error`), `aria-invalid="true"`, `aria-describedby` vers le message
  (critère WCAG 3.3.1 *Error Identification*, A) ; résumé en tête de formulaire si plusieurs erreurs, avec focus
  sur le premier champ fautif (3.3.3 *Error Suggestion*, AA : le message dit comment corriger).
- **Erreurs serveur** : un 422 qui correspond à un champ connu est rattaché au champ (table de correspondance
  clé serveur → libellé : `apply_daily_max` → « Candidatures / jour », etc.) ; sinon `ErrorBox`.
- **Préservation de la saisie** : jamais de réinitialisation sur erreur (déjà le cas) ; les valeurs numériques
  vides ne sont pas transformées silencieusement en `0` `[ACTUEL]` (`num("")` donne `NaN`, refusé par `settingsSchema`, cf. § 3.1).
- **Prévention des erreurs pour les actions engageantes** (3.3.4, AA) : envoi de message et soumission de
  candidature passent par une étape de relecture réversible (« Modifier » / « Annuler ») — I-04.

## 3. Formulaires

### 3.1 Paramètres (`SettingsPage`, `PUT /api/settings`)

Formulaire entièrement chargé depuis `GET /api/settings`, envoyé **en entier** par « Enregistrer » (en-tête de
page). Succès : Notice « Paramètres enregistrés. » ; erreur : `ErrorBox` (en haut).

| Champ (libellé UI) | Clé | Contrôle actuel | Règle serveur (`settings.validate_settings`) | Message serveur |
| :--- | :--- | :--- | :--- | :--- |
| Mot-clé du poste | `keyword` | texte | chaîne, `strip()` ; vide admis ici, mais requis au lancement (`build_request`) | « Renseigne le mot-clé du poste visé (Paramètres ou lancement d'analyse). » (au lancement) |
| Localisation (« Vide = monde entier ») | `location` | texte | chaîne, `strip()` | — |
| Ton lieu de résidence (« Utilisé dans les analyses et les CV ; vide = non mentionné ») `[ACTUEL]` vague 6 | `home_location` | texte | `strip()`, espaces réduits, ≤ 80, ni saut de ligne ni caractère de contrôle | « home_location ne peut pas dépasser 80 caractères. » / « home_location ne doit contenir ni saut de ligne ni caractère de contrôle. » |
| Ton fuseau horaire (« Ex. Africa/Abidjan, Europe/Paris ; vide = non mentionné ») `[ACTUEL]` vague 6 | `timezone` | texte | vide ou fuseau IANA connu de `zoneinfo` | « timezone doit être un fuseau horaire IANA valide (ex. Europe/Paris) ou vide. » |
| Full remote uniquement | `remote_only` | case | `bool()` | — |
| Prudence | `caution` | `select` prudent / normal / agressif | ∈ `CAUTION_LEVELS` | « caution doit être l'un de : prudent, normal, agressif. » |
| Éléments par source (« Vide = selon la prudence ») | `max_items` | `number min=1 max=200`, vide → `null` | `null` ou entier **1–200** | « max_items doit être un entier. » / « … compris entre 1 et 200. » |
| Source du profil | `profile_source` | `select` auto / file / linkedin | ∈ (`auto`, `file`, `linkedin`) | « profile_source doit être l'un de : … » |
| Interprétation / CV | `interpreter` | `select` claude / local / **none** | ∈ (`claude`, `local`, `none`) | « interpreter doit être l'un de : … » |
| Score minimal pour proposer une offre | `apply_min_score` | `number min=0 max=100`, vide → **0** | nombre (décimal admis) **0–100** | « apply_min_score doit être un nombre. » / « … compris entre 0 et 100. » |
| Candidatures / jour | `apply_daily_max` | `number min=0 max=200`, vide → **0** | entier **0–200** | « apply_daily_max doit être compris entre 0 et 200. » |
| Messages / jour | `message_daily_max` | `number min=0 max=200`, vide → **0** | entier **0–200** | idem |
| Email de contact | `contact_email` | `type="email"` (non bloquant) | chaîne, `strip()`, **aucun contrôle de format** | — |
| Téléphone | `contact_phone` | texte | chaîne, `strip()` | — |
| Nom complet (carte « Coordonnées du CV ») `[ACTUEL]` 2026-10-15 | `full_name` | texte | ≤ 80, une ligne, sans caractère de contrôle | « full_name ne peut pas dépasser 80 caractères. » / « full_name ne doit contenir ni saut de ligne ni caractère de contrôle. » |
| Profil LinkedIn (carte « Coordonnées du CV ») `[ACTUEL]` 2026-10-15 | `linkedin_url` | `type="url"` | vide ou `https://…linkedin.com/in/…` | « linkedin_url doit être l'URL de ton profil (https://www.linkedin.com/in/...) ou vide. » |
| Ajouter à mon profil LinkedIn chaque compétence… | `linkedin_skill_sync` | case | `bool()` | — |
| Signature des emails | `signature` | `textarea` 3 lignes | chaîne, `strip()` | — |
| Réponses mémorisées (« Oublier ») | `answers` | liste + bouton | objet ; clé normalisée tronquée à 200, réponse à 500 caractères | « answers doit être un objet question -> réponse. » |

Écarts relevés (état au 2026-10-13) :
- l'option « Aucune » (`none`) de l'interprète a disparu `[ACTUEL]` (G-4) : le `select` propose « Claude » et « Local
  (règles) » ; l'avertissement s'affiche quand `claude` est choisi ;
- un champ numérique vidé n'est plus transformé en 0 `[ACTUEL]` : `num("")` donne `NaN`, la valeur est affichée vide et
  refusée par `settingsSchema` (champ requis) ;
- `max_items` borné par la prudence choisie (15 / 30 / 60) `[ACTUEL]` côté serveur et côté UI (E-3, G-3, I-07) ;
- « Oublier » une réponse envoie seulement `{ answers }`, affiche « Paramètres enregistrés. », puis l'invalidation
  de `["settings"]` recharge les réglages et le `useEffect` écrase le formulaire : les **autres modifications non
  enregistrées sont perdues** sans avertissement (toujours vrai au 2026-10-13 : toute nouvelle version des réglages enregistrés remplace le formulaire) → `[CIBLE]` « Oublier »
  désactivé tant que le formulaire est modifié, ou fusion des réglages rechargés avec les champs modifiés ;
- la prudence par défaut du serveur est `prudent` `[ACTUEL]` (`config.DEFAULT_CAUTION`, E-3, R-1) ; la case « Passer la
  prudence à prudent » de la popup reste utile si l'utilisateur l'a relevée.

### 3.2 Profil candidat (`Skills`)

**a) Ajout de compétence** — `<form onSubmit>`, champ `aria-label="Nouvelle compétence"`, bouton « Ajouter »
désactivé si vide → `POST /api/profile/skills` `{ skill }`. Serveur : `nlp.normalize(skill)` non vide sinon 422
« Compétence vide. » ; à l'enregistrement, les compétences de plus de **60 caractères** sont **ignorées sans
erreur** (`_skill_list`). Réponse : profil + `linkedin_sync` (`queued` → tâche suivie, `unavailable` → Notice
explicative, `skipped` / `off` → rien). Retrait : `DELETE /api/profile/skills/{skill}` (aucune validation).

**b) Identité professionnelle** — « Enregistrer » → `PUT /api/profile` :

| Champ | Contrôle | Règle serveur (`save_profile_settings`) | Message |
| :--- | :--- | :--- | :--- |
| Titre | texte | chaîne, `strip()`, **tronquée à 120 caractères** | — |
| Années d'expérience | `number min=0 max=60` ; vide → `clear_years: true` | `null` ou entier **0–60** ; Pydantic `int` (un décimal ⇒ 422 Pydantic en anglais) | « years_experience doit être compris entre 0 et 60. » |
| Séniorité | `select` « Détectée automatiquement » (vide) / junior / confirme / senior / lead | vide ou ∈ `SENIORITY_LEVELS` | « seniority doit être l'un de : junior, confirme, senior, lead. » |

Les libellés de séniorité sont affichés bruts (`confirme` sans accent, identifiant du glossaire) →
`[CIBLE]` libellés « Junior », « Confirmé », « Senior », « Lead » avec les mêmes valeurs.

### 3.3 Lancement d'analyse (`RunModal`, `POST /api/runs`)
> **Vague 2 `[ACTUEL]`** : `runBodySchema` ; `max_items` ∈ [1, plafond de la prudence] ; messages : `max_items doit
> être au moins 1.` (minimum) et `max_items ne peut pas dépasser 15 en prudence « prudent ».` (maximum, G-3).
> Option « Aucune » de l'interprète retirée (Paramètres).


| Champ | Défaut | Règle client | Règle serveur |
| :--- | :--- | :--- | :--- |
| Poste visé (mot-clé) | `settings.keyword` | requis (bouton désactivé si vide après `trim`) | requis : 422 « Renseigne le mot-clé du poste visé… » |
| Localisation | `settings.location` | — | chaîne |
| Prudence anti-blocage | `settings.caution` | `select` (« Prudent (15 offres) », « Normal (30) », « Agressif (60) ») | ∈ `CAUTION_LEVELS` |
| Full remote uniquement | `settings.remote_only` | case | `bool` |
| Collecter de nouvelles données sur LinkedIn | `true` | case ; avertissement CGU si cochée | `scrape: true` ⇒ `require_browser_ready()` : 409 identifiants absents ou blocage actif |
| Éléments par source `[ACTUEL]` | `max_items` (vide = plafond de la prudence) | `number` 1 à plafond de la prudence | `max_items` ≤ plafond (G-3) |
| — (non exposé) | `interpreter` | — | accepté par `RunBody`, validé |

Autres réponses : 409 « Une analyse est déjà en cours. ».
`[ACTUEL]` (UI et backend, vagues 2-3) : `SourcePicker` (au moins une source si collecte, message
client « Choisis au moins une source. ») ; champ « Éléments par source » borné par la prudence (vide = plafond) ;
popup de risques **avant** la requête pour la première source `navigateur` cochée sans consentement à jour, sans
lancement automatique après acceptation ; si la popup a passé la prudence à « prudent »
(`onAccepted({ switchedToPrudent: true })`), le formulaire passe à « prudent » et vide un `max_items` > 15 ; un 428
passe par `ConsentGate` (relance explicite, corps de `/runs` réécrit en `caution: "prudent"` dans le même cas).
Reste `[CIBLE]` : choix de l'interprète (`claude` / `local`) dans `RunModal`.

### 3.4 Réponse à une tâche (`TaskDetail` et modale de relecture)

| Cas | Contrôle | Envoi | Règles |
| :--- | :--- | :--- | :--- |
| Question avec options (`needs_input.options` non vide, champ ≠ `confirm*`) | un bouton par option | `POST /api/tasks/{id}/answer` `{ value: option, remember: true }` | Erreur affichée dans l'`ErrorBox` du détail `[ACTUEL]` (revue du 2026-10-09). |
| Question libre | champ texte (placeholder « Suggestion : … » ou « Ta réponse (mémorisée pour la suite) ») + « Envoyer » désactivé si vide | `{ value, remember: true }` | Serveur : `value: str` requis ; mémorise si `remember` et champ ≠ `confirm*` (clé tronquée à 200, réponse à 500). 409 « Cette tâche n'attend pas de réponse. » / « … n'attend plus de réponse. » ; délai 600 s. |
| Confirmation `confirm_*` | modale « Relire avant l'envoi » **uniquement**, boutons des options ; le tiroir n'affiche aucune option (« Relecture ouverte : confirme ou annule dans la fenêtre « Relire avant l'envoi » ») | `{ value, remember: false }` | **jamais mémorisée** (client et serveur, I-04). Fermer (Échap, « ✕ ») = **toujours `"Annuler"`** ; focus initial sur « Annuler » (sinon sur le titre), jamais sur « Soumettre » ; voile sans effet. `[ACTUEL]` |

`[CIBLE]` : case « Mémoriser cette réponse pour les prochaines candidatures » (cochée par défaut) au lieu d'un
`remember: true` implicite ; indication du temps restant ; pour une suggestion, bouton « Utiliser la suggestion »
(remplit le champ, ne l'envoie pas — I-05 : rien n'est deviné ni envoyé sans action).

### 3.5 Message à l'entreprise (`ComposeModal`, `POST /api/applications/{id}/messages`)

| Champ | Règle client `[ACTUEL]` | Règle serveur |
| :--- | :--- | :--- |
| Canal | onglets Email (SMTP) / Message LinkedIn ; défaut email si le contact a un email ou pas d'URL de profil | `Literal["email", "linkedin"]` |
| Destinataire (email) | requis (non vide) ; `type="email"` non bloquant | `outreach.validate_recipient` : `^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$` sinon 422 « Adresse email invalide : « … ». » |
| Profil LinkedIn du destinataire | requis (non vide) | contrôlé pendant la tâche : « Profil LinkedIn invalide : renseigne l'URL du profil du recruteur (https://www.linkedin.com/in/…). » |
| Objet (email) | requis (non vide) | non vide et **sans saut de ligne** sinon 422 « Objet requis, sur une seule ligne. » |
| Joindre le CV (PDF) | case, cochée | `attach_cv: bool` |
| Message | requis ; compteur ; avertissement au-delà de **300** caractères en LinkedIn | `body` requis ; note de connexion LinkedIn tronquée à `LINKEDIN_NOTE_MAX = 300` |
| Confirmation | « Relire avant l'envoi » puis « Confirmer l'envoi » | `confirm: true` sinon 422 « Confirmation explicite requise : relis le message puis confirme l'envoi. » |

Gardes : plafond `message_daily_max` (bouton désactivé + Notice ; serveur 429), SMTP absent (bouton désactivé ;
serveur 503), envoi déjà en cours (409). Le brouillon est chargé depuis `message-draft` : objet
« Candidature — {titre} », corps = message du dossier + signature.

`[ACTUEL]` (UI et backend, I-12, I-13, vague 4, revue du 2026-10-09) :
- relecture : origine de l'adresse et pied imposé lus par `GET …/message-draft?to=` à chaque relecture, affichés
  en lecture seule ; « Confirmer l'envoi » désactivé tant qu'ils ne sont pas chargés ;
- case obligatoire « J'ai relu ce message et je confirme l'envoi », jamais pré-cochée, décochée à chaque retour à
  la relecture et après tout échec ; aucun rejeu ;
- destinataire opposé (`contact.opposed`, `draft.opposed` ou 409 `Ce destinataire a demandé à ne plus être
  contacté.`) : bandeau `danger` « Cette personne a demandé à ne plus être contactée : envoi impossible. », case et
  bouton désactivés, aucun moyen de forcer ;
- longueur LinkedIn comptée comme le serveur (V4-4) : `Array.from(s).length` ; dépassement si
  `len(body.trimEnd()) + 2 + len(footer) > 300` ; le compteur affiche la place utile `300 − 2 − len(footer)`
  (« mention déduite ») ; le corps est tronqué avant la mention, jamais la mention ;
- message LinkedIn : popup de risques préventive à l'ouverture de la relecture et à l'envoi.
Reste `[CIBLE]` : validation de l'URL LinkedIn côté client avant la relecture.

### 3.6 Contacts et dossier (`Applications`)

- **Contacts** (`PATCH /api/applications/{id}` `{ contacts }`) : lignes Nom / Profil LinkedIn / Email, ajout et
  suppression locales, « Enregistrer les contacts ». Serveur (Pydantic `Contact`) : email vide ou conforme à
  `^[^@\s]+@[^@\s]+\.[^@\s]+$` (« Adresse email invalide. ») ; URL vide ou
  `^https://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+` (« L'URL doit être un profil LinkedIn
  (https://www.linkedin.com/in/...). ») ; erreurs renvoyées en 422 Pydantic et concaténées par `api.ts`. Les
  deux expressions d'email (contacts vs `outreach`) diffèrent légèrement → `[CIBLE]` une seule expression partagée.
- **Message de candidature / CV** (`PATCH { message }` / `{ cv }`) : boutons « Enregistrer » actifs seulement si
  modifié ; avertissement au-delà de 1 500 caractères pour le message ; aucun contrôle serveur de longueur.
- **Statut** (`select`, `PATCH { status }`) : valeurs de `application.STATUSES` ; changement immédiat, sans
  confirmation (changement libre autorisé par I-03). `[CIBLE]` confirmation si le statut **régresse**
  (ex. `postulée` → `préparée`).

### 3.7 Import manuel d'une offre Indeed `[ACTUEL]` (R-6, vague 5)
`ManualImportModal` ; schéma `manualImportSchema` (`src/schemas.ts`, § 4) ; les messages reprennent ceux du serveur
(`champ : message`, `backend/01` § 5.8.2). Sur un 422 rattaché à un champ, le message s'affiche sous ce champ (`aria-describedby`) et
le **focus** passe au premier champ en erreur (`aria-invalid="true"`). Le champ `keyword` (≤ 80, facultatif) est omis du corps s'il est vide.

| Champ | Règle |
| :--- | :--- |
| URL de l'offre | requise, `https://`, hôte `indeed.com` ou sous-domaine national (`fr.indeed.com`…), port numérique admis, ≤ 2 000 points de code |
| Texte de l'offre (collé) | requis, 200 à 20 000 caractères |
| Titre | requis, ≤ 150 caractères (`ManualItem.title`) |
| Entreprise, localisation, mot-clé | facultatifs (≤ 80 / 120 / 80 caractères) |
| Envoi | `POST /api/items/import` (cf. `backend/01-api-contracts.md`) `{ source_code: "indeed", url, text, title, company?, location? }` → `201`, `409` si déjà dans l'historique |

## 4. Schémas Zod `[ACTUEL]` (vagues 2 à 6)
> Vague 5 `[ACTUEL]` : `manualImportSchema` (champs `url`, `title`, `company`, `location`, `text`, `keyword` ; `url` :
> `INDEED_HOST_RE = /^https:\/\/([a-z0-9-]+\.)*indeed\.com(:\d+)?(\/|$|\?|#)/i` (un port numérique est admis, comme côté serveur), ≤ 2 000 ; `text` : `MANUAL_TEXT_MIN` 200 à
> `MANUAL_TEXT_MAX` 20 000 ; messages en minuscule initiale, comme le `detail` serveur : `url : colle l'adresse https
> de l'offre Indeed.`…). La forme Zod 4 ci-dessous est indicative ; le fichier fait foi.
> Vague 6 `[ACTUEL]` : `settingsSchema` gagne `home_location` (espaces internes réduits puis ≤ 80 ; sans caractère de
> contrôle ; messages `home_location ne peut pas dépasser 80 caractères.` et `home_location ne doit contenir ni saut de
> ligne ni caractère de contrôle.`) et `timezone` (vide ou nom connu d'`Intl.DateTimeFormat` ; message `timezone doit
> être un fuseau horaire IANA valide (ex. Europe/Paris) ou vide.`), identiques mot pour mot aux 422 de
> `settings.validate_settings` (`backend/01` § 5.2). Le client refuse aussi les séparateurs de ligne et de paragraphe Unicode
> (U+2028 / U+2029), comme le serveur (catégories Unicode `Cc`, `Zl`, `Zp`) ; les longueurs de `home_location` et des champs
> de l'import Indeed sont comptées en **points de code** (`Array.from(valeur).length`, comme le `len()` du serveur : un
> emoji compte 1, non 2) ; le serveur reste juge.
> Vague 4 `[ACTUEL]` : `suppressionSchema` (`recipient` après `trim` = `EMAIL_RE` ou `LINKEDIN_PROFILE_RE`, message
> `recipient : adresse email ou URL de profil LinkedIn attendue.` — avec point final, comme le message du
> validateur serveur repris tel quel par G-1 `value_error` ; `reason` ∈ `désinscription`, `refus_oral`, `plainte`,
> message `reason : doit être l'un de : désinscription, refus_oral, plainte` — sans point, forme G-1
> `literal_error`). `LINKEDIN_PROFILE_RE = /^https:\/\/([a-z]{2,3}\.)?linkedin\.com\/in\/[^/?#\s]+/` =
> `outreach._PROFILE_URL` (`backend/01` § 5.7.1, V4-3).
> Vagues 2-3 : `EMAIL_RE` identique à `outreach._EMAIL` (sans `,;<>`) ; `contact_email` (G-12) ; `max_items` :
> `max_items doit être au moins 1.` (G-11) / G-3 ; `runBodySchema.sources` : tableau non vide de codes, défaut
> `linkedin` (`backend/01` § 5.8.1).

> **`[ACTUEL]` depuis la vague 2** : `src/schemas.ts` (Zod 4) contient `settingsSchema`, `runBodySchema`,
> `reportSchema` et les constantes `CAUTIONS`, `CAUTION_MAX`, `DEFAULT_CAUTION`. Les messages sont ceux du serveur
> (`backend/01` § 1.2, § 5.2). `contact_email` : même expression que le serveur, validée **aussi** côté serveur
> (G-12). Le code ci-dessous reste la référence de conception.


```ts
// src/schemas.ts — bornes identiques à settings.py, outreach.py et web/routers/applications.py
import { z } from "zod";

export const CAUTION_MAX = { prudent: 15, normal: 30, agressif: 60 } as const;
const entier = (libelle: string, min: number, max: number) =>
  z.number({ error: `${libelle} : saisis un nombre.` })
    .int(`${libelle} : saisis un nombre entier.`)
    .min(min, `${libelle} : minimum ${min}.`)
    .max(max, `${libelle} : maximum ${max}.`);

export const EMAIL_RE = /^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$/;
export const LINKEDIN_PROFILE_RE = /^https:\/\/([a-z]{2,3}\.)?linkedin\.com\/in\/[^/?#\s]+/;

export const settingsSchema = z.object({
  keyword: z.string().trim(),
  location: z.string().trim(),
  home_location: z.string().trim(),   // vague 6 : ≤ 80, sans caractère de contrôle (cf. note ci-dessus)
  timezone: z.string().trim(),        // vague 6 : vide ou fuseau IANA
  remote_only: z.boolean(),
  caution: z.enum(["prudent", "normal", "agressif"]),
  max_items: entier("Éléments par source", 1, 200).nullable(),
  profile_source: z.enum(["auto", "file", "linkedin"]),
  interpreter: z.enum(["claude", "local"]),                       // « none » retiré (I-11)
  apply_min_score: z.number({ error: "Score minimal : saisis un nombre." })
    .min(0, "Score minimal : minimum 0.").max(100, "Score minimal : maximum 100."),
  apply_daily_max: entier("Candidatures / jour", 0, 200),
  message_daily_max: entier("Messages / jour", 0, 200),
  contact_email: z.union([z.literal(""), z.string().trim().regex(EMAIL_RE, "Email de contact invalide.")]),
  contact_phone: z.string().trim(),
  signature: z.string().trim(),
  linkedin_skill_sync: z.boolean(),
}).refine((s) => s.max_items === null || s.max_items <= CAUTION_MAX[s.caution], {
  path: ["max_items"],
  message: "Éléments par source : ne peut pas dépasser le plafond de la prudence choisie.",   // choix à valider (R-2)
});
export type SettingsForm = z.infer<typeof settingsSchema>;

export const profileSchema = z.object({
  title: z.string().trim().max(120, "Titre : 120 caractères maximum."),
  years_experience: entier("Années d'expérience", 0, 60).nullable(),
  seniority: z.enum(["", "junior", "confirme", "senior", "lead"]),
});

export const skillSchema = z.object({
  skill: z.string().trim().min(1, "Saisis une compétence.").max(60, "Compétence : 60 caractères maximum."),
});

export const runSchema = z.object({
  keyword: z.string().trim().min(1, "Renseigne le poste visé."),
  location: z.string().trim(),
  remote_only: z.boolean(),
  caution: z.enum(["prudent", "normal", "agressif"]),
  scrape: z.boolean(),
  sources: z.array(z.enum(["linkedin", "france_travail", "malt", "wttj"])),
  interpreter: z.enum(["claude", "local"]),
}).refine((r) => !r.scrape || r.sources.length > 0, { path: ["sources"], message: "Choisis au moins une source." });

export const messageSchema = z.discriminatedUnion("channel", [
  z.object({
    channel: z.literal("email"),
    to: z.string().trim().regex(EMAIL_RE, "Adresse email invalide."),
    subject: z.string().trim().min(1, "Objet requis.").regex(/^[^\r\n]*$/, "L'objet doit tenir sur une seule ligne."),
    body: z.string().trim().min(1, "Le message est vide."),
    attach_cv: z.boolean(),
    reviewed: z.literal(true, { error: "Coche « J'ai relu ce message » pour confirmer." }),
  }),
  z.object({
    channel: z.literal("linkedin"),
    to: z.string().trim().regex(LINKEDIN_PROFILE_RE, "L'URL doit être un profil LinkedIn (https://www.linkedin.com/in/…)."),
    subject: z.string().optional(),
    body: z.string().trim().min(1, "Le message est vide."),
    attach_cv: z.boolean(),
    reviewed: z.literal(true, { error: "Coche « J'ai relu ce message » pour confirmer." }),
  }),
]);
// À l'envoi : { ...valeurs sans `reviewed`, confirm: true } — confirm n'est jamais pré-rempli ailleurs (I-04).

export const contactSchema = z.object({
  name: z.string().trim(),
  profile_url: z.union([z.literal(""), z.string().trim().regex(LINKEDIN_PROFILE_RE, "L'URL doit être un profil LinkedIn.")]),
  email: z.union([z.literal(""), z.string().trim().regex(EMAIL_RE, "Adresse email invalide.")]).nullable(),
});

export const riskConsentSchema = z.object({
  accepted: z.literal(true, { error: "Coche la case pour accepter les risques." }),
});

export const manualImportSchema = z.object({
  url: z.url({ protocol: /^https$/, hostname: /(^|\.)indeed\.com$/, error: "Colle l'adresse https de l'offre Indeed." }),
  text: z.string().trim().min(200, "Colle le texte complet de l'offre (200 caractères minimum).").max(20000, "Texte trop long (20 000 caractères maximum)."),
  title: z.string().trim().min(1, "Indique le titre de l'offre.").max(150),
  company: z.string().trim().max(80).optional(),
  location: z.string().trim().max(120).optional(),
});
```

Notes : l'API Zod 4 (`error` au lieu de `message`/`invalid_type_error`, `z.url({ protocol, hostname })`) est à
vérifier sur la version installée ; les champs numériques HTML renvoient des chaînes → conversion explicite
(`e.target.valueAsNumber`, `NaN` → erreur « saisis un nombre ») avant validation, jamais `Number("") = 0`.

## 5. Upload de Fichiers & Pièces Jointes

- `[ACTUEL]` **aucun upload** dans l'interface. Les fichiers sources restent sur disque, déposés par
  l'utilisateur : `profil.json/.md/.txt` ou `cv.pdf` à la racine du projet (état affiché « Fichier de profil :
  trouvé / aucun » dans Paramètres). La pièce jointe des emails est le CV **généré** au format ATS (`cv_export.ensure`, fichier
  `<Prénom-Nom>-CV-<Titre>.pdf`) à
  partir du CV Markdown du dossier, choisie par la case « Joindre le CV (PDF) » ; l'utilisateur ne choisit pas de
  fichier.
- `[CIBLE]` si un import de CV est ajouté : type MIME `application/pdf` uniquement (vérifié aussi côté serveur par
  la signature `%PDF-`), 5 Mo maximum, un seul fichier, copie dans le dossier de données local ; pas de barre de
  progression nécessaire (transfert local), mais bouton en `loading` et message de succès.

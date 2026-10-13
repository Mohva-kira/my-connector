# Design System & Charte UI

> Sources vérifiées : `web/frontend/tailwind.config.js`, `src/index.css`, `src/ui.tsx`, `src/main.tsx`,
> `src/App.tsx`, `index.html`. Tailwind CSS **3.4** (`3.4.19` installé), PostCSS + Autoprefixer.
> Le fichier `tailwind.config.js` indique être une copie des jetons de `services/email_analyzer/frontend` :
> l'outil reste autonome, les deux copies peuvent diverger.

## 1. Tokens de Design

### 1.1 Palette de couleurs `[ACTUEL]`

Chaque couleur est une **variable CSS RGB** (`--color-*-rgb`) définie dans `index.css` pour `:root` (clair) et
`[data-theme="dark"]` (sombre), exposée à Tailwind par `withOpacity()` : les modificateurs d'opacité fonctionnent
(`bg-danger/15`, `border-warning/40`, `bg-text-muted/40`). Exception : `accent-soft` a une opacité **figée à 0,08**.

| Classe Tailwind (jeton) | Rôle | Clair (RGB) | Sombre (RGB) |
| :--- | :--- | :--- | :--- |
| `bg-primary` | Fond de page (`body`) | 250, 250, 248 | 11, 15, 26 |
| `bg-secondary` | Survol, blocs `pre`, fond des journaux | 244, 244, 241 | 16, 22, 42 |
| `bg-tertiary` | Pistes des barres, badge `neutral` | 236, 236, 231 | 26, 35, 56 |
| `surface` | Cartes, modales, navigation, tiroir | 255, 255, 255 | 19, 26, 44 |
| `text-primary` | Texte principal | 30, 37, 48 | 232, 234, 240 |
| `text-secondary` | Texte secondaire, libellés de champ | 89, 99, 111 | 167, 176, 192 |
| `text-muted` | Aides, dates, états vides | 138, 148, 160 | 107, 118, 144 |
| `border-default` | Bordures de cartes et champs | 226, 225, 220 | 35, 44, 66 |
| `border-subtle` | Séparateurs de listes | 236, 235, 230 | 28, 36, 56 |
| `accent-primary` | Couleur primaire (boutons `primary`, liens, focus des champs) | 29, 78, 216 | 59, 130, 246 |
| `accent-hover` | Survol du bouton `primary` | 30, 64, 175 | 96, 165, 250 |
| `accent-soft` | Fond de l'élément de navigation actif, badge `accent` (opacité 0,08) | 29, 78, 216 | 59, 130, 246 |
| `success` | Succès, compétence acquise, statut `postulée`, tâche `terminée` | 22, 163, 74 | 34, 197, 94 |
| `warning` | Attention, tâche en attente, plafond > 70 % | 217, 119, 6 | 245, 158, 11 |
| `danger` | Erreurs, statut `refusée` / `échec`, bandeau de blocage | 220, 38, 38 | 239, 68, 68 |
| `info` | Information, statut `contactée`, tâche `en_cours` | 37, 99, 235 | 59, 130, 246 |

Correspondance avec le modèle générique : `primary` = `accent-primary` ; `secondary` = `text-secondary` /
bouton `secondary` (bordure + `surface`) ; `accent` = `accent-soft` + `accent-primary` ; `neutral / background` =
`bg-primary` / `surface` ; `destructive` = `danger` ; `success` = `success`. Il n'y a pas de jeton `secondary`
dédié.

**Couleurs hors jetons** : **aucune** `[ACTUEL]` (vague 7 ; revue du 2026-10-09 : aucune classe Tailwind brute ni
hexadécimal hors `index.css`). Ombres : `shadow-lift` (Modal, bouton Tâches, tiroir) et `shadow-card`. Les
graphiques SVG utilisent directement `rgb(var(--color-…-rgb))` (Dashboard `KIND_COLORS`, Market `ScoreRing`,
`WeeklyChart`) : c'est admis, la valeur reste thémée.

**Vague 7, partie 1 (2026-10-08) `[ACTUEL]`** : couleurs hors jetons résorbées. Nouveaux jetons, définis dans
les **deux** thèmes de `index.css` et exposés par `tailwind.config.js` : `accent-strong`, `border-strong`,
`success-strong`, `warning-strong`, `danger-strong`, `info-strong` (texte des badges et notices sur fond à 15 %),
`on-accent` (texte des boutons pleins, remplace `text-white` ; quasi noir en sombre), `overlay` (voile des
modales, remplace `bg-black/40`). Valeurs claires : celles du § 4.1 ; valeurs sombres calculées par lm-frontend
(≥ 4,5:1 texte, ≥ 3:1 bordures) — re-mesurées à la revue du 2026-10-09 (§ 4.1).

### 1.2 Tons sémantiques (`ui.tsx`) `[ACTUEL]`

Type `Tone = "neutral" | "success" | "warning" | "danger" | "info" | "accent"`, table `TONES` partagée par
`Badge` et `Notice` : fond à 15 % + texte en jeton `*-strong` (`bg-success/15 text-success-strong`…), `neutral` =
`bg-bg-tertiary text-text-secondary`, `accent` = `bg-accent-soft text-accent-strong`. `Notice` porte `role="alert"`
(assertif) pour le ton `danger` et `role="status"` (poli) pour tous les autres tons (`success`, `info`, `warning`, `neutral`,
`accent`) ; ne jamais l'envelopper dans un autre `role="alert"`.

| Fonction | Règle |
| :--- | :--- |
| `scoreTone(score)` | `≥ 75` → `success` ; `≥ 50` → `warning` ; sinon `neutral`. |
| `statusTone(status)` (candidature) | `postulée` → `success` ; `entretien` → `accent` ; `contactée` → `info` ; `refusée`, `échec` → `danger` ; `abandonnée`, `externe` → `neutral` ; autre (dont `préparée`) → `warning`. |
| `toneOf(status)` (tâche, `TaskDrawer.tsx`) | `terminée` → `success` ; `échec` → `danger` ; `annulée` → `neutral` ; `en_cours` → `info` ; `en_attente`, `attente_saisie`, `attente_verification_manuelle` → `warning`. |

### 1.3 Typographie `[ACTUEL]`

- **Police principale** : `Inter` variable, **auto-hébergée** (`@fontsource-variable/inter`, importée dans
  `main.tsx`), puis `system-ui, sans-serif` (`fontFamily.sans`) ; aucun appel à un serveur de polices tiers. **Monospace** : `JetBrains Mono, monospace` (`fontFamily.mono`)
  — JetBrains Mono n'est **pas** chargée : rendu réel = police monospace du système.
- **Échelle utilisée** (classes Tailwind par défaut, aucune taille personnalisée) :

| Usage | Classe | Taille / graisse |
| :--- | :--- | :--- |
| Titre de page (`PageHeader`, Marché) | `text-xl font-semibold` | 20 px / 600 |
| Titre de candidature | `text-lg font-semibold` | 18 px / 600 |
| Titre de section (Marché), titre de modale | `text-base font-semibold` | 16 px / 600 |
| Titre de carte (`Card`, `Panel`) | `text-sm font-semibold` | 14 px / 600 |
| Corps, champs, boutons | `text-sm` | 14 px / 400 (boutons 500) |
| Aides, badges, journaux, boutons `small` | `text-xs` | 12 px |
| KPI (Marché) | `text-2xl font-semibold tabular-nums` | 24 px / 600 |
| Intitulés en capitales | `text-xs font-semibold uppercase` | 12 px / 600 |

- **Markdown** (conseils, CV, interprétation) : classe utilitaire `.prose-md` définie dans `index.css`
  (h1 `text-lg`, h2 `text-base`, h3 `text-sm`, listes à puces / numérotées, liens `text-accent-primary underline`).
  Pas de plugin `@tailwindcss/typography`.
- `[ACTUEL]` (vague 7) **Inter auto-hébergée** : `index.html` n'appelle plus `fonts.googleapis.com` ; un outil
  local ne doit pas transmettre l'adresse IP de l'utilisateur à un tiers à chaque ouverture (RGPD, cohérent avec
  R-3 et « aucune donnée ne sort de la machine », lot 1 § 04-1).

### 1.4 Espacements, rayons, ombres `[ACTUEL]`

- **Espacements** : échelle Tailwind par défaut (base 4 px). Usages récurrents : cartes `p-4` (`Panel` `sm:p-5`),
  modales `p-5`, grilles `gap-3` / `gap-4`, page `p-4 md:p-6`, entre sections `space-y-3` / `space-y-4`.
- **Rayons** (surchargés dans `tailwind.config.js`) : `rounded-sm` 6 px · `rounded-md` 10 px (boutons, champs,
  notices) · `rounded-lg` 14 px (`Card`) · `rounded-xl` 18 px (modales, `Panel`, cartes d'offre, tiroir) ·
  `rounded-full` 999 px (badges, puces de compétences). `rounded` (sans suffixe, 4 px par défaut) n'est utilisé
  que par le bouton de fermeture des modales.
- **Ombres** : `shadow-card` (`var(--shadow-card)`) et `shadow-lift` (`var(--shadow-lift)`), redéfinies en mode
  sombre. `.card-lift:hover` élève une carte de 2 px avec `--shadow-lift`.
- **Animations** (`index.css`) : `.animate-in` (fondu + translation 10 px, décalage `--i × 60 ms`), `.bar-grow`
  (barres), `.ring-draw` (anneau de score), `.card-lift`. Toutes neutralisées sous
  `@media (prefers-reduced-motion: reduce)` ; `useCountUp` (Marché) affiche directement la valeur finale dans ce
  cas. Conforme au critère WCAG 2.3.3 (AAA, bonne pratique).

## 2. Gestion des Thèmes `[ACTUEL]`

- **Mode clair / sombre** : attribut `data-theme` sur `<html>`.
  - Au démarrage (script inline de `index.html`) : valeur de `localStorage["lm-theme"]` si présente, sinon
    `prefers-color-scheme: dark` ; en cas d'exception (stockage indisponible) → clair.
  - Bascule manuelle : bouton « Thème clair / sombre » en bas de la navigation (`App.tsx`, `toggleTheme()`),
    persistée dans `localStorage["lm-theme"]`.
  - `[ACTUEL]` (vague 7) un script inline dans `index.html` pose `data-theme` avant le premier rendu (lecture de
    `localStorage` dans un `try`, sinon `prefers-color-scheme`) : plus de flash clair en sombre. Aucune CSP n'est servie pour la SPA : le script inline est admis. Si une CSP
    `script-src 'self'` est ajoutée un jour, y mettre le **hash sha256** de ce script (pas de `'unsafe-inline'`).
- **Règle** : toujours utiliser les classes sémantiques (`bg-surface`, `text-text-secondary`, `bg-danger/15`…),
  jamais de couleur Tailwind brute (`bg-blue-600`) ni d'hexadécimal. Plus d'exception : texte des boutons pleins
  `text-on-accent`, voile des modales `backdrop:bg-overlay/40` (vague 7).
- **Nouvelles couleurs** : ajouter la variable dans **les deux** blocs de `index.css` puis l'exposer dans
  `tailwind.config.js` via `withOpacity`.

## 3. Responsive Breakpoints `[ACTUEL]`

Points de rupture Tailwind par défaut (non surchargés) : `sm` 640 px, `md` 768 px, `lg` 1024 px, `xl` 1280 px,
`2xl` 1536 px (non utilisé). Conception *mobile-first*.

| Point | Effet principal |
| :--- | :--- |
| `< md` | Navigation en **barre horizontale défilante** en haut (`overflow-x-auto`), titre de l'application masqué ; bouton de thème à droite. Tableaux avec défilement horizontal (`Offres` : `min-w-[720px]`). |
| `md` | Navigation en **colonne latérale** de 224 px (`md:w-56`), contenu `md:p-6`. |
| `sm` | Grilles à 2 colonnes (filtres, KPI, détail d'offre, contacts à 3 colonnes). |
| `lg` | Tableau de bord sur 3 colonnes, Paramètres / Compétences / Marché sur 2–3 colonnes, Candidatures en maître-détail (`lg:grid-cols-[18rem_1fr]`). |
| `xl` | Cartes d'offres sur 3 colonnes (Marché). |

Le tiroir des tâches est contraint à `min(28rem, 100vw − 2rem)` et 75 vh ; les modales à 90 vh avec défilement
interne. Critère WCAG 1.4.10 *Reflow* : satisfait sauf pour le tableau des Offres (défilement horizontal admis pour
les tableaux de données).

## 4. Règles d'Accessibilité (a11y) — WCAG 2.2 niveau AA

### 4.1 Contrastes mesurés (ratio WCAG, calculé sur les jetons) `[ACTUEL]`

Re-mesurés à la revue du 2026-10-09 sur les valeurs de `index.css` (vague 7, partie 1). Les badges et notices sont
mesurés sur le fond du ton à 15 % (8 % pour `accent`) composé sur `surface`.

| Combinaison | Clair | Sombre | Verdict (texte < 18 px : 4,5:1 ; composant : 3:1) |
| :--- | :--- | :--- | :--- |
| `text-muted` / `surface` | 4,8 | 5,1 | OK |
| Texte `success` / `warning` / `danger` sur `surface` | 5,0 / 5,0 / 4,8 | 7,6 / 8,1 / 4,6 | OK |
| Badge / notice `success-strong` / `warning-strong` (fond 15 %) | 5,8 / 5,8 | 7,7 / 8,0 | OK |
| Badge / notice `danger-strong` / `info-strong` (fond 15 %) | 5,1 / 5,4 | 5,4 / 5,6 | OK |
| Badge `accent-strong` (fond `accent-soft`) | 5,9 | 6,2 | OK |
| `on-accent` sur `accent-primary` (bouton `primary`) | 6,7 | 5,2 | OK |
| `on-accent` sur `danger` (bouton `danger`) | 4,8 | 5,1 | OK |
| `border-strong` / `surface` (contour des champs, critère 1.4.11) | 3,3 | 3,3 | OK |

Le bouton « Tâches » en attente utilise `bg-warning/15 text-warning-strong border-warning`. `ErrorBox` et les
erreurs de la liste des tâches utilisent `text-danger-strong` (fond `danger/10`) depuis la revue du 2026-10-09. Reste `[CIBLE]` : test
automatisé de contraste des jetons (aujourd'hui seul `eslint-plugin-jsx-a11y` est branché sur `npm run lint`).

### 4.2 Focus, clavier, libellés `[ACTUEL]` (vague 7, revue du 2026-10-09)

- **Focus visible (2.4.7, AA)** : règle globale de `index.css` (`:focus-visible` = contour 2 px `accent-primary`,
  décalé de 2 px ; décalage 0 pour les champs). Les champs ont une bordure `border-strong` (≥ 3:1).
- **Focus non masqué (2.4.11, AA)** : `html { scroll-padding-bottom: 4rem }` (bouton flottant « Tâches ») ; page
  Marché enveloppée dans `.sticky-nav-offset` (`scroll-margin-top: 4.5rem` sous la barre de sections collante).
- **Taille des cibles (2.5.8, AA)** : `min-h-6 min-w-6` sur le « ✕ » des modales et le retrait d'une compétence.
- **Modales (2.1.2, 2.4.3)** : élément natif `<dialog>` + `showModal()` (piège de focus, page inerte, Échap natif),
  `aria-labelledby` vers le titre, `aria-describedby` facultatif, focus initial (`initialFocus` ou titre avec
  `focusTitle`), restitution du focus au déclencheur, `dismissible = false` pour les dialogues de décision ; détail
  dans `02-components-catalog.md` § 3.6. Dialogues de décision (revue du 2026-10-09) : aucun bouton d'envoi
  n'a le focus initial — relecture `confirm_*` : focus sur « Annuler » (sinon titre), Échap = « Annuler » ;
  popup de risques : focus sur le titre, Échap = « Refuser » ; `ComposeModal` : focus sur le début de la relecture.
- **Libellés (1.3.1, 3.3.2, 4.1.2)** : `Field` enveloppe le champ dans un `<label>` et relie l'erreur par
  `aria-describedby` + `aria-invalid` ; les champs sans `Field` ont un `aria-label` (« Rechercher », « Annotation »,
  « Message de candidature », « Nouvelle compétence », « Filtrer par statut », « Statut », « Rapport »). Les boutons
  d'annotation des pages Offres et Marché portent `aria-label` + `aria-pressed`.
- **Annonce des changements (4.1.3, AA)** : `ErrorBox` `role="alert"` ; `Notice` : `role="alert"` si `danger`,
  `role="status"` sinon ; statut courant du tiroir des tâches dans une région `aria-live="polite"`.
- **Onglets** : `Tabs` complet (WAI-ARIA) : flèches, Début / Fin, `tabIndex` itinérant, `role="tabpanel"` +
  `aria-controls` quand le contenu est passé en `children` (candidature, `ComposeModal`).
- **Graphiques** : SVG avec `role="img"` + `aria-label` et `<title>` par barre / point `[ACTUEL]` ;
  `QuotaBar` utilise `role="progressbar"` avec `aria-valuenow` / `aria-valuemax` `[ACTUEL]`.
- **Icônes** : `Icon` est `aria-hidden="true"` ; tout bouton purement iconographique porte un `aria-label` (lien
  « Ouvrir l'offre … sur LinkedIn (nouvel onglet) » de `OfferCard` compris).
- **Saisie redondante (3.3.7, A, nouveau en 2.2)** : satisfait par les réponses mémorisées (I-05) et le
  pré-remplissage de `RunModal` depuis les Paramètres ; à conserver pour les nouveaux formulaires.
- **Authentification accessible (3.3.8, AA)** : sans objet (aucun mot de passe dans l'UI, jeton injecté).
- **Couleur seule (1.4.1)** : statuts toujours accompagnés d'un texte (badges libellés, « ✓ acquise »,
  « manquante ») `[ACTUEL]` ; le seuil de quotas (`QuotaBar`) a aussi le texte `used / limit`.

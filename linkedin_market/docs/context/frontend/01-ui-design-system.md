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

**Couleurs hors jetons relevées** `[ACTUEL]` (à résorber `[CIBLE]`) : `text-white` sur boutons `primary` /
`danger` et sur le bouton « Tâches » en attente ; `bg-black/40` du voile des modales ; ombres Tailwind par défaut
`shadow-xl` (Modal), `shadow-lg` (bouton Tâches), `shadow-2xl` (tiroir). Les graphiques SVG utilisent
directement `rgb(var(--color-…-rgb))` (Dashboard `KIND_COLORS`, Market `ScoreRing`, `WeeklyChart`) : c'est
admis, la valeur reste thémée.

### 1.2 Tons sémantiques (`ui.tsx`) `[ACTUEL]`

Type `Tone = "neutral" | "success" | "warning" | "danger" | "info" | "accent"`, table `TONES` partagée par
`Badge` et `Notice` : fond à 15 % + texte plein (`bg-success/15 text-success`…), `neutral` =
`bg-bg-tertiary text-text-secondary`, `accent` = `bg-accent-soft text-accent-primary`.

| Fonction | Règle |
| :--- | :--- |
| `scoreTone(score)` | `≥ 75` → `success` ; `≥ 50` → `warning` ; sinon `neutral`. |
| `statusTone(status)` (candidature) | `postulée` → `success` ; `entretien` → `accent` ; `contactée` → `info` ; `refusée`, `échec` → `danger` ; `abandonnée`, `externe` → `neutral` ; autre (dont `préparée`) → `warning`. |
| `toneOf(status)` (tâche, `TaskDrawer.tsx`) | `terminée` → `success` ; `échec` → `danger` ; `annulée` → `neutral` ; `en_cours` → `info` ; `en_attente`, `attente_saisie`, `attente_verification_manuelle` → `warning`. |

### 1.3 Typographie `[ACTUEL]`

- **Police principale** : `Inter` (400, 500, 600, 700) chargée depuis **Google Fonts** dans `index.html`, puis
  `system-ui, sans-serif` (`fontFamily.sans`). **Monospace** : `JetBrains Mono, monospace` (`fontFamily.mono`)
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
- `[CIBLE]` **auto-héberger Inter** (paquet `@fontsource-variable/inter` importé dans `main.tsx`) et supprimer
  l'appel à `fonts.googleapis.com` : un outil local ne doit pas transmettre l'adresse IP de l'utilisateur à un
  tiers à chaque ouverture (RGPD, cohérent avec R-3 et « aucune donnée ne sort de la machine », lot 1 § 04-1).

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
  - Au démarrage (`main.tsx`) : valeur de `localStorage["lm-theme"]` si présente, sinon
    `prefers-color-scheme: dark` ; en cas d'exception (stockage indisponible) → clair.
  - Bascule manuelle : bouton « Thème clair / sombre » en bas de la navigation (`App.tsx`, `toggleTheme()`),
    persistée dans `localStorage["lm-theme"]`.
  - Le thème n'est appliqué qu'après le chargement du JS : un bref flash clair est possible en sombre.
    `[CIBLE]` script inline de 3 lignes dans `index.html` pour poser `data-theme` avant le premier rendu, et
    `aria-pressed` sur le bouton de bascule.
- **Règle** : toujours utiliser les classes sémantiques (`bg-surface`, `text-text-secondary`, `bg-danger/15`…),
  jamais de couleur Tailwind brute (`bg-blue-600`) ni d'hexadécimal. Exceptions tolérées : `text-white` sur un fond
  plein `accent-primary` / `danger`, voile `bg-black/40`.
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

| Combinaison | Clair | Sombre | Verdict (texte < 18 px : 4,5:1 requis, critère 1.4.3) |
| :--- | :--- | :--- | :--- |
| `text-primary` / `surface` | 15,4 | 14,4 | OK |
| `text-secondary` / `surface` | 6,1 | 7,9 | OK |
| `text-muted` / `surface` | **3,1** | **3,8** | **Échec** — utilisé pour les aides de champ, dates, états vides |
| `success` texte / `surface` | **3,3** | 7,6 | Échec en clair |
| `warning` texte / `surface` | **3,2** | 8,1 | Échec en clair |
| `danger` texte / `surface` | 4,8 | 4,6 | OK |
| `info` texte / `surface` | 5,2 | 4,7 | OK |
| Badge `success` / `warning` (texte sur fond 15 %) | **2,8 / 2,7** | 5,9 / 6,2 | Échec en clair |
| Badge `danger` / `info` (fond 15 %) | **3,8 / 4,2** | **4,0 / 3,9** | Échec |
| Blanc sur `accent-primary` (bouton `primary`) | 6,7 | **3,7** | Échec en sombre |
| Blanc sur `danger` (bouton `danger`) | 4,8 | **3,8** | Échec en sombre |
| Blanc sur `warning` (bouton « Tâches » en attente) | **3,2** | **2,2** | Échec |
| `border-default` / `surface` (contour des champs, critère 1.4.11 : 3:1) | **1,3** | **1,3** | Échec pour identifier un champ |

`[CIBLE]` corrections minimales (sans changer l'identité visuelle) :
- clair : `--color-text-muted-rgb: 107, 114, 128` (4,8:1 sur `surface`), `--color-success-rgb: 21, 128, 61`
  (5,0:1), `--color-warning-rgb: 180, 83, 9` (5,0:1) ; pour le **texte des badges et notices** (fond à 15 %),
  nouveaux jetons `--color-success-strong-rgb: 22, 101, 52` (5,8:1), `--color-warning-strong-rgb: 146, 64, 14`
  (5,8:1), `--color-danger-strong-rgb: 185, 28, 28` (5,1:1), `--color-info-strong-rgb: 29, 78, 216` (5,4:1),
  utilisés par `TONES` ; à recalculer de même pour le thème sombre ;
- sombre : texte des boutons pleins en `text-bg-primary` (quasi noir) au lieu de `text-white`, ou
  `--color-accent-primary-rgb: 37, 99, 235` pour le fond des boutons ;
- bouton « Tâches » en attente : `bg-warning/15 text-warning border-warning` au lieu de `bg-warning text-white` ;
- champs : bordure `border-text-muted` (≥ 3:1) ou fond `bg-bg-secondary` distinct.
- vérification automatisée : `eslint-plugin-jsx-a11y` + test de contraste des jetons dans la CI.

### 4.2 Focus, clavier, libellés

- **Focus visible (2.4.7, AA)** `[ACTUEL]` : les boutons gardent le contour natif du navigateur (pas de
  `outline-none`) ; les champs (`inputClass`) suppriment le contour (`outline-none`) et ne changent que la couleur
  de bordure (`focus:border-accent-primary`), indicateur faible. `[CIBLE]` ajouter à `Button`, `inputClass`,
  onglets et liens-boutons : `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-primary
  focus-visible:ring-offset-2 focus-visible:ring-offset-surface`.
- **Focus non masqué (2.4.11, AA, nouveau en 2.2)** : la barre de sections collante de Marché (`sticky top-0`) et
  le bouton flottant « Tâches » ne doivent pas recouvrir l'élément focalisé — `[CIBLE]` `scroll-margin-top` sur les
  éléments focusables de Marché (les titres ont déjà `scroll-mt-20`) et `scroll-padding-bottom: 4rem` sur `html`.
- **Taille des cibles (2.5.8, AA, nouveau en 2.2)** : minimum 24 × 24 px. Points à vérifier : bouton « ✕ » de
  retrait d'une compétence (`px-1.5`, ≈ 18 px de large), bouton de fermeture des modales (`p-1`), boutons
  d'annotation de Marché (`p-1.5` + icône 16 px = 28 px, OK). `[CIBLE]` `min-h-6 min-w-6` sur ces boutons.
- **Modales (2.1.2, 2.4.3)** : `Modal` pose `role="dialog"`, `aria-modal="true"`, `aria-label={title}` et ferme sur
  Échap `[ACTUEL]`, mais **sans piège de focus, sans focus initial, sans restitution du focus** à l'élément
  déclencheur. `[CIBLE]` spécification dans `02-components-catalog.md` § 3.6.
- **Libellés (1.3.1, 3.3.2, 4.1.2)** : `Field` enveloppe le champ dans un `<label>` (association implicite)
  `[ACTUEL]` ; les champs sans `Field` ont un `aria-label` (« Rechercher », « Annotation », « Candidature »,
  « Nouvelle compétence », « Filtrer par statut », « Statut », « Rapport ») `[ACTUEL]`. Les boutons d'annotation 👍 /
  👎 de la page Offres n'ont qu'un `title` → `[CIBLE]` `aria-label` + `aria-pressed` comme dans Marché.
- **Annonce des changements (4.1.3, AA)** : `ErrorBox` a `role="alert"` `[ACTUEL]` ; les `Notice` de succès et le
  changement de statut d'une tâche ne sont pas annoncés → `[CIBLE]` `role="status"` sur `Notice` de ton `success`
  / `info` et région `aria-live="polite"` dans le tiroir pour le statut courant.
- **Onglets** : `Tabs` pose `role="tablist"`, `role="tab"`, `aria-selected` `[ACTUEL]` ; manquent
  `aria-controls`, `role="tabpanel"` et la navigation aux flèches → `[CIBLE]`.
- **Graphiques** : SVG avec `role="img"` + `aria-label` et `<title>` par barre / point `[ACTUEL]` ;
  `QuotaBar` utilise `role="progressbar"` avec `aria-valuenow` / `aria-valuemax` `[ACTUEL]`.
- **Icônes** : `Icon` est `aria-hidden="true"` ; tout bouton purement iconographique doit porter `aria-label`
  (le lien « Ouvrir sur LinkedIn » de `OfferCard` n'a qu'un `title` → `[CIBLE]`).
- **Saisie redondante (3.3.7, A, nouveau en 2.2)** : satisfait par les réponses mémorisées (I-05) et le
  pré-remplissage de `RunModal` depuis les Paramètres ; à conserver pour les nouveaux formulaires.
- **Authentification accessible (3.3.8, AA)** : sans objet (aucun mot de passe dans l'UI, jeton injecté).
- **Couleur seule (1.4.1)** : statuts toujours accompagnés d'un texte (badges libellés, « ✓ acquise »,
  « manquante ») `[ACTUEL]` ; le seuil de quotas (`QuotaBar`) a aussi le texte `used / limit`.

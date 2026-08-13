# UI Design Tokens & Context

Ce fichier centralise les jetons de design de l'application Web (ReactJS,
`services/email_analyzer/frontend/`). L'identité visuelle est un thème
clair et sobre par défaut — "bureau", pas "dashboard technique" : fond
blanc cassé/gris très clair, bleu profond pour les actions/l'IA, et les
couleurs vives (vert/orange/rouge) réservées aux statuts sémantiques
(santé projet, attention, urgence critique). Depuis la refonte
"chat-first" (Unit 33+, voir `context/progress-tracker.md`), un vrai
thème sombre existe aussi (bascule utilisateur, voir §1 et §4) — sa
palette reste dans le même esprit sobre, pas un retour au thème
"Tech/Security" (violet électrique + vert cyber) qui a existé plus tôt
dans le projet et qui a été abandonné.

Les noms de jetons ci-dessous sont ceux réellement définis dans
`services/email_analyzer/frontend/tailwind.config.js` (source de vérité —
pas de jetons `bg-base`/`ai-surface`/`status-*` distincts de ceux du code :
ce document décrivait auparavant une palette qui n'a jamais correspondu au
code réel, voir `context/rfc-email-pipeline-v2.md` §0 pour le même constat
côté backend). Depuis Unit 33, chaque jeton Tailwind ne pointe plus vers un
hex/rgba statique mais vers une variable CSS `--color-<slug>-rgb` (triplet
"R, G, B", parfois enveloppée d'une opacité fixe pour les halos comme
`ai-glow`) définie dans `services/email_analyzer/frontend/src/index.css` —
`:root` pour le clair, `[data-theme="dark"]` pour le sombre. Les noms de
jetons et leur usage restent identiques dans les deux thèmes ; seule la
valeur RGB change.

---

## 1. Palette de Couleurs & Jetons Sémantiques (Color Tokens)

| Catégorie             | Jeton Tailwind     | Valeur HEX                    | Usage & Rôle UI                                                                 |
| :--------------------- | :------------------ | :----------------------------- | :-------------------------------------------------------------------------------- |
| **Fond (Background)** | `bg-primary`        | `#FAFAF8`                      | Fond d'écran principal (blanc cassé).                                             |
|                        | `bg-secondary`      | `#F4F4F1`                      | Fonds secondaires (bandeaux, zones de repli).                                     |
|                        | `bg-tertiary`       | `#ECECE7`                      | Champs de saisie, badges neutres, skeletons de chargement.                        |
|                        | `surface`           | `#FFFFFF`                      | Cartes (projets, panneaux, modales) — contraste avec le fond off-white.           |
| **Texte (Text)**       | `text-primary`      | `#1E2530`                      | Titres, corps de texte principal.                                                 |
|                        | `text-secondary`    | `#59636F`                      | Descriptions, métadonnées, labels de section.                                     |
|                        | `text-muted`        | `#8A94A0`                      | Texte d'aide, placeholders, éléments désactivés ("Bientôt").                      |
| **Bordures (Border)** | `border-default`    | `#E2E1DC`                      | Bordures de cartes, séparateurs, champs de saisie.                                |
|                        | `border-subtle`     | `#ECEBE6`                      | Séparateurs discrets à l'intérieur d'une carte (ex. chat).                        |
| **Statuts (Semantic)** | `success`           | `#16A34A`                      | Vert : projet sur la bonne voie (`sentiment: on_track`).                          |
|                        | `warning`           | `#D97706`                      | Orange : point d'attention (échéance proche, action en attente).                  |
|                        | `danger`            | `#DC2626`                      | Rouge : urgence critique uniquement (`sentiment: under_tension`, risque critique). |
|                        | `info`              | `#2563EB`                      | Bleu : informations neutres, compteurs du Brief.                                  |
| **Accent (Actions/IA)** | `accent-primary`   | `#1D4ED8`                      | Bleu profond : boutons principaux, liens actifs, sidebar active.                  |
|                        | `accent-hover`      | `#1E40AF`                      | État survolé des éléments `accent-primary`.                                       |
|                        | `accent-soft`       | `rgba(29, 78, 216, 0.08)`      | Fond des éléments actifs légers (item de nav sélectionné).                        |
|                        | `ai-primary`        | `#1E3A8A`                      | Bleu plus profond : indicateurs "assistant"/IA (points de saisie, bulles).        |
|                        | `ai-secondary`      | `#3B5BDB`                      | Variante IA secondaire.                                                           |
|                        | `ai-glow`           | `rgba(30, 58, 138, 0.08)`      | Fond des blocs liés à l'IA (bandeau réponse rapide).                              |
| **Priorité (legacy)**  | `priority-high/medium/low` | `#DC2626` / `#D97706` / `#16A34A` | Alias de `danger`/`warning`/`success`, utilisés par l'ancien tableau de bord.  |
| **Fast-Track**         | `fasttrack-active`  | `#2563EB`                      | Pulsation d'une carte projet en cours de rafraîchissement.                        |
|                        | `fasttrack-bg`      | `rgba(37, 99, 235, 0.08)`      | Fond associé à `fasttrack-active`.                                                 |

Les valeurs HEX de la colonne ci-dessus sont celles du **thème clair**
(`:root` dans `index.css`) ; le tableau ne duplique pas la palette sombre
pour rester lisible — voir `[data-theme="dark"]` dans `index.css` pour les
valeurs exactes (fond navy/indigo profond `#0B0F1A`/`#131A2C`, texte clair
`#E8EAF0`, accent/IA éclaircis vers un bleu/indigo plus lumineux `#3B82F6`/
`#60A5FA`). Aucune classe Tailwind `dark:` n'est utilisée dans le code : le
basculement passe entièrement par l'attribut `data-theme` sur `<html>`
(posé par `theme/ThemeProvider.tsx`) et les variables CSS ci-dessus — les
composants n'ont donc rien à connaître du thème actif.
`priority-*`/`danger` restent dans la même famille de teinte (rouge) dans
les deux thèmes, réservés à l'urgence réelle (voir §4).

---

## 2. Recommandations Typographiques (Typography Scale)

Police système définie dans `tailwind.config.js::theme.extend.fontFamily` :

- **Sans-serif (`font-sans`, par défaut) :** `Inter`, puis `system-ui`, `sans-serif`.
- **Monospace (`font-mono`) :** `JetBrains Mono`, puis `monospace` — code, IDs, contenu brut.

Le reste de l'échelle (tailles/graisses) suit les classes Tailwind standard
(`text-xs` → `text-2xl`, `font-medium`/`font-semibold`) directement dans les
composants plutôt que des jetons nommés séparés.

---

## 3. Échelle des Rayons de Bordure (Border Radius Scale)

Définie dans `tailwind.config.js::theme.extend.borderRadius` :

| Jeton Tailwind | Valeur (Pixel) | Usage typique                                              |
| :-------------- | :-------------- | :----------------------------------------------------------- |
| `rounded-sm`     | `6px`            | Badges, petits éléments interactifs.                          |
| `rounded-md`     | `10px`           | Boutons standards, champs de saisie.                           |
| `rounded-lg`     | `14px`           | Cartes de projet, panneaux (sidebar, assistant).                |
| `rounded-xl`     | `18px`           | Cartes principales (Brief, modales), boutons d'action primaires.|
| `rounded-full`   | `9999px`         | Avatars, pastilles de statut, boutons ronds.                    |

---

## 4. Principes de mise en page

Hérités de la vision produit "assistante proactive", puis de la refonte
"chat-first" (voir `context/project-overview.md`) :

- Beaucoup d'espace blanc, cartes peu chargées, une information par bloc.
- La sidebar (`components/AppShell.tsx`) est la seule navigation persistante :
  Accueil / Projets / Agenda / Actions / Paramètres en mode SaaS (item
  "Assistant" retiré — l'accueil chat-first EST l'assistant portefeuille
  persistant, plus besoin d'une page séparée), Accueil seul en mode legacy
  (.env, sans DB — les autres routes redirigent vers `/` sans SaaS). Une
  barre de recherche en langage naturel en haut (SaaS uniquement) envoie
  vers `/?q=…`, lu par l'accueil au montage. Le pied de la sidebar porte le
  bouton de bascule clair/sombre (`components/ThemeToggle.tsx`).
- L'accueil (`pages/ChatHome.tsx`, routé sur `/` dans les deux modes) est
  chat-first : salutation centrée + avatar "orbe" en dégradé `ai-primary`→
  `ai-secondary`, champ de saisie unique, cartes d'action rapide, puis un
  fil de conversation (`components/chat/ChatThread.tsx`) une fois le
  premier message envoyé. En mode legacy, les résultats d'analyse
  apparaissent comme des cartes inline dans le fil
  (`components/chat/AnalysisCard.tsx`) plutôt qu'un panneau séparé.
- Le rouge (`danger`) est réservé aux urgences réelles — ne jamais l'utiliser
  pour un état neutre ou informatif, dans les deux thèmes.

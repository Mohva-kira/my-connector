# UI Design Tokens & Context

Ce fichier centralise les jetons de design de l'application Web (ReactJS,
`services/email_analyzer/frontend/`). L'identité visuelle est un thème clair
et sobre — "bureau", pas "dashboard technique" : fond blanc cassé/gris très
clair, bleu profond pour les actions/l'IA, et les couleurs vives (vert/
orange/rouge) réservées aux statuts sémantiques (santé projet, attention,
urgence critique). Ce choix remplace un thème sombre "Tech/Security" (violet
électrique + vert cyber) qui a existé plus tôt dans le projet.

Les noms de jetons ci-dessous sont ceux réellement définis dans
`services/email_analyzer/frontend/tailwind.config.js` (source de vérité —
pas de jetons `bg-base`/`ai-surface`/`status-*` distincts de ceux du code :
ce document décrivait auparavant une palette qui n'a jamais correspondu au
code réel, voir `context/rfc-email-pipeline-v2.md` §0 pour le même constat
côté backend).

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

Le thème est unique (pas de variante sombre) : aucune classe Tailwind `dark:`
n'est utilisée dans le code, tout passe par ces jetons nommés.

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

Hérités de la vision produit "assistante proactive" (voir
`context/project-overview.md`) :

- Beaucoup d'espace blanc, cartes peu chargées, une information par bloc.
- La sidebar (`components/AppShell.tsx`) est la seule navigation persistante :
  Brief / Projets / Agenda / Actions / Assistant / Paramètres, plus une barre
  de recherche en langage naturel en haut.
- Le rouge (`danger`) est réservé aux urgences réelles — ne jamais l'utiliser
  pour un état neutre ou informatif.

# UI Design Tokens & Context

Ce fichier centralise l'ensemble des jetons de design graphiques pour l'application Web (ReactJS) et Mobile (React Native). L'identité visuelle repose sur un mode sombre "Tech/Security" rehaussé par un vert cyber (Fiabilité/Chiffrement) et un violet électrique (Intelligence Artificielle).

---

## 1. Palette de Couleurs & Jetons Sémantiques (Color Tokens)

| Catégorie                 | Jeton (Token Name)    | Valeur HEX | Usage & Rôle UI                                                                   |
| :------------------------ | :-------------------- | :--------- | :-------------------------------------------------------------------------------- |
| **Fond (Background)**     | `bg-base`             | `#0B0F19`  | Fond d'écran principal de l'application (Sombre profond).                         |
|                           | `bg-surface`          | `#161B26`  | Arrière-plan des cartes (Cards), encadrés de projets et lignes de tableau.        |
|                           | `bg-surface-elevated` | `#1F2635`  | État survolé (Hover), menus déroulants et modales de validation.                  |
| **Texte (Text)**          | `text-primary`        | `#F3F4F6`  | Titres majeurs, texte principal des emails et données critiques (Lisibilité max). |
|                           | `text-secondary`      | `#9CA3AF`  | Descriptions de projets, métadonnées IMAP, horodatages (Timestamps).              |
|                           | `text-muted`          | `#6B7280`  | Texte d'aide secondaire, placeholders de formulaires et éléments désactivés.      |
| **Bordures (Border)**     | `border-dim`          | `#242C3D`  | Séparateurs discrets entre les projets ou lignes de logs.                         |
|                           | `border-focus`        | `#10B981`  | Contour actif lors de la sélection d'un champ ou d'une boîte mail connectée.      |
| **Accent IA (AI Assist)** | `ai-primary`          | `#7C3AED`  | Violet électrique : Boutons d'analyse IA, badges d'insights, triggers de RAG.     |
|                           | `ai-surface`          | `#2E1065`  | Fond des blocs de résumés d'emails ou de suggestions de projets par l'IA.         |
|                           | `ai-text`             | `#DDD6FE`  | Texte explicatif généré par le LLM ou suggestions de brouillons.                  |
| **Statuts (Semantic)**    | `status-success`      | `#059669`  | Vert : Synchronisation réussie, email chiffré stocké en DB, alias résolu.         |
|                           | `status-warning`      | `#D97706`  | Orange : Nouveau projet détecté en attente de validation manuelle.                |
|                           | `status-error`        | `#DC2626`  | Rouge : Échec de connexion IMAP/OAuth, tâche Celery en échec, token expiré.       |

---

## 2. Recommandations Typographiques (Typography Scale)

L'utilisation de polices de caractères géométriques et prévisibles renforce l'aspect technique et facilite la lecture des blocs d'emails denses.

| Jeton Typo          | Taille (Web) | Taille (Mobile) | Graisse (Weight) | Usage UI                                                                 |
| :------------------ | :----------- | :-------------- | :--------------- | :----------------------------------------------------------------------- |
| `font-mono-code`    | `13px`       | `13pt`          | Regular (400)    | Contenu brut de l'email, logs des tâches Celery, IDs de projets.         |
| `font-sans-body`    | `14px`       | `14pt`          | Regular (400)    | Corps des résumés d'IA, listes de projets, texte général de l'interface. |
| `font-sans-label`   | `12px`       | `12pt`          | Medium (500)     | Badges de statut, en-têtes de colonnes, noms des alias de projets.       |
| `font-sans-title`   | `18px`       | `18pt`          | SemiBold (600)   | Titres des cartes de projets, objets des emails prioritaires.            |
| `font-sans-display` | `24px`       | `24pt`          | Bold (700)       | Titre principal du tableau de bord, vue macro de l'assistant.            |

- **Police recommandée (Sans-Serif) :** `Inter` ou `Geist Sans` (Web), `System Default` (Mobile).
- **Police recommandée (Monospace) :** `JetBrains Mono` ou `Fira Code` (Web/Mobile).

---

## 3. Échelle des Rayons de Bordure (Border Radius Scale)

Une échelle de coins subtilement arrondis maintient un aspect "logiciel pro / dashboard technique" sans tomber dans un style trop grand public ou trop rigide.

| Jeton Rayon   | Valeur (Pixel) | Application UI Concrete                                                            |
| :------------ | :------------- | :--------------------------------------------------------------------------------- |
| `radius-none` | `0px`          | Séparateurs stricts, bordures de l'écran ou de conteneurs pleine largeur.          |
| `radius-sm`   | `4px`          | Cases à cocher (Checkboxes), badges de statut (Success/Warning) et infobulles.     |
| `radius-md`   | `8px`          | Boutons d'action ("Synchroniser", "Valider le projet"), champs de saisie (Inputs). |
| `radius-lg`   | `12px`         | Boîtes de dialogue, modales d'analyse approfondie, cartes de projet (Cards).       |
| `radius-full` | `9999px`       | Boutons d'avatars utilisateurs, indicateurs de bascule (Toggle switches).          |

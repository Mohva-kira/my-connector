# Design System & Charte UI

## 1. Tokens de Design
- **Palette de Couleurs** :
  - `primary` : `[Ex: #2563EB / HSL / Tailwind class]`
  - `secondary` : `[Ex: #64748B]`
  - `accent` : `[Ex: #F59E0B]`
  - `neutral / background` : `[Ex: #FFFFFF / #0F172A (dark)]`
  - `destructive / error` : `[Ex: #EF4444]`
  - `success` : `[Ex: #10B981]`
- **Typographie** :
  - Police principale : `[Ex: Inter, Roboto, sans-serif]`
  - Titres (Headings) : `[Tailles et graisses : H1 font-bold 32px, H2 font-semibold 24px...]`
  - Corps de texte : `[font-normal 14px / 16px]`
- **Espacements & Rayons de bordure** :
  - Unités de base : `[Ex: 4px base scale, tailwind space]`
  - Radius : `[Ex: rounded-md (6px), rounded-lg (8px)]`

## 2. Gestion des Thèmes
- **Mode Sombre / Clair** : `[Ex: Détection système par défaut + toggle manuel via cookie/localStorage]`
- **Tokens sémantiques** : Toujours utiliser les classes sémantiques plutôt que des couleurs brutes codées en dur.

## 3. Responsive Breakpoints
- `sm` : 640px (Mobile grand écran)
- `md` : 768px (Tablette)
- `lg` : 1024px (Desktop standard)
- `xl` : 1280px (Écrans larges)

## 4. Règles d'Accessibilité (a11y)
- Contraste minimal WCAG AA (4.5:1 pour le texte normal).
- États de focus visibles obligatoires (`focus-visible:ring-2`).
- Remplissage systématique des attributs `aria-label`, `alt` et rôles sémantiques.

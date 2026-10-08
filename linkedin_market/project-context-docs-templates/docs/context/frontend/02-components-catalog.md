# Catalogue de Composants UI

## 1. Organisation des Composants
- `components/ui/` : Composants de base atomes/molécules (Button, Input, Modal, Badge).
- `components/forms/` : Formulaires et champs complexes contrôlés.
- `components/layout/` : Header, Sidebar, Footer, Breadcrumbs, PageContainer.
- `components/features/[domaine]/` : Composants dédiés aux fonctionnalités métier.

## 2. États Obligatoires par Composant
Chaque composant interactif doit implémenter et gérer les états suivants :
- `Default` : État nominal au repos.
- `Hover` : Survol souris avec transition fluide.
- `Active / Pressed` : Clic ou activation.
- `Disabled` : Non interactif, opacité réduite, curseur `not-allowed`.
- `Loading / Busy` : Indicateur de chargement / skeleton loader sans layout shift.
- `Error` : Contour ou signalétique rouge avec message d'erreur associé.
- `Empty State` : Affichage explicatif lorsqu'aucune donnée n'est disponible.

## 3. Conventions de Nommage & Exports
- **Fichiers** : PascalCase (ex: `TransactionTable.tsx`) ou kebab-case selon la convention du projet.
- **Props interface** : `[NomComposant]Props` (ex: `ButtonProps`).
- Toujours exporter le composant avec typage TypeScript strict des props.

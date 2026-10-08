# Gestion d'État & Data Fetching

## 1. Répartition de l'État
- **État Local** : `useState`, `useReducer` pour l'état interne d'un composant isolé.
- **État d'Interface Global** : `[Ex: Zustand / Redux]` pour sidebar open/close, préférences de modale, thèmes.
- **État Serveur (Server State)** : `[Ex: TanStack Query (React Query) / SWR]` pour toutes les données distantes.

## 2. Stratégie de Requêtage (Query & Mutations)
- **Clés de cache** : Conventions de clés hiérarchiques (ex: `['tickets', 'list', { status, page }]`).
- **Invalidation après mutation** :
  - Toute modification de donnée doit explicitement invalider les listes et entités impactées.
  - Utiliser les mises à jour optimistes (optimistic updates) uniquement pour les actions réversibles rapides.
- **Gestion des requêtes en cours** : Affichage d'états `isPending` / `isFetching` ciblés pour ne pas figer toute la page.

## 3. Gestion des Erreurs Réseau
- Intercepteur HTTP global pour capturer les codes HTTP `401` (redirection login) et `403` (accès refusé).
- Toasts d'erreur automatiques pour les codes `500` et pannes de connectivité.
- Utilisation de React Error Boundaries pour isoler les crashs de composants critiques.

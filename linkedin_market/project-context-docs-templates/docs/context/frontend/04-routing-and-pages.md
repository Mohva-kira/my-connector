# Routage, Pages & Navigation

## 1. Arborescence des Routes
- `/` : Page d'accueil / Landing page publique.
- `/auth/login` : Authentification.
- `/auth/register` : Inscription.
- `/app/dashboard` : Tableau de bord principal (protégé).
- `/app/[feature]` : Pages fonctionnelles métier.
- `/admin/*` : Espace d'administration réservé aux super-utilisateurs.

## 2. Guards & Middlewares de Navigation
- **Redirection Non-Authentifié** : Rediriger systématiquement vers `/auth/login?redirect=[URL]` si session absente.
- **Contrôle d'Accès par Rôle (RBAC UI)** : Masquer les liens du menu et bloquer la navigation si le rôle utilisateur est insuffisant.

## 3. SEO, Titres & Métadonnées
- Définition systématique de balises `<title>` claires : `[Titre de la Page] | [Nom Application]`.
- Configuration des loaders de route / `loading.tsx` pour éliminer les écrans blancs pendant la navigation.

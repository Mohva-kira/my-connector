# Authentification & Matrice de Permissions

## 1. Stratégie d'Authentification
- Mécanisme : `[Ex: JWT (Access Token court 15 min + Refresh Token sécurisé HttpOnly Cookie)]`
- Algorithme de signature : `[Ex: RS256 ou HS256 avec secret fort]`
- Révocation : Blacklist de tokens sur Redis en cas de déconnexion ou changement de mot de passe critique.

## 2. Matrice RBAC (Rôles & Permissions)
| Rôle | Permission | Périmètre |
| :--- | :--- | :--- |
| `SuperAdmin` | `*` (Toutes permissions) | Système complet |
| `Admin` | `organization:*`, `users:manage` | Organisation courante |
| `Manager` | `items:read`, `items:write` | Équipe / Département |
| `Viewer` | `items:read` | Lecture seule |

## 3. Rate Limiting & Protection Anti-Abus
- Limite publique (routes d'authentification) : `[Ex: Max 5 tentatives par minute par IP]`.
- Limite authentifiée (API standard) : `[Ex: Max 100 requêtes par minute par utilisateur]`.
- En-têtes standards retournés : `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After`.

# Infrastructure, Déploiement & Environnements

## 1. Topologie d'Hébergement
- **Environnements** :
  - `Development` : Environnement local (Docker Compose).
  - `Staging` : Environnement de recette miroir de la production.
  - `Production` : Haute disponibilité, redondance et sauvegardes automatiques.
- **Conteneurisation** : Images Docker multi-stage builds optimisées pour la légèreté et la sécurité.

## 2. Variables d'Environnement Requises
| Variable | Description | Exemple / Format | Obligatoire |
| :--- | :--- | :--- | :--- |
| `NODE_ENV` | Environnement d'exécution | `production` / `development` | Oui |
| `PORT` | Port d'écoute du serveur | `3000` | Oui |
| `DATABASE_URL` | URL de connexion à la base | `postgresql://user:pass@host:5432/db` | Oui |
| `REDIS_URL` | URL du serveur Redis | `redis://default:pass@host:6379` | Oui |
| `JWT_SECRET` | Secret de signature des tokens | Chaîne aléatoire cryptographique | Oui |

## 3. Pipeline CI/CD
1. **Lint & Format Check** : Vérification ESLint / Prettier.
2. **Type Check** : `tsc --noEmit` sans erreurs tolérées.
3. **Tests Automatisés** : Exécution des tests unitaires et d'intégration.
4. **Build** : Compilation du bundle frontend et de l'artefact backend.
5. **Déploiement Automatique** : Déploiement en production si la branche `main` est mise à jour et que les checks sont verts.

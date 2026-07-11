# Code Standards & Engineering Guidelines

Ce document définit les standards de développement, les règles de formatage et les exigences de qualité logicielle pour l'écosystème de l'application. L'objectif est de garantir la maintenabilité d'une architecture hybride impliquant un backend Python asynchrone (FastAPI/Celery) et des interfaces clients en JavaScript/TypeScript (React et React Native), tout en assurant une étanchéité absolue lors du traitement et du chiffrement des données sensibles (emails et pièces jointes).

## 1. Objectifs de Qualité du Code (Engineering Goals)

1. **Typage Stricte Obligatoire :** Maximiser la détection d'erreurs à la compilation et à l'analyse statique en imposant un typage à 100% sur le backend (Type Hints Python avec MyPy) et le frontend (TypeScript en mode strict).
2. **Découplage des Effets de Bord :** Isoler systématiquement la logique métier pure des accès à la base de données PostgreSQL ou des appels d'API (LLM, Google), en appliquant les principes de la Clean Architecture.
3. **Sécurité par Défaut (Chiffrement) :** Garantir qu'aucune fonction de manipulation de texte ou d'indexation ne puisse manipuler des données d'emails sans passer explicitement par la couche de déchiffrement temporaire en mémoire vive (RAM).
4. **Performance du Thread Principal :** Interdire toute opération d'I/O bloquante ou synchrone dans les contrôleurs d'API et les composants UI afin de maintenir un temps de réponse inférieur à 200ms.

## 2. Flux de Développement de Bout en Bout (Core Development Workflow)

1. **Création de Branche :** Le développeur crée une branche issue de `main` en respectant la nomenclature stricte : `feat/nom-fonctionnalite`, `fix/nom-bug`, ou `chore/nom-tâche`.
2. **Écriture du Code et Tests :** Développement de la fonctionnalité accompagnée obligatoirement de ses tests unitaires. Pour le backend Python, chaque nouveau worker Celery ou service doit atteindre une couverture minimale de 80%.
3. **Analyse Statique Locale :** Exécution des outils de peluchage (Linting) et de formatage avant chaque commit via des hooks de pré-engagement (Pre-commit hooks) : Black/Ruff pour Python, ESLint/Prettier pour React.
4. **Soumission de Pull Request (PR) :** Ouverture d'une PR vers `main`. Le modèle de PR doit obligatoirement lier l'ID de la tâche et décrire l'impact sur les performances ou la sécurité des données.
5. **Intégration Continue (CI) :** Les tests automatisés, la vérification du typage (`mypy` et `tsc`), et l'audit des vulnérabilités des dépendances s'exécutent dans la pipeline de CI.
6. **Relecture Technique & Fusion :** Validation obligatoire par au moins un pair (Peer Review). La fusion dans `main` se fait exclusivement par *Squash and Merge* pour garder un historique de commit propre.

## 3. Standards Spécifiques par Catégorie Technique

### Python Backend (FastAPI / Celery)
* **Formatage :** Strict respect de la PEP 8 via l'outil `Ruff` ou `Black`. Longueur maximale des lignes fixée à 88 caractères.
* **Asynchronisme :** Utilisation préférentielle de `async/await` pour les routes de l'API. Les fonctions effectuant des calculs lourds (ex: parsing PDF) doivent être définies comme des tâches Celery synchrones s'exécutant dans des workers isolés.
* **Gestion des Exceptions :** Interdiction d'utiliser des blocs `try/except Exception: pass` génériques. Chaque exception doit être capturée explicitement, loggée de manière anonymisée (sans données d'emails), et retourner un code d'erreur standardisé.

### Frontend TypeScript (ReactJS / React Native)
* **Composants :** Utilisation exclusive de composants fonctionnels avec des Hooks personnalisés (`custom hooks`) pour séparer la logique d'état et de requêtage de la couche graphique.
* **Gestion d'État Local (Device) :** Toute interaction avec la base SQLite locale ou l'index vectoriel (Orama/Voy) doit être encapsulée dans un *Provider* ou un *Context* React dédié, avec gestion explicite des états de chargement (`loading`) et d'erreur.
* **Typage des API :** Interdiction d'utiliser le type `any`. Tous les payloads retournés par le backend Python doivent avoir une interface TypeScript correspondante générée ou partagée.

### Sécurité et Cryptographie
* **Variables d'Environnement :** Aucune clé secrète ne doit être écrite en dur dans le code. Les clés de chiffrement et tokens d'API doivent être lus via un gestionnaire de configuration (`Pydantic Settings` pour Python, `process.env` pour React).
* **Isolation Temporelle :** Les variables contenant du texte d'email déchiffré en RAM doivent être nettoyées ou écrasées dès que le traitement par le LLM ou l'affichage est terminé.

## 4. Pratiques Incluses (In-Scope Verification)
* Validation automatique du code à chaque commit via des scripts de peluchage (Linter) locaux.
* Écriture systématique de mocks pour les API externes (Gmail, OpenAI, Anthropic) dans l'environnement de test.
* Utilisation d'outils de migration de base de données (Alembic pour PostgreSQL) pour toute modification de schéma.
* Documentation des fonctions complexes et des points d'accès API directement dans le code via des Docstrings au format Google.

## 5. Pratiques Exclues (Out-of-Scope Practices)
* **Pas de micro-optimisation prématurée :** Ne pas réécrire de modules de parsing en C/Rust tant que les performances du worker Python actuel respectent les SLAs établis.
* **Pas de stockage de données sensibles en clair dans les logs :** L'affichage des variables contenant des tokens, des adresses emails, ou des contenus de messages dans la console (`print` ou `logger.info`) est strictement banni.
* **Pas de manipulation directe de la DB sans ORM :** Interdiction d'exécuter des requêtes SQL brutes sous forme de chaînes de caractères sans passer par l'ORM (SQLAlchemy/SQLModel) afin de prévenir les injections SQL.

## 6. Critères de Validation du Code (Definition of Done)

* **Zéro Erreur de Typage :** Les vérifications `mypy` (backend) et `tsc` (frontend) s'exécutent sans aucun avertissement ni contournement (`# type: ignore` ou `ts-ignore` interdits).
* **Couverture de Tests Minimale :** La suite de tests unitaires et d'intégration affiche un taux de couverture minimal de 80% sur l'ensemble de la logique métier (dossier `core/` et `workers/`).
* **Zéro Fuite de Données en Cache :** Les tests d'intégration du frontend valident que le script de nettoyage supprime effectivement les données de plus de 30 jours du simulateur de stockage local.
* **Validation de Sécurité :** L'analyseur de code statique de sécurité (comme `Bandit` pour Python) est exécuté et ne remonte aucune faille de sévérité "Moyenne" ou "Haute".
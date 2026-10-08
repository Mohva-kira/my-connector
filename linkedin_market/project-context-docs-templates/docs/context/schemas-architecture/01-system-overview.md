# Vue d'Ensemble du Système & Architecture

## 1. Style Architectural Global
- Modèle général : `[Ex: Monolithe Modulaire / Microservices / Serverless]`
- Frontends connectés : `[Ex: Web App (Next.js), Mobile App (React Native)]`
- Backend principal : `[Ex: API REST Node.js / NestJS ou Python / FastAPI]`
- Persistance & Caches : `[Ex: PostgreSQL + Redis]`

## 2. Diagramme C4 Mermaid (Niveau Système & Conteneurs)
```mermaid
C4Context
  title Diagramme Système Global

  Person(user, "Utilisateur Final", "Utilise la plateforme via son navigateur ou mobile")
  System_Boundary(platform, "Plateforme Applicative") {
    Container(webApp, "Frontend Web App", "React / Next.js", "Interface utilisateur responsive")
    Container(apiService, "Backend Core API", "NestJS / FastAPI", "Logique métier, auth, points d'API")
    ContainerDb(database, "Base de Données", "PostgreSQL", "Stockage persistant relationnel")
    ContainerDb(cacheQueue, "Cache & Broker", "Redis", "Gestion des sessions, cache et files d'attente")
  }

  System_Ext(externalService, "Services Tiers", "Passerelles de paiement, envoi d'emails")

  Rel(user, webApp, "Navigue et interagit", "HTTPS")
  Rel(webApp, apiService, "Consomme les APIs", "JSON/HTTPS")
  Rel(apiService, database, "Lit et écrit", "TCP/SSL")
  Rel(apiService, cacheQueue, "Enqueue tâches et cache", "TCP")
  Rel(apiService, externalService, "Appelle via API", "HTTPS")
```

## 3. Frontières et Responsabilités
- **Frontend** : Rendu visuel, capture des intentions utilisateur, validations de premier niveau, gestion de session locale.
- **Backend Core** : Garant des invariants de données, autorisations strictes, opérations transactionnelles.
- **Systèmes Externes** : Délégués uniquement pour les compétences non-core (livraison de mails, encaissement bancaire).

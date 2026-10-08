# Flux de Données & Diagrammes de Séquence

## 1. Flux Critique 1 : Authentification & Accès aux Données
```mermaid
sequenceDiagram
    autonumber
    actor User as Utilisateur
    participant Front as Frontend (Client)
    participant API as Backend API
    participant DB as Base de Données

    User->>Front: Saisit identifiants et valide
    Front->>API: POST /api/v1/auth/login
    API->>DB: Recherche utilisateur & vérifie mot de passe
    DB-->>API: Utilisateur validé
    API-->>Front: 200 OK + JWT Tokens (Cookie HttpOnly)
    Front-->>User: Redirection vers le tableau de bord
```

## 2. Flux Critique 2 : Traitement d'une Action Métier Asynchrone
```mermaid
sequenceDiagram
    autonumber
    actor User as Utilisateur
    participant Front as Frontend
    participant API as Backend API
    participant Queue as Redis Queue (BullMQ)
    participant Worker as Background Worker

    User->>Front: Déclenche une opération lourde
    Front->>API: POST /api/v1/jobs/trigger (avec Idempotency-Key)
    API->>Queue: Push tâche dans la file
    API-->>Front: 202 Accepted (jobId)
    Front-->>User: Notification 'Traitement en cours'
    Queue->>Worker: Dépilage du job
    Worker->>Worker: Traitement asynchrone
    Worker-->>API: Notifie statut terminé / Webhook
```

## 3. Résilience & Gestion des Pannes Réseau
- Stratégie de timeout : 10 secondes maximum par requête API synchrone.
- Circuit Breaker configuré pour isoler les services tiers défaillants sans bloquer le reste de l'application.

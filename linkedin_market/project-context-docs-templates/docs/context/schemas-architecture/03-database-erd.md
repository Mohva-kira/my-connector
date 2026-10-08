# Schéma Entité-Relation (ERD)

## 1. Diagramme Entité-Relation Mermaid
```mermaid
erDiagram
    ORGANIZATIONS ||--o{ USERS : "contient"
    ORGANIZATIONS ||--o{ ITEMS : "possède"
    USERS ||--o{ ITEMS : "crée"
    USERS ||--o{ AUDIT_LOGS : "génère"

    ORGANIZATIONS {
        uuid id PK
        string name
        string slug UK
        string status
        timestamp created_at
        timestamp updated_at
    }

    USERS {
        uuid id PK
        uuid organization_id FK
        string email UK
        string password_hash
        string role
        string status
        timestamp created_at
        timestamp updated_at
    }

    ITEMS {
        uuid id PK
        uuid organization_id FK
        uuid created_by FK
        string title
        string status
        numeric amount
        timestamp created_at
        timestamp updated_at
    }

    AUDIT_LOGS {
        uuid id PK
        uuid user_id FK
        string action
        string entity_name
        uuid entity_id
        jsonb metadata
        timestamp created_at
    }
```

## 2. Énumérations & Dictionnaire de Données
- **Status (ORGANIZATIONS)** : `ACTIVE`, `SUSPENDED`, `TRIAL`.
- **Status (USERS)** : `PENDING_ACTIVATION`, `ACTIVE`, `BLOCKED`.
- **Roles (USERS)** : `SUPER_ADMIN`, `ADMIN`, `MEMBER`, `VIEWER`.
- **Status (ITEMS)** : `DRAFT`, `PUBLISHED`, `ARCHIVED`.

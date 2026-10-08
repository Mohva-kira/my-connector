# Base de Données, Modélisation & Migrations

## 1. Stack Base de Données
- SGBD : `[Ex: PostgreSQL 16]`
- Outil d'ORM / Migration : `[Ex: Prisma / TypeORM / Drizzle / Flyway]`

## 2. Conventions de Schéma
- Noms de tables : snake_case, pluriel (ex: `users`, `invoices`, `audit_logs`).
- Clés primaires : UUIDv4 ou UUIDv7 recommandés (`id`).
- Colonnes d'audit obligatoires sur chaque table :
  - `created_at` (TIMESTAMP WITH TIME ZONE, DEFAULT NOW())
  - `updated_at` (TIMESTAMP WITH TIME ZONE, DEFAULT NOW())
- Soft-delete : Si activé, utiliser `deleted_at` (NULLABLE).

## 3. Politique des Migrations
- Tout changement de schéma passe par un script de migration versionné et traçable dans Git.
- Les migrations en production doivent être rétrocompatibles (principe d'expansion puis contraction).
- Tout champ utilisé fréquemment dans les clauses `WHERE` ou `JOIN` doit posséder un index dédié ou composite.
- Mise en œuvre de politiques RLS (Row-Level Security) pour l'isolation multi-tenant le cas échéant.

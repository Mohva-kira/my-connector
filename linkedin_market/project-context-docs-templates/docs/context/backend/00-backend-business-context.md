# Contexte Métier & Règles Système - Backend

## 1. Concepts Métier du Domaine (Core Domain Entities)
- **[Entité Principale 1 - ex: Compte / Dossier / Transaction]** :
  - Définition : [Rôle de cette entité dans le système]
  - Cycle de vie / Statuts possibles : `[STATUS_DRAFT -> STATUS_ACTIVE -> STATUS_ARCHIVED]`
  - Règles d'appartenance : [Ex: Appartient obligatoirement à un Tenant / Organisation]
- **[Entité Principale 2]** :
  - Définition :
  - Statuts :

## 2. Invariants & Règles Métier Strictes
Les règles suivantes ne doivent JAMAIS être enfreintes par aucun endpoint ni script :
- **Invariant 1** : [Ex: Un solde de compte ne peut jamais devenir négatif sans autorisation de découvert préalable].
- **Invariant 2** : [Ex: Une facture émise ne peut plus être modifiée ni supprimée (uniquement annulée par un avoir)].
- **Invariant 3** : [Ex: L'isolation multi-locataire (multi-tenant) doit être garantie sur 100% des requêtes en base].

## 3. Actions & Cas d'Usage Métier (Use-Cases)
### Cas d'Usage : `[Nom de l'action métier - ex: ExecuterPaiement]`
- **Conditions préalables (Pre-conditions)** :
  - Utilisateur authentifié possédant la permission requise.
  - Ressource cible dans un état éligible : `[Statut attendu]`.
- **Exécution transactionnelle** :
  - Toutes les écritures doivent être exécutées dans une transaction ACID unifiée.
- **Effets de bord autorisés** :
  - Émission de l'événement système : `[NomDeLEvenement]`.
  - Notification asynchrone transmise à la file d'attente.

## 4. Résultats Finaux Attendus
- **Persistance** : Mise à jour atomique en base avec horodatage UTC et identifiant d'audit.
- **Garantie d'Idempotence** : Utilisation d'une clé d'idempotence (`Idempotency-Key`) pour éviter tout double traitement lors des réessais réseau.
- **Réponse HTTP** : Payload unifié retournant l'état final stabilisé de l'entité.

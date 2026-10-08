# Contexte Métier Frontend

## 1. Périmètre & Utilisateurs Cibles (Personas)
- **Persona 1 ([Nom / Rôle])** :
  - Profil & Contexte d'usage : [Ex: Sur desktop au bureau, sur mobile sur le terrain]
  - Objectifs principaux : [Ex: Saisie rapide, consultation de tableaux de bord, traitement d'alertes]
  - Contraintes d'ergonomie : [Ex: Faible bande passante, besoin de mode sombre, accessibilité clavier]
- **Persona 2 ([Nom / Rôle])** :
  - Profil & Contexte d'usage :
  - Objectifs principaux :

## 2. Parcours Clés & Actions Utilisateur
### Parcours 1 : [Nom du parcours - ex: Création de dossier / Checkout]
- **Déclencheur (Trigger)** : [Ex: Clic sur 'Nouveau dossier', soumission du panier]
- **Étapes utilisateur** :
  1. [Étape 1 : Saisie des données de base]
  2. [Étape 2 : Validation en ligne des contraintes]
  3. [Étape 3 : Confirmation explicite / Modale récapitulative]
- **Feedback & Interactions** : [Ex: Spinner inline, désactivation du bouton pendant le traitement]

### Parcours 2 : [Nom du parcours]
- **Déclencheur** :
- **Étapes** :
- **Feedback** :

## 3. Résultats Finaux Attendus (Expected Client Outcomes)
- **États Visuels de Succès** :
  - Notification toast de confirmation contextualisée : `[Format du message de confirmation]`.
  - Redirection automatique vers : `[Route cible avec refresh des données de vue]`.
- **Résilience & Gestion des Échecs** :
  - Message d'erreur clair et actionnable (pas de code technique brut affiché à l'utilisateur).
  - Préservation intégrale de la saisie utilisateur en cas d'erreur réseau pour éviter la ressaisie.
- **Accessibilité & Conformité** :
  - Navigation complète au clavier assurée sur l'ensemble des parcours critiques.

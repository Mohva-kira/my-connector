# Contexte Architectural du Domaine & Glossaire

## 1. Glossaire Métier Partagé (Ubiquitous Language)
Définition formelle des termes pour aligner l'équipe technique, l'IA et le métier :
- **[Terme Métier 1]** : [Définition exacte et sans ambiguïté - interdit d'utiliser un autre mot dans le code].
- **[Terme Métier 2]** : [Définition].
- **[Terme Métier 3]** : [Définition].

## 2. Matrice Actions Métier -> Impact Système
| Action Métier | Entrée Frontend | Services Backend Impactés | Tables Modifiées | Événements / Files |
| :--- | :--- | :--- | :--- | :--- |
| `[Ex: ValiderCommande]` | Clic 'Payer' | OrderService, PaymentGateway | `orders`, `transactions` | `OrderPlacedEvent` -> BullMQ |
| `[Ex: InviterMembre]` | Formulaire email | UserService, MailService | `users`, `invitations` | `SendInviteEmailJob` |

## 3. Niveaux de Consistance & SLA Système
- **Consistance Forte (ACID)** : [Ex: Pour les soldes financiers, les autorisations d'accès et les réservations de stock].
- **Consistance Éventuelle (Eventual Consistency)** : [Ex: Pour la mise à jour des métriques de dashboard, les index de recherche plein texte et les notifications].
- **Traçabilité Globale** : Un identifiant de corrélation (`X-Correlation-Id`) est propagé du frontend jusqu'aux workers backend pour tout audit.

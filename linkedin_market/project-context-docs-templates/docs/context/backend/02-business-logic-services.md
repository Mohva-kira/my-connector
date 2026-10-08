# Logique Métier & Architecture des Services

## 1. Découpage en Couches (Clean / Hexagonal)
1. **Controller / Handler Layer** :
   - Reçoit la requête HTTP, valide le format du DTO d'entrée, appelle le use-case, formate le code HTTP.
   - Ne contient AUCUNE logique métier ni calcul.
2. **Service / Use-Case Layer** :
   - Contient les règles métier, valide les invariants et gère les transactions.
3. **Repository / Data Access Layer** :
   - Isole les requêtes SQL/ORM de la logique métier.
4. **Integration / Gateway Layer** :
   - Gère les appels vers les APIs tierces (Stripe, Twilio, S3, etc.).

## 2. Gestion des Transactions & Verrous
- Délimiter explicitement le scope des transactions de base de données.
- Privilégier les verrous optimistes (`version` column) pour les ressources à faible concurrence.
- Utiliser les verrous pessimistes (`SELECT FOR UPDATE`) pour les flux de balance financière ou d'allocation de stock critique.

## 3. Logging & Traçabilité
- Logs obligatoirement structurés en JSON :
  `{"timestamp": "...", "level": "INFO", "traceId": "...", "message": "...", "context": {...}}`
- Ne JAMAIS journaliser de mots de passe, clés secrètes, tokens ou données personnelles hautement sensibles.

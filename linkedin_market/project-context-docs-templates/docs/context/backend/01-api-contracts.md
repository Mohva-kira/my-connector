# Contrats d'API & Conventions

## 1. Standards d'API
- Style d'architecture : RESTful (JSON sur HTTP/HTTPS).
- Préfixe de versioning : `/api/v1/...`
- Conventions de casse :
  - URLs : kebab-case (ex: `/api/v1/user-profiles`)
  - Clés JSON : camelCase (ex: `{"userId": "123", "createdAt": "2026-01-01T00:00:00Z"}`)

## 2. Format Standard des Réponses
### Réponse de Succès
```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 100
  }
}
```

### Réponse d'Erreur
```json
{
  "success": false,
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "La ressource demandée n'existe pas.",
    "details": []
  }
}
```

## 3. Headers HTTP Standards
- `Authorization: Bearer <token>`
- `Content-Type: application/json`
- `X-Correlation-Id: <uuid>` (Traçabilité des requêtes)
- `Idempotency-Key: <uuid>` (Obligatoire pour les opérations critiques `POST` / `PATCH`)

## 4. Codes de Statut Utilisés
- `200 OK` : Succès lecture ou mise à jour synchrone.
- `201 Created` : Ressource créée avec succès.
- `202 Accepted` : Tâche acceptée en arrière-plan (asynchrone).
- `400 Bad Request` : Erreur de validation de payload.
- `401 Unauthorized` : Token manquant ou expiré.
- `403 Forbidden` : Token valide mais permissions insuffisantes.
- `404 Not Found` : Ressource inexistante.
- `409 Conflict` : Conflit d'état ou violation d'unicité.
- `422 Unprocessable Entity` : Violation d'une règle métier.
- `500 Internal Server Error` : Erreur serveur non rattrapée.

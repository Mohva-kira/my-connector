# Formulaires & Validation

## 1. Bibliothèques et Standards
- Gestionnaire de formulaire : `[Ex: React Hook Form]`
- Schéma de validation : `[Ex: Zod]`
- Toujours inférer les types TypeScript directement depuis le schéma Zod :
  `type FormValues = z.infer<typeof formSchema>;`

## 2. Règles de Validation UX
- **Validation à la soumission initiale** : Ne pas agresser l'utilisateur d'erreurs avant la première tentative d'envoi.
- **Re-validation en direct (onChange / onBlur)** : Réactiver la validation dynamique dès qu'un champ en erreur est modifié.
- **Affichage des erreurs** : Message textuel explicite directement sous le champ concerné.

## 3. Upload de Fichiers & Pièces Jointes
- Validation stricte côté client :
  - Types MIME acceptés : `[Ex: application/pdf, image/png, image/jpeg]`.
  - Taille maximale : `[Ex: 5 Mo par fichier]`.
- Indicateur de progression pour les uploads asynchrones.

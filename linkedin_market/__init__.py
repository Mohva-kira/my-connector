"""Analyse du marché LinkedIn pour un mot-clé de poste (freelance, full remote).

Pipeline : saisie utilisateur -> scraping (offres, posts, profils) ->
extraction statistique locale -> tendances 30 jours -> matching par
régression logistique (descente de gradient) -> interprétation (Claude
ou modèle local) -> rapport HTML + JSON.

Lancement : ``python -m linkedin_market``
"""

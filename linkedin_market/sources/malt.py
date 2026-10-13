"""Malt (source ``malt``, mode ``navigateur``) : lecture seule des offres de mission (``backend/05`` § 2.6).

Malt est surtout une place de marché de profils de freelances : seules les **offres** sont lues, jamais un profil
(I-09) ; les liens vers un profil sont ignorés par ``BrowserJobSource``. Mêmes règles que LinkedIn : consentement
versionné (I-16), compte propre (I-08), plafond de la prudence (I-07), arrêt et repos de 6 h au premier blocage
(I-06). Aucune candidature ni aucun message sur la plateforme.
"""

from __future__ import annotations

import re

from .base import BrowserJobSource

# NON VÉRIFIÉ EN RÉEL : adresse de recherche de missions et forme des liens d'offre à confirmer sur le site.
SEARCH_URL = "https://www.malt.fr/s/missions?q={keyword}"
OFFER_PATH = re.compile(r"/missions?/(?!(?:search|new|create)/?$)(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)/?$")


class MaltSource(BrowserJobSource):
    code = "malt"
    search_url = SEARCH_URL
    offer_pattern = OFFER_PATH

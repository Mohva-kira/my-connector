"""Welcome to the Jungle (source ``wttj``, mode ``navigateur``) : lecture seule des offres (``backend/05`` § 2.6).

Mêmes règles que LinkedIn : consentement versionné (I-16), compte propre (I-08), plafond de la prudence (I-07),
arrêt et repos de 6 h au premier blocage (I-06). Aucune candidature ni aucun message sur la plateforme.
"""

from __future__ import annotations

import re

from .base import BrowserJobSource, Page

# NON VÉRIFIÉ EN RÉEL : adresse de recherche d'offres et forme des liens d'offre à confirmer sur le site.
SEARCH_URL = "https://www.welcometothejungle.com/fr/jobs?query={keyword}"
# Lien d'offre : /fr/companies/<entreprise>/jobs/<offre> (le texte visible du lien d'entreprise donne son nom).
OFFER_PATH = re.compile(r"/companies/[^/]+/jobs/(?P<slug>[^/]+)/?$")
COMPANY_PATH = re.compile(r"/companies/(?P<company>[^/]+)/?$")
OFFER_COMPANY = re.compile(r"/companies/(?P<company>[^/]+)/jobs/")


class WttjSource(BrowserJobSource):
    code = "wttj"
    search_url = SEARCH_URL
    offer_pattern = OFFER_PATH

    def company_of(self, url: str, page: Page) -> str:
        named = super().company_of(url, page)
        if named:
            return named
        match = OFFER_COMPANY.search(url)
        if not match:
            return ""
        for href, text in page.links:
            link = COMPANY_PATH.search(href.split("?", 1)[0])
            if text and link and link["company"] == match["company"]:
                return text
        return match["company"].replace("-", " ").title()

"""Sources de collecte d'offres (vague 5) : France Travail (API officielle) et Malt / WTTJ (navigateur, lecture seule).

LinkedIn reste collecté par ``scraper.LinkedInScraper`` via ``pipeline.scrape`` ; Indeed n'a aucune collecte
(import manuel, R-6). Ce paquet est la couche intégrations : il n'importe ni FastAPI ni le pipeline.
"""

from __future__ import annotations

from .base import ID_PREFIXES, LABELS, BrowserJobSource, SourceFetchError, label_of, source_of_ref
from .france_travail import FranceTravailSource
from .malt import MaltSource
from .wttj import WttjSource

# Sources collectées par un navigateur piloté (consentement I-16, repos de 6 h par source).
BROWSER_SOURCES: dict[str, type[BrowserJobSource]] = {"malt": MaltSource, "wttj": WttjSource}

__all__ = [
    "BROWSER_SOURCES",
    "ID_PREFIXES",
    "LABELS",
    "BrowserJobSource",
    "FranceTravailSource",
    "MaltSource",
    "SourceFetchError",
    "WttjSource",
    "france_travail_source",
    "label_of",
    "source_of_ref",
]


def france_travail_source() -> FranceTravailSource:
    """Client France Travail construit depuis l'environnement (``SourceFetchError`` si les identifiants manquent).

    Point d'injection des tests : ils remplacent cette fonction par une fabrique qui fournit un transport factice.
    """
    from .. import config

    client_id, client_secret = config.france_travail_credentials()
    if not (client_id and client_secret):
        raise SourceFetchError(
            "Identifiants France Travail absents : renseigne FRANCE_TRAVAIL_CLIENT_ID et "
            "FRANCE_TRAVAIL_CLIENT_SECRET dans le fichier .env."
        )
    return FranceTravailSource(client_id, client_secret)

"""API France Travail « Offres d'emploi v2 » (source ``france_travail``, mode ``api_officielle``).

API officielle : aucun risque de compte, aucun navigateur, aucune popup. Le client est synchrone (``httpx``) et
appelé dans la tâche ``run``. Débit : 0,5 s au moins entre deux appels (≤ 2 appels/s, jeton compris) ; délai de 10 s ;
3 essais au total (attentes de 1 s puis 2 s) sur ``429``, ``500``, ``502``, ``503``, ``504`` et erreurs réseau ;
``Retry-After`` remplace l'attente s'il vaut 30 s ou moins, sinon abandon immédiat. Un ``401`` renouvelle le jeton et
accorde une seule nouvelle tentative, non comptée. Le jeton reste en mémoire ; ni lui ni le secret ne sont journalisés
(I-15). Le client, le transport, ``sleep`` et l'horloge sont injectables pour les tests hors ligne.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime

import httpx

from .. import config
from ..models import Item, SearchRequest
from .base import ID_PREFIXES, SourceFetchError

log = logging.getLogger(__name__)

CODE = "france_travail"
TOKEN_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"
SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
SCOPE = "api_offresdemploiv2 o2dsoffre"
TIMEOUT_SECONDS = 10.0
MIN_INTERVAL_SECONDS = 0.5  # ≤ 2 appels par seconde
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (1.0, 2.0)
MAX_RETRY_AFTER_SECONDS = 30.0
TOKEN_MARGIN_SECONDS = 60.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RANGE = 150  # offres par appel (limite de l'API)
PUBLISHED_SINCE_DAYS = 31

# Code de département : 01-95 (métropole), 2A/2B (Corse), 971-976 (outre-mer).
_DEPARTMENT = r"(?:0[1-9]|[1-8]\d|9[0-5]|2[AB]|97[1-6])"
_BARE_DEPARTMENT = re.compile(rf"^{_DEPARTMENT}$", re.IGNORECASE)
_PAREN_DEPARTMENT = re.compile(rf"\(\s*({_DEPARTMENT})\s*\)\s*$", re.IGNORECASE)

Sleep = Callable[[float], None]
Clock = Callable[[], float]


def department_of(location: str) -> str | None:
    """Code de département de ``location`` (``75``, ``2A``, ``Lyon (69)``), sinon ``None``."""
    text = (location or "").strip()
    if _BARE_DEPARTMENT.match(text):
        return text.upper()
    found = _PAREN_DEPARTMENT.search(text)
    return found.group(1).upper() if found else None


def configured() -> bool:
    """Identifiants présents dans l'environnement (``FRANCE_TRAVAIL_CLIENT_ID`` et ``..._SECRET``)."""
    return config.france_travail_configured()


class FranceTravailSource:
    """Recherche d'offres France Travail pour une analyse (un seul appel de recherche en pratique)."""

    code = CODE

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._client = client or httpx.Client(timeout=TIMEOUT_SECONDS, transport=transport)
        self._owns_client = client is None
        self._sleep, self._clock = sleep, clock
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._last_call: float | None = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> FranceTravailSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ HTTP
    def _throttle(self) -> None:
        """Garantit 0,5 s au moins entre deux appels (jeton compris)."""
        if self._last_call is not None:
            wait = MIN_INTERVAL_SECONDS - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def _send(self, send: Callable[[], httpx.Response], action: str) -> httpx.Response:
        """Appel avec cadence et relances ; renvoie la réponse (``401`` comprise) ou lève ``SourceFetchError``."""
        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            failure = ""
            try:
                response = send()
            except httpx.HTTPError as exc:
                failure = f"{action} : France Travail ne répond pas ({exc.__class__.__name__})."
                response = None
            if response is not None:
                if response.status_code not in RETRY_STATUSES:
                    return response
                failure = f"{action} : France Travail a répondu {response.status_code}."
            if attempt >= MAX_ATTEMPTS:
                raise SourceFetchError(f"{failure} {attempt} essais sans succès : réessaie plus tard.")
            wait = BACKOFF_SECONDS[attempt - 1]
            retry_after = _retry_after(response)
            if retry_after is not None:
                if retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise SourceFetchError(
                        f"{failure} France Travail demande d'attendre {retry_after:.0f} s : réessaie plus tard."
                    )
                wait = retry_after
            log.warning("%s Nouvel essai dans %.0f s (essai %d sur %d).", failure, wait, attempt + 1, MAX_ATTEMPTS)
            self._sleep(wait)

    def _fetch_token(self) -> str:
        response = self._send(
            lambda: self._client.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": SCOPE,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                timeout=TIMEOUT_SECONDS,
            ),
            "Connexion à l'API",
        )
        if response.status_code != 200:
            raise SourceFetchError(
                f"Connexion à l'API : France Travail a refusé la demande de jeton ({response.status_code}) : "
                "vérifie FRANCE_TRAVAIL_CLIENT_ID et FRANCE_TRAVAIL_CLIENT_SECRET dans le fichier .env."
            )
        try:
            payload = response.json()
            token = str(payload["access_token"])
            expires_in = float(payload.get("expires_in", 0))
        except ValueError, KeyError, TypeError:
            raise SourceFetchError("Connexion à l'API : réponse de France Travail illisible (jeton absent).") from None
        self._token = token
        self._token_expires_at = self._clock() + max(0.0, expires_in - TOKEN_MARGIN_SECONDS)
        return token

    def _bearer(self) -> str:
        if self._token is None or self._clock() >= self._token_expires_at:
            return self._fetch_token()
        return self._token

    def _search_once(self, params: dict[str, str | int]) -> httpx.Response:
        token = self._bearer()  # avant l'appel : le jeton et la recherche sont deux appels espacés de 0,5 s
        return self._send(
            lambda: self._client.get(
                SEARCH_URL,
                params=params,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=TIMEOUT_SECONDS,
            ),
            "Recherche d'offres",
        )

    def _search(self, params: dict[str, str | int]) -> httpx.Response:
        response = self._search_once(params)
        if response.status_code == 401:
            # Jeton expiré ou révoqué : un renouvellement puis une seule nouvelle tentative (non comptée).
            log.info("Jeton France Travail refusé : renouvellement puis une nouvelle tentative.")
            self._token = None
            response = self._search_once(params)
            if response.status_code == 401:
                raise SourceFetchError(
                    "Recherche d'offres : France Travail refuse le jeton renouvelé (401) : vérifie l'abonnement de "
                    "ton application à l'API « Offres d'emploi v2 »."
                )
        return response

    # --------------------------------------------------------------- collecte
    def fetch(self, request: SearchRequest) -> list[Item]:
        """Offres publiées depuis 31 jours pour le mot-clé, ``max_items`` au plus (un seul appel de recherche)."""
        count = max(1, min(request.max_items, MAX_RANGE))
        params: dict[str, str | int] = {
            "motsCles": request.keyword,
            "publieeDepuis": PUBLISHED_SINCE_DAYS,
            "range": f"0-{count - 1}",
            "sort": 1,
        }
        department = department_of(request.location)
        if department:
            params["departement"] = department
        response = self._search(params)
        if response.status_code == 204:
            log.info("France Travail : aucune offre pour ce mot-clé.")
            return []
        if response.status_code == 400:
            raise SourceFetchError(
                "Recherche d'offres : France Travail a refusé les paramètres de la recherche (400) : "
                "modifie le mot-clé ou le lieu."
            )
        if response.status_code not in (200, 206):
            raise SourceFetchError(f"Recherche d'offres : France Travail a répondu {response.status_code}.")
        try:
            rows = response.json().get("resultats") or []
        except ValueError, AttributeError:
            raise SourceFetchError("Recherche d'offres : réponse de France Travail illisible.") from None
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        items = _items(rows, request, count)
        log.info("France Travail : %d offres lues%s.", len(items), f" sur {total} disponibles" if total else "")
        return items


def _retry_after(response: httpx.Response | None) -> float | None:
    """``Retry-After`` en secondes (la forme date HTTP est ignorée : repli sur l'attente habituelle)."""
    if response is None:
        return None
    try:
        value = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return max(0.0, value)


def _items(rows: list[dict], request: SearchRequest, limit: int) -> list[Item]:
    now = datetime.now(UTC)
    items: dict[str, Item] = {}
    for row in rows:
        offer_id = str(row.get("id") or "").strip()
        if not offer_id or len(items) >= limit:
            continue
        skills = [str(s.get("libelle", "")).strip() for s in row.get("competences") or [] if s.get("libelle")]
        text = str(row.get("description") or "").strip()
        if skills:
            text = f"{text}\n\nCompétences : {', '.join(skills)}".strip()
        origin = row.get("origineOffre") or {}
        external_id = f"{ID_PREFIXES[CODE]}{offer_id}"
        items[external_id] = Item(
            kind="job",
            id=f"job:{external_id}",
            url=str(origin.get("urlOrigine") or "")
            or f"https://candidat.francetravail.fr/offres/recherche/detail/{offer_id}",
            title=str(row.get("intitule") or "").strip()[:150],
            text=text,
            keyword=request.keyword,
            company=str((row.get("entreprise") or {}).get("nom") or "").strip()[:100],
            location=str((row.get("lieuTravail") or {}).get("libelle") or "").strip()[:120],
            posted_at=str(row.get("dateCreation") or "")[:10],
            scraped_at=now.isoformat(),
            source_code=CODE,
        )
    return list(items.values())
